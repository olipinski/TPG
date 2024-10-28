"""Model for the base networks."""
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tpg.utils import gumbel_softmax_sample


class BaseLSTMNetwork(L.LightningModule):
    """The BaseLSTM Network class."""

    def __init__(
        self,
        seq_length: int = 100,
        seq_window: int = 2,
        num_distractors: int = 3,
        vocab_size: int = 5,
        max_length: int = 6,
        gs_temperature: float = 1.0,
        sender_hidden: int = 64,
        receiver_hidden: int = 64,
        length_penalty: float = 0.001,
        eos_char: int = 0,
    ):
        """
        Network for analysing the understanding of temporal progression in emergent communication.

        Parameters
        ----------
        seq_length: int
            The length of the sequence for the receiver.
        seq_window: int
            Size of the sequence window for the sender.
        num_distractors: int
            Number of distractors for the receiver.
        vocab_size: int
            Vocabulary size for the agents.
        max_length: int
            Maximum length of the sender message.
        gs_temperature: float
            Gumbel-Softmax temperature.
        sender_hidden: int
            Size of the sender hidden units. This will apply to all parts of the sender network.
        receiver_hidden: int
            Size of the receiver hidden units. This will apply to all parts of the receiver network.
        length_penalty: float
            Length penalty when evaluating linguistic parsimony.
        eos_char: int
            End-of-sentence character.
        """
        super().__init__()

        self.seq_length = seq_length
        self.seq_window = seq_window
        self.num_distractors = num_distractors
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.gs_temperature = gs_temperature
        self.sender_hidden = sender_hidden
        self.receiver_hidden = receiver_hidden

        # Linguistic parsimony pressure
        self.eos_char = eos_char
        # Values higher than 0.001 make training unstable
        self.length_penalty = length_penalty

        # Agent 1
        # LSTM to extract the position and neighbours of the requested number.
        self.sender_meaning_lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.sender_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Gumbel-Softmax related stuff, adapted from EGG
        self.sos_embedding = nn.Parameter(torch.zeros(self.sender_hidden))
        self.embedding = nn.Linear(
            in_features=self.vocab_size, out_features=self.sender_hidden
        )
        self.message_lstm_sender = nn.LSTMCell(
            input_size=self.sender_hidden, hidden_size=self.sender_hidden
        )
        self.lstm_to_msg = nn.Linear(
            in_features=self.sender_hidden, out_features=self.vocab_size
        )

        # Agent 2
        # LSTM to process the incoming message.
        self.receiver_message_lstm = nn.LSTM(
            input_size=self.vocab_size,
            hidden_size=self.receiver_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Network to process sequence together with message embedding
        # Currently set to 1 for single float/ints
        self.receiver_seq_msg_process = nn.LSTM(
            input_size=1,
            hidden_size=self.receiver_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Network to embed objects.
        # Currently set to 1 for single float/ints
        self.receiver_obj_embed = nn.Linear(1, self.receiver_hidden)

        # Dictionary for saving interactions and other statistics.
        self.exchange_dict = {}
        self.exchange_count = 0
        self.eval_mode = False

        self.save_hyperparameters()

    def infer(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the target integer for a given batch.

        This is a convenience function which is used in all the steps for PyTorch Lightning to save on duplicate code.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Batch to process, consisting of the sequence, inputs for sender, targets and distractors, and target labels.

        Returns
        -------
        Tuple(guess, message):
            Tuple of integer label guess, and the message.
        """
        # Unpack the tuple.

        full_sequence, cut_inputs, tds, target_labels = batch

        # Cast the inputs from int to float for learning.
        full_sequence = full_sequence.float()
        cut_inputs = cut_inputs.float()
        tds = tds.float()

        # Cast the labels from int to long, pytorch requires longs.
        target_labels = target_labels.long()

        # We start in the sender agent.
        # Agent 1
        # We can play with what the meaning LSTM gets and see the convergence change significantly.
        # Depending on the input shape we may not need to unsqueeze.
        _, (embedding_meaning_object, _) = self.sender_meaning_lstm(
            cut_inputs.unsqueeze(-1)
        )
        embedding_meaning_object = embedding_meaning_object.squeeze()

        # Temporary holder for generated probabilities over vocabulary space.
        # The probabilities are generated with the Gumbel-Softmax trick from EGG.
        sequence = []
        # Pre-seed the hidden state of the LSTM with the embedding understanding.
        prev_hidden = embedding_meaning_object
        # The previous cell state is all zeros. We need to provide a cell state if we provide the hidden state.
        prev_c = torch.zeros_like(prev_hidden)
        # Start of sentence embedding is passed first. This follows from how EGG does this.
        character_to_process = torch.stack([self.sos_embedding] * full_sequence.size(0))

        # Let's generate the message character by character to use Gumbel-Softmax after each character.
        # This will allow for later discretisation of the messages.
        for step in range(self.max_length):
            h_t, prev_c = self.message_lstm_sender(
                character_to_process, (prev_hidden, prev_c)
            )
            # Process the LSTM hidden state into vocab size.
            step_logits = self.lstm_to_msg(h_t)
            # Here we generate the character probabilities using the Gumbel-Softmax trick.
            character = gumbel_softmax_sample(
                logits=step_logits,
                training=self.training,
                temperature=self.gs_temperature,
            )
            # Use the resulting hidden state.
            prev_hidden = h_t
            # Process the character back into an embedding for the LSTM.
            character_to_process = self.embedding(character)
            # Append character to create a message later.
            sequence.append(character)

        # Create a message from all appended characters, and permute back into batch shape.
        message = torch.stack(sequence).permute(1, 0, 2)

        # Now we move onto the second agent.
        # Agent 2
        # Process message and get the last hidden state for each message.
        # We don't care about per-character hidden states in this case.
        _, (message_decoded, _) = self.receiver_message_lstm(message)

        # Permute and reshape back to batch shape.
        message_decoded = message_decoded.permute(1, 0, 2)
        message_decoded = message_decoded.reshape(full_sequence.shape[0], -1)

        # Process message together with sequence
        # Pre-seed the hidden state of the LSTM with the embedding.
        prev_hidden_r = message_decoded.unsqueeze(0)
        # The previous cell state is all zeros. We need to provide a cell state if we provide the hidden state.
        prev_c_r = torch.zeros_like(prev_hidden_r)
        full_sequence = full_sequence.unsqueeze(-1)
        _, (msg_seq_tgt, _) = self.receiver_seq_msg_process(
            full_sequence, (prev_hidden_r, prev_c_r)
        )

        # Embed the objects to an embedding space.
        embedding_obj = self.receiver_obj_embed(tds.unsqueeze(dim=-1)).relu()
        # Produce guess by using the two embeddings and multiplying.
        guess = torch.matmul(embedding_obj, msg_seq_tgt.permute(1, 2, 0))
        guess = guess.squeeze()

        # In evaluation mode we want to save as much info as possible.
        # Using a dict will make it easier to export to json later.
        # Extending this also is just adding a key-value pair.
        if self.eval_mode:
            for i in range(full_sequence.size(0)):
                self.exchange_dict[f"exchange_{self.exchange_count}"] = {
                    "sequence": full_sequence[i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32),
                    "cut_inputs": cut_inputs[i].detach().cpu().numpy().astype(np.int32),
                    "tds": tds[i].detach().cpu().numpy().astype(np.int32),
                    "message": message[i].argmax(dim=1).detach().cpu().numpy(),
                    "guess": guess.argmax(dim=1)[i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32),
                    "target": tds[i][target_labels[i]]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32),
                    "target_id": target_labels[i]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32),
                }
                self.exchange_count += 1

        return guess, message

    def loss_accuracy(
        self,
        guess: torch.Tensor,
        message: torch.Tensor,
        target_labels: torch.Tensor,
    ):
        """
        Calculate the loss and accuracy.

        Loss includes the integer guess, and the length penalty.
        Uses the cross entropy loss.

        Parameters
        ----------
        guess: torch.Tensor
        message: torch.Tensor
        target_labels: torch.Tensor

        Returns
        -------
        Tuple(loss, accuracy):
            Tuple of loss, accuracy.

        """
        target_labels = target_labels.long().squeeze()

        loss = F.cross_entropy(guess, target_labels)
        accuracy = (guess.argmax(dim=1) == target_labels).detach().float().mean()

        # Length cost
        # In EGG the loss is calculated per step, so we can fake this by creating
        # a per step loss.

        expected_length = 0.0
        step_loss = loss / message.shape[1]
        length_loss = 0
        eos_val_mask = torch.ones(message.shape[0], device=self.device)
        pos = 0
        for pos in range(message.shape[1]):
            eos_mask = message[:, pos, self.eos_char]
            add_mask = eos_mask * eos_val_mask
            length_loss += (
                step_loss * add_mask + self.length_penalty * (1.0 + pos) * add_mask
            )
            expected_length += add_mask.detach() * (1.0 + pos)
            eos_val_mask = eos_val_mask * (1.0 - eos_mask)

        length_loss += (
            step_loss * eos_val_mask + self.length_penalty * (pos + 1.0) * eos_val_mask
        )
        expected_length += (pos + 1) * eos_val_mask

        if self.length_penalty > 0:
            loss = torch.mean(length_loss)

        expected_length = torch.mean(expected_length)

        return loss, accuracy, expected_length

    def forward(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Predict the missing integer from the sequence.

        This function is used only for post-training inferences. Currently, this function still requires a
        batch with shape like in training, so is not a true inference function.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            atch to process, consisting of the sequence, inputs for sender, targets and distractors, and target labels.

        Returns
        -------
        Tuple(guess, message):
            Tuple of integer label guess, and the message.
        """
        self.eval_mode = True
        guess, message = self.infer(batch)
        guess = guess.argmax(dim=1)
        return guess, message

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        """
        Run the training for a single batch, with a given batch id.

        This is overridden from PyTorch Lightning.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Batch to process, consisting of the sequence, inputs for sender, targets and distractors, and target labels.
        batch_idx: int
            ID of the batch.

        Returns
        -------
        loss: torch.Tensor
            The loss for a given batch.
        """
        _, _, _, target_labels = batch
        guess, message = self.infer(batch)
        loss, acc, expected_length = self.loss_accuracy(guess, message, target_labels)
        values = {
            "train_loss": loss,
            "train_acc": acc,
            "expected_length": expected_length,
        }
        self.log_dict(values, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        """
        Run the validation for a single batch, with a given batch id.

        This function does not return anything, just logs the loss and accuracies to Lightning.

        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Batch to process, consisting of the sequence, inputs for sender, targets and distractors, and target labels.
        batch_idx: int
            ID of the batch.
        """
        _, _, _, target_labels = batch
        guess, message = self.infer(batch)
        loss, acc, expected_length = self.loss_accuracy(guess, message, target_labels)
        values = {"val_loss": loss, "val_acc": acc, "expected_length": expected_length}
        self.log_dict(values, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizers to be used for the training.

        Returns
        -------
        optimizer: torch.optim.Optimizer
            Optimizer to be used for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
