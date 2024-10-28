"""Model for the temporal networks."""
from typing import Tuple

import numpy as np
import torch
from torch import nn

from tpg.models import BaseGRUNetwork
from tpg.utils import gumbel_softmax_sample


class TemporalGRUNetwork(BaseGRUNetwork):
    """The Temporal GRU Network class."""

    def __init__(self, *args, **kwargs):
        """
        Network for analysing the understanding of temporal progression in emergent communication.

        Will use the same arguments as BaseGRU Network.
        """
        super().__init__(*args, **kwargs)

        # Agent 1
        # LSTM to extract temporal relationships.
        # E.g. do certain objects come more often together?
        # Do they repeat?
        self.sender_temporal_gru = nn.GRU(
            input_size=(self.seq_window * 2) + 1,
            hidden_size=self.sender_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Agent 2
        # LSTM to process messages temporally
        self.receiver_temporal_gru = nn.GRU(
            input_size=self.receiver_hidden,
            hidden_size=self.receiver_hidden,
            num_layers=1,
            batch_first=True,
        )

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
        # We can play with what the meaning GRU gets and see the convergence change significantly.
        # Depending on the input shape we may not need to unsqueeze.
        _, embedding_meaning_object = self.sender_meaning_gru(cut_inputs.unsqueeze(-1))
        embedding_meaning_object = embedding_meaning_object.squeeze()

        # Allow for temporal understanding by different batching.
        # Notice the difference in unsqueeze calls:
        # In the first case (meaning) we pass in the batch in form [128,1,4] for one target
        # So the meaning GRU processes 128 sequences of length 1.
        # Below we change to [1,128,4]. So the GRU processes one continuous sequence
        # to understand the relationships within.
        embedding_meaning_temporal, _ = self.sender_temporal_gru(
            cut_inputs.unsqueeze(0)
        )

        # Let's get it back into a batch shape. The return does not have batch first etc.
        embedding_meaning_temporal = embedding_meaning_temporal.permute(1, 0, 2)
        embedding_meaning_temporal = embedding_meaning_temporal.reshape(
            cut_inputs.shape[0], -1
        )
        embedded_meaning_together = torch.mul(
            embedding_meaning_temporal, embedding_meaning_object
        )

        # Temporary holder for generated probabilities over vocabulary space.
        # The probabilities are generated with the Gumbel-Softmax trick from EGG.
        sequence = []
        # Pre-seed the hidden state of the LSTM with the embedding and temporal understanding.
        prev_hidden = embedded_meaning_together
        # Start of sentence embedding is passed first. This follows from how EGG does this.
        character_to_process = torch.stack([self.sos_embedding] * full_sequence.size(0))

        # Let's generate the message character by character to use Gumbel-Softmax after each character.
        # This will allow for later discretisation of the messages.
        for step in range(self.max_length):
            h_t = self.message_gru_sender(character_to_process, prev_hidden)
            # Process the GRU hidden state into vocab size.
            step_logits = self.gru_to_msg(h_t)
            # Here we generate the character probabilities using the Gumbel-Softmax trick.
            character = gumbel_softmax_sample(
                logits=step_logits,
                training=self.training,
                temperature=self.gs_temperature,
            )
            # Use the resulting hidden state.
            prev_hidden = h_t
            # Process the character back into an embedding for the GRU.
            character_to_process = self.embedding(character)
            # Append character to create a message later.
            sequence.append(character)

        # Create a message from all appended characters, and permute back into batch shape.
        message = torch.stack(sequence).permute(1, 0, 2)

        # Now we move onto the second agent.
        # Agent 2
        # Process message and get the last hidden state for each message.
        # We don't care about per-character hidden states in this case.
        _, message_decoded = self.receiver_message_gru(message)

        # Process the messages through time
        message_temporal, _ = self.receiver_temporal_gru(message_decoded)
        message_decoded_together = torch.mul(message_decoded, message_temporal)

        # Permute and reshape back to batch shape.
        message_decoded_together = message_decoded_together.permute(1, 0, 2)
        message_decoded_together = message_decoded_together.reshape(
            full_sequence.shape[0], -1
        )

        # Process message together with sequence
        # Pre-seed the hidden state of the GRU with the embedding and temporal understanding.
        prev_hidden_r = message_decoded_together.unsqueeze(0)
        full_sequence = full_sequence.unsqueeze(-1)
        _, msg_seq_tgt = self.receiver_seq_msg_process(full_sequence, prev_hidden_r)

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
