"""Model for the temporal networks."""
from typing import Tuple

import numpy as np
import torch
from torch import nn

from tpg.models.base_hybrid import BaseHybridNetwork
from tpg.utils import PositionalEncoding, get_causal_mask, gumbel_softmax_sample


class TemporalAttentionHybridNetwork(BaseHybridNetwork):
    """The Temporal Attention Hybrid Network class."""

    def __init__(
        self,
        attention_sender_n_heads: int = 8,
        attention_sender_dropout: float = 0.1,
        attention_receiver_n_heads: int = 8,
        attention_receiver_dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        """
        Network for analysing the understanding of temporal progression in emergent communication.

        Parameters
        ----------
        attention_sender_n_heads:
            Number of attention heads for the sender.
        attention_sender_dropout
            Dropout value for the sender attention.
        attention_receiver_n_heads
            Number of attention heads for the receiver.
        attention_receiver_dropout
            Dropout value for the receiver attention.
        """
        super().__init__(*args, **kwargs)

        self.attention_sender_n_heads = attention_sender_n_heads
        self.attention_sender_dim = self.sender_hidden
        self.attention_sender_dropout = attention_sender_dropout
        self.attention_receiver_n_heads = attention_receiver_n_heads
        self.attention_receiver_dim = self.receiver_hidden
        self.attention_receiver_dropout = attention_receiver_dropout

        # Agent 1
        # Attention layer after GRU
        self.sender_pos_encoding_layer = PositionalEncoding(
            d_model=self.attention_sender_dim, dropout=self.attention_sender_dropout
        )
        self.sender_attention_embedding_layer = nn.Linear(
            self.sender_hidden, self.attention_sender_dim
        )
        self.sender_attention_layer = nn.MultiheadAttention(
            num_heads=self.attention_sender_n_heads,
            embed_dim=self.attention_sender_dim,
            dropout=self.attention_sender_dropout,
        )

        # Agent 2
        # Attention Layer to process the output of the GRU
        self.receiver_pos_encoding_layer = PositionalEncoding(
            d_model=self.attention_receiver_dim, dropout=self.attention_receiver_dropout
        )
        self.receiver_attention_embedding_layer = nn.Linear(
            self.receiver_hidden, self.attention_receiver_dim
        )
        self.receiver_attention_layer = nn.MultiheadAttention(
            num_heads=self.attention_receiver_n_heads,
            embed_dim=self.attention_receiver_dim,
            dropout=self.attention_receiver_dropout,
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
        _, (embedding_meaning_object, _) = self.sender_meaning_lstm(
            cut_inputs.unsqueeze(-1)
        )
        embedding_meaning_object = embedding_meaning_object.squeeze()

        # Embed targets into the attention space, and pass through multi-head attention
        # Pos encoding: [seq_len, batch_size, embedding_dim]
        attn_mask = get_causal_mask(embedding_meaning_object.shape[0]).to(self.device)
        meaning_embedded = self.sender_attention_embedding_layer(
            embedding_meaning_object
        )
        meaning_pos_encoded = self.sender_pos_encoding_layer(
            meaning_embedded.unsqueeze(1)
        ).squeeze()
        attention_meaning, _ = self.sender_attention_layer(
            meaning_pos_encoded,
            meaning_pos_encoded,
            meaning_pos_encoded,
            attn_mask=attn_mask,
            need_weights=False,
        )
        embedding_meaning_object = torch.mul(
            embedding_meaning_object, attention_meaning
        )

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

        # Pos encoding: [seq_len, batch_size, embedding_dim]
        message_embedded = self.receiver_attention_embedding_layer(message_decoded)
        message_pos_encoded = self.receiver_pos_encoding_layer(
            message_embedded.permute(1, 0, 2)
        ).permute(1, 0, 2)
        message_decoded_attention, _ = self.receiver_attention_layer(
            message_pos_encoded,
            message_pos_encoded,
            message_pos_encoded,
            need_weights=False,
        )
        message_decoded = torch.mul(message_decoded, message_decoded_attention)

        # Permute and reshape back to batch shape.
        message_decoded = message_decoded.permute(1, 0, 2)
        message_decoded = message_decoded.reshape(full_sequence.shape[0], -1)

        # Process message together with sequence
        # Pre-seed the hidden state of the GRU with the embedding.
        prev_hidden_r = message_decoded.unsqueeze(0)
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
