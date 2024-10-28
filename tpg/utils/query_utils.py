"""Helper functions to query models."""
import lightning as L
import torch
import torch.nn.functional as F


def query_agent(
    full_sequence: torch.Tensor,
    tds: torch.Tensor,
    message: torch.Tensor,
    model: L.LightningModule,
    agent: str,
) -> (torch.Tensor, torch.Tensor):
    """
    Query agent, based on the model used.

    Parameters
    ----------
    full_sequence: torch.Tensor
    tds: torch.Tensor
    message: torch.Tensor
    model: L.LightningModule
    agent: str

    Returns
    -------
    guess: torch.Tensor
    message: torch.Tensor

    """
    if "Base" in agent:
        # Skip unnecessary stuff
        if "OHV" in agent:
            message = F.one_hot(message, model.hparams["seq_length"])
            full_sequence = F.one_hot(full_sequence, model.hparams["seq_length"])
            tds = F.one_hot(tds, model.hparams["seq_length"])
        else:
            message = message.to(dtype=torch.float32)
            full_sequence = full_sequence.to(dtype=torch.float32)
            tds = tds.to(dtype=torch.float32)

        if "GRU" in agent:  # Base GRU
            _, message_decoded = model.receiver_message_gru(message)
        elif "LSTM" in agent or "Hybrid" in agent:  # BaseHybrid or BaseLSTM
            _, (
                message_decoded,
                _,
            ) = model.receiver_message_lstm(message)
        else:
            raise NotImplementedError("Unknown architecture!")

        message_decoded = message_decoded.permute(1, 0, 2)
        message_decoded = message_decoded.reshape(full_sequence.shape[0], -1)
        prev_hidden_r = message_decoded.unsqueeze(0)
        full_sequence = full_sequence.unsqueeze(-1)

        if "GRU" in agent or "Hybrid" in agent:  # BaseGRU or BaseHybrid
            _, msg_seq_tgt = model.receiver_seq_msg_process(
                full_sequence, prev_hidden_r
            )
        elif "LSTM" in agent:  # BaseLSTM
            prev_c_r = torch.zeros_like(prev_hidden_r)
            _, (
                msg_seq_tgt,
                _,
            ) = model.receiver_seq_msg_process(full_sequence, (prev_hidden_r, prev_c_r))
        else:
            raise NotImplementedError("Unknown architecture!")

        embedding_obj = model.receiver_obj_embed(tds.unsqueeze(dim=-1)).relu()
        guess = torch.matmul(embedding_obj, msg_seq_tgt.permute(1, 2, 0))
        guess = guess.squeeze()
    elif "Attention" in agent:
        if "GRU" in agent:  # Attention GRU
            _, message_decoded = model.receiver_message_gru(message)
        elif "LSTM" in agent or "Hybrid" in agent:  # AttentionHybrid or AttentionLSTM
            _, (
                message_decoded,
                _,
            ) = model.receiver_message_lstm(message)
        else:
            raise NotImplementedError("Unknown architecture!")

        message_embedded = model.receiver_attention_embedding_layer(message_decoded)
        message_pos_encoded = model.receiver_pos_encoding_layer(
            message_embedded.permute(1, 0, 2)
        ).permute(1, 0, 2)
        (
            message_decoded_attention,
            _,
        ) = model.receiver_attention_layer(
            message_pos_encoded,
            message_pos_encoded,
            message_pos_encoded,
            need_weights=False,
        )
        message_decoded = torch.mul(message_decoded, message_decoded_attention)
        message_decoded = message_decoded.permute(1, 0, 2)
        message_decoded = message_decoded.reshape(full_sequence.shape[0], -1)
        prev_hidden_r = message_decoded.unsqueeze(0)
        full_sequence = full_sequence.unsqueeze(
            -1
        )  # Reshape to [batch, seq_len, input size]

        if "GRU" in agent or "Hybrid" in agent:  # AttentionGRU or AttentionHybrid
            _, msg_seq_tgt = model.receiver_seq_msg_process(
                full_sequence, prev_hidden_r
            )
        elif "LSTM" in agent:  # AttentionLSTM
            prev_c_r = torch.zeros_like(prev_hidden_r)
            _, (
                msg_seq_tgt,
                _,
            ) = model.receiver_seq_msg_process(full_sequence, (prev_hidden_r, prev_c_r))
        else:
            raise NotImplementedError("Unknown architecture!")

        embedding_obj = model.receiver_obj_embed(tds.unsqueeze(dim=-1)).relu()
        guess = torch.matmul(embedding_obj, msg_seq_tgt.permute(1, 2, 0))
        guess = guess.squeeze()
    else:
        # TemporalGRU
        if "GRU" in agent:  # Attention GRU
            _, message_decoded = model.receiver_message_gru(message)
            (
                message_temporal,
                _,
            ) = model.receiver_temporal_gru(message_decoded)
        elif "LSTM" in agent or "Hybrid" in agent:  # AttentionHybrid or AttentionLSTM
            _, (
                message_decoded,
                _,
            ) = model.receiver_message_lstm(message)
            (
                message_temporal,
                _,
            ) = model.receiver_temporal_lstm(message_decoded)
        else:
            raise NotImplementedError("Unknown architecture!")

        message_decoded_together = torch.mul(message_decoded, message_temporal)
        message_decoded_together = message_decoded_together.permute(1, 0, 2)
        message_decoded_together = message_decoded_together.reshape(
            full_sequence.shape[0], -1
        )
        prev_hidden_r = message_decoded_together.unsqueeze(0)
        full_sequence = full_sequence.unsqueeze(-1)

        if "GRU" in agent or "Hybrid" in agent:  # AttentionGRU or AttentionHybrid
            _, msg_seq_tgt = model.receiver_seq_msg_process(
                full_sequence, prev_hidden_r
            )
        elif "LSTM" in agent:  # AttentionLSTM
            prev_c_r = torch.zeros_like(prev_hidden_r)
            _, (
                msg_seq_tgt,
                _,
            ) = model.receiver_seq_msg_process(full_sequence, (prev_hidden_r, prev_c_r))
        else:
            raise NotImplementedError("Unknown architecture!")

        embedding_obj = model.receiver_obj_embed(tds.unsqueeze(dim=-1)).relu()
        guess = torch.matmul(embedding_obj, msg_seq_tgt.permute(1, 2, 0))
        guess = guess.squeeze()

    return guess, message
