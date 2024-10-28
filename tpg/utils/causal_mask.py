"""Generation of causal masks for attention layers."""

import torch


def get_causal_mask(target_length: int) -> torch.Tensor:
    """
    Generate an attention mask.

    Parameters
    ----------
    target_length: int
        Length of sequence to mask

    Returns
    -------
    Attention mask
    """
    attn_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1)
    attn_mask = attn_mask.masked_fill(attn_mask == 1, True).bool()

    return attn_mask
