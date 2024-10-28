"""
Module containing the Gumbel-Softmax utilities.

This code is adapted from EGG: https://github.com/facebookresearch/EGG
"""

import torch
from torch.distributions import RelaxedOneHotCategorical


def gumbel_softmax_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    training: bool = True,
):
    """
    Sample from the Gumbel-Softmax distribution for discretising the communication later.

    Parameters
    ----------
    logits: torch.Tensor
    temperature: float
    training: bool
    """
    size = logits.size()

    if not training:
        indexes = logits.argmax(dim=-1)
        one_hot = torch.zeros_like(logits).view(-1, size[-1])
        one_hot.scatter_(1, indexes.view(-1, 1), 1)
        one_hot = one_hot.view(*size)
        return one_hot

    sample = RelaxedOneHotCategorical(logits=logits, temperature=temperature).rsample()

    return sample
