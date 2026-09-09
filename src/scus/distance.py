"""Distances between paired U/S token representations (not RNA velocity)."""

import torch
import torch.nn.functional as F


def cosine_distance(u: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Return 1-cosine along the last axis, clipped to [0, 2].

    Uses float32 and PyTorch's epsilon normalization: a zero vector has
    similarity zero and hence distance one. Missing tokens must be masked
    separately; a distance of zero denotes a valid identical representation.
    """
    return (1 - (F.normalize(u.float(), dim=-1) * F.normalize(s.float(), dim=-1)).sum(-1)).clamp(0, 2)


def us_distance(hidden: torch.Tensor, value_bins: torch.Tensor):
    """Return (distances, valid_pairs) for interleaved U,S tokens.

    hidden: [B, 2G, D], value_bins: [B, 2G]. Both expression bins must be
    positive for a pair to be valid. Invalid entries are NaN, not zero.
    Inputs should contain observed bins, never mask tokens.
    """
    if hidden.ndim != 3 or value_bins.shape != hidden.shape[:2] or hidden.shape[1] % 2:
        raise ValueError('Expected hidden [B, 2G, D] and value_bins [B, 2G]')
    valid = (value_bins[:, 0::2] > 0) & (value_bins[:, 1::2] > 0)
    distance = cosine_distance(hidden[:, 0::2], hidden[:, 1::2])
    return distance.masked_fill(~valid, float('nan')), valid
