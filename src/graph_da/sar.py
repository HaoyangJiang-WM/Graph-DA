"""Explicit SAR footprint observation operators.

The observation remains on the raster support. ``W[s, i]`` is the fraction of
SAR pixel ``s`` represented by hydraulic control volume ``i``. The code never
pretends that a SAR class is a water-depth label at a mesh node.
"""

from __future__ import annotations

import torch
from torch import Tensor


def footprint_average(cell_value: Tensor, footprint_operator: Tensor) -> Tensor:
    """Map control-volume values ``[..., nodes]`` to SAR pixels ``[..., pixels]``."""

    if footprint_operator.ndim != 2:
        raise ValueError("footprint_operator must have shape [pixels,nodes]")
    if cell_value.shape[-1] != footprint_operator.shape[1]:
        raise ValueError("cell values and footprint operator use different nodes")
    if torch.any(footprint_operator < 0.0):
        raise ValueError("footprint weights must be nonnegative")
    row_sum = footprint_operator.sum(dim=1)
    if torch.any(row_sum <= 0.0):
        raise ValueError("every retained SAR pixel needs positive model overlap")
    normalized = footprint_operator / row_sum[:, None]
    return cell_value @ normalized.T


def wet_response(depth_m: Tensor, *, wet_depth_m: float, transition_depth_m: float) -> Tensor:
    """Smooth sub-grid wet response used by the registered OSSE observation law."""

    if wet_depth_m < 0.0 or transition_depth_m <= 0.0:
        raise ValueError("wet and transition depths must be physically valid")
    return torch.sigmoid((depth_m - wet_depth_m) / transition_depth_m)


def binary_sar_probability(
    depth_m: Tensor,
    footprint_operator: Tensor,
    *,
    wet_depth_m: float = 0.05,
    transition_depth_m: float = 0.02,
    sensitivity: float = 0.95,
    false_positive_rate: float = 0.02,
) -> Tensor:
    """Return classifier-positive probability for each member and SAR pixel.

    ``sensitivity`` and ``false_positive_rate`` define the classifier channel.
    A production experiment should replace these constants with a registered or
    independently calibrated scene-level error model.
    """

    if not (0.0 < sensitivity <= 1.0):
        raise ValueError("sensitivity must lie in (0,1]")
    if not (0.0 <= false_positive_rate < sensitivity):
        raise ValueError("false_positive_rate must lie in [0,sensitivity)")
    wet_fraction = footprint_average(
        wet_response(
            depth_m,
            wet_depth_m=wet_depth_m,
            transition_depth_m=transition_depth_m,
        ),
        footprint_operator,
    )
    return false_positive_rate + (sensitivity - false_positive_rate) * wet_fraction


def sample_binary_sar(probability: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
    """Sample a binary SAR OSSE realization from registered pixel probabilities."""

    if torch.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must lie in [0,1]")
    return torch.bernoulli(probability, generator=generator)
