"""Deterministic nonlinear-moment ETKF baseline for binary SAR footprints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .sar import binary_sar_probability


@dataclass(frozen=True)
class BinarySARETKFResult:
    analysis_depth_m: Tensor
    member_mass_change_m3: Tensor
    prior_probability: Tensor
    analysis_probability: Tensor
    observation_conditional_variance: Tensor


def binary_sar_etkf(
    forecast_depth_m: Tensor,
    observation_class: Tensor,
    footprint_operator: Tensor,
    cell_area_m2: Tensor,
    *,
    wet_depth_m: float = 0.05,
    transition_depth_m: float = 0.02,
    sensitivity: float = 0.95,
    false_positive_rate: float = 0.02,
    conditional_variance_floor: float = 1.0e-6,
) -> BinarySARETKFResult:
    """Assimilate one SAR scene in ensemble space.

    The ensemble anomalies supply state--observation covariance. The Bernoulli
    conditional variance supplies ``R``. All inverses are at most M x M, where
    M is the ensemble size.
    """

    if forecast_depth_m.ndim != 2 or forecast_depth_m.shape[0] < 3:
        raise ValueError("forecast_depth_m must have shape [members,nodes], M >= 3")
    members, nodes = forecast_depth_m.shape
    if cell_area_m2.shape != (nodes,) or torch.any(cell_area_m2 <= 0.0):
        raise ValueError("cell_area_m2 must be positive and match the mesh")
    if observation_class.ndim != 1 or torch.any(
        (observation_class != 0.0) & (observation_class != 1.0)
    ):
        raise ValueError("observation_class must be a binary pixel vector")

    def observe(depth: Tensor) -> Tensor:
        return binary_sar_probability(
            depth,
            footprint_operator,
            wet_depth_m=wet_depth_m,
            transition_depth_m=transition_depth_m,
            sensitivity=sensitivity,
            false_positive_rate=false_positive_rate,
        )

    predicted = observe(forecast_depth_m)
    if predicted.shape != (members, observation_class.numel()):
        raise ValueError("observation and footprint shapes are inconsistent")
    state_mean = forecast_depth_m.mean(dim=0)
    state_anomaly = (forecast_depth_m - state_mean) / (members - 1) ** 0.5
    predicted_mean = predicted.mean(dim=0)
    observation_anomaly = (predicted - predicted_mean) / (members - 1) ** 0.5
    conditional_variance = (
        predicted * (1.0 - predicted)
    ).mean(dim=0).clamp_min(conditional_variance_floor)

    whitened = observation_anomaly / conditional_variance.sqrt()[None]
    system = torch.eye(
        members, dtype=forecast_depth_m.dtype, device=forecast_depth_m.device
    ) + whitened @ whitened.T
    rhs = observation_anomaly @ (
        (observation_class - predicted_mean) / conditional_variance
    )
    eigenvalue, eigenvector = torch.linalg.eigh(system)
    eigenvalue = eigenvalue.clamp_min(torch.finfo(eigenvalue.dtype).eps)
    inverse = (eigenvector * eigenvalue.reciprocal()[None]) @ eigenvector.T
    inverse_sqrt = (eigenvector * eigenvalue.rsqrt()[None]) @ eigenvector.T

    analysis_mean = state_mean + state_anomaly.T @ (inverse @ rhs)
    analysis_anomaly = (
        (members - 1) ** 0.5 * state_anomaly.T @ inverse_sqrt
    ).T
    analysis = (analysis_mean[None] + analysis_anomaly).clamp_min(0.0)
    prior_mass = (forecast_depth_m * cell_area_m2[None]).sum(dim=1)
    analysis_mass = (analysis * cell_area_m2[None]).sum(dim=1)
    return BinarySARETKFResult(
        analysis_depth_m=analysis,
        member_mass_change_m3=analysis_mass - prior_mass,
        prior_probability=predicted,
        analysis_probability=observe(analysis),
        observation_conditional_variance=conditional_variance,
    )
