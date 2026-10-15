"""Graph-conditioned stochastic completion of simulator restart variables.

The currently defensible decomposition is

    p(s, theta | y, c) = p(theta | s, c) p(s | y, c),

when SAR is conditionally independent of latent restart variables ``theta``
given the hydraulic state ``s``. A registered Bayesian/ensemble step updates
``s``; this module learns only the conditional completion ``p(theta | s, c)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RestartSamples:
    parameter_coordinate: Tensor
    bankfull_coordinate: Tensor

    @property
    def joint_coordinate(self) -> Tensor:
        return torch.cat((self.parameter_coordinate, self.bankfull_coordinate), dim=-1)


class _GraphBlock(nn.Module):
    def __init__(self, width: int, edge_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * width + edge_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.update = nn.Sequential(
            nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, node: Tensor, edge_index: Tensor, edge_feature: Tensor) -> Tensor:
        source, destination = edge_index
        edge = edge_feature[None].expand(node.shape[0], -1, -1)
        forward = self.message(torch.cat((node[:, source], node[:, destination], edge), -1))
        backward = self.message(torch.cat((node[:, destination], node[:, source], edge), -1))
        aggregate = node.new_zeros(node.shape)
        aggregate.index_add_(1, destination, forward)
        aggregate.index_add_(1, source, backward)
        degree = torch.bincount(
            torch.cat((source, destination)), minlength=node.shape[1]
        ).to(node).clamp_min(1.0)
        aggregate = aggregate / degree[None, :, None]
        return self.norm(node + self.update(torch.cat((node, aggregate), -1)))


class AreaWeightedDepthConditionalGenerator(nn.Module):
    """Generate bounded restart coordinates from an unstructured depth graph."""

    parameter_width = 5
    bankfull_width = 2

    def __init__(
        self,
        node_static_dim: int,
        edge_static_dim: int,
        *,
        width: int = 64,
        layers: int = 3,
        noise_dim: int = 8,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_static_dim + 1, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            [_GraphBlock(width, edge_static_dim) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(width + noise_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, self.parameter_width + self.bankfull_width),
        )

    def _encode(
        self,
        depth_m: Tensor,
        cell_area_m2: Tensor,
        static_node_features: Tensor,
        edge_index: Tensor,
        static_edge_features: Tensor,
        depth_scale_m: float,
    ) -> Tensor:
        batch, nodes = depth_m.shape
        if cell_area_m2.shape != (nodes,) or torch.any(cell_area_m2 <= 0.0):
            raise ValueError("positive cell areas must match the graph")
        if static_node_features.shape[0] != nodes:
            raise ValueError("static node features do not match the graph")
        dynamic = torch.log1p(depth_m / depth_scale_m).unsqueeze(-1)
        static = static_node_features.to(depth_m)[None].expand(batch, -1, -1)
        node = self.node_encoder(torch.cat((static, dynamic), -1))
        for block in self.blocks:
            node = block(node, edge_index, static_edge_features.to(depth_m))
        weight = cell_area_m2.to(depth_m) / cell_area_m2.sum()
        return torch.sum(node * weight[None, :, None], dim=1)

    def forward(
        self,
        depth_m: Tensor,
        cell_area_m2: Tensor,
        static_node_features: Tensor,
        edge_index: Tensor,
        static_edge_features: Tensor,
        random_coordinate: Tensor,
        *,
        depth_scale_m: float = 1.0,
    ) -> RestartSamples:
        if depth_m.ndim != 2 or torch.any(depth_m < 0.0):
            raise ValueError("depth_m must have shape [batch,nodes] and be nonnegative")
        if random_coordinate.ndim != 3 or random_coordinate.shape[:1] != depth_m.shape[:1]:
            raise ValueError("random_coordinate must have shape [batch,samples,noise]")
        if random_coordinate.shape[-1] != self.noise_dim or depth_scale_m <= 0.0:
            raise ValueError("noise or depth scale has the wrong size")
        pooled = self._encode(
            depth_m,
            cell_area_m2,
            static_node_features,
            edge_index,
            static_edge_features,
            depth_scale_m,
        )
        condition = pooled[:, None].expand(-1, random_coordinate.shape[1], -1)
        coordinate = torch.sigmoid(self.head(torch.cat((condition, random_coordinate), -1)))
        return RestartSamples(
            coordinate[..., : self.parameter_width], coordinate[..., self.parameter_width :]
        )


class PriorAnchoredDepthConditionalTransport(AreaWeightedDepthConditionalGenerator):
    """Transport a registered prior draw; the zero-initialized map starts as identity."""

    coordinate_width = 7

    def __init__(self, node_static_dim: int, edge_static_dim: int, **kwargs: int) -> None:
        super().__init__(
            node_static_dim,
            edge_static_dim,
            noise_dim=self.coordinate_width,
            **kwargs,
        )
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, *args: Tensor, depth_scale_m: float = 1.0) -> RestartSamples:
        depth_m, cell_area_m2, static_node, edge_index, static_edge, reference = args
        if torch.any((reference < 0.0) | (reference > 1.0)):
            raise ValueError("reference prior coordinates must lie in [0,1]")
        pooled = self._encode(
            depth_m,
            cell_area_m2,
            static_node,
            edge_index,
            static_edge,
            depth_scale_m,
        )
        condition = pooled[:, None].expand(-1, reference.shape[1], -1)
        increment = self.head(torch.cat((condition, reference.to(depth_m)), -1))
        epsilon = torch.finfo(reference.dtype).eps
        coordinate = torch.sigmoid(
            torch.logit(reference.clamp(epsilon, 1.0 - epsilon)) + increment
        )
        coordinate = torch.where((reference == 0.0) | (reference == 1.0), reference, coordinate)
        return RestartSamples(coordinate[..., :5], coordinate[..., 5:])


def conditional_restart_fair_energy(samples: Tensor, truth: Tensor) -> Tensor:
    """Fair Energy Score for conditionally independent generated samples."""

    if samples.ndim != 3 or truth.shape != (samples.shape[0], samples.shape[2]):
        raise ValueError("samples/truth have incompatible shapes")
    if samples.shape[1] < 2:
        raise ValueError("fair Energy Score requires at least two samples")
    attraction = torch.linalg.vector_norm(samples - truth[:, None], dim=2).mean(1)
    pair = torch.cdist(samples, samples)
    members = samples.shape[1]
    repulsion = pair.sum((1, 2)) / (2.0 * members * (members - 1))
    return (attraction - repulsion).mean()
