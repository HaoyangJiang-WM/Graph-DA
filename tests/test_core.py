import torch

from graph_da.etkf import binary_sar_etkf
from graph_da.restart_completion import PriorAnchoredDepthConditionalTransport
from graph_da.sar import footprint_average


def test_footprint_average_preserves_constant_field() -> None:
    value = torch.full((3, 4), 2.5)
    overlap = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.5, 0.5, 1.0]])
    observed = footprint_average(value, overlap)
    assert torch.allclose(observed, torch.full((3, 2), 2.5))


def test_etkf_is_nonnegative_and_restartable() -> None:
    depth = torch.tensor(
        [[0.00, 0.01, 0.10], [0.00, 0.04, 0.16], [0.01, 0.07, 0.24], [0.02, 0.10, 0.32]]
    )
    overlap = torch.eye(3)
    result = binary_sar_etkf(depth, torch.tensor([0.0, 1.0, 1.0]), overlap, torch.ones(3))
    assert result.analysis_depth_m.shape == depth.shape
    assert torch.all(result.analysis_depth_m >= 0.0)
    assert torch.isfinite(result.member_mass_change_m3).all()


def test_prior_anchored_transport_starts_as_identity() -> None:
    model = PriorAnchoredDepthConditionalTransport(2, 1, width=8, layers=1)
    reference = torch.rand(2, 4, 7).clamp(0.01, 0.99)
    sample = model(
        torch.rand(2, 3),
        torch.ones(3),
        torch.rand(3, 2),
        torch.tensor([[0, 1], [1, 2]]),
        torch.ones(2, 1),
        reference,
    )
    assert torch.allclose(sample.joint_coordinate, reference, atol=1.0e-6)
