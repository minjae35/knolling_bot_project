import pytest
import torch
from torch.nn import functional as F

from knolling_model.deterministic_sampling import bilinear_sample, DeterministicDeformableAttention


def test_bilinear_forward_and_both_gradients_match_grid_sample():
    torch.manual_seed(31)
    value = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    grid = (torch.rand(2, 4, 6, 2, dtype=torch.float64)*3-1.5).requires_grad_()
    with torch.no_grad():
        grid[:, 0, :3] = torch.tensor([[-1., -1.], [1., 1.], [0., 0.]])
    expected = F.grid_sample(value, grid, align_corners=False)
    actual = bilinear_sample(value, grid)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    upstream = torch.randn_like(expected)
    a = torch.autograd.grad(actual, (value, grid), upstream, retain_graph=True)
    b = torch.autograd.grad(expected, (value, grid), upstream)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-13, rtol=1e-13)


def test_multiscale_core_matches_stock_forward_and_gradients():
    from transformers.models.rt_detr.modeling_rt_detr import MultiScaleDeformableAttention
    torch.manual_seed(32)
    shapes = [(3, 5), (2, 2)]
    value = torch.randn(2, 19, 2, 4, dtype=torch.float64, requires_grad=True)
    locations = torch.rand(2, 7, 2, 2, 3, 2, dtype=torch.float64, requires_grad=True)
    weights = torch.rand(2, 7, 2, 2, 3, dtype=torch.float64, requires_grad=True)
    args = (value, torch.tensor(shapes), shapes, torch.tensor([0, 15]), locations, weights, 64)
    expected = MultiScaleDeformableAttention()(*args)
    actual = DeterministicDeformableAttention()(*args)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    for left, right in zip(torch.autograd.grad(actual.square().sum(), (value, locations, weights)),
                           torch.autograd.grad(expected.square().sum(), (value, locations, weights))):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_backward_is_exactly_repeatable_under_strict_determinism():
    from knolling_model.runner import seed_everything
    previous = torch.are_deterministic_algorithms_enabled()
    seed_everything(1203)
    try:
        value = torch.randn(2, 3, 5, 7, device='cuda', requires_grad=True)
        # Repeated locations exercise the collision/reduction backward path.
        grid = torch.zeros(2, 4, 6, 2, device='cuda', requires_grad=True)
        first = torch.autograd.grad(bilinear_sample(value, grid).square().sum(), (value, grid))
        second = torch.autograd.grad(bilinear_sample(value, grid).square().sum(), (value, grid))
        assert all(torch.equal(a, b) and torch.isfinite(a).all() for a, b in zip(first, second))
    finally:
        torch.use_deterministic_algorithms(previous)
