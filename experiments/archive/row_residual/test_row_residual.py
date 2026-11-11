"""A zero residual leaves the frozen relation head unchanged."""
import torch
from knolling_model.experiments.relation_pose import NeuralRelationPose
from knolling_model.experiments.row_residual import FrozenRowModel, RowResidual, residual_loss


def test_zero_residual_matches_the_frozen_head_and_learns_only_the_add_on():
    torch.manual_seed(2)
    head = NeuralRelationPose(12)
    head.requires_grad_(False)
    residual = RowResidual()
    model = FrozenRowModel(head, residual)
    features = torch.randn(2, 4, 12)
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    goal = torch.zeros(2, 13)
    goal[:, 6] = 1
    order = torch.tensor([[0, 1, 2, -1], [3, 1, 0, 2]])
    breaks = torch.zeros(2, 4)
    breaks[0, 2] = 1
    breaks[1, 0] = 1
    target = torch.rand(2, 4, 2)
    base = head(features, valid, goal, order)
    wrapped = model(features, valid, goal, order, breaks)
    torch.testing.assert_close(wrapped['xy'], base['xy'])
    free = model(features, valid, goal)
    torch.testing.assert_close(free['xy'], head(features, valid, goal)['xy'])
    residual_loss(model(features, valid, goal, order, breaks), target, order, breaks)['loss'].backward()
    assert residual.residual[-1].weight.grad.abs().sum() > 0
    assert residual.classifier[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in head.parameters())
