"""The free pose path does not read a target, a teacher row, or a teacher slot."""
import inspect

import torch

from knolling_model.experiments.end_to_end import (
    arrange_free, arrangement_supervision_loss, predict_poses)
from knolling_model.experiments.row_layout import NeuralRowPlacement


def _scene():
    generator = torch.Generator().manual_seed(1203)
    hidden = torch.randn(1, 3, 128, generator=generator)
    valid = torch.ones(1, 3, dtype=torch.bool)
    goal = torch.zeros(1, 13)
    goal[0, 6] = 1
    widths = torch.tensor([[.04, .05, .03]])
    heights = torch.tensor([[.02, .025, .02]])
    row = torch.zeros(1, 3, dtype=torch.long)
    slot = torch.tensor([[0, 1, 2]])
    target_xy = torch.tensor([[[.10, .20], [.16, .20], [.23, .20]]])
    target_angle = torch.zeros(1, 3)
    return hidden, valid, goal, widths, heights, row, slot, target_xy, target_angle


def test_free_arrangement_has_no_teacher_arguments():
    names = inspect.signature(arrange_free).parameters
    assert set(names) == {'row_model', 'hidden', 'valid', 'goal', 'widths', 'heights'}
    source = inspect.getsource(predict_poses)
    for word in ('target_xy', 'target_angle', 'teacher_row', 'teacher_slot', 'supervision'):
        assert word not in source


def test_a_target_tensor_does_not_affect_the_free_pose():
    model = NeuralRowPlacement()
    hidden, valid, goal, widths, heights, row, slot, target_xy, target_angle = _scene()
    target_xy.requires_grad_(True)
    free = arrange_free(model, hidden, valid, goal, widths, heights)
    free['xy'].sum().backward()
    assert target_xy.grad is None
    assert torch.isfinite(free['xy']).all() and torch.isfinite(free['angle']).all()
    supervised = model(hidden.detach(), valid, goal, widths, heights, row, slot, True)
    loss = arrangement_supervision_loss(supervised, target_xy.detach(), target_angle, valid, row, slot)
    loss['loss'].backward()
    assert model.row_layout[-1].weight.grad.abs().sum() > 0
