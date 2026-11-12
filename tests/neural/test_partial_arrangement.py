"""Partial and no-op routing. No checkpoint and no GPU."""
import torch

from knolling_model.experiments.partial_arrangement import clear_held_objects, command_is_full


def _goal():
    goal = torch.zeros(1, 13)
    goal[0, 1] = 1
    goal[0, 3] = 1
    goal[0, 6] = 1
    goal[0, 9] = 1
    goal[0, 11] = 1
    return goal


def test_a_full_command_stays_on_the_full_path():
    valid = torch.tensor([[True, True, False]])
    selected = torch.tensor([[0.9, 0.7, 0.0]])
    assert command_is_full(torch.tensor([-2.]), selected, valid)
    assert not command_is_full(torch.tensor([2.]), selected, valid)
    assert not command_is_full(torch.tensor([-2.]), torch.tensor([[0.9, 0.2, 0.0]]), valid)


def test_a_moved_object_shifts_off_a_fixed_object_and_a_clear_one_stays():
    goal = _goal()
    angle = torch.zeros(1, 2)
    size = torch.tensor([[[0.08, 0.04], [0.08, 0.04]]])
    overlap = torch.tensor([[[0.25, 0.20], [0.25, 0.20]]])
    cleared = clear_held_objects(overlap, angle, size, torch.tensor([0]), goal)
    torch.testing.assert_close(cleared[0, 1], overlap[0, 1])
    assert float(torch.linalg.vector_norm(cleared[0, 0] - overlap[0, 0])) > 0.04
    apart = torch.tensor([[[0.12, 0.20], [0.40, 0.20]]])
    stayed = clear_held_objects(apart, angle, size, torch.tensor([0]), goal)
    torch.testing.assert_close(stayed, apart)
