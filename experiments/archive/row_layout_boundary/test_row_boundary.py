"""Boundary cuts become row ids. A per-object classifier is a different family."""
import torch

from knolling_model.experiments.row_layout import (
    NeuralRowPlacement, boundary_loss, load_row_weights, rows_from_boundaries)


def test_cuts_become_rows_and_slots_for_two_through_eight_objects():
    placed = torch.arange(8)[None]
    lengths = torch.tensor([8])
    none = torch.zeros(1, 7, dtype=torch.bool)
    row, slot = rows_from_boundaries(placed, lengths, none)
    assert row[0].tolist() == [0]*8
    assert slot[0].tolist() == list(range(8))
    two = torch.zeros(1, 1, dtype=torch.bool)
    two[0, 0] = True
    row, slot = rows_from_boundaries(torch.arange(2)[None], torch.tensor([2]), two)
    assert row[0].tolist() == [0, 1] and slot[0].tolist() == [0, 0]
    cuts = torch.zeros(1, 7, dtype=torch.bool)
    cuts[0, 2] = cuts[0, 5] = True
    row, slot = rows_from_boundaries(placed, lengths, cuts)
    assert row[0].tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    assert slot[0].tolist() == [0, 1, 2, 0, 1, 2, 0, 1]


def test_padding_is_excluded_and_the_cut_follows_the_predicted_order():
    placed = torch.tensor([[2, 0, 1, 0]])
    lengths = torch.tensor([3])
    cuts = torch.tensor([[1, 0, 0]], dtype=torch.bool)
    row, slot = rows_from_boundaries(placed, lengths, cuts)
    assert row[0].tolist() == [1, 1, 0, -1]
    assert slot[0].tolist() == [0, 1, 0, -1]
    produced = []
    for pattern in range(4):
        choice = torch.tensor([[bool(pattern & 1), bool(pattern & 2)]])
        seq, _ = rows_from_boundaries(torch.arange(3)[None], torch.tensor([3]), choice)
        values = seq[0].tolist()
        assert values[0] == 0
        assert all(b-a in (0, 1) for a, b in zip(values, values[1:]))
        produced.append(values)
    assert [0, 1, 0] not in produced


def test_boundary_loss_trains_only_the_new_head():
    torch.manual_seed(0)
    model = NeuralRowPlacement()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('boundary.'))
    hidden = torch.randn(1, 4, 128)
    valid = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    goal = torch.zeros(1, 13)
    goal[0, 6] = 1
    widths = torch.tensor([[.04, .05, .03, 0.]])
    heights = torch.tensor([[.02, .03, .04, 0.]])
    row = torch.tensor([[0, 0, 1, -1]])
    slot = torch.tensor([[0, 1, 0, -1]])
    output = model(hidden, valid, goal, widths, heights, row, slot, False)
    assert torch.equal(output['row'], row)
    loss = boundary_loss(output, row)
    loss.backward()
    assert model.boundary[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if not name.startswith('boundary.'))
    state = {key: value for key, value in model.state_dict().items() if not key.startswith('boundary.')}
    fresh = NeuralRowPlacement()
    load_row_weights(fresh, state)
    assert torch.count_nonzero(fresh.boundary[-1].weight) == 0
    assert torch.equal(fresh.assign[0].weight, model.assign[0].weight)
