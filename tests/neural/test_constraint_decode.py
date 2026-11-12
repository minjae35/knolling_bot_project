"""Constraint checks use the predicted command and do not reject a legal row."""
import torch

from knolling_model.experiments.constraint_decode import ACTIONABLE, order_by_area, structure_issues


def _goal(group='color', sort='asc'):
    goal = torch.zeros(13)
    goal[3] = 1
    goal[6] = 1
    goal[9 if group == 'none' else 10] = 1
    goal[11 if sort == 'asc' else 12] = 1
    return goal


def test_a_same_color_ascending_row_is_left_alone():
    issues = structure_issues(
        torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([.04, .05]), torch.tensor([.03, .03]),
        torch.zeros(2), torch.tensor([1, 1]), _goal())
    assert not (issues & ACTIONABLE)


def test_a_near_tie_takes_the_predicted_area_order():
    row, slot = order_by_area(
        torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([.00126, .00120]),
        torch.tensor([1, 1]), _goal(group='none'))
    assert row.tolist() == [0, 0]
    assert slot.tolist() == [1, 0]
    kept_row, kept_slot = order_by_area(
        torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([.00120, .00126]),
        torch.tensor([1, 1]), _goal(group='none'))
    assert kept_row.tolist() == [0, 0]
    assert kept_slot.tolist() == [0, 1]


def test_a_mixed_color_row_is_illegal_but_a_row_id_gap_is_not_searched():
    mixed = structure_issues(
        torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([.04, .05]), torch.tensor([.03, .03]),
        torch.zeros(2), torch.tensor([0, 1]), _goal())
    assert 'mixed_color' in mixed and mixed & ACTIONABLE
    gap = structure_issues(
        torch.tensor([0, 2]), torch.tensor([0, 0]), torch.tensor([.04, .05]), torch.tensor([.03, .03]),
        torch.zeros(2), torch.tensor([1, 1]), _goal())
    assert 'row_gap' in gap and not (gap & ACTIONABLE)
