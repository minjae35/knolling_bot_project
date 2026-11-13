"""Constraint checks use the predicted command and do not reject a legal row."""
import torch

from knolling_model.experiments.constraint_decode import (
    ACTIONABLE, apply_color_separation, apply_row_repair, order_by_area, repair_rows,
    separate_color_rows, structure_issues)
from knolling_model.recovery.contracts import Objects


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


def test_a_wide_row_offers_a_split_and_keeps_colors_apart():
    goal = _goal()
    goal[3] = 0
    goal[4] = 1
    logits = torch.full((2, 8), -4.)
    logits[:, 0] = 4.
    major, minor = torch.tensor([.16, .16]), torch.tensor([.04, .04])
    angle, color = torch.zeros(2), torch.tensor([0, 1])
    found = repair_rows(logits, angle, major, minor, color, goal)
    assert found
    row, slot = found[0]
    assert row.tolist() == [0, 1] or row.tolist() == [1, 0]
    assert 'mixed_color' not in structure_issues(row, slot, major, minor, angle, color, goal)
    kept = {'geometry_status': ['placed'], 'xy': torch.zeros(1, 2, 2), 'row': torch.zeros(1, 2, dtype=torch.long)}
    assert apply_row_repair(kept, None, goal) is kept


def test_a_confident_mixed_row_splits_only_for_that_sentence():
    goal = _goal()
    logits = torch.full((2, 8), -4.)
    logits[:, 0] = 4.
    major, minor = torch.tensor([.05, .05]), torch.tensor([.04, .04])
    angle, color = torch.zeros(2), torch.tensor([0, 1])
    found = separate_color_rows(logits, angle, major, minor, color, goal)
    assert found
    assert found[0][0].tolist() in ([0, 1], [1, 0])
    assert 'mixed_color' not in structure_issues(found[0][0], found[0][1], major, minor, angle, color, goal)
    color_p = torch.zeros(1, 2, 3)
    color_p[0, 0, 0] = 1
    color_p[0, 1, 1] = 1
    objects = Objects(
        xy=torch.tensor([[[.20, .20], [.30, .20]]]), size=torch.tensor([[[.05, .04], [.05, .04]]]),
        angle=torch.zeros(1, 2), color=color_p, shape=torch.tensor([[[1., 0, 0], [1., 0, 0]]]),
        valid=torch.ones(1, 2, dtype=torch.bool), query_index=torch.arange(2)[None])
    output = dict(
        row=torch.zeros(1, 2, dtype=torch.long), slot=torch.tensor([[0, 1]]),
        xy=objects.xy.clone(), angle=objects.angle.clone(), row_logits=logits[None],
        geometry_status=['placed'])
    plain = 'Arrange all objects in rows in the full workspace; long axes horizontal.'
    assert apply_color_separation(output, objects, goal[None], [plain]) is output
    asked = plain + ' Use single-color rows, with each color\'s rows together.'
    separated = apply_color_separation(output, objects, goal[None], [asked])
    assert separated is not output
    assert separated['row'][0].tolist() in ([0, 1], [1, 0])
    flat = Objects(
        xy=objects.xy, size=objects.size, angle=objects.angle, color=torch.full((1, 2, 3), 1/3),
        shape=objects.shape, valid=objects.valid, query_index=objects.query_index)
    assert apply_color_separation(output, flat, goal[None], [asked]) is output


def test_a_mixed_color_row_is_illegal_but_a_row_id_gap_is_not_searched():
    mixed = structure_issues(
        torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([.04, .05]), torch.tensor([.03, .03]),
        torch.zeros(2), torch.tensor([0, 1]), _goal())
    assert 'mixed_color' in mixed and mixed & ACTIONABLE
    gap = structure_issues(
        torch.tensor([0, 2]), torch.tensor([0, 0]), torch.tensor([.04, .05]), torch.tensor([.03, .03]),
        torch.zeros(2), torch.tensor([1, 1]), _goal())
    assert 'row_gap' in gap and not (gap & ACTIONABLE)
