"""CPU checks for the geometry decoder. No checkpoint and no GPU."""
import torch

from knolling_dataset.schema import Pose
from knolling_dataset.validator import polygon
from knolling_model.experiments.geometry_decoder import decode_scene, extent_offsets


def _goal(region='full', gap=0, group='none'):
    goal = torch.zeros(13)
    goal[1] = 1
    goal[3+{'full': 0, 'left': 1, 'right': 2}[region]] = 1
    goal[6+gap] = 1
    goal[9+{'none': 0, 'color': 1}[group]] = 1
    goal[11] = 1
    return goal


class _Asset:
    def __init__(self, major, minor):
        self.a_m, self.b_m = major, minor


class _Object:
    def __init__(self, major, minor):
        self.asset = _Asset(major, minor)


def test_offsets_match_the_evaluator_envelope():
    angle, major, minor = torch.tensor(0.4), 0.060, 0.040
    xmin, xmax, ymin, ymax = extent_offsets(angle, major, minor)
    bounds = polygon(_Object(major, minor), Pose(0.2, 0.2, float(angle))).bounds
    torch.testing.assert_close(torch.tensor(bounds[0]), torch.tensor(0.2)+xmin, atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.tensor(bounds[2]), torch.tensor(0.2)+xmax, atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.tensor(bounds[1]), torch.tensor(0.2)+ymin, atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.tensor(bounds[3]), torch.tensor(0.2)+ymax, atol=1e-6, rtol=0)


def test_an_already_legal_gap_does_not_move():
    goal = _goal()
    xy = torch.tensor([[.15, .20], [.15+.030+.030+.015, .20]])
    angle = torch.zeros(2)
    row, slot = torch.tensor([0, 0]), torch.tensor([0, 1])
    major, minor = torch.tensor([.060, .060]), torch.tensor([.040, .040])
    color = torch.zeros(2, dtype=torch.long)
    decoded, status = decode_scene(xy, angle, row, slot, major, minor, color, goal)
    assert status == 'kept'
    torch.testing.assert_close(decoded, xy)


def test_a_wide_gap_moves_only_to_the_edge_of_the_band():
    goal = _goal()
    exact = .030+.030+.015
    xy = torch.tensor([[.15, .20], [.15+exact+.010, .20]])
    angle = torch.zeros(2)
    row, slot = torch.tensor([0, 0]), torch.tensor([0, 1])
    major, minor = torch.full((2,), .060), torch.full((2,), .040)
    decoded, status = decode_scene(xy, angle, row, slot, major, minor, torch.zeros(2, dtype=torch.long), goal)
    assert status == 'placed'
    opening = (decoded[1, 0]-.030)-(decoded[0, 0]+.030)
    torch.testing.assert_close(opening, torch.tensor(.017), atol=1e-5, rtol=0)
    assert float((decoded-xy).abs().sum()) < .020


def test_color_rows_use_twice_the_gap_and_a_too_wide_row_is_rejected():
    goal = _goal(group='color')
    xy = torch.tensor([[.20, .30], [.20, .20]])
    angle = torch.zeros(2)
    row, slot = torch.tensor([0, 1]), torch.tensor([0, 0])
    major, minor = torch.full((2,), .050), torch.full((2,), .040)
    decoded, status = decode_scene(xy, angle, row, slot, major, minor, torch.tensor([0, 1]), goal)
    assert status == 'placed'
    opening = (decoded[0, 1]-.020)-(decoded[1, 1]+.020)
    torch.testing.assert_close(opening, torch.tensor(.032), atol=1e-5, rtol=0)
    wide = _goal(region='left')
    crowded, failure = decode_scene(
        torch.tensor([[.10, .20], [.25, .20]]), torch.zeros(2), torch.zeros(2, dtype=torch.long),
        torch.tensor([0, 1]), torch.full((2,), .200), torch.full((2,), .040),
        torch.zeros(2, dtype=torch.long), wide)
    assert failure == 'infeasible:horizontal_room'
    torch.testing.assert_close(crowded, torch.tensor([[.10, .20], [.25, .20]]))


def test_width_still_moves_the_following_object():
    goal = _goal()
    xy = torch.tensor([[.15, .20], [.40, .20]])
    angle = torch.zeros(2)
    row, slot = torch.tensor([0, 0]), torch.tensor([0, 1])
    minor = torch.full((2,), .040)
    color = torch.zeros(2, dtype=torch.long)
    narrow, _ = decode_scene(xy, angle, row, slot, torch.full((2,), .060), minor, color, goal)
    wide, status = decode_scene(xy, angle, row, slot, torch.tensor([.080, .060]), minor, color, goal)
    assert status == 'placed'
    assert float(wide[1, 0]) > float(narrow[1, 0])
