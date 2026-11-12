"""CPU checks for the geometry decoder. No checkpoint and no GPU."""
import math

import torch

from knolling_dataset.schema import Pose
from knolling_dataset.validator import polygon
from knolling_model.experiments.geometry_decoder import (
    apply_geometry, decode_scene, extent_offsets)
from knolling_model.recovery.contracts import Objects


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


def _objects(major, minor, angle):
    count = major.shape[-1]
    size = torch.stack((major, minor), -1)
    color = torch.zeros(1, count, 3)
    color[..., 0] = 1
    return Objects(
        torch.zeros(1, count, 2), size, angle, color, color.clone(),
        torch.ones(1, count, dtype=torch.bool), torch.arange(count)[None])


def _output(xy, angle, row, slot):
    return dict(xy=xy, angle=angle, row=row, slot=slot)


def test_a_requested_axis_turns_the_major_side_without_swapping_it():
    major = torch.tensor([[.080, .060]])
    minor = torch.tensor([[.040, .030]])
    xy = torch.tensor([[[.20, .20], [.40, .20]]])
    row, slot = torch.zeros(1, 2, dtype=torch.long), torch.tensor([[0, 1]])
    horizontal = apply_geometry(
        _output(xy, torch.full((1, 2), math.pi / 2), row, slot),
        _objects(major, minor, torch.full((1, 2), math.pi / 2)), _goal()[None])
    torch.testing.assert_close(horizontal['angle'], torch.zeros(1, 2))
    vertical_goal = _goal()
    vertical_goal[1], vertical_goal[2] = 0, 1
    vertical = apply_geometry(
        _output(xy, torch.zeros(1, 2), row, slot),
        _objects(major, minor, torch.zeros(1, 2)), vertical_goal[None])
    torch.testing.assert_close(vertical['angle'], torch.full((1, 2), math.pi / 2))
    for result, along_x in ((horizontal, major), (vertical, minor)):
        for index in range(2):
            xmin, xmax, ymin, ymax = extent_offsets(
                result['angle'][0, index], major[0, index], minor[0, index])
            span_x, span_y = float(xmax - xmin), float(ymax - ymin)
            assert abs(span_x - float(along_x[0, index])) < 1e-6
            other = minor if along_x is major else major
            assert abs(span_y - float(other[0, index])) < 1e-6


def test_an_aligned_angle_and_a_keep_command_stay_put():
    major, minor = torch.tensor([[.060, .060]]), torch.tensor([[.040, .040]])
    angle = torch.tensor([[0.01, -0.01]])
    xy = torch.tensor([[[.15, .20], [.15 + .030 + .030 + .015, .20]]])
    row, slot = torch.zeros(1, 2, dtype=torch.long), torch.tensor([[0, 1]])
    objects = _objects(major, minor, angle)
    kept = apply_geometry(_output(xy, angle, row, slot), objects, _goal()[None])
    assert kept['geometry_status'] == ['kept']
    torch.testing.assert_close(kept['xy'], xy)
    assert torch.equal(kept['angle'], angle)
    keep = torch.zeros(1, 13)
    keep[0, 0] = 1
    upright = torch.full((1, 2), math.pi / 2)
    untouched = apply_geometry(_output(xy, upright, row, slot), _objects(major, minor, upright), keep)
    assert torch.equal(untouched['angle'], upright)
    tied = _goal()[None].clone()
    tied[0, 1] = tied[0, 2] = 0.5
    same = apply_geometry(_output(xy, upright, row, slot), _objects(major, minor, upright), tied)
    assert torch.equal(same['angle'], upright)


def test_a_short_vertical_opening_uses_the_request_when_it_fits():
    goal = _goal()
    xy = torch.tensor([[[.20, .30], [.20, .25]]])
    angle = torch.zeros(1, 2)
    row, slot = torch.tensor([[0, 1]]), torch.zeros(1, 2, dtype=torch.long)
    major, minor = torch.full((1, 2), .050), torch.full((1, 2), .040)
    result = apply_geometry(_output(xy, angle, row, slot), _objects(major, minor, angle), goal[None])
    opening = (result['xy'][0, 0, 1]-.020)-(result['xy'][0, 1, 1]+.020)
    torch.testing.assert_close(opening, torch.tensor(.015), atol=1e-5, rtol=0)
    legal = torch.tensor([[[.20, .30], [.20, .244]]])
    untouched = apply_geometry(_output(legal, angle, row, slot), _objects(major, minor, angle), goal[None])
    torch.testing.assert_close(untouched['xy'], legal)
    crowded = _goal(gap=2)
    tall = torch.full((1, 2), .166)
    band, status = decode_scene(
        torch.tensor([[.20, .24], [.20, .08]]), torch.zeros(2), torch.tensor([0, 1]), torch.zeros(2, dtype=torch.long),
        torch.full((2,), .050), tall[0], torch.zeros(2, dtype=torch.long), crowded)
    assert status == 'placed'
    kept = apply_geometry(
        _output(torch.tensor([[[.20, .24], [.20, .08]]]), angle, row, slot),
        _objects(torch.full((1, 2), .050), tall, angle), crowded[None])
    torch.testing.assert_close(kept['xy'][0], band)


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
