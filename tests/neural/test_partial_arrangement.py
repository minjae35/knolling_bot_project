"""Partial and no-op routing. No checkpoint and no GPU."""
import math

import torch

from knolling_dataset.schema import angle_error
from knolling_model.experiments.partial_arrangement import (
    clear_held_objects, command_is_full, named_color, supplement_moving, text_requests_every_object)
from knolling_model.experiments.size_refinement import silhouette_angle


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
    near = torch.tensor([[1.0, 0.465, 0.0]])
    assert command_is_full(torch.tensor([-2.]), near, valid)


def test_an_all_objects_command_is_not_a_partial_command():
    assert text_requests_every_object(['Arrange all objects in rows in the full workspace.'])
    assert not text_requests_every_object(['Arrange only the red objects in rows.'])
    assert not text_requests_every_object(['Keep every object at its current position and orientation.'])


def test_a_colored_silhouette_reports_its_major_axis():
    image = torch.full((3, 384, 576), 179 / 255)
    theta, center_x, center_y = 0.4, 0.30, 0.20
    major, minor = 0.08, 0.03
    columns = torch.arange(576)
    rows = torch.arange(384)
    x_m = (columns + 0.5) / 576 * 0.6
    y_m = (1 - (rows + 0.5) / 384) * 0.4
    grid_y, grid_x = torch.meshgrid(y_m, x_m, indexing='ij')
    cosine, sine = math.cos(theta), math.sin(theta)
    local_x = cosine * (grid_x - center_x) + sine * (grid_y - center_y)
    local_y = -sine * (grid_x - center_x) + cosine * (grid_y - center_y)
    inside = (local_x.abs() <= major / 2) & (local_y.abs() <= minor / 2)
    image[0, inside] = 0.85
    image[1, inside] = 0.08
    image[2, inside] = 0.08
    found = silhouette_angle(image, torch.tensor([center_x, center_y]))
    assert found is not None
    assert angle_error(found, theta) < math.radians(1)


def test_a_named_color_is_added_and_a_tie_or_another_color_is_not():
    color = torch.tensor([[0.9, 0.05, 0.05], [0.2, 0.6, 0.2], [0.4, 0.4, 0.2]])
    moving = supplement_moving(torch.tensor([1]), color, 'Arrange only the red objects in rows.')
    assert moving.tolist() == [0, 1]
    assert supplement_moving(torch.tensor([0]), color, 'Keep every object at its current position and orientation.').tolist() == [0]
    assert named_color('Arrange only the red objects and only the blue objects.') is None
    assert supplement_moving(torch.zeros(0, dtype=torch.long), color, 'Arrange only the blue objects.').tolist() == []


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
