"""The shipped inference path states its scope and does not take labels."""
import inspect

import torch

from knolling_model.experiments.inference import (
    CAPTION, SCOPE, accepted_success, format_poses, instruction_supported, render_comparison)
from knolling_model.experiments.size_refinement import poses_from_detection


def _record(mode, count=4, locked=(), keep=False, selected=None):
    identities = [f'o{index}' for index in range(count)]
    return dict(
        mode=mode,
        source=dict(objects=[dict(object_id=name) for name in identities]),
        instruction=dict(keep_all=keep, locked_ids=list(locked),
                         selected_ids=identities if selected is None else list(selected)))


def test_only_a_full_arrangement_is_supported():
    assert instruction_supported(_record('full'))
    assert not instruction_supported(_record('partial', selected=['o0'], locked=['o1', 'o2', 'o3']))
    assert not instruction_supported(_record('no_op', keep=True))
    assert not instruction_supported(_record('full', count=1))
    assert not instruction_supported(_record('full', count=9))
    assert 'partial' in SCOPE['not_supported'] and 'no_op' in SCOPE['not_supported']


def test_an_unsupported_instruction_is_not_a_success():
    partial = _record('partial', selected=['o0'], locked=['o1', 'o2', 'o3'])
    assert accepted_success(_record('full'), True)
    assert not accepted_success(_record('full'), False)
    assert not accepted_success(partial, True)
    assert not accepted_success(_record('no_op', keep=True), True)


def test_pose_formatting_and_the_comparison_image_use_no_labels():
    assert 'target_xy' not in inspect.getsource(poses_from_detection)
    assert 'geometry' not in inspect.getsource(poses_from_detection)
    xy = torch.tensor([[0.30, 0.20]])
    poses = format_poses(xy, torch.zeros(1), torch.tensor([[0.04, 0.02]]),
                         torch.tensor([[0., 0., 1.]]), torch.tensor([True]))
    assert poses[0]['color'] == 'blue'
    assert torch.allclose(torch.tensor([poses[0][key] for key in ('x_m', 'y_m', 'theta_rad', 'width_m', 'height_m')]),
                          torch.tensor([0.30, 0.20, 0.0, 0.04, 0.02]))
    image = torch.zeros(3, 32, 48)
    image[0, :, :24] = 1.
    panel = render_comparison(image, poses, supported=True)
    assert panel.size == (48*2+8, 32+28)
    assert CAPTION.startswith('Supported: full arrangement')
    refused = render_comparison(image, poses, supported=False)

    def red_marks(picture):
        banner = list(picture.crop((0, 0, picture.size[0], 28)).getdata())
        return sum(pixel[0] > pixel[1]+40 for pixel in banner)

    assert red_marks(refused) > red_marks(panel)
