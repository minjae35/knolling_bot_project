"""The size residual starts at the frozen prediction and does not read labels."""
import inspect

import torch
from torch.nn import functional as F

from knolling_model.experiments.row_layout import NeuralRowPlacement
from knolling_model.experiments.size_refinement import (
    ObjectSizeRefinement, horizontal_extent, install_refined_size, live_poses,
    matched_queries, poses_from_detection, scatter_refined, suppress_duplicate_detections)


def _inputs():
    generator = torch.Generator().manual_seed(1203)
    tokens = torch.randn(2, 4, 256, generator=generator)
    images = torch.rand(2, 3, 32, 48, generator=generator)
    boxes = torch.tensor([[[.2, .3, .1, .2], [.5, .5, .2, .15], [.7, .4, .12, .1], [.4, .8, .08, .1]],
                          [[.3, .3, .1, .12], [.6, .6, .18, .1], [.2, .7, .1, .1], [.8, .2, .09, .08]]])
    base = torch.tensor([[[.040, .020], [.055, .030], [.035, .018], [.048, .022]],
                         [[.042, .021], [.060, .028], [.033, .019], [.050, .025]]])
    return tokens, images, boxes, base


def test_a_repeated_detection_is_dropped_and_a_separate_one_stays():
    source = dict(
        objectness_logits=torch.tensor([[2.0, 0.2, 3.0]]),
        uv=torch.tensor([[[0.20, 0.50], [0.210, 0.50], [0.80, 0.20]]]))
    updated = suppress_duplicate_detections(source)
    assert float(updated['objectness_logits'][0, 1]) == -20
    assert torch.equal(updated['objectness_logits'][0, [0, 2]], source['objectness_logits'][0, [0, 2]])
    apart = dict(
        objectness_logits=torch.tensor([[2.0, 2.0]]),
        uv=torch.tensor([[[0.20, 0.50], [0.40, 0.50]]]))
    assert torch.equal(suppress_duplicate_detections(apart)['objectness_logits'], apart['objectness_logits'])


def test_zero_residual_keeps_the_detector_size():
    head = ObjectSizeRefinement()
    tokens, images, boxes, base = _inputs()
    assert torch.equal(head(tokens, images, boxes, base), base)


def test_forward_does_not_take_a_ground_truth_size():
    names = inspect.signature(ObjectSizeRefinement.forward).parameters
    assert set(names) == {'self', 'tokens', 'images', 'boxes', 'base_size'}
    for function in (poses_from_detection, live_poses):
        source = inspect.getsource(function)
        for word in ('target_xy', 'target_angle', 'teacher_row', 'teacher_slot', 'supervision', 'source_xy'):
            assert word not in source


def test_one_step_changes_refined_size_without_touching_the_query():
    head = ObjectSizeRefinement()
    tokens, images, boxes, base = _inputs()
    tokens = tokens.detach()
    frozen = torch.nn.Linear(4, 4)
    frozen_weight = frozen.weight.detach().clone()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2)
    refined = head(tokens, images, boxes, base)
    loss = F.smooth_l1_loss(refined, base+0.004)
    loss.backward()
    assert tokens.grad is None
    assert frozen.weight.grad is None
    optimizer.step()
    assert not torch.equal(head(tokens, images, boxes, base), refined.detach())
    assert torch.equal(frozen.weight, frozen_weight)


def test_installation_replaces_size_only_for_selected_queries():
    head = ObjectSizeRefinement()
    with torch.no_grad():
        head.readout[-1].bias.copy_(torch.tensor([0.003, -0.002]))
    tokens, images, boxes, base = _inputs()
    objectness = torch.full((2, 4), -4.)
    objectness[:, 0] = 4.
    visual = dict(source=dict(size=base.clone(), objectness_logits=objectness,
                              uv=torch.zeros(2, 4, 2), angle=torch.zeros(2, 4)),
                  stages=[dict(tokens=tokens, boxes=boxes)])
    updated = install_refined_size(head, images, visual, .5)
    assert torch.allclose(updated['source']['size'][:, 0], (base[:, 0]+torch.tensor([0.003, -0.002])).clamp_min(1e-4))
    assert torch.equal(updated['source']['size'][:, 1:], base[:, 1:])
    assert torch.equal(updated['source']['uv'], visual['source']['uv'])
    assert torch.equal(updated['source']['angle'], visual['source']['angle'])
    assert torch.equal(updated['source']['objectness_logits'], objectness)


def test_matched_queries_do_not_keep_a_gradient_into_detector_tokens():
    tokens, images, boxes, base = _inputs()
    tokens.requires_grad_(True)
    rgb, query_tokens, query_boxes, query_size = matched_queries(tokens, images, boxes, base, [(0, 1), (1, 2)])
    head = ObjectSizeRefinement()
    head(query_tokens, rgb, query_boxes, query_size).sum().backward()
    assert tokens.grad is None
    assert query_tokens.shape == (2, 1, 256)


def test_pose_loss_reaches_the_size_residual_and_not_the_row_weights():
    generator = torch.Generator().manual_seed(3)
    head = ObjectSizeRefinement()
    row = NeuralRowPlacement()
    row.requires_grad_(False)
    with torch.no_grad():
        row.assign[-1].weight.zero_()
        row.assign[-1].bias.zero_()
        row.assign[-1].bias[0] = 10.
        row.step[-1].weight.fill_(.5)
    tokens = torch.randn(1, 2, 256, generator=generator)
    images = torch.rand(1, 3, 24, 32, generator=generator)
    boxes = torch.tensor([[[.4, .4, .2, .15], [.7, .6, .18, .12]]])
    base = torch.tensor([[[.05, .02], [.04, .025]]])
    refined = head(tokens, images, boxes, base)[:, 0]
    size = scatter_refined(base, [(0, 0), (0, 1)], refined)
    hidden = torch.randn(1, 2, 128, generator=generator)
    valid = torch.ones(1, 2, dtype=torch.bool)
    goal = torch.zeros(1, 13)
    goal[0, 6] = 1.
    output = row(hidden, valid, goal, size[..., 0], size[..., 1], None, None, True)
    target = output['xy'].detach()+.02
    F.smooth_l1_loss(output['xy'], target).backward()
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
               for parameter in head.parameters())
    assert all(parameter.grad is None for parameter in row.parameters())


def test_horizontal_extent_uses_both_axes():
    size = torch.tensor([[0.04, 0.02]])
    assert torch.allclose(horizontal_extent(size, torch.zeros(1)), torch.tensor([0.04]))
    assert torch.allclose(horizontal_extent(size, torch.full((1,), torch.pi/2)), torch.tensor([0.02]))
