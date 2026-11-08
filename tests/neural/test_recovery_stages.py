from types import SimpleNamespace

import pytest
import torch
from torch import nn

from knolling_model.recovery.data import RecoveryDataset
from knolling_model.recovery.model import RecoveryModel, CapacityError, legacy_evaluation_view
from knolling_model.recovery.stages import (configure_stage, parameter_groups, read_review_config,
                                          stage_gate, stage_loss)
from test_recovery_perception import tiny_detector, truth


class TextFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 384)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def inputs():
    return dict(images=torch.rand(1, 3, 384, 576), input_ids=torch.tensor([[1, 2, 0]]),
                attention_mask=torch.tensor([[True, True, False]]))


def selected_fixture(model, batch):
    """Synthetic valid detections for loss wiring; not a trained detector claim."""
    original = model.perception.forward
    def forward(images):
        out = original(images)
        scores = torch.full_like(out['source']['objectness_logits'], -10.)
        scores[:, :2] = 10.
        out['source']['objectness_logits'] = scores
        return out
    model.perception.forward = forward
    with torch.no_grad():
        s = model.perception(batch['images'])['source']
    t = truth()
    t.update(source_xy=s['uv'][0, :2].detach()*torch.tensor([.6, .4], device=s['uv'].device),
             source_angle=s['angle'][0, :2].detach(), size=s['size'][0, :2].detach())
    t['target_xy'] = (t['source_xy']+.01).clamp_min(.01)
    return {k: v.to(s['uv'].device) for k, v in t.items()}


@pytest.fixture
def model():
    return RecoveryModel(TextFixture(), tiny_detector(), dimension=32, layers=1, num_denoising=4)


@pytest.mark.parametrize('stage', ['oracle_layout', 'perception', 'predicted_layout', 'joint'])
def test_stage_freeze_shape_gradient_and_no_parameter_updates(model, stage):
    configure_stage(model, stage)
    batch = inputs()
    labels = selected_fixture(model, batch) if stage in ('predicted_layout', 'joint') else truth()
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    terms, out = stage_loss(model, batch, [labels], replay=(batch, [labels]) if stage == 'joint' else None)
    assert terms and all(torch.isfinite(v) for v in terms.values())
    sum(terms.values()).backward()
    assert all(p.grad is None for p in model.text_encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name])  # backward only, never optimizer.step
        if not p.requires_grad:
            assert p.grad is None
    expected = ('perception',) if stage == 'perception' else ('layout', 'perception') if stage == 'joint' else ('layout',)
    assert tuple(g['name'] for g in parameter_groups(model)) == expected
    for group in parameter_groups(model):
        assert sum(p.grad.abs().sum() for p in group['params'] if p.grad is not None) > 0
    if stage != 'perception':
        assert out['layout']['xy'].shape == (1, 2, 2)
        assert out['layout']['rollout'] == 'free_rollout'
    if stage == 'joint':
        assert not any(p.requires_grad for p in model.perception.detector.model.backbone.parameters())


def test_rgb_target_gradient_and_inference_has_no_supervision_interface(model):
    # Exercise the entire differentiable RGB path, independently of stage freezes.
    batch = inputs()
    batch['images'].requires_grad_()
    with torch.no_grad():
        # Test fixture forces detections; production threshold stays fixed at .5.
        for head in model.perception.detector.class_embed:
            head.bias.fill_(10.)
    out = model(**batch)
    assert out['layout']['xy'].shape == (1, 7, 2)
    objective = out['layout']['candidate_xy'].square().sum()+out['layout']['angle_logits'].square().mean()
    objective.backward()
    assert batch['images'].grad is not None and batch['images'].grad.abs().sum() > 0
    for module in (model.perception.detector.model.backbone, model.perception.detector.model.enc_output,
                   model.perception.detector.model.decoder, model.layout.text_projection):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.text_encoder.parameters())
    view = legacy_evaluation_view(out)
    assert len(view['targets']) == 1
    indices = out['layout']['query_index'][0]
    torch.testing.assert_close(view['targets'][0]['uv'][0, indices], out['layout']['xy'][0]/torch.tensor([.6, .4]))
    with pytest.raises(TypeError):
        model(**batch, truths=[truth()])
    model.max_objects = 4
    with pytest.raises(CapacityError) as error:
        model(**batch)
    assert error.value.counts == [7]


def test_stage_guards_and_fail_closed_review_gates(model, s0_dataset):
    config = read_review_config('configs/recovery_review.json')
    assert config['allow_training'] is False and config['device'] == 'cpu'
    wrapped = RecoveryDataset(s0_dataset)
    assert 'move' in wrapped[0]['supervision'] and 'move' not in s0_dataset[0]['supervision']
    configure_stage(model, 'joint')
    batch = inputs()
    labels = selected_fixture(model, batch)
    with pytest.raises(ValueError, match='replay'):
        stage_loss(model, batch, [labels])
    with pytest.raises(ValueError, match='teacher forcing'):
        stage_loss(model, inputs(), [truth()], teacher_forcing=True)
    for stage in config['stages']:
        assert not stage_gate(stage, {})
    metrics = dict(evaluation_mode='free_rollout', complete_set=True, source_mode='oracle',
                   target_position_p95_mm=.5, target_angle_p95_deg=1., move_accuracy=1., pure_axis_success=1.)
    assert stage_gate('oracle_layout', metrics)
    assert not stage_gate('oracle_layout', {**metrics, 'evaluation_mode': 'teacher_forced'})
    assert not stage_gate('predicted_layout', metrics)
    assert not stage_gate('oracle_layout', {**metrics, 'target_position_p95_mm': float('nan')})
    from knolling_model.runner import read_config
    with pytest.raises(ValueError):
        read_config('configs/recovery_review.json')


def test_source_noise_is_explicit_in_layout_warmup(model):
    configure_stage(model, 'oracle_layout')
    batch = inputs()
    terms, output = stage_loss(model, batch, [truth()], teacher_forcing=True, source_noise=dict(
        xy_std_m=.001, size_std_m=.0001, angle_std_rad=.01, generator=torch.Generator().manual_seed(1)))
    assert output['layout']['rollout'] == 'teacher_forced'
    assert not torch.allclose(output['objects'].xy[0], truth()['source_xy'])
    sum(terms.values()).backward()
    assert model.layout.xy.weight.grad.abs().sum() > 0
