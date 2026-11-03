"""Small real RT-DETR modules; pretrained CUDA checks are opt-in separately."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from knolling_model.model import RTDetrKnollingModel, decode_predictions
from knolling_model.losses import S0Criterion
from knolling_model.runner import make_optimizer, read_config


class TextFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 384)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@pytest.fixture
def model():
    from transformers import RTDetrConfig, RTDetrResNetConfig, RTDetrForObjectDetection
    torch.manual_seed(12)
    backbone = RTDetrResNetConfig(embedding_size=8, hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 1, 1], layer_type='basic', out_features=['stage2', 'stage3', 'stage4'])
    config = RTDetrConfig(backbone_config=backbone, encoder_in_channels=[16, 32, 64],
        encoder_hidden_dim=32, encoder_ffn_dim=64, decoder_in_channels=[32, 32, 32],
        decoder_layers=2, decoder_ffn_dim=64, d_model=256, num_queries=7,
        num_labels=5, hidden_expansion=.5, num_denoising=0, disable_custom_kernels=True,
        attention_dropout=0., activation_dropout=0., dropout=0.)
    return RTDetrKnollingModel(TextFixture(), RTDetrForObjectDetection(config))


def inputs():
    return dict(images=torch.rand(1, 3, 384, 576), input_ids=torch.tensor([[1, 2, 0]]),
                attention_mask=torch.tensor([[True, True, False]]))


def truth():
    return dict(source_xy=torch.tensor([[.15, .2], [.35, .3]]), source_angle=torch.tensor([.2, -.3]),
        target_xy=torch.tensor([[.2, .2], [.3, .2]]), target_angle=torch.tensor([0., 0.]),
        size=torch.tensor([[.06, .035], [.055, .025]]), color=torch.tensor([0, 1]), shape=torch.tensor([1, 2]))


def test_pretrained_module_forward_equivalence_and_image_to_workspace_coordinates(model):
    model.eval()
    batch = inputs()
    with torch.no_grad():
        native = model.perception.detector(pixel_values=F.interpolate(batch['images'], (640, 640),
                                           mode='bilinear', align_corners=False))
        output = model(**batch)
    stage = output['detector_stages'][-1]
    torch.testing.assert_close(stage['boxes'], native.pred_boxes, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(stage['logits'], native.logits, atol=2e-5, rtol=2e-5)
    assert output['object_tokens'].shape == (1, 7, 256)
    torch.testing.assert_close(output['source']['uv'][..., 0], stage['boxes'][..., 0])
    torch.testing.assert_close(output['source']['uv'][..., 1], 1-stage['boxes'][..., 1])
    torch.testing.assert_close(output['source']['objectness_logits'], stage['logits'].max(-1).values)
    for target in output['targets']:
        torch.testing.assert_close(target['uv'], output['source']['uv'].clamp(1e-4, 1-1e-4))
        torch.testing.assert_close(target['angle'], output['source']['angle'])


def test_target_only_gradient_reaches_proposals_decoder_backbone_and_all_query_rows(model):
    model.train()
    # Open residual heads to test every differentiable route, including scores.
    with torch.no_grad():
        for head in model.arrangement.residuals:
            head[-1].weight.normal_(std=.002)
    output = model(**inputs())
    watched = [s[k] for s in output['detector_stages'] for k in ('tokens', 'boxes')]
    # Earlier classifier heads only feed discrete native proposal selection or
    # auxiliary Detection supervision, not the Target loss's continuous path.
    watched.append(output['detector_stages'][-1]['logits'])
    for tensor in watched:
        tensor.retain_grad()
    terms, _ = S0Criterion()(output, [truth()])
    terms['target'].backward()
    for tensor in watched:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    for stage in output['detector_stages']:
        assert (stage['tokens'].grad.abs().sum(-1) > 0).all()
    for module in (model.perception.detector.model.backbone, model.perception.detector.model.encoder,
                   model.perception.detector.model.decoder, model.perception.detector.model.enc_output,
                   model.text_projection, model.arrangement):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and sum(g.abs().sum() for g in gradients) > 0
    assert all(p.grad is None for p in model.text_encoder.parameters())


def test_auxiliary_detection_reuses_final_assignment_and_keeps_all_queries(model, monkeypatch):
    import knolling_model.losses as losses
    original = losses.linear_sum_assignment
    calls = []
    def counted(cost):
        calls.append(cost.shape)
        return original(cost)
    monkeypatch.setattr(losses, 'linear_sum_assignment', counted)
    output = model(**inputs())
    terms, _ = S0Criterion()(output, [truth()])
    assert calls == [(7, 2)]
    terms['loss'].backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # Inference filtering happens only after all queries completed arrangement.
    output['source']['objectness_logits'] = torch.full((1, 7), -100.)
    assert output['targets'][-1]['uv'].shape[1] == 7
    assert decode_predictions(output)[0]['query_index'].numel() == 0


def test_frozen_normalization_statistics_and_optimizer_groups(model):
    model.train()
    assert not model.text_encoder.training
    assert all(not m.training for m in model.perception.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm))
    opt = make_optimizer(model, read_config('configs/s0_rtdetr.json'))
    entries = {id(p): group for group in opt.param_groups for p in group['params']}
    assert len(entries) == sum(len(group['params']) for group in opt.param_groups)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert id(parameter) not in entries
            continue
        expected = 1e-5 if name.startswith('perception.detector.model.backbone.') else 1e-4
        assert entries[id(parameter)]['lr'] == expected


def test_target_query_permutation_preserves_source_identity(model):
    model.eval()
    with torch.no_grad():
        for head in model.arrangement.residuals:
            head[-1].weight.normal_(std=.002)
        output = model(**inputs())
        permutation = torch.tensor([6, 0, 5, 1, 4, 2, 3])
        source = {k: v[:, permutation] for k, v in output['source'].items()}
        text = torch.randn(1, 4, 256)
        mask = torch.ones(1, 4, dtype=torch.bool)
        first = model.arrangement(output['object_tokens'], output['source'], text, mask)
        second = model.arrangement(output['object_tokens'][:, permutation], source, text, mask)
        for a, b in zip(first, second):
            for key in a:
                torch.testing.assert_close(a[key][:, permutation], b[key], atol=2e-5, rtol=2e-5)
