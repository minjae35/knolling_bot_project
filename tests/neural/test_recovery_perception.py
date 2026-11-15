import pytest
import torch
from torch.nn import functional as F

from knolling_model.arrangement.perception import RecoveryPerception


def tiny_detector():
    from transformers import RTDetrConfig, RTDetrResNetConfig, RTDetrForObjectDetection
    torch.manual_seed(12)
    backbone = RTDetrResNetConfig(embedding_size=8, hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 1, 1], layer_type='basic', out_features=['stage2', 'stage3', 'stage4'])
    config = RTDetrConfig(backbone_config=backbone, encoder_in_channels=[16, 32, 64],
        encoder_hidden_dim=32, encoder_ffn_dim=64, decoder_in_channels=[32, 32, 32],
        decoder_layers=2, decoder_ffn_dim=64, d_model=256, num_queries=7, num_labels=5,
        hidden_expansion=.5, num_denoising=0, disable_custom_kernels=True,
        attention_dropout=0., activation_dropout=0., dropout=0.)
    return RTDetrForObjectDetection(config)


def truth():
    return dict(source_xy=torch.tensor([[.15, .2], [.35, .3]]), source_angle=torch.tensor([.2, -.3]),
        target_xy=torch.tensor([[.2, .2], [.3, .2]]), target_angle=torch.tensor([0., 0.]),
        size=torch.tensor([[.06, .035], [.055, .025]]), color=torch.tensor([0, 1]),
        shape=torch.tensor([1, 2]), move=torch.tensor([True, True]), selected=torch.tensor([True, True]),
        axis=torch.tensor(0), region=torch.tensor(0))


@pytest.fixture
def perception():
    return RecoveryPerception(tiny_detector(), num_denoising=4)


def test_foreground_heads_keep_geometry_and_native_eval_equivalence():
    detector = tiny_detector()
    original = detector.bbox_embed[0].layers[0].weight.detach().clone()
    model = RecoveryPerception(detector, num_denoising=4).eval()
    torch.testing.assert_close(original, detector.bbox_embed[0].layers[0].weight)
    assert detector.model.decoder.class_embed is detector.class_embed
    assert all(h.out_features == 1 for h in detector.class_embed)
    image = torch.rand(1, 3, 384, 576)
    with torch.no_grad():
        native = detector(pixel_values=F.interpolate(image, (640, 640), mode='bilinear', align_corners=False))
        output = model(image)
    assert output['source']['uv'].shape == (1, 7, 2)
    torch.testing.assert_close(output['stages'][-1]['boxes'], native.pred_boxes, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(output['stages'][-1]['logits'], native.logits, atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError, match='evaluation'):
        model.supervised(image, [truth()])


def test_dn_isolation_native_losses_and_gradients(perception, monkeypatch):
    from transformers.loss.loss_rt_detr import RTDetrHungarianMatcher
    image = torch.rand(2, 3, 384, 576)
    truths = [truth(), {k: (v[:1] if v.ndim else v) for k, v in truth().items()}]
    model = perception.train()
    seen = []
    original = RTDetrHungarianMatcher.forward
    def capture(self, output, labels):
        seen.append(output['logits'].shape[1])
        return original(self, output, labels)
    monkeypatch.setattr(RTDetrHungarianMatcher, 'forward', capture)
    with torch.no_grad():
        clean = model(image)
    output = model.supervised(image, truths)
    assert output['dn_meta'] is not None
    assert output['dn_stages'][0]['logits'].shape[1] > 0
    # Normal queries must be independent of injected ground truth, including B>1.
    for a, b in zip(clean['stages'], output['stages']):
        torch.testing.assert_close(a['boxes'], b['boxes'], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(a['logits'], b['logits'], atol=3e-5, rtol=3e-5)
    losses, matches = model.detection_loss(output, truths)
    assert set(seen) == {7}  # DN never participates in Hungarian matching.
    assert any('_dn_' in k for k in losses) and any('_aux_' in k for k in losses)
    assert [len(p) for p, _ in matches] == [2, 1]
    sum(losses.values()).backward()
    for module in (model.detector.model.backbone, model.detector.model.enc_score_head,
                   model.detector.class_embed, model.physical,
                   model.detector.model.denoising_class_embed):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
