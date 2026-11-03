from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torchvision.ops import FrozenBatchNorm2d
from knolling_model.model import KnollingModel, SourceHeads, ArrangementDecoder, decode_predictions


class TextFixture(nn.Module):
    """Test-only small encoder; production factory always loads the pinned model."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 384)
        self.dropout = nn.Dropout(.5)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


def fixture_model():
    torch.manual_seed(5)
    return KnollingModel(TextFixture(), pretrained_backbone=False).train()


def inputs():
    return {'images': torch.rand(1, 3, 384, 576),
            'input_ids': torch.tensor([[1, 2, 3, 0, 0]]),
            'attention_mask': torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)}


def open_heads(model):
    # Simulate nonzero residual weights WITHOUT running an optimization step.
    with torch.no_grad():
        for head in model.arrangement.residuals:
            head[-1].weight.normal_(std=.002)


def test_exact_architecture_dimensions_frozen_bn_and_initial_copy():
    model = fixture_model()
    batch = inputs()
    bn = {n: v.clone() for n, v in model.backbone.named_buffers() if 'running_' in n}
    output = model(**batch)
    memory, positions = model.backbone(batch['images'])
    assert memory.shape == (1, 4536, 256) and positions.shape == (1, 4536, 256)
    assert output['object_tokens'].shape == (1, 12, 256)
    assert len(model.object_blocks) == 4 and len(output['targets']) == 3
    assert output['source']['size'].shape == (1, 12, 2)
    assert output['source']['color_logits'].shape == (1, 12, 3)
    assert output['source']['shape_logits'].shape == (1, 12, 3)
    assert torch.allclose(output['source']['size'], torch.tensor([.060, .035]).expand(1, 12, 2), atol=1e-7)
    for stage in output['targets']:
        assert stage['uv'].shape == (1, 12, 2) and stage['angle'].shape == (1, 12)
        assert torch.allclose(stage['uv'], output['source']['uv'], atol=1e-6)
        assert torch.allclose(stage['angle'], output['source']['angle'], atol=1e-6)
    assert not model.text_encoder.training
    assert all(not p.requires_grad for p in model.text_encoder.parameters())
    assert any(isinstance(m, FrozenBatchNorm2d) for m in model.backbone.modules())
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.backbone.modules())
    for n, v in model.backbone.named_buffers():
        if n in bn:
            assert torch.equal(v, bn[n])
    with pytest.raises(TypeError):
        model(**batch, selected_ids=['gt'])


def test_query_identity_is_preserved_under_query_permutation():
    model = fixture_model()
    open_heads(model)
    batch = inputs()
    with torch.no_grad():
        first = model(**batch)
        permutation = torch.tensor([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6])
        model.queries.copy_(model.queries[permutation])
        second = model(**batch)
    for key in first['source']:
        assert torch.allclose(first['source'][key][:, permutation], second['source'][key], atol=2e-5), key
    for a, b in zip(first['targets'], second['targets']):
        for key in a:
            assert torch.allclose(a[key][:, permutation], b[key], atol=2e-5)


def test_text_padding_and_instruction_effect_after_residual_heads_open():
    model = fixture_model()
    open_heads(model)
    batch = inputs()
    with torch.no_grad():
        first = model(**batch)
        padded = dict(batch, input_ids=torch.tensor([[1, 2, 3, 30, 31]]))
        second = model(**padded)
        changed = model(**dict(batch, input_ids=torch.tensor([[5, 6, 7, 0, 0]])))
    assert torch.allclose(first['targets'][-1]['uv'], second['targets'][-1]['uv'], atol=1e-7)
    assert torch.equal(first['source']['uv'], changed['source']['uv'])
    assert not torch.allclose(first['targets'][-1]['uv'], changed['targets'][-1]['uv'], atol=1e-6)


def test_initial_residual_heads_get_gradients_and_open_heads_connect_entire_graph():
    model = fixture_model()
    batch = inputs()
    out = model(**batch)
    sum(t['uv'].square().mean() + t['angle'].mean() for t in out['targets']).backward()
    for head in model.arrangement.residuals:
        assert head[-1].weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    open_heads(model)
    out = model(**batch)
    for t in out['targets']:
        t['uv'].retain_grad()
    for key in ('uv', 'size', 'angle', 'objectness_logits'):
        out['source'][key].retain_grad()
    # Final stage alone must reach earlier poses and predicted source geometry.
    (out['targets'][-1]['uv'].square().mean() + out['targets'][-1]['angle'].mean()).backward()
    for stage in out['targets']:
        assert stage['uv'].grad is not None and stage['uv'].grad.abs().sum() > 0
    for key in ('uv', 'size', 'angle', 'objectness_logits'):
        grad = out['source'][key].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, key
    for module in (model.backbone, model.object_blocks, model.arrangement.blocks,
                   model.arrangement.geometry, model.text_projection):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert all(p.grad is None for p in model.text_encoder.parameters())


def test_angle_zero_vector_size_positivity_and_threshold_only_decode():
    heads = SourceHeads()
    with torch.no_grad():
        heads.direction[-1].bias.zero_()
    output = heads(torch.randn(1, 12, 256, requires_grad=True))
    output['angle'].sum().backward()
    assert torch.isfinite(heads.direction[-1].bias.grad).all()
    assert (output['size'][..., 0] >= output['size'][..., 1]).all()
    assert (output['size'] > 0).all()
    with torch.no_grad():
        output['objectness_logits'].fill_(-10)
        output['objectness_logits'][0, [1, 7]] = 0
    decoded = decode_predictions({'source': output, 'targets': [{'uv': output['uv'], 'angle': output['angle']}]})
    assert decoded[0]['query_index'].tolist() == [1, 7]


def test_refinement_step_scales_and_no_hard_objectness_mask():
    decoder = ArrangementDecoder()
    with torch.no_grad():
        for head in decoder.residuals:
            head[-1].bias.copy_(torch.tensor([.1, -.2, .3]))
    source = {'uv': torch.full((1, 12, 2), .5), 'size': torch.tensor([.06, .035]).expand(1, 12, 2),
              'angle': torch.zeros(1, 12), 'objectness_logits': torch.full((1, 12), -100.)}
    stages = decoder(torch.zeros(1, 12, 256), source, torch.randn(1, 3, 256), torch.ones(1, 3, dtype=torch.bool))
    for stage, scale in zip(stages, (1., 1.5, 1.75)):
        assert torch.allclose(stage['uv'], (scale * torch.tensor([.1, -.2])).sigmoid().expand(1, 12, 2))
    assert not torch.equal(stages[-1]['uv'], source['uv'])
