import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from knolling_model import runner
from knolling_model.__main__ import parser
from knolling_model.data import Batch


def config():
    return runner.read_config('configs/s0.json')


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.body = nn.Linear(2, 2)
        self.projection = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(2)
        self.attention = nn.MultiheadAttention(2, 1)
        self.text_encoder = nn.Linear(2, 2).requires_grad_(False)


def test_optimizer_covers_trainable_once_and_all_bias_norm_exclusions():
    model = TinyModel()
    opt = runner.make_optimizer(model, config())
    entries = {id(p): group for group in opt.param_groups for p in group['params']}
    assert len(entries) == sum(len(group['params']) for group in opt.param_groups)
    assert set(entries) == {id(p) for p in model.parameters() if p.requires_grad}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = entries[id(parameter)]
        assert group['lr'] == (1e-5 if name.startswith('backbone.body.') else 1e-4)
        assert group['weight_decay'] == (0 if name.endswith('bias') or name.startswith('norm.') else 1e-4)


@pytest.mark.parametrize('n,micro', [(16,2), (9,2), (9,4), (9,1)])
def test_accumulation_is_exact_scene_mean_even_for_tail(n, micro):
    # A scalar algebra fixture, not neural overfitting or optimizer updates.
    model = nn.Linear(1, 1, bias=False)
    values = [torch.tensor([[float(i)]]) for i in range(n)]
    class InputOnly(nn.Module):
        def forward(self, images):
            return model(images)
    def collate(items):
        return Batch({'images': torch.cat(items)}, [{} for _ in items], [])
    def criterion(out, labels):
        return {'loss': out.square().mean()}, []
    result = runner.backward_group(InputOnly(), criterion, values, collate, list(range(n)), micro, 'cpu')
    accumulated = model.weight.grad.clone()
    model.zero_grad()
    expected = model(torch.cat(values)).square().mean()
    expected.backward()
    assert result['loss'] == pytest.approx(float(expected.detach()), rel=1e-6)
    assert torch.allclose(accumulated, model.weight.grad)
    assert [len(g) for g in runner.update_groups(list(range(25)), 16)] == [16,9]


def test_checkpoint_roundtrip_cursor_rng_optimizer_and_provenance(tmp_path):
    runner.seed_everything(1203)
    model = TinyModel()
    opt = runner.make_optimizer(model, config())
    # Populate Adam buffers without an optimizer update, then verify serialization.
    p = next(p for p in model.parameters() if p.requires_grad)
    opt.state[p] = {'step': torch.tensor(7.), 'exp_avg': torch.ones_like(p), 'exp_avg_sq': torch.full_like(p, 2)}
    evidence = runner.provenance(config())
    cursor = {'epoch': 4, 'group': 1, 'updates': 9}
    path = tmp_path/'last.pt'
    runner.save_checkpoint(path, model, opt, cursor, (0., 0., 7.), evidence)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    saved = torch.load(path, weights_only=True)
    assert not any(k.startswith('text_encoder.') for k in saved['model'])
    restored = TinyModel()
    other = runner.make_optimizer(restored, config())
    actual_cursor, score = runner.load_checkpoint(path, restored, other, evidence)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])
    assert actual_cursor == cursor and score == (0., 0., 7.)
    for key, value in saved['model'].items():
        assert torch.equal(restored.state_dict()[key], value)
    assert torch.equal(next(iter(other.state.values()))['exp_avg_sq'], torch.full_like(p, 2))
    with pytest.raises(ValueError, match='selection_sha256'):
        runner.load_checkpoint(path, restored, other, {**evidence, 'selection_sha256': 'changed'})
    assert not list(tmp_path.glob('*.tmp'))


def test_cli_cannot_start_training_implicitly_and_config_budget(tmp_path):
    with pytest.raises(SystemExit):
        parser().parse_args(['train', '--run-dir', str(tmp_path)])
    args = parser().parse_args(['preflight'])
    assert args.device == 'cpu' and not args.download_weights
    invalid = {**config(), 'max_updates': 2001}
    path = tmp_path/'invalid.json'
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match='2000'):
        runner.read_config(path)
    with pytest.raises(FileExistsError):
        runner.train(config(), tmp_path)
    with pytest.raises(ValueError, match='immutable'):
        runner.train(config(), 'artifacts/pilot_v1_3_en/forbidden-run')
    with pytest.raises(ValueError, match='immutable'):
        runner.preflight(config(), 'artifacts/pilot_v1_3_en/forbidden-report.json')


def test_training_orchestration_resume_without_real_training(tmp_path, monkeypatch):
    # Exercise loop/cursor/evaluation/checkpoint control with no network forward,
    # backward or weight update. Real model tests cover forward/backward separately.
    dataset = SimpleNamespace(records=[{'scene_family_id': str(i)} for i in range(25)])
    class FakeDataset:
        records = dataset.records
        def __len__(self):
            return 32
    model = TinyModel()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    steps, groups = [], []
    monkeypatch.setattr(runner, 'prepare', lambda *args: (FakeDataset(), model, None))
    real_optimizer = runner.make_optimizer
    def optimizer(*args):
        opt = real_optimizer(*args)
        opt.step = lambda: steps.append(True)
        return opt
    monkeypatch.setattr(runner, 'make_optimizer', optimizer)
    def backward(*args):
        groups.append(args[4])
        return {'loss': 3.}
    monkeypatch.setattr(runner, 'backward_group', backward)
    monkeypatch.setattr(runner, 'batches', lambda *args: [])
    metrics = {'count_accuracy': 0., 'correspondence_coverage': 0., 's0_gate': False,
               **{k: {'p95': None} for k in ('source_position_mm','target_position_mm','source_size_mm',
                                           'source_angle_deg','target_angle_deg')}}
    monkeypatch.setattr(runner, 'evaluate', lambda *args: (metrics, []))
    cfg = {**config(), 'max_updates': 3, 'checkpoint_every': 1}
    result = runner.train(cfg, tmp_path/'first', device='cpu')
    assert len(steps)==3 and [len(g) for g in groups]==[16,9,16]
    assert result['cursor'] == {'epoch': 1, 'group': 1, 'updates': 3}
    # Rewind the test checkpoint to after one update, then verify next group is
    # epoch zero's exact nine-element tail, followed by epoch one's first group.
    checkpoint = torch.load(tmp_path/'first/last.pt', weights_only=True)
    checkpoint['cursor'] = {'epoch': 0, 'group': 1, 'updates': 1}
    path = tmp_path/'resume-fixture.pt'
    torch.save(checkpoint, path)
    original_groups = groups[:]
    groups.clear(); steps.clear()
    resumed = runner.train(cfg, tmp_path/'continued', device='cpu', resume=path)
    assert groups == original_groups[1:] and len(steps)==2
    assert resumed['cursor']==result['cursor']
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
    # Resuming a completed budget performs evaluation but no phantom update.
    steps.clear()
    finished = runner.train(cfg, tmp_path/'finished', device='cpu', resume=tmp_path/'continued/last.pt')
    assert not steps and finished['last_evaluation'] is not None
