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


def config(tmp_path=None):
    cfg = runner.read_config('configs/s0.json')
    if tmp_path is not None:
        # Provenance needs a manifest, but runner unit tests need no pilot data.
        root = tmp_path/'synthetic-data'
        root.mkdir(exist_ok=True)
        (root/'manifest.jsonl').write_text('{}\n')
        cfg = {**cfg, 'dataset': str(root)}
    return cfg


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
    evidence = runner.provenance(config(tmp_path))
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


def test_checkpoint_accepts_implicit_legacy_policy_and_rejects_reweighting(tmp_path):
    cfg = config(tmp_path)
    model = TinyModel()
    opt = runner.make_optimizer(model, cfg)
    evidence = runner.provenance(cfg)
    old = dict(evidence, loss={k: v for k, v in evidence['loss'].items()
                               if k != 'objectness_normalization'})
    path = tmp_path/'legacy.pt'
    runner.save_checkpoint(path, model, opt, {'updates': 0}, None, old)
    runner.load_checkpoint(path, model, opt, evidence)
    changed = dict(evidence, loss={**evidence['loss'], 'objectness_normalization': 'balanced_groups'})
    with pytest.raises(ValueError, match='loss'):
        runner.load_checkpoint(path, model, opt, changed)
    assert runner.make_criterion(cfg).objectness_normalization == 'legacy_query_weight'
    assert runner.make_criterion({**cfg, 'objectness_normalization': 'balanced_groups'}).objectness_normalization == 'balanced_groups'
    grouped = runner.provenance({**cfg, 'objectness_normalization': 'balanced_groups'})
    assert grouped['loss']['negative_objectness_weight'] is None


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
               'task_success': 0., 'raw_geometry_success': 0.,
               **{k: {'p95': None} for k in ('source_position_mm','target_position_mm','source_size_mm',
                                           'source_angle_deg','target_angle_deg')}}
    monkeypatch.setattr(runner, 'evaluate', lambda *args: (metrics, []))
    cfg = {**config(tmp_path), 'max_updates': 3, 'checkpoint_every': 1}
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


def evaluation_metrics(gate=False, task=0., geometry=0., target=3.):
    return {'count_accuracy': 1., 'correspondence_coverage': 1., 's0_gate': gate,
            'task_success': task, 'raw_geometry_success': geometry,
            **{k: {'p95': target} for k in ('source_position_mm', 'target_position_mm',
                                           'source_size_mm', 'source_angle_deg', 'target_angle_deg')}}


def test_checkpoint_gate_admission_and_spec_tie_breaks():
    rank = runner.checkpoint_score
    # Previously a failing model with a single 1.1 mm error could outrank
    # a passing model with five sub-threshold errors by their summed errors.
    failing = evaluation_metrics(task=1., geometry=1., target=0.)
    failing['source_position_mm']['p95'] = 1.1
    passing = evaluation_metrics(gate=True, task=.5, geometry=.5, target=.8)
    assert rank(passing) < rank(failing)
    assert rank(evaluation_metrics(task=.9, geometry=.1, target=5.)) < rank(
        evaluation_metrics(task=.8, geometry=1., target=.1))
    assert rank(evaluation_metrics(task=.9, geometry=1., target=5.)) < rank(
        evaluation_metrics(task=.9, geometry=.9, target=.1))
    assert rank(evaluation_metrics(target=.5)) < rank(evaluation_metrics(target=1.))
    assert rank(evaluation_metrics(target=1.)) < rank(evaluation_metrics(target=None))


class OptimizerFixtureModel(nn.Module):
    """Small CPU regression fixture, exercising actual AdamW and RNG replay."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 1)
        self.text_encoder = nn.Linear(1, 1).requires_grad_(False)

    def forward(self, images):
        jitter = torch.rand_like(images)*.01 if self.training else 0.
        return self.projection(images+jitter)


def install_optimizer_fixture(monkeypatch, tmp_path, metrics):
    class Dataset:
        records = [{'scene_family_id': str(i)} for i in range(25)]
        def __len__(self):
            return 25
        def __getitem__(self, i):
            return torch.tensor([[i/25+1.]])
    def collate(items):
        return Batch({'images': torch.cat(items)}, [{} for _ in items], [])
    models, optimizers, visits = [], [], []
    def prepare(cfg, *args):
        runner.seed_everything(cfg['seed'])
        model = OptimizerFixtureModel()
        models.append(model)
        return Dataset(), model, collate
    class Criterion:
        def __call__(self, output, _):
            return {'loss': (output-2.).square().mean()}, []
    real_optimizer, real_backward = runner.make_optimizer, runner.backward_group
    def optimizer(*args):
        opt = real_optimizer(*args)
        optimizers.append(opt)
        return opt
    def backward(*args):
        visits.append(list(args[4]))
        return real_backward(*args)
    monkeypatch.setattr(runner, 'prepare', prepare)
    monkeypatch.setattr(runner, 'S0Criterion', Criterion)
    monkeypatch.setattr(runner, 'make_optimizer', optimizer)
    monkeypatch.setattr(runner, 'backward_group', backward)
    monkeypatch.setattr(runner, 'evaluate', lambda *args: (metrics, []))
    cfg = {**config(tmp_path), 'max_updates': 5, 'checkpoint_every': 1}
    return cfg, models, optimizers, visits


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_successful_resume_preserves_weights_optimizer_cursor_and_best(tmp_path, monkeypatch):
    metrics = evaluation_metrics(gate=True, task=1., geometry=1., target=.5)
    cfg, models, optimizers, visits = install_optimizer_fixture(monkeypatch, tmp_path, metrics)
    first = runner.train(cfg, tmp_path/'successful', device='cpu')
    assert first['cursor']['updates'] == 2 < cfg['max_updates']
    saved = torch.load(tmp_path/'successful/last.pt', weights_only=True)
    assert all(state['step'] == 2 for state in saved['optimizer']['state'].values())
    def diagnostic_evaluation(*args):
        random.random(); np.random.rand(); torch.rand(1)
        return metrics, []
    monkeypatch.setattr(runner, 'evaluate', diagnostic_evaluation)
    visits.clear()
    resumed = runner.train(cfg, tmp_path/'resumed-success', device='cpu', resume=tmp_path/'successful/last.pt')
    after = torch.load(tmp_path/'resumed-success/last.pt', weights_only=True)
    assert not visits
    assert resumed['stop_reason'] == 'memorization_gate' and resumed['cursor'] == first['cursor']
    for field in ('model', 'optimizer', 'cursor', 'rng'):
        assert_nested_equal(saved[field], after[field])
    best = torch.load(tmp_path/'resumed-success/best.pt', weights_only=True)
    assert best['best_score'][0] is False
    assert_nested_equal(best['model'], saved['model'])
    assert_nested_equal(best['rng'], saved['rng'])


@pytest.mark.parametrize('interrupt_after', [1, 2, 3])
def test_real_optimizer_interrupted_resume_matches_uninterrupted(tmp_path, monkeypatch, interrupt_after):
    cfg, models, optimizers, visits = install_optimizer_fixture(
        monkeypatch, tmp_path, evaluation_metrics())
    result = runner.train(cfg, tmp_path/'uninterrupted', device='cpu')
    baseline = torch.load(tmp_path/'uninterrupted/last.pt', weights_only=True)
    baseline_visits = visits[:]
    visits.clear()
    save = runner.save_checkpoint
    class Interrupted(Exception):
        pass
    def interrupt(path, *args):
        save(path, *args)
        if path.name == 'last.pt' and args[2]['updates'] == interrupt_after:
            raise Interrupted
    monkeypatch.setattr(runner, 'save_checkpoint', interrupt)
    with pytest.raises(Interrupted):
        runner.train(cfg, tmp_path/'interrupted', device='cpu')
    monkeypatch.setattr(runner, 'save_checkpoint', save)
    assert visits == baseline_visits[:interrupt_after]
    visits.clear()
    resumed = runner.train(cfg, tmp_path/'continued', device='cpu', resume=tmp_path/'interrupted/last.pt')
    actual = torch.load(tmp_path/'continued/last.pt', weights_only=True)
    assert visits == baseline_visits[interrupt_after:]
    assert resumed['cursor'] == result['cursor']
    for field in ('model', 'optimizer', 'cursor', 'rng'):
        assert_nested_equal(baseline[field], actual[field])
    assert all(state['step'] == 5 for state in actual['optimizer']['state'].values())


def test_best_checkpoint_promotes_gate_passing_state(tmp_path, monkeypatch):
    failing = evaluation_metrics(task=1., geometry=1., target=0.)
    failing['source_position_mm']['p95'] = 1.1
    passing = evaluation_metrics(gate=True, task=.5, geometry=.5, target=.8)
    cfg, _, _, _ = install_optimizer_fixture(monkeypatch, tmp_path, failing)
    reports = iter([failing, passing])
    monkeypatch.setattr(runner, 'evaluate', lambda *args: (next(reports), []))
    result = runner.train(cfg, tmp_path/'promotion', device='cpu')
    best = torch.load(tmp_path/'promotion/best.pt', weights_only=True)
    last = torch.load(tmp_path/'promotion/last.pt', weights_only=True)
    assert result['s0_gate'] and result['cursor']['updates'] == 4
    assert best['cursor'] == last['cursor'] == result['cursor']
    assert best['best_score'] == runner.checkpoint_score(passing)
    assert_nested_equal(best['model'], last['model'])


def test_progress_resume_restores_actual_updates_and_running_average(tmp_path, monkeypatch):
    metrics = evaluation_metrics()
    cfg, _, _, _ = install_optimizer_fixture(monkeypatch, tmp_path, metrics)
    runner.train(cfg, tmp_path/'logged', device='cpu')
    events = [json.loads(line) for line in (tmp_path/'logged/metrics.jsonl').read_text().splitlines()]
    updates = [e for e in events if e['event'] == 'update']
    progress = json.loads((tmp_path/'logged/progress.json').read_text())
    assert progress['updates'] == progress['loss_updates'] == 5
    assert progress['percent'] == 100
    assert progress['average_loss'] == pytest.approx(sum(e['losses']['loss'] for e in updates)/5)
    saved = torch.load(tmp_path/'logged/last.pt', weights_only=True)
    assert saved['progress']['updates'] == 5
    runner.train(cfg, tmp_path/'logged-resumed', device='cpu', resume=tmp_path/'logged/last.pt')
    continued = json.loads((tmp_path/'logged-resumed/progress.json').read_text())
    assert continued['updates'] == 5 and continued['loss_updates'] == 5
    assert continued['average_loss'] == progress['average_loss']
    assert continued['latest_evaluation']['s0_gate'] is False


def test_progress_counts_updates_not_micro_batches_and_rejects_jumps(tmp_path):
    from knolling_model.progress import TrainingProgress
    with TrainingProgress(tmp_path, 2000, 850, 'cpu',
                          {'loss_sum': 850., 'loss_updates': 850, 'elapsed_seconds': 100.}) as progress:
        assert progress.bar.n == 850
        progress.update(851, 2.)
        assert progress.bar.n == 851
        assert progress.snapshot()['average_loss'] == pytest.approx(852/851)
        assert progress.snapshot()['elapsed_seconds'] >= 100
        with pytest.raises(ValueError, match='actual optimizer'):
            progress.update(853, 2.)
