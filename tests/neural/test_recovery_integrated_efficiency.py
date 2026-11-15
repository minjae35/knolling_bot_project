"""Bounded synthetic CPU/CUDA tests; no experiment dataset or pretrained model."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from knolling_dataset.schema import digest
from knolling_model.arrangement import integrated
from knolling_model.arrangement.driver import save_exclusive
from knolling_model.arrangement.readout import HeadFitter
from test_recovery_integrated import setup, make, equal, metrics
from test_recovery_readout import DEVICES, Fixture


def solver_config(tmp_path, device):
    c, m, parent = setup(tmp_path, device)
    c['integrated_pose'].update(head_time_mode='solver', finalize_reserve_seconds=10)
    raw = torch.load(parent, weights_only=True)
    raw.update(config=deepcopy(c), config_sha256=digest(c))
    parent = tmp_path/'solver-ground.pt'; save_exclusive(parent, raw)
    return c, m, parent


@pytest.mark.parametrize('device', DEVICES)
def test_overhead_does_not_consume_solver_allowance_and_hashes_are_reused(tmp_path, monkeypatch, device):
    c, m, parent = solver_config(tmp_path, device)
    whole, _ = make(tmp_path, 'reference', c, m)
    reference = whole.run('micro_pose', parent=parent, micro_case='axis')
    now = [0.]; hashes = {}; original_hash = integrated.file_hash
    original_save = integrated.save_exclusive; original_step = HeadFitter.step
    def file_hash(path):
        key = str(path); hashes[key] = hashes.get(key, 0) + 1
        now[0] += .2
        return original_hash(path)
    def save(path, value):
        original_save(path, value); now[0] += 1.5
    def step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs); now[0] += .25
        return result
    monkeypatch.setattr(integrated, 'file_hash', file_hash)
    monkeypatch.setattr(integrated, 'save_exclusive', save)
    monkeypatch.setattr(HeadFitter, 'step', step)
    e, events = make(tmp_path, 'slow_io', c, Fixture().to(device), clock=lambda: now[0])
    actual = e.run('micro_pose', parent=parent, micro_case='axis')
    assert actual['updates'] == reference['updates'] and actual['status'] == reference['status']
    assert all(n == 1 for p, n in hashes.items() if '/slow_io/checkpoint-' in p)
    a = integrated.load_integrated(reference['checkpoint'], c, 'fixture', 'test')
    b = integrated.load_integrated(actual['checkpoint'], c, 'fixture', 'test')
    equal(a['model'], b['model']); equal(a['optimizer'], b['optimizer'])
    phases = b['integrated']['phase_results']
    for row in phases:
        if row['phase'] in integrated.HEADS:
            assert row['timing']['solver'] == row['updates'] * .25
            assert row['timing'].get('checkpoint_io', 0) + row['timing'].get('journal_io', 0) > 0
    import json
    timing = json.loads((tmp_path/'slow_io/timing.json').read_text())
    assert sum(timing['timing']['total'].values()) == pytest.approx(timing['elapsed_seconds'])
    assert len(list((tmp_path/'slow_io').glob('features-*.pt'))) == 2


@pytest.mark.parametrize('device', DEVICES)
def test_solver_deadline_aborts_trial_and_is_retained_on_resume(tmp_path, monkeypatch, device):
    c, m, parent = solver_config(tmp_path, device)
    now = [0.]; original_loss = HeadFitter.loss
    def loss(self):
        result = original_loss(self); now[0] += 11.
        return result
    monkeypatch.setattr(HeadFitter, 'loss', loss)
    phase = ['coarse']
    def observe(event, **r): phase[0] = r['phase']
    e, events = make(tmp_path, 'deadline', c, m, clock=lambda: now[0], observer=observe,
                     evaluate=lambda *_: metrics(angle=1 if phase[0] == 'coarse' else 3))
    result = e.run('micro_pose', parent=parent, micro_case='axis')
    assert not any(ev == 'update' for ev, _ in events)
    saved = integrated.load_integrated(result['checkpoint'], c, 'fixture', 'test')
    equal(saved['model'], torch.load(parent, weights_only=True, map_location='cpu')['model'])
    assert saved['integrated']['timing']['total']['solver'] >= 10
    assert saved['integrated']['phase_results'][0]['readout']['reason'] == 'head_time_budget'


@pytest.mark.parametrize('device', DEVICES)
def test_immutable_base_mutation_is_not_hidden_by_digest_cache(tmp_path, device):
    c, m, parent = solver_config(tmp_path, device)
    touched = []
    def observe(event, **r):
        if event == 'update' and r['phase'] == 'head_xy' and not touched:
            raw = torch.load(r['journal'], weights_only=True)
            path = Path(raw['base']); path.write_bytes(b'corrupt fixture'); touched.append(path)
    e, _ = make(tmp_path, 'corrupt', c, m, observer=observe)
    result = e.run('micro_pose', parent=parent, micro_case='axis')
    assert result['status'] == 'rolled_back' and 'immutable artifact changed' in result['error']
    with pytest.raises(ValueError, match='base mismatch'):
        integrated.load_integrated(next((tmp_path/'corrupt').glob('readout-*.pt')), c, 'fixture', 'test')


def test_global_deadline_keeps_finalization_reserve_even_with_solver_time_policy(tmp_path):
    c, m, parent = solver_config(tmp_path, 'cpu'); now = [0.]
    def observe(event, **r):
        if event == 'update': now[0] = 111.
    e, events = make(tmp_path, 'reserve', c, m, observer=observe, clock=lambda: now[0])
    result = e.run('micro_pose', parent=parent, micro_case='axis')
    assert result['status'] == 'budget_exhausted' and result['updates'] == 1
    assert any(ev == 'evaluation' and r['elapsed_seconds'] >= 110 for ev, r in events)


@pytest.mark.parametrize('device', DEVICES)
def test_solver_allowance_survives_journal_crash_without_extra_steps(tmp_path, monkeypatch, device):
    from test_recovery_integrated_resume import ProcessCrash
    c, m, parent = solver_config(tmp_path, device)
    c['integrated_pose']['head_seconds'] = .3
    raw = torch.load(parent, weights_only=True); raw.update(config=deepcopy(c), config_sha256=digest(c))
    parent = tmp_path/'short-solver-ground.pt'; save_exclusive(parent, raw)
    now = [0.]; original = HeadFitter.step
    def step(self, *args, **kwargs):
        now[0] += .2
        return original(self, *args, **kwargs)
    monkeypatch.setattr(HeadFitter, 'step', step)
    whole, _ = make(tmp_path, 'whole', c, m, clock=lambda: now[0])
    expected = whole.run('micro_pose', parent=parent, micro_case='axis')
    now[0] = 0.; journals = []
    def crash(event, **r):
        if event == 'update' and r['phase'] == 'head_xy':
            journals.append(r['journal']); raise ProcessCrash()
    part, _ = make(tmp_path, 'part', c, Fixture().to(device), clock=lambda: now[0], observer=crash)
    with pytest.raises(ProcessCrash): part.run('micro_pose', parent=parent, micro_case='axis')
    snapshot = integrated.load_integrated(journals[0], c, 'fixture', 'test')
    assert snapshot['integrated']['timing']['phase']['solver'] == pytest.approx(.2)
    now[0] = 0.
    resumed, _ = make(tmp_path, 'resumed', c, Fixture().to(device), clock=lambda: now[0])
    actual = resumed.run('micro_pose', resume=journals[0], micro_case='axis')
    assert actual['status'] == expected['status'] and actual['updates'] == expected['updates']
    a = integrated.load_integrated(expected['checkpoint'], c, 'fixture', 'test')
    b = integrated.load_integrated(actual['checkpoint'], c, 'fixture', 'test')
    for key in ('model', 'optimizer', 'rng'): equal(a[key], b[key])
    assert a['integrated']['closure_evaluations'] == b['integrated']['closure_evaluations']


def test_cache_preparation_journal_recovers_before_first_candidate(tmp_path, monkeypatch):
    from test_recovery_integrated_resume import ProcessCrash
    c, m, parent = solver_config(tmp_path, 'cpu')
    whole, _ = make(tmp_path, 'whole', c, m)
    expected = whole.run('micro_pose', parent=parent, micro_case='axis')
    original = integrated.IntegratedEngine.journal; journals = []
    def crash(self):
        path = original(self)
        if self.state['phase_update'] == 0:
            journals.append(path); raise ProcessCrash()
        return path
    monkeypatch.setattr(integrated.IntegratedEngine, 'journal', crash)
    part, _ = make(tmp_path, 'part', c, Fixture())
    with pytest.raises(ProcessCrash): part.run('micro_pose', parent=parent, micro_case='axis')
    monkeypatch.setattr(integrated.IntegratedEngine, 'journal', original)
    resumed, _ = make(tmp_path, 'resumed', c, Fixture())
    actual = resumed.run('micro_pose', resume=journals[0], micro_case='axis')
    assert actual['updates'] == expected['updates'] and actual['status'] == expected['status']
    a = integrated.load_integrated(expected['checkpoint'], c, 'fixture', 'test')
    b = integrated.load_integrated(actual['checkpoint'], c, 'fixture', 'test')
    equal(a['model'], b['model']); equal(a['optimizer'], b['optimizer']); equal(a['rng'], b['rng'])


def test_timing_report_failure_does_not_discard_terminal_result(tmp_path, monkeypatch):
    from knolling_model.arrangement import driver
    c, m, parent = solver_config(tmp_path, 'cpu')
    def fail(*args, **kwargs): raise OSError('diagnostic disk error')
    monkeypatch.setattr(driver, 'save_json_exclusive', fail)
    e, events = make(tmp_path, 'result', c, m, evaluate=lambda *_: metrics(passed=True))
    result = e.run('micro_pose', parent=parent, micro_case='axis')
    assert result['status'] == 'passed' and Path(result['checkpoint']).exists()
    assert any(ev == 'timing_report_error' for ev, _ in events)
