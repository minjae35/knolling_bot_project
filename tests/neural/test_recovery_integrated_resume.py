"""Crash boundaries on CPU/CUDA tensor fixtures; no actual model training."""
from copy import deepcopy

import pytest
import torch

from knolling_model.recovery.driver import save_exclusive
from knolling_model.recovery.integrated import HEADS, load_integrated
from knolling_model.recovery.readout import HeadFitter
from test_recovery_integrated import setup, make, metrics, equal
from test_recovery_readout import Fixture, DEVICES


class ProcessCrash(BaseException):
    """Bypass normal pause/exception cleanup, like a process dying after fsync."""


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('phase', HEADS)
@pytest.mark.parametrize('outcome', ['pass', 'regression', 'converged', 'cap'])
def test_journal_crash_preserves_decisions_weights_and_optimizers(tmp_path, monkeypatch, device, phase, outcome):
    c, model, parent = setup(tmp_path, device)
    target = {'update': None}
    active = {}
    original_step = HeadFitter.step

    def step(self, *args, **kwargs):
        value, converged = original_step(self, *args, **kwargs)
        return value, converged or (outcome == 'converged' and active['phase'] == phase)
    monkeypatch.setattr(HeadFitter, 'step', step)

    def runner(name, model, crash=False):
        active.update(phase='coarse', update=0)
        target_steps = 0
        journal = []
        def observe(event, **r):
            nonlocal target_steps
            active.update(phase=r['phase'], update=r['update'])
            if event == 'update' and r['phase'] == phase:
                target_steps += 1
                if target_steps == (2 if outcome == 'cap' else 1):
                    if target['update'] is None: target['update'] = r['update']
                    journal.append(r['journal'])
                    if crash: raise ProcessCrash()
        def evaluate(*_):
            candidate = active['phase'] == phase and target['update'] is not None and active['update'] >= target['update']
            return metrics(position=20 if active['update'] == 0 else 5,
                success=[] if candidate and outcome == 'regression' else ['stable'],
                passed=candidate and outcome == 'pass')
        engine, events = make(tmp_path, name, c, model, observer=observe, evaluate=evaluate)
        return engine, events, journal

    whole, _, _ = runner('whole', model)
    expected = whole.run('micro_pose', parent=parent, micro_case='axis')
    assert target['update'] is not None
    assert expected['status'] == {'pass':'passed', 'regression':'rolled_back'}.get(outcome, 'budget_exhausted'), expected
    part, _, journals = runner('part', Fixture().to(device), crash=True)
    with pytest.raises(ProcessCrash): part.run('micro_pose', parent=parent, micro_case='axis')
    saved_journal = load_integrated(journals[0], c, 'fixture', 'test')
    assert saved_journal['integrated']['pending_head']['phase_done'] == (outcome in ('converged', 'cap'))
    resumed, events, _ = runner('resumed', Fixture().to(device))
    actual = resumed.run('micro_pose', resume=journals[0], micro_case='axis')
    assert actual['status'] == expected['status'] and actual['updates'] == expected['updates']
    order = [event for event, _ in events]
    assert order.index('evaluation') < (order.index('update') if 'update' in order else len(order))
    if outcome in ('pass', 'regression'):
        assert 'update' not in order and actual['updates'] == target['update']
    else:
        # The completed solver must not run again in the interrupted phase.
        assert not any(e == 'update' and r['phase'] == phase for e, r in events)
    a = load_integrated(expected['checkpoint'], c, 'fixture', 'test')
    b = load_integrated(actual['checkpoint'], c, 'fixture', 'test')
    for key in ('model', 'optimizer', 'rng', 'cursor', 'metrics'):
        equal(a[key], b[key])
    for key in ('phase', 'phase_update', 'fine_resets', 'closure_evaluations', 'solver', 'pending_head'):
        equal(a['integrated'][key], b['integrated'][key])
    assert b['integrated']['pending_head'] is None


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('mode', ['double_crash', 'legacy_journal', 'expired'])
def test_pending_candidate_survives_resume_checkpoint_and_budget(tmp_path, device, mode):
    c, model, parent = setup(tmp_path, device)
    original = model.layout.pose.output[1].weight.detach().clone()
    def evaluate(model):
        return lambda *_: metrics(passed=not torch.equal(model.layout.pose.output[1].weight, original))
    whole, _ = make(tmp_path, 'whole', c, model, evaluate=evaluate(model))
    expected = whole.run('micro_pose', parent=parent, micro_case='axis')
    journals = []
    def crash(event, **r):
        if event == 'update':
            journals.append(r['journal']); raise ProcessCrash()
    part_model = Fixture().to(device)
    part, _ = make(tmp_path, 'part', c, part_model, evaluate=evaluate(part_model), observer=crash)
    with pytest.raises(ProcessCrash): part.run('micro_pose', parent=parent, micro_case='axis')
    path = journals[0]
    if mode == 'legacy_journal':
        raw = torch.load(path, weights_only=True)
        raw['integrated'].pop('pending_head')
        from pathlib import Path
        path = Path(path).with_name('legacy-journal.pt'); save_exclusive(path, raw)
    if mode == 'double_crash':
        checkpoints = []
        def crash_again(event, **r):
            if event == 'checkpoint':
                checkpoints.append(r['checkpoint']); raise ProcessCrash()
        again_model = Fixture().to(device)
        again, _ = make(tmp_path, 'again', c, again_model, evaluate=evaluate(again_model), observer=crash_again)
        with pytest.raises(ProcessCrash): again.run('micro_pose', resume=path, micro_case='axis')
        path = checkpoints[0]
        assert load_integrated(path, c, 'fixture', 'test')['integrated']['pending_head'] is not None
    now = [0.]
    def expire(event, **r):
        if event == 'start' and mode == 'expired': now[0] = 121.
    resumed_model = Fixture().to(device)
    resumed, events = make(tmp_path, 'resumed', c, resumed_model, evaluate=evaluate(resumed_model),
                           observer=expire, clock=lambda: now[0])
    actual = resumed.run('micro_pose', resume=path, micro_case='axis')
    assert actual['status'] == expected['status'] == 'passed'
    assert actual['updates'] == expected['updates'] == 1
    assert not any(e == 'update' for e, _ in events)
    a = load_integrated(expected['checkpoint'], c, 'fixture', 'test')
    b = load_integrated(actual['checkpoint'], c, 'fixture', 'test')
    equal(a['model'], b['model']); equal(a['optimizer'], b['optimizer'])
    equal(a['integrated']['solver'], b['integrated']['solver'])
