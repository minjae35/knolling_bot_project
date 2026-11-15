"""Synthetic CPU/CUDA updates only; no real Knolling training."""
from copy import deepcopy
import pytest
import torch
import test_recovery_alternating as legacy
from test_recovery_readout import DEVICES
from test_recovery_integrated import equal
from knolling_model.arrangement import alternating as alt
from knolling_model.arrangement.integrated import load_integrated


def xy_build(tmp_path, monkeypatch, device, outcome):
    original_fixture, original_batch, original_config = legacy.Fixture, legacy.batch_for, legacy.read_config
    monkeypatch.setattr(legacy, 'Fixture', lambda: original_fixture().to(device))
    monkeypatch.setattr(legacy, 'batch_for', lambda m: original_batch(m, device))
    def config(path):
        value = original_config(path)
        value['device'] = device
        value['alternating_pose'] = deepcopy(alt.XY_POLICY)
        return value
    monkeypatch.setattr(legacy, 'read_config', config)
    result = legacy.build(tmp_path, monkeypatch, outcome)
    # Any CE optimization in this mode is a bug (including after Feature updates).
    monkeypatch.setattr(legacy.HeadFitter, 'step', lambda *a, **k: pytest.fail('orientation solver entered'))
    return result


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('outcome', ['pass', 'regression'])
@pytest.mark.parametrize('boundary', ['head_candidate', 'feature_update', 'transition'])
def test_xy_only_reaches_feature_and_resumes_exactly(tmp_path, monkeypatch, device, outcome, boundary):
    c, source, make = xy_build(tmp_path, monkeypatch, device, outcome)
    whole = make('whole').run('oracle_layout', parent=source)
    assert whole['updates'] == 1
    assert whole['status'] == ('passed' if outcome == 'pass' else 'rolled_back')
    paths = []
    def crash(event, **r):
        if ((boundary == 'head_candidate' and event == 'head_candidate' and r['update'] == 1)
            or (boundary == 'feature_update' and event == 'update')
            or (boundary == 'transition' and event == 'phase_transition' and r['next_phase'] == 'feature')):
            paths.append(r['journal'] if event == 'head_candidate' else sorted((tmp_path/'part').glob('checkpoint-*.pt'))[-1])
            raise legacy.Crash()
    with pytest.raises(legacy.Crash):
        make('part', observer=crash).run('oracle_layout', parent=source)
    events = []
    resumed = make('resume', observer=lambda event, **r: events.append(event)).run('oracle_layout', resume=paths[0])
    assert (whole['status'], whole['updates']) == (resumed['status'], resumed['updates'])
    if boundary == 'head_candidate': assert 'update' not in events
    a = load_integrated(whole['checkpoint'], c, 'fixture', 'test')
    b = load_integrated(resumed['checkpoint'], c, 'fixture', 'test')
    for key in ('model', 'optimizer', 'rng', 'cursor', 'metrics'): equal(a[key], b[key])
    assert a['purpose'] == 'diagnostic'
    assert all(x['phase'] == 'head_xy' for x in a['integrated']['calibrations'])
    original = torch.load(source, map_location='cpu', weights_only=False)['model']
    for key, value in original.items():
        if key.startswith('layout.pose.output.1.'):
            equal(value[2:], a['model'][key][2:])
        elif not key.startswith('layout.pose.'):
            equal(value, a['model'][key])
    if outcome == 'pass':
        assert any(not torch.equal(v, a['model'][k]) for k,v in original.items()
                   if k.startswith('layout.pose.') and not k.startswith('layout.pose.output.1.'))
    else:
        equal(a['optimizer'], a['integrated']['anchor_optimizer'])
        equal(a['rng'], a['integrated']['anchor_rng'])


def test_xy_policy_is_separate_and_not_training_approved():
    from knolling_model.arrangement.driver import read_config, authorize
    c = read_config('configs/recovery_v4_xy_diagnostic_review.json')
    alt.validate_config(c)
    assert c['alternating_pose'] == alt.XY_POLICY and not c['allow_training']
    with pytest.raises(PermissionError): authorize(c, {}, 'oracle_layout', 'data', 'commit')
    original = read_config('configs/recovery_v4_alternating_review.json')
    assert original['alternating_pose'] == alt.POLICY
    c['alternating_pose']['diagnostic'] = 'bypass'
    with pytest.raises(ValueError): alt.validate_config(c)


@pytest.mark.parametrize('device', DEVICES)
def test_xy_only_deadline_preserves_anchor(tmp_path, monkeypatch, device):
    c, source, make = xy_build(tmp_path, monkeypatch, device, 'pass')
    now = [0.]
    def expire(event, **r):
        if event == 'update': now[0] = 120.
    result = make('expired', observer=expire, clock=lambda: now[0]).run('oracle_layout', parent=source)
    assert result['status'] == 'rolled_back' and result['updates'] == 1
    s = load_integrated(result['checkpoint'], c, 'fixture', 'test')
    equal(s['optimizer'], s['integrated']['anchor_optimizer'])
    equal(s['rng'], s['integrated']['anchor_rng'])


@pytest.mark.parametrize('device', DEVICES)
def test_reload_evaluation_normalizes_flags_and_restores_state(tmp_path, device):
    from test_recovery_readout import Fixture, batch_for
    from knolling_model.arrangement.stages import configure_stage
    torch.set_num_threads(2); torch.manual_seed(14)
    m = Fixture().to(device); b = batch_for(m, device)
    b.supervision[1] = {k:v[:2] if isinstance(v,torch.Tensor) and v.ndim else v for k,v in b.supervision[1].items()}
    b.metadata[1]['object_ids'] = ['0','1']
    for meta in b.metadata: meta['record']['supervision_v3'] = {'copy_required':{'0':True,'1':False}}
    pairs = [dict(factor='axis_alignment',sample_ids=['0','1'])]
    configure_stage(m, 'oracle_layout'); alt.FeatureAdam(m,1e-4,1e-4)
    before = alt.trial_measurements(m,[b],['0','1'],'fixture',pairs)
    saved = tmp_path/'reload.pt'; torch.save(m.state_dict(),saved)
    n = Fixture().to(device); n.load_state_dict(torch.load(saved,weights_only=True))
    n.requires_grad_(False); n.eval()
    flags = [p.requires_grad for p in n.parameters()]; modes = [q.training for q in n.modules()]
    after = alt.trial_measurements(n,[b],['0','1'],'fixture',pairs)
    equal(before,after)
    assert flags == [p.requires_grad for p in n.parameters()]
    assert modes == [q.training for q in n.modules()]
    assert all(p.grad is None for p in n.parameters())
    with pytest.raises(RuntimeError,match='probe'):
        with alt.alternating_evaluation(n):
            assert not torch.is_grad_enabled()
            assert all(not p.requires_grad for p in n.layout.pose.output[1].parameters())
            raise RuntimeError('probe')
    assert flags == [p.requires_grad for p in n.parameters()]
    assert modes == [q.training for q in n.modules()]


@pytest.mark.parametrize('device', DEVICES)
def test_xy_rounding_stall_still_reaches_feature(tmp_path, monkeypatch, device):
    c, source, make = xy_build(tmp_path, monkeypatch, device, 'pass')
    def stall(*args, **kwargs):
        raise alt.SVDNumericalStall({'fixture': True})
    monkeypatch.setattr(legacy.SVDXYFitter, 'step', stall)
    end = make('stalled', observer=None).run('oracle_layout', parent=source)
    assert end['status'] == 'passed' and end['updates'] == 1
    s = load_integrated(end['checkpoint'], c, 'fixture', 'test')
    assert all(x['diagnostic']['reason'] == 'fp32_numerical_stall' for x in s['integrated']['calibrations'])
    from knolling_model.arrangement.driver import validate_checkpoint
    bad = deepcopy(s); bad['purpose'] = 'production'
    with pytest.raises(ValueError, match='purpose'):
        validate_checkpoint(bad, c, 'fixture', 'test')


def test_xy_feature_cap_cannot_extend(tmp_path, monkeypatch):
    c, source, make = xy_build(tmp_path, monkeypatch, 'cpu', 'fail')
    def stall(*args, **kwargs): raise alt.SVDNumericalStall({'fixture': True})
    monkeypatch.setattr(legacy.SVDXYFitter, 'step', stall)
    end = make('capped').run('oracle_layout', parent=source)
    assert end['status'] == 'budget_exhausted' and end['updates'] == 10
    with pytest.raises(ValueError, match='terminal'):
        make('forbidden').run('oracle_layout', resume=end['checkpoint'])
