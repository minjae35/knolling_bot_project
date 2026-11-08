"""Lifecycle tests update a scalar fixture only, never the Knolling model."""
import subprocess
import sys
import threading
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from knolling_dataset.schema import digest
from knolling_model.recovery.driver import (StageEngine, authorize, load_checkpoint,
    read_config, save_exclusive, CONFIG_SCHEMA)


class ScalarFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.layout = nn.Linear(1, 1, bias=False)
        self.perception = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.layout.weight.fill_(.4)
            self.perception.weight.fill_(.2)


def config():
    c = read_config('configs/recovery_execution.json')
    c.update(evaluate_every=1,checkpoint_every=1,stop_on_initial_pass=False,layout_lr=.01)
    for s in c['stages'].values():
        s.update(max_updates=4,max_seconds=600)
    return c


def configure(model, stage):
    model.stage=stage
    model.requires_grad_(False)
    model.layout.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    model.train()


def batches(stage, epoch):
    for index in range(3):
        yield SimpleNamespace(loss_arguments=lambda index=index: ({'x':torch.tensor([[index+1.]])},[],None))


def loss(model, inputs, truths, replay=None):
    # RNG consumption proves resume restores RNG, rather than merely weights.
    return {'loss':(model.layout(inputs['x'])-torch.rand(1)).square().mean()}, {}


def engine(tmp_path, name, model=None, c=None, event=None, evaluator=None, **kwargs):
    model = model or ScalarFixture()
    evaluator = evaluator or (lambda stage, baseline: {'stage':stage,'pass':False,'full_set_no_regression':True})
    return StageEngine(model,c or config(),'fixture-data','fixture-commit',tmp_path/name,batches,evaluator,
        loss_fn=loss,configure_fn=configure,gate_fn=lambda stage,m: bool(m.get('pass')),
        stop_event=event,**kwargs)


def test_pause_resume_matches_uninterrupted_weights_optimizer_and_cursor(tmp_path):
    torch.manual_seed(123)
    full = engine(tmp_path,'full')
    expected = full.run('oracle_layout')
    torch.manual_seed(123)
    stop = threading.Event()
    seen = 0
    def pause(stage, baseline):
        nonlocal seen
        seen += 1
        if seen == 4:  # baseline, initial metrics, update 1, update 2
            stop.set()
        return {'pass':False,'full_set_no_regression':True}
    first = engine(tmp_path,'paused',event=stop,evaluator=pause)
    interrupted = first.run('oracle_layout')
    assert interrupted['status'] == 'paused' and interrupted['updates'] == 2
    resumed = engine(tmp_path,'resumed')
    actual = resumed.run('oracle_layout',resume=interrupted['checkpoint'])
    assert actual['updates'] == expected['updates'] == 4
    torch.testing.assert_close(full.model.layout.weight,resumed.model.layout.weight,rtol=0,atol=0)
    a=load_checkpoint(expected['checkpoint'],config(),'fixture-data','fixture-commit')
    b=load_checkpoint(actual['checkpoint'],config(),'fixture-data','fixture-commit')
    for key in ('update','epoch','batch'):
        assert a['cursor'][key] == b['cursor'][key]
    for key in a['optimizer']['state'][0]:
        torch.testing.assert_close(a['optimizer']['state'][0][key],b['optimizer']['state'][0][key],rtol=0,atol=0)


def test_previous_gate_full_set_regression_and_selection_failure_rollback(tmp_path):
    c=config()
    checkpoint=None
    for stage in ('oracle_layout','perception','predicted_layout'):
        e=engine(tmp_path,stage,c=c,evaluator=lambda *_:{'pass':True,'full_set_no_regression':True})
        checkpoint=e.run(stage,parent=checkpoint)['checkpoint']
    reference=load_checkpoint(checkpoint,c,'fixture-data','fixture-commit')['model']
    calls=0
    def regress(stage,baseline):
        nonlocal calls
        calls+=1
        return {'pass':False,'full_set_no_regression':calls<=2}
    joint=engine(tmp_path,'joint',c=c,evaluator=regress)
    result=joint.run('joint',parent=checkpoint)
    assert result['status']=='rolled_back' and result['updates']==1
    for name,value in joint.model.state_dict().items():
        torch.testing.assert_close(value,reference[name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='previous stage'):
        engine(tmp_path,'no-parent').run('predicted_layout')
    from knolling_model.recovery.selection import SourceSelectionError
    failed=engine(tmp_path,'selection-failure',c=c)
    def bad(*args,**kwargs):
        raise SourceSelectionError('missing foreground object')
    failed.loss_fn=bad
    result=failed.run('joint',parent=checkpoint)
    assert result['status']=='rolled_back' and 'SourceSelectionError' in result['error']


def test_stop_during_forward_replays_uncommitted_batch_rng(tmp_path):
    torch.manual_seed(321)
    full=engine(tmp_path,'uninterrupted')
    full.run('oracle_layout')
    torch.manual_seed(321)
    event=threading.Event()
    interrupted=engine(tmp_path,'in-flight',event=event)
    def stop_after_forward(*args,**kwargs):
        terms=loss(*args,**kwargs)
        event.set()
        return terms
    interrupted.loss_fn=stop_after_forward
    saved=interrupted.run('oracle_layout')
    assert saved['status']=='paused' and saved['updates']==0
    resumed=engine(tmp_path,'replayed')
    resumed.run('oracle_layout',resume=saved['checkpoint'])
    torch.testing.assert_close(full.model.layout.weight,resumed.model.layout.weight,rtol=0,atol=0)


def test_time_budget_nonfinite_guard_and_immutable_checkpoints(tmp_path):
    c=config()
    c['stages']['oracle_layout']['max_seconds']=1
    ticks=iter(range(100))
    e=engine(tmp_path,'budget',c=c,clock=lambda:next(ticks))
    result=e.run('oracle_layout')
    assert result['status']=='budget_exhausted' and result['updates']==0
    with pytest.raises(FileExistsError):
        save_exclusive(result['checkpoint'],{})
    with pytest.raises(ValueError,match='contract'):
        load_checkpoint(result['checkpoint'],config(),'wrong-data','fixture-commit')
    e=engine(tmp_path,'nan')
    def nan(model,*args,**kwargs):
        return {'loss':model.layout.weight.sum()*float('nan')},{}
    e.loss_fn=nan
    assert e.run('oracle_layout')['status']=='rolled_back'


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    os.environ.get('RUN_RECOVERY_CUDA_TESTS') != '1', reason='explicit GPU verification only'))])
@pytest.mark.parametrize('expire_at', ['forward', 'backward'])
@pytest.mark.parametrize('outcome', ['regression', 'budget_exhausted', 'passed', 'evaluation_error'])
def test_joint_timeout_final_evaluation_and_rollback(tmp_path, device, expire_at, outcome):
    """Only a scalar fixture updates; expire before an uncommitted second step."""
    if device == 'cuda':
        assert torch.cuda.is_available()
    c=config()
    c.update(evaluate_every=25, checkpoint_every=25, stop_on_initial_pass=True)
    c['stages']['joint']['max_seconds']=1
    parent=None
    for stage in ('oracle_layout', 'perception', 'predicted_layout'):
        previous=engine(tmp_path, stage, c=c,
            evaluator=lambda *_: {'pass':True, 'full_set_no_regression':True})
        parent=previous.run(stage, parent=parent)['checkpoint']
    reference=load_checkpoint(parent, c, 'fixture-data', 'fixture-commit')['model']
    now=[0.]
    calls={'loss':0, 'evaluate':0}
    def evaluate(stage, baseline):
        calls['evaluate']+=1
        if calls['evaluate'] <= 2:
            return {'pass':False, 'full_set_no_regression':True}
        if outcome == 'evaluation_error':
            raise RuntimeError('final evaluation failed')
        return {'pass':outcome == 'passed', 'full_set_no_regression':outcome != 'regression'}
    joint=engine(tmp_path, 'timeout', model=ScalarFixture().to(device), c=c,
                 evaluator=evaluate, clock=lambda:now[0])
    def expire():
        now[0]=2.
    def timed_loss(model, inputs, truths, replay=None):
        calls['loss']+=1
        value=(model.layout(inputs['x'].to(device))-1.).square().mean()
        if calls['loss']==2:
            if expire_at == 'forward':
                expire()
            else:
                def hook(gradient):
                    expire()
                    return gradient
                value.register_hook(hook)
        return {'loss':value}, {}
    joint.loss_fn=timed_loss
    result=joint.run('joint', parent=parent)
    saved=load_checkpoint(result['checkpoint'], c, 'fixture-data', 'fixture-commit')
    assert calls == {'loss':2, 'evaluate':3}
    assert result['updates']==1 and saved['cursor']['batch']==1
    assert saved['metrics']==result['metrics'] and saved['status']==result['status']
    assert int(next(iter(saved['optimizer']['state'].values()))['step'])==1
    assert all(p.grad is None for p in joint.model.parameters())
    if outcome in ('regression', 'evaluation_error'):
        assert result['status']=='rolled_back'
        for name, value in joint.model.state_dict().items():
            torch.testing.assert_close(value.cpu(), reference[name], rtol=0, atol=0)
            torch.testing.assert_close(saved['model'][name], reference[name], rtol=0, atol=0)
        assert 'joint' not in saved['completed']
        if outcome=='regression':
            assert result['metrics']['full_set_no_regression'] is False
        else:
            assert 'final evaluation failed' in result['error']
    else:
        assert result['status']==outcome and result['metrics']['full_set_no_regression'] is True
        assert not torch.equal(saved['model']['layout.weight'], reference['layout.weight'])
        assert ('joint' in saved['completed']) == (outcome=='passed')


def test_default_config_and_stale_receipts_cannot_train():
    c=config()
    receipt=dict(approved=True,reviewer='unit-test',commit='fixture-commit',
                 config_sha256=digest(c),dataset_fingerprint='fixture-data',stages=['oracle_layout'])
    with pytest.raises(PermissionError,match='disabled'):
        authorize(c,receipt,'oracle_layout','fixture-data','fixture-commit')
    c.update(execution='approved_training',allow_training=True)
    with pytest.raises(PermissionError,match='receipt'):
        authorize(c,receipt,'oracle_layout','fixture-data','fixture-commit')
    receipt['config_sha256']=digest(c)
    authorize(c,receipt,'oracle_layout','fixture-data','fixture-commit')
    with pytest.raises(PermissionError):
        authorize(c,receipt,'joint','fixture-data','fixture-commit')


def test_default_cli_stops_before_loading_receipt_or_creating_output(tmp_path):
    output=tmp_path/'must-not-exist'
    result=subprocess.run([sys.executable,'-m','knolling_model.recovery.driver',
        '--stage','oracle_layout','--output',str(output),'--approval',str(tmp_path/'absent.json')],
        capture_output=True,text=True)
    assert result.returncode != 0 and 'training disabled' in result.stderr
    assert not output.exists()
