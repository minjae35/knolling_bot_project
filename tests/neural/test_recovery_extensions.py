"""Approved extension lifecycle on CPU/CUDA scalar fixtures, not Knolling training."""
from copy import deepcopy
import os
import threading

import pytest
import torch

from knolling_dataset.builder import file_hash
from knolling_dataset.schema import digest
from knolling_model.recovery.driver import load_checkpoint
from knolling_model.recovery.storage import load_artifact
from knolling_model.recovery.extensions import EXTENSION_SCHEMA, assess_progress
from test_recovery_driver import config, engine, ScalarFixture


def metrics(position=10., move=.4, axis=0.):
    return dict(schema='recovery_metrics_v2', stage='oracle_layout', complete_set=True,
                source_mode='oracle', evaluation_mode='free_rollout', sample_ids=['fixture'],
                target_position_p95_mm=position, move_accuracy=move, pure_axis_success=axis,
                **{'pass':False}, full_set_no_regression=True)


def settings(device='cpu', cap=2, seconds=600):
    c=config()
    c.update(device=device, execution='approved_training', allow_training=True, evaluate_every=3)
    c['stages']['oracle_layout'].update(max_updates=cap, max_seconds=seconds)
    return c


def make_engine(root, name, c, *, receipt=None, evaluator=None, commit='fixture-commit', clock=lambda:0., **kwargs):
    e=engine(root,name,c=c,model=ScalarFixture().to(c['device']),
             evaluator=evaluator or (lambda *_:metrics()),review_receipt=receipt,clock=clock,**kwargs)
    e.commit=commit
    def loss(model, inputs, truths, replay=None):
        return {'loss':(model.layout(inputs['x'].to(c['device']))-torch.rand(1,device=c['device'])).square().mean()},{}
    e.loss_fn=loss
    return e


def grant(path, c, window=2):
    old=load_artifact(path)
    return dict(approved=True,reviewer='unit-test explicit extension',commit='extension-code',
        config_sha256=digest(c),dataset_fingerprint='fixture-data',stages=['oracle_layout'],gpu_training=True,
        extension=dict(schema=EXTENSION_SCHEMA,stage='oracle_layout',source_checkpoint_sha256=file_hash(path),
            source_commit=old['commit'],source_config_sha256=old['config_sha256'],
            source_update=old['cursor']['update'],source_elapsed_seconds=old['cursor']['elapsed_seconds'],
            additional_updates=c['stages']['oracle_layout']['max_updates']-old['cursor']['update'],
            total_max_updates=c['stages']['oracle_layout']['max_updates'],
            total_max_seconds=c['stages']['oracle_layout']['max_seconds'],progress_after_updates=window))


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    os.environ.get('RUN_RECOVERY_CUDA_TESTS')!='1',reason='explicit GPU verification only'))])
def test_extension_matches_uninterrupted_weights_optimizer_rng_and_cursor(tmp_path,device):
    old,new=settings(device),settings(device,cap=6)
    torch.manual_seed(51)
    whole=make_engine(tmp_path,'whole',new)
    full=whole.run('oracle_layout')
    torch.manual_seed(51)
    first=make_engine(tmp_path,'source',old).run('oracle_layout')
    source_hash=file_hash(first['checkpoint'])
    receipt=grant(first['checkpoint'],new)
    prepared=[]
    def prepare(stage):
        prepared.append(stage)
        torch.rand(3,device=device)  # Setup consumption must precede RNG restoration.
    continued=make_engine(tmp_path,'extended',new,receipt=receipt,commit='extension-code',
                          evaluator=lambda *_:metrics(position=9.),prepare_continuation=prepare)
    result=continued.run('oracle_layout',extend_from=first['checkpoint'])
    assert result['updates']==6 and result['status']=='budget_exhausted'
    assert prepared==['oracle_layout']
    a=load_checkpoint(full['checkpoint'],new,'fixture-data','fixture-commit')
    b=load_checkpoint(result['checkpoint'],new,'fixture-data','extension-code')
    for name in a['model']:
        torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)
    for key in a['optimizer']['state'][0]:
        torch.testing.assert_close(a['optimizer']['state'][0][key],b['optimizer']['state'][0][key],rtol=0,atol=0)
    assert a['cursor']==b['cursor']
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    assert all(torch.equal(x,y) for x,y in zip(a['rng']['cuda'],b['rng']['cuda']))
    assert b['extension']['checked_at']==4 and b['extension']['improved']['target_position_p95_mm']
    assert file_hash(first['checkpoint'])==source_hash


def test_normal_resume_stays_blocked_and_extension_grants_fail_closed(tmp_path):
    old,new=settings(),settings(cap=6)
    source=make_engine(tmp_path,'source',old).run('oracle_layout')['checkpoint']
    with pytest.raises(ValueError,match='interrupted'):
        make_engine(tmp_path,'ordinary-resume',old).run('oracle_layout',resume=source)
    good=grant(source,new)
    for key,value in [('source_checkpoint_sha256','wrong'),('source_commit','wrong'),
                      ('source_config_sha256','wrong'),('source_update',0),
                      ('source_elapsed_seconds',-1.),('additional_updates',5),('total_max_seconds',601)]:
        bad=deepcopy(good);bad['extension'][key]=value
        with pytest.raises((ValueError,PermissionError)):
            make_engine(tmp_path,'bad-'+key,new,receipt=bad,commit='extension-code').run('oracle_layout',extend_from=source)
        assert not (tmp_path/('bad-'+key)).exists()
    changed=deepcopy(new);changed['layout_lr']*=2
    bad=deepcopy(good);bad['config_sha256']=digest(changed)
    with pytest.raises(ValueError,match='only this stage'):
        make_engine(tmp_path,'changed-lr',changed,receipt=bad,commit='extension-code').run('oracle_layout',extend_from=source)
    with pytest.raises(PermissionError):
        make_engine(tmp_path,'missing-grant',new,commit='extension-code').run('oracle_layout',extend_from=source)


def test_extension_window_survives_pause_and_forces_off_schedule_evaluation(tmp_path):
    old,new=settings(),settings(cap=6)
    source=make_engine(tmp_path,'source',old).run('oracle_layout')['checkpoint']
    receipt=grant(source,new)
    stop=threading.Event()
    def pause_at_three(*_):
        stop.set()
        return metrics()
    paused=make_engine(tmp_path,'pause',new,receipt=receipt,commit='extension-code',
                       evaluator=pause_at_three,event=stop).run('oracle_layout',extend_from=source)
    assert paused['status']=='paused' and paused['updates']==3
    resumed=make_engine(tmp_path,'resume',new,receipt=receipt,commit='extension-code')
    result=resumed.run('oracle_layout',resume=paused['checkpoint'])
    assert result['status']=='early_stopped' and result['updates']==4  # evaluate_every is 3.
    saved=load_checkpoint(result['checkpoint'],new,'fixture-data','extension-code')
    assert saved['extension']['origin_update']==2 and saved['extension']['checked_at']==4
    assert 'oracle_layout' not in saved['completed']
    with pytest.raises(ValueError):
        make_engine(tmp_path,'blocked',new,commit='extension-code').run('oracle_layout',resume=result['checkpoint'])


def test_extension_keeps_consumed_wall_time(tmp_path):
    old,new=settings(seconds=5),settings(cap=6,seconds=5)
    def timed(name,c,**kwargs):
        now=[0.]
        e=make_engine(tmp_path,name,c,clock=lambda:now[0],**kwargs)
        original=e.loss_fn
        def loss(*args,**kw):
            now[0]+=1.
            return original(*args,**kw)
        e.loss_fn=loss
        return e
    source=timed('source',old).run('oracle_layout')['checkpoint']
    receipt=grant(source,new,window=4)
    result=timed('extended',new,receipt=receipt,commit='extension-code').run('oracle_layout',extend_from=source)
    assert result['status']=='budget_exhausted' and result['elapsed_seconds']==5.
    assert result['updates']==4  # Two old steps + two new; fifth second cannot commit a step.


def test_extension_rejects_changed_anchor_before_creating_output(tmp_path):
    old,new=settings(),settings(cap=6)
    source=make_engine(tmp_path,'source',old).run('oracle_layout')['checkpoint']
    receipt=grant(source,new)
    (tmp_path/'source'/'anchor.pt').write_bytes(b'corrupted fixture anchor')
    with pytest.raises(ValueError,match='anchor hash'):
        make_engine(tmp_path,'bad-anchor',new,receipt=receipt,commit='extension-code').run('oracle_layout',extend_from=source)
    assert not (tmp_path/'bad-anchor').exists()


@pytest.mark.parametrize('field', ['none','position','move','axis'])
def test_endpoint_requires_at_least_one_measured_improvement(field):
    state=dict(checked=False,progress_update=700,reference_metrics={
        'target_position_p95_mm':10.,'move_accuracy':.4,'pure_axis_success':0.},sample_ids=['fixture'])
    m=metrics(position=9. if field=='position' else 10.,move=.5 if field=='move' else .4,axis=.1 if field=='axis' else 0.)
    assert not assess_progress(state,699,m) and not state['checked']
    assert assess_progress(state,700,m)==(field=='none')
    assert state['checked_at']==700
