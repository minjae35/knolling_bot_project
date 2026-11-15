"""Bounded CPU tensor fixtures only; never run a Knolling training experiment."""
from copy import deepcopy
import threading

import pytest
import torch

from knolling_dataset.builder import file_hash
from knolling_dataset.schema import digest
from knolling_model.arrangement.driver import read_config, save_exclusive
from knolling_model.arrangement.storage import load_artifact
from knolling_model.arrangement.integrated import load_integrated
from knolling_model.arrangement.coarse_extension import load_extension, validate_policy, record_progress
from test_recovery_integrated import setup, make, metrics, equal
from test_recovery_readout import Fixture, DEVICES


class Crash(BaseException): pass


@pytest.fixture
def source(tmp_path, monkeypatch):
    from knolling_model.arrangement import driver, integrated, coarse_extension
    c,m,parent=setup(tmp_path)
    c['integrated_pose']['xy_solver']='svd_huber_v1'
    raw=load_artifact(parent);raw['config']=deepcopy(c);raw['config_sha256']=digest(c)
    parent=tmp_path/'ground-svd.pt';save_exclusive(parent,raw)
    evidence={k:dict(path=str(parent),sha256=file_hash(parent)) for k in ('translation','keep','axis')}
    monkeypatch.setattr(driver,'validate_micro_checkpoints',lambda *a,**k:deepcopy(evidence))
    monkeypatch.setattr(integrated,'validate_micro_checkpoints',lambda *a,**k:deepcopy(evidence))
    monkeypatch.setattr(coarse_extension,'contract_hashes',lambda *a:{'fixture':'same'})
    def crash(event,**r):
        if event=='phase_transition' and r['next_phase']=='head_xy':raise Crash()
    e,_=make(tmp_path,'source',c,m,evaluate=lambda *_:metrics(position=20),observer=crash)
    with pytest.raises(Crash):e.run('oracle_layout',parent=parent,micro_checkpoints=list(evidence))
    path=next((tmp_path/'source').glob('checkpoint-000200-*.pt'))
    old=load_artifact(path)
    save_exclusive(path.parent/'terminal.pt',{'status':'rolled_back'})
    target=deepcopy(c);target['stages']['oracle_layout']['max_updates']=1300
    target['integrated_pose']['stages']['oracle_layout']['coarse']=1000
    target['coarse_extension']=dict(schema='coarse_extension_v1',source_sha256=file_hash(path),interval_updates=200,min_relative_improvement=.05,patience=2)
    receipt={'coarse_extension':dict(approved=True,sha256=file_hash(path),commit='test',config_sha256=digest(c),contracts={'fixture':'same'})}
    return target,path,receipt,old


def test_transfer_preserves_state_and_requires_explicit_new_run(tmp_path,source):
    c,path,receipt,old=source;before=file_hash(path)
    transferred=load_extension(path,c,'fixture',receipt)
    for k in ('model','optimizer','rng','cursor','micro_evidence'):equal(old[k],transferred[k])
    assert transferred['integrated']['phase']=='coarse' and transferred['integrated']['phase_update']==200
    assert file_hash(path)==before
    stop=threading.Event();stop.set()
    e,_=make(tmp_path,'paused',c,Fixture(),stop_event=stop,review_receipt=receipt,evaluate=lambda *_:metrics(position=20))
    result=e.run('oracle_layout',extend_from=path)
    assert result['status']=='paused' and result['updates']==200
    saved=load_integrated(result['checkpoint'],c,'fixture','test')
    for k in ('model','optimizer','rng','cursor'):equal(old[k],saved[k])
    ordinary,_=make(tmp_path,'forbidden',c,Fixture(),review_receipt=receipt)
    with pytest.raises(ValueError,match='terminal'):ordinary.run('oracle_layout',resume=path)
    with pytest.raises(ValueError):ordinary.run('micro_pose',extend_from=path,micro_case='axis')


@pytest.mark.parametrize('change',['approval','hash','lr','clip','loss_config','budget','data','commit','anchor','head'])
def test_extension_rejects_unapproved_changes(source,change,tmp_path):
    c,path,receipt,old=source
    if change=='approval':receipt['coarse_extension']['approved']=False
    elif change=='hash':receipt['coarse_extension']['sha256']='0'*64
    elif change=='lr':c['layout_lr']=1e-3
    elif change=='clip':c['gradient_clip']=2.
    elif change=='loss_config':c['weight_decay']=.1
    elif change=='budget':c['stages']['oracle_layout']['max_seconds']=601
    elif change=='data':
        with pytest.raises(ValueError):load_extension(path,c,'other',receipt)
        return
    elif change=='commit':receipt['coarse_extension']['commit']='wrong'
    else:
        bad=deepcopy(old)
        if change=='anchor':bad['anchor_sha256']='0'*64
        else:bad['model']['layout.pose.output.1.weight'][0,0]+=.1
        path=tmp_path/'tampered.pt';save_exclusive(path,bad)
        receipt['coarse_extension']['sha256']=c['coarse_extension']['source_sha256']=file_hash(path)
    with pytest.raises((ValueError,PermissionError)):load_extension(path,c,'fixture',receipt)


def test_extension_resume_keeps_adam_rng_and_next_batch(tmp_path,source):
    c,path,receipt,_=source
    def runner(name,stop_at):
        stop=threading.Event()
        def observe(event,**r):
            if event=='update' and r['update']==stop_at:stop.set()
        return make(tmp_path,name,c,Fixture(),observer=observe,stop_event=stop,review_receipt=receipt,evaluate=lambda *_:metrics(position=20))
    whole,_=runner('whole',204);a=whole.run('oracle_layout',extend_from=path)
    part,_=runner('part',202);b=part.run('oracle_layout',extend_from=path)
    resumed,_=runner('resumed',204);end=resumed.run('oracle_layout',resume=b['checkpoint'])
    aa=load_integrated(a['checkpoint'],c,'fixture','test');bb=load_integrated(end['checkpoint'],c,'fixture','test')
    for k in ('model','optimizer','rng','cursor'):equal(aa[k],bb[k])
    equal(aa['integrated']['coarse_progress'],bb['integrated']['coarse_progress'])
    assert aa['cursor']['epoch']==204 and aa['cursor']['batch']==0


def test_monitor_stops_at_second_weak_interval_even_after_crash(tmp_path,source,monkeypatch):
    from knolling_model.arrangement import coarse_extension
    c,path,receipt,_=source
    monkeypatch.setattr(coarse_extension,'readout_diagnostic',lambda *a:dict(xy_svd_rms_mm=20.,losses={},angle_bin_accuracy=.5))
    checkpoints=[]
    def crash(event,**r):
        if event=='checkpoint' and r['update']==600:
            checkpoints.append(r['checkpoint']);raise Crash()
    whole,_=make(tmp_path,'whole',c,Fixture(),review_receipt=receipt,evaluate=lambda *_:metrics(position=20))
    a=whole.run('oracle_layout',extend_from=path)
    assert a['status']=='coarse_stalled' and a['updates']==600
    part,_=make(tmp_path,'part',c,Fixture(),review_receipt=receipt,evaluate=lambda *_:metrics(position=20),observer=crash)
    with pytest.raises(Crash):part.run('oracle_layout',extend_from=path)
    resumed,events=make(tmp_path,'resumed',c,Fixture(),review_receipt=receipt,evaluate=lambda *_:metrics(position=20))
    b=resumed.run('oracle_layout',resume=checkpoints[-1])
    assert b['status']=='coarse_stalled' and b['updates']==600
    assert not any(e=='update' for e,r in events)
    aa=load_integrated(a['checkpoint'],c,'fixture','test');bb=load_integrated(b['checkpoint'],c,'fixture','test')
    for k in ('model','optimizer','rng','cursor'):equal(aa[k],bb[k])
    equal(aa['integrated']['coarse_progress'],bb['integrated']['coarse_progress'])


def test_progress_boundary_and_review_config():
    c=read_config('configs/recovery_v4_coarse_extension_review.json');validate_policy(c)
    assert not c['allow_training']
    h=[]
    for step,value in [(200,20.),(400,19.5),(600,18.),(800,17.8),(1000,17.7)]:record_progress(h,step,dict(xy_svd_rms_mm=value))
    assert [v['weak_intervals'] for v in h]==[0,1,0,1,2]
    with pytest.raises(ValueError):record_progress(h,1000,dict(xy_svd_rms_mm=17.))


def test_cumulative_time_budget_is_not_reset(tmp_path,source):
    c,path,receipt,old=source
    raw=deepcopy(old);raw['cursor']['elapsed_seconds']=50.
    path=tmp_path/'elapsed-source.pt';save_exclusive(path,raw)
    receipt['coarse_extension']['sha256']=c['coarse_extension']['source_sha256']=file_hash(path)
    now=[0.]
    def observe(event,**r):
        if event=='start':now[0]=550.
    e,events=make(tmp_path,'expired',c,Fixture(),review_receipt=receipt,
        evaluate=lambda *_:metrics(position=20),clock=lambda:now[0],observer=observe)
    result=e.run('oracle_layout',extend_from=path)
    assert result['status']=='budget_exhausted' and result['updates']==200
    assert result['elapsed_seconds']==600 and not any(e=='update' for e,r in events)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('outcome', ['pass', 'transition', 'continue'])
def test_diagnostic_checkpoint_resolves_gate_before_next_update(tmp_path, source, device, outcome):
    c,path,receipt,_=source
    checkpoints=[]
    def runner(name, crash=False, double_crash=False):
        active={'update':200,'phase':'coarse'};stop=threading.Event()
        def observe(event,**r):
            active.update(update=r['update'],phase=r['phase'])
            if event=='checkpoint' and r['update']==400 and (crash or double_crash):
                checkpoints.append(r['checkpoint']);raise Crash()
            if outcome=='transition' and event=='phase_transition' and r['next_phase']=='head_xy':stop.set()
            if outcome=='continue' and event=='update' and r['update']==401:stop.set()
        def evaluate(*_):
            at_boundary=active['update']==400
            return metrics(position=5 if at_boundary and outcome=='transition' else 20,
                           passed=at_boundary and outcome=='pass')
        e,events=make(tmp_path,name,c,Fixture().to(device),review_receipt=receipt,
                      observer=observe,evaluate=evaluate,stop_event=stop)
        return e,events
    whole,_=runner('gate-whole');a=whole.run('oracle_layout',extend_from=path)
    part,_=runner('gate-part',crash=True)
    with pytest.raises(Crash):part.run('oracle_layout',extend_from=path)
    pending=load_integrated(checkpoints[-1],c,'fixture','test')
    assert pending['integrated']['pending_coarse']==400
    # A second interruption at the Resume checkpoint must retain the decision.
    again,_=runner('gate-again',double_crash=True)
    with pytest.raises(Crash):again.run('oracle_layout',resume=checkpoints[-1])
    resumed,events=runner('gate-resumed');b=resumed.run('oracle_layout',resume=checkpoints[-1])
    assert a['status']==b['status']==('passed' if outcome=='pass' else 'paused')
    assert a['updates']==b['updates']==(401 if outcome=='continue' else 400)
    order=[e for e,_ in events]
    assert order.index('evaluation') < (order.index('update') if 'update' in order else len(order))
    if outcome=='transition':
        assert not any(e=='update' for e,_ in events)
        assert any(e=='phase_transition' and r['next_phase']=='head_xy' for e,r in events)
    aa=load_integrated(a['checkpoint'],c,'fixture','test');bb=load_integrated(b['checkpoint'],c,'fixture','test')
    for k in ('model','optimizer','rng','cursor','metrics'):equal(aa[k],bb[k])
    for k in ('phase','phase_update','pending_coarse','coarse_progress','fine_resets'):
        equal(aa['integrated'][k],bb['integrated'][k])
    assert bb['integrated']['pending_coarse'] is None


def test_pending_coarse_gate_precedes_expired_budget(tmp_path,source):
    c,path,receipt,_=source;active={'update':200};checkpoints=[]
    def observe(event,**r):
        active['update']=r['update']
        if event=='checkpoint' and r['update']==400:
            checkpoints.append(r['checkpoint']);raise Crash()
    def evaluate(*_):return metrics(position=20,passed=active['update']==400)
    part,_=make(tmp_path,'expired-part',c,Fixture(),review_receipt=receipt,observer=observe,evaluate=evaluate)
    with pytest.raises(Crash):part.run('oracle_layout',extend_from=path)
    now=[0.]
    def expire(event,**r):
        active['update']=r['update']
        if event=='start':now[0]=601.
    resumed,events=make(tmp_path,'expired-resume',c,Fixture(),review_receipt=receipt,observer=expire,
                        evaluate=evaluate,clock=lambda:now[0])
    result=resumed.run('oracle_layout',resume=checkpoints[-1])
    assert result['status']=='passed' and result['updates']==400
    assert not any(e=='update' for e,r in events)
