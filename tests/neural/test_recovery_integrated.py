"""Lifecycle tests on random tensor fixtures, never experiment training."""
from copy import deepcopy
import threading

import pytest
import torch

from knolling_dataset.schema import digest
from knolling_model.recovery.batches import StageBatch
from knolling_model.recovery.driver import StageEngine, CHECKPOINT_SCHEMA, read_config, save_exclusive, load_checkpoint
from knolling_model.recovery.integrated import PHASES, load_integrated, retained
from knolling_model.recovery.evaluation_v4 import MICRO_CASES
from test_recovery_readout import Fixture, batch_for, DEVICES


def setup(tmp_path, device='cpu'):
    torch.set_num_threads(2); torch.manual_seed(113)
    c=read_config('configs/recovery_v4_integrated_review.json')
    c['integrated_pose']['stages']['micro_pose']={p:2 for p in PHASES}
    c['micro_pose']['max_updates']=12
    m=Fixture().to(device)
    parent=tmp_path/'ground.pt'
    save_exclusive(parent,dict(schema=CHECKPOINT_SCHEMA,config=deepcopy(c),config_sha256=digest(c),
        dataset_fingerprint='fixture',commit='test',architecture=m.architecture,evaluation_schema=c['evaluation_schema'],
        goal_contract=c['goal_contract'],purpose='production',stage='grounding',status='passed',
        model=m.state_dict(),completed={'grounding':{'pass':True}}))
    return c,m,parent


def make(tmp_path,name,c,m,**kwargs):
    batch=batch_for(m,next(m.parameters()).device)
    for meta,sid in zip(batch.metadata,MICRO_CASES['axis']):meta['sample_id']=sid
    events=[]; external=kwargs.pop('observer',None)
    def observe(event,**r):
        events.append((event,deepcopy(r)))
        if external:external(event,**r)
    def evaluate(*_):
        return metrics(position=20 if not any(e=='update' for e,_ in events) else 5)
    e=StageEngine(m,c,'fixture','test',tmp_path/name,
        lambda stage,epoch:[StageBatch(batch,None)],kwargs.pop('evaluate',evaluate),
        readout_batches=lambda:[batch],expected_ids=list(MICRO_CASES['axis']),
        gate_fn=lambda stage,x:x.get('pass',False),clock=kwargs.pop('clock',lambda:0.),observer=observe,**kwargs)
    return e,events


def metrics(position=5,angle=1,success=(),pairs=(),violations=0,passed=False):
    return dict(pass_=passed,**{'pass':passed},complete_set=True,integrated_evidence=dict(
        sample_ids=list(MICRO_CASES['axis']),successful_scenes=list(success),successful_pairs=list(pairs),
        keep_violations=violations,max_position_mm=position,max_angle_deg=angle,
        angle_errors_deg={'a/0':angle,'a/1':angle,'b/0':angle,'b/1':angle}))


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('pause_phase',PHASES)
def test_all_phase_resume_and_fresh_final_cache(tmp_path,device,pause_phase):
    c,m,parent=setup(tmp_path,device)
    whole,events=make(tmp_path,'whole',c,m)
    result=whole.run('micro_pose',parent=parent,micro_case='axis')
    assert result['status']=='budget_exhausted',result
    phases=[r['next_phase'] for e,r in events if e=='phase_transition']
    assert phases==list(PHASES[1:])
    state=load_integrated(result['checkpoint'],c,'fixture','test')
    assert state['integrated']['fine_resets']==1 and result['updates']<=12
    caches=[torch.load(p,weights_only=True) for p in (tmp_path/'whole').glob('features-*.pt')]
    assert len(caches)==4 and caches[0]['producer']!=caches[2]['producer']
    stop=threading.Event()
    def pause(event,**r):
        if event=='update' and r['phase']==pause_phase:stop.set()
    part,_=make(tmp_path,'part',c,Fixture().to(device),observer=pause,stop_event=stop)
    paused=part.run('micro_pose',parent=parent,micro_case='axis')
    assert paused['status']=='paused',paused
    resume,_=make(tmp_path,'resume',c,Fixture().to(device))
    final=resume.run('micro_pose',resume=paused['checkpoint'],micro_case='axis')
    assert final['status']=='budget_exhausted',final
    saved=load_integrated(final['checkpoint'],c,'fixture','test')
    equal(state['model'],saved['model']);equal(state['optimizer'],saved['optimizer'])
    equal(state['rng'],saved['rng'])
    for k in ('update','epoch','batch'):assert state['cursor'][k]==saved['cursor'][k]
    assert state['integrated']['closure_evaluations']==saved['integrated']['closure_evaluations']
    assert state['integrated']['solver']['cache_key']==saved['integrated']['solver']['cache_key']
    with pytest.raises(ValueError,match='terminal'):
        resume.run('micro_pose',resume=paused['checkpoint'] if False else final['checkpoint'],micro_case='axis')


def test_head_journal_resume_and_report_failure_preserves_weights(tmp_path):
    c,m,parent=setup(tmp_path);stop=threading.Event();journals=[]
    def observer(event,**r):
        if event=='update' and r['phase']=='head_xy':
            journals.append(r['journal']);stop.set();raise OSError('audit report failed')
    e,events=make(tmp_path,'part',c,m,observer=observer,stop_event=stop)
    part=e.run('micro_pose',parent=parent,micro_case='axis')
    assert part['status']=='paused' and part['observer_errors']
    journal=load_integrated(journals[0],c,'fixture','test')
    equal(journal['model'],load_integrated(part['checkpoint'],c,'fixture','test')['model'])
    resumed,_=make(tmp_path,'resumed',c,Fixture())
    end=resumed.run('micro_pose',resume=journals[0],micro_case='axis')
    assert end['status']=='budget_exhausted',end
    whole,_=make(tmp_path,'whole',c,Fixture());result=whole.run('micro_pose',parent=parent,micro_case='axis')
    equal(load_integrated(end['checkpoint'],c,'fixture','test')['model'],
          load_integrated(result['checkpoint'],c,'fixture','test')['model'])


@pytest.mark.parametrize('where',['coarse','head_xy','fine'])
def test_timeout_final_evaluation_regression_rollback(tmp_path,where):
    c,m,parent=setup(tmp_path);now=[0.];phase=['coarse'];expired=[False];checks=[]
    def observe(event,**r):
        phase[0]=r['phase']
        if event=='update' and phase[0]==where:now[0]=121;expired[0]=True
    def evaluate(*_):
        checks.append(expired[0]);return metrics(position=20 if phase[0]=='coarse' else 5,
            success=() if expired[0] else ('stable',))
    e,_=make(tmp_path,'timeout',c,m,observer=observe,evaluate=evaluate,clock=lambda:now[0])
    result=e.run('micro_pose',parent=parent,micro_case='axis')
    assert checks[-1] is True
    if where=='coarse':assert result['status']=='budget_exhausted'
    else:
        assert result['status']=='rolled_back',result
        saved=load_integrated(result['checkpoint'],c,'fixture','test')
        anchor=torch.load(saved['integrated']['anchor_path'],weights_only=True)
        equal(saved['model'],anchor)


def test_not_near_stops_after_heads_without_fine(tmp_path):
    c,m,parent=setup(tmp_path)
    e,events=make(tmp_path,'far',c,m,evaluate=lambda *_:metrics(position=11,angle=3))
    result=e.run('micro_pose',parent=parent,micro_case='axis')
    assert result['status']=='fine_ineligible'
    assert [r['next_phase'] for e,r in events if e=='phase_transition']==['head_xy','head_angle']


def test_scene_identity_retention_and_fail_closed():
    assert not retained(metrics(success=['other']),metrics(success=['old']))
    assert not retained(metrics(pairs=['other']),metrics(pairs=['old']))
    assert not retained(metrics(violations=1),metrics())
    assert not retained(metrics(angle=3),metrics(),angles=True)
    assert retained(metrics(angle=1),metrics(angle=3),angles=True)


def test_initial_gate_stops_without_optimizer_step(tmp_path):
    c,m,parent=setup(tmp_path)
    e,events=make(tmp_path,'pass',c,m,evaluate=lambda *_:metrics(passed=True))
    result=e.run('micro_pose',parent=parent,micro_case='axis')
    assert result['status']=='passed' and result['updates']==0
    assert not any(event=='update' for event,_ in events)


def test_atomic_result_and_disabled_integrated_cli(tmp_path):
    import json,subprocess,sys
    from knolling_model.recovery.driver import save_json_exclusive
    path=tmp_path/'result.json';save_json_exclusive(path,{'status':'passed'})
    with pytest.raises(FileExistsError):save_json_exclusive(path,{'status':'failed'})
    assert json.loads(path.read_text())=={'status':'passed'}
    with pytest.raises(ValueError):save_json_exclusive(tmp_path/'bad.json',{'loss':float('nan')})
    assert not (tmp_path/'bad.json').exists() and not list(tmp_path.glob('.report-*'))
    out=tmp_path/'forbidden'
    run=subprocess.run([sys.executable,'-m','knolling_model.recovery.driver','--config',
        'configs/recovery_v4_integrated_review.json','--stage','micro_pose','--micro-case','axis',
        '--output',str(out),'--approval','missing'],capture_output=True,text=True)
    assert run.returncode!=0 and 'training disabled' in run.stderr and not out.exists()


def test_deadline_inside_line_search_restores_accepted_weights(tmp_path,monkeypatch):
    from knolling_model.recovery.readout import HeadFitter
    c,m,parent=setup(tmp_path);now=[0.];original=HeadFitter.loss;expired=[False]
    def deadline(self):
        value=original(self);now[0]=121.;expired[0]=True;return value
    monkeypatch.setattr(HeadFitter,'loss',deadline)
    e,events=make(tmp_path,'deadline',c,m,clock=lambda:now[0],evaluate=lambda *_:metrics())
    result=e.run('micro_pose',parent=parent,micro_case='axis')
    assert expired[0] and result['status']=='budget_exhausted' and result['updates']==0
    saved=load_integrated(result['checkpoint'],c,'fixture','test')
    equal(saved['model'],torch.load(parent,weights_only=True)['model'])
    assert saved['integrated']['closure_evaluations']==1
    assert events[-2][0]=='evaluation' or any(e=='evaluation' and r['elapsed_seconds']>=120 for e,r in events)


def test_cache_corruption_and_micro_production_prerequisite(tmp_path):
    c,m,parent=setup(tmp_path);stop=threading.Event()
    def observer(event,**r):
        if event=='update' and r['phase']=='head_xy':stop.set()
    e,_=make(tmp_path,'part',c,m,observer=observer,stop_event=stop)
    result=e.run('micro_pose',parent=parent,micro_case='axis')
    saved=load_integrated(result['checkpoint'],c,'fixture','test')
    from pathlib import Path
    cache=Path(saved['integrated']['cache_path']);cache.write_bytes(b'corrupted fixture')
    again,_=make(tmp_path,'resume',c,Fixture())
    end=again.run('micro_pose',resume=result['checkpoint'],micro_case='axis')
    assert end['status']=='rolled_back' and 'cache artifact hash' in end['error']
    oracle,_=make(tmp_path,'oracle',c,Fixture())
    with pytest.raises(ValueError,match='three'):
        oracle.run('oracle_layout',parent=parent)
    assert not (tmp_path/'oracle').exists()


@pytest.mark.parametrize('change',['cap','lr','bf16','mixed','head_time'])
def test_invalid_integrated_contract_fails_closed(tmp_path,change):
    import json
    c=read_config('configs/recovery_v4_integrated_review.json')
    if change=='cap':c['integrated_pose']['stages']['micro_pose']['coarse']+=1
    elif change=='lr':c['layout_lr']=2e-4
    elif change=='bf16':c['precision']['micro_pose']='bf16'
    elif change=='mixed':c['micro_precision']={}
    else:c['integrated_pose']['head_seconds']=11
    path=tmp_path/'bad.json';path.write_text(json.dumps(c))
    with pytest.raises(ValueError):read_config(path)
