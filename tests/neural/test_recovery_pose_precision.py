"""Only synthetic scalar optimizer steps; cached-model tests remain forward/backward only."""
from copy import deepcopy
import json
import os
import threading

import pytest
import torch

from knolling_dataset.schema import digest
from knolling_dataset.builder import file_hash
from knolling_model.recovery.driver import read_config, load_checkpoint, load_grounding_transfer
from knolling_model.recovery.pose_precision import PosePrecision, POLICY
from knolling_model.recovery.stages import parameter_groups
from test_recovery_driver_v4 import ScalarV4, config, configure, engine


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    os.environ.get('RUN_RECOVERY_CUDA_TESTS') != '1', reason='explicit CUDA'))]


def metric(position=10., angle=2.):
    return dict(schema='recovery_metrics_v4', stage='micro_pose', complete_set=True,
                source_mode='oracle', evaluation_mode='free_rollout',
                micro=dict(max_position_mm=position, max_angle_deg=angle))


@pytest.mark.parametrize('device', DEVICES)
def test_one_transition_preserves_weights_second_moment_step_and_frozen_goal(device):
    m=ScalarV4().to(device);configure(m,'micro_pose')
    opt=torch.optim.AdamW(parameter_groups(m),weight_decay=1e-4,foreach=False)
    for _ in range(3):
        opt.zero_grad();m.layout.pose.weight.square().sum().backward();opt.step()
    weights=deepcopy(m.state_dict());before=deepcopy(opt.state_dict())
    phase=PosePrecision(m,opt)
    for bad in (metric(10.001,2),metric(10,2.001),metric(float('nan')),metric(None),
                dict(metric(),complete_set=False),dict(metric(),stage='grounding')):
        assert not phase.consider(bad,3)
    assert phase.consider(metric(),3)
    after=opt.state_dict()
    assert all(g['lr']==1e-6 for g in after['param_groups'])
    for key,v in before['state'][0].items():
        if key=='exp_avg':assert torch.count_nonzero(after['state'][0][key])==0
        else:torch.testing.assert_close(after['state'][0][key],v,atol=0,rtol=0)
    for key,v in weights.items():torch.testing.assert_close(m.state_dict()[key],v,atol=0,rtol=0)
    opt.zero_grad();m.layout.pose.weight.square().sum().backward();opt.step()
    moment=opt.state_dict()['state'][0]['exp_avg'].clone()
    assert torch.count_nonzero(moment)>0
    assert not phase.consider(metric(0,0),4)
    torch.testing.assert_close(opt.state_dict()['state'][0]['exp_avg'],moment,atol=0,rtol=0)
    restored=PosePrecision(m,opt,saved=phase.state,continuing=True)
    assert not restored.consider(metric(),4)
    with pytest.raises(ValueError,match='missing'):PosePrecision(m,opt,continuing=True)
    with pytest.raises(ValueError,match='mismatch'):PosePrecision(m,opt,saved=dict(phase='coarse',transition_update=None),continuing=True)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('pause_at',[1,2,3])
def test_resume_before_at_after_transition_matches_uninterrupted(tmp_path,device,pause_at):
    c=config();c.update(layout_lr=1e-4,micro_precision=deepcopy(POLICY),evaluate_every=99)
    c['micro_pose']['max_updates']=6;c['stop_on_initial_pass']=True
    ground=engine(tmp_path,'ground',c=c,evaluate=lambda *_:{'pass_fixture':True}).run('grounding')['checkpoint']
    def run(name,resume=None,stop_at=None):
        stop=threading.Event();current=[0];events=[]
        def observe(event,**r):
            current[0]=r['update'];events.append((event,r))
            if event=='evaluation' and r['update']==stop_at:stop.set()
        def evaluate(*_):
            out=metric(9 if current[0]>=2 else 11,1)
            out['pass_fixture']=current[0]>=4
            return out
        e=engine(tmp_path,name,c=c,model=ScalarV4().to(device),evaluate=evaluate,stop_event=stop,observer=observe)
        r=e.run('micro_pose',parent=None if resume else ground,resume=resume,micro_case='translation')
        return r,load_checkpoint(r['checkpoint'],c,'fixture-data','fixture-commit'),events
    torch.manual_seed(42);whole,a,events=run('whole')
    torch.manual_seed(42);part,p,first=run('part',stop_at=pause_at)
    assert part['status']=='paused' and part['updates']==pause_at
    resumed,b,second=run('resume',resume=part['checkpoint'])
    assert whole['status']==resumed['status']=='passed' and whole['updates']==resumed['updates']==4
    assert a['pose_precision']==b['pose_precision']==dict(phase='fine',transition_update=2)
    assert sum(e=='precision_transition' for e,_ in first+second)==1
    assert [r['update'] for e,r in events if e=='evaluation']==[0,0,1,2,3,4]
    for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],atol=0,rtol=0)
    for k in a['optimizer']['state'][0]:torch.testing.assert_close(a['optimizer']['state'][0][k],b['optimizer']['state'][0][k],atol=0,rtol=0)
    assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])


def test_precision_budget_no_extension_and_other_stage_isolation(tmp_path):
    c=config();c.update(layout_lr=1e-4,micro_precision=deepcopy(POLICY),stop_on_initial_pass=True)
    root=engine(tmp_path,'root',c=c,evaluate=lambda *_:{'pass_fixture':True}).run('grounding')['checkpoint']
    r=engine(tmp_path,'micro',c=c,evaluate=lambda *_:metric()).run('micro_pose',parent=root,micro_case='translation')
    assert r['status']=='budget_exhausted' and r['updates']==4
    saved=load_checkpoint(r['checkpoint'],c,'fixture-data','fixture-commit')
    assert saved['pose_precision']==dict(phase='fine',transition_update=0)
    with pytest.raises(ValueError,match='interrupted'):
        engine(tmp_path,'denied',c=c).run('micro_pose',resume=r['checkpoint'],micro_case='translation')
    assert load_checkpoint(root,c,'fixture-data','fixture-commit')['pose_precision'] is None
    with pytest.raises(PermissionError,match='case'):
        engine(tmp_path,'wrong-case',c=c,review_receipt={'cases':['translation']}).run('micro_pose',parent=root,micro_case='keep')


def test_pinned_grounding_transfer_and_unchanged_strict_resume(tmp_path):
    c=config();c.update(layout_lr=1e-4,stop_on_initial_pass=True)
    metrics=dict(metric(),stage='grounding',architecture='recovery_v4_goal_joint',grounding={'passed':True},pass_fixture=True)
    root=engine(tmp_path,'root',c=c,evaluate=lambda *_:metrics).run('grounding')['checkpoint']
    target=deepcopy(c);target['micro_precision']=deepcopy(POLICY)
    receipt=dict(grounding_transfer=dict(sha256=file_hash(root),commit='fixture-commit',config_sha256=digest(c)))
    before=file_hash(root)
    saved=load_grounding_transfer(root,target,'fixture-data',receipt)
    assert saved['commit']=='fixture-commit' and file_hash(root)==before
    with pytest.raises(ValueError,match='contract'):load_checkpoint(root,target,'fixture-data','new-commit')
    for key,value in [('dataset','other'),('layout_lr',2e-4),('micro_pose',dict(max_updates=201,max_seconds=120))]:
        bad=deepcopy(target);bad[key]=value
        with pytest.raises(ValueError,match='only'):load_grounding_transfer(root,bad,'fixture-data',receipt)
    bad=deepcopy(receipt);bad['grounding_transfer']['sha256']='wrong'
    with pytest.raises(ValueError,match='unapproved'):load_grounding_transfer(root,target,'fixture-data',bad)
    with pytest.raises(ValueError,match='contract'):load_grounding_transfer(root,target,'wrong-data',receipt)
    failed=engine(tmp_path,'failed',c=c).run('grounding')['checkpoint']
    bad=deepcopy(receipt);bad['grounding_transfer']['sha256']=file_hash(failed)
    with pytest.raises(ValueError,match='passed'):load_grounding_transfer(failed,target,'fixture-data',bad)


def test_policy_config_validation(tmp_path):
    c=read_config('configs/recovery_v4_micro_precision.json')
    assert c['micro_precision']==POLICY and c['micro_pose']==dict(max_updates=200,max_seconds=120)
    for key,value in [('fine_lr',0),('position_mm',100),('unexpected',1)]:
        bad=deepcopy(c);bad['micro_precision'][key]=value
        path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='policy'):read_config(path)


def test_reject_partial_pose_or_extra_trainable_groups():
    m=ScalarV4();m.layout.pose=torch.nn.Linear(2,2);configure(m,'micro_pose')
    partial=torch.optim.AdamW([dict(name='layout',params=[m.layout.pose.weight],lr=1e-4)])
    with pytest.raises(ValueError,match='exactly all'):PosePrecision(m,partial)
    m.layout.goal_encoder.requires_grad_(True)
    extra=torch.optim.AdamW(parameter_groups(m))
    with pytest.raises(ValueError,match='exactly all'):PosePrecision(m,extra)


def test_precision_timeout_final_evaluation_without_extra_update(tmp_path):
    c=config();c.update(layout_lr=1e-4,micro_precision=deepcopy(POLICY),stop_on_initial_pass=True)
    c['micro_pose']['max_seconds']=1
    parent=engine(tmp_path,'root',c=c,evaluate=lambda *_:{'pass_fixture':True}).run('grounding')['checkpoint']
    now=[0.];evaluations=[]
    def evaluate(*_):
        evaluations.append(now[0]);return metric()
    e=engine(tmp_path,'timeout',c=c,evaluate=evaluate,clock=lambda:now[0])
    original=e.loss_fn
    def expire(*args,**kw):
        terms=original(*args,**kw);now[0]=2.;return terms
    e.loss_fn=expire
    r=e.run('micro_pose',parent=parent,micro_case='translation')
    assert r['status']=='budget_exhausted' and r['updates']==0
    assert evaluations==[0.,0.,2.] and r['metrics']['micro']['max_position_mm']==10
    saved=load_checkpoint(r['checkpoint'],c,'fixture-data','fixture-commit')
    assert saved['pose_precision']==dict(phase='fine',transition_update=0)
