"""Only scalar fixture optimizer updates; no Knolling model training."""
from copy import deepcopy
import json
import os
import subprocess
import sys
import threading

import pytest
import torch
from torch import nn

from knolling_dataset.builder import file_hash
from knolling_model.arrangement.driver import StageEngine,read_config,load_checkpoint,save_exclusive
from knolling_model.arrangement.evaluation_v4 import MICRO_CASES
from knolling_model.arrangement.goal_joint_layout import ARCHITECTURE
from test_recovery_driver import batches
from test_recovery_dataset_v3 import suite_v3


class ScalarV4(nn.Module):
    architecture=ARCHITECTURE
    def __init__(self):
        super().__init__()
        self.layout=nn.Module()
        self.layout.pose=nn.Linear(1,1,bias=False)
        self.layout.goal_encoder=nn.Linear(1,1,bias=False)
        self.perception=nn.Linear(1,1,bias=False)
        with torch.no_grad():
            self.layout.pose.weight.fill_(.4);self.layout.goal_encoder.weight.fill_(.2)


def config():
    c=read_config('configs/recovery_v4_execution.json')
    c.update(evaluate_every=1,checkpoint_every=1,stop_on_initial_pass=False,layout_lr=.01,goal_lr=.01)
    for spec in [*c['stages'].values(),c['micro_pose']]:spec.update(max_updates=4,max_seconds=600)
    return c


def configure(m,stage):
    m.stage=stage;m.requires_grad_(False);m.zero_grad(set_to_none=True);m.train()
    (m.layout.goal_encoder if stage=='grounding' else m.layout.pose).requires_grad_(True)


def loss(m,x,truth,replay=None):
    layer=m.layout.goal_encoder if m.stage=='grounding' else m.layout.pose
    return {'fixture':(layer(x['x'].to(next(m.parameters()).device))-torch.rand(1,device=next(m.parameters()).device)).square().mean()},{}


def micro_metric(case):
    return dict(schema='recovery_metrics_v4',architecture=ARCHITECTURE,stage='micro_pose',complete_set=True,
                evaluation_mode='free_rollout',source_mode='oracle',pass_fixture=True,
                micro=dict(passed=True,case=case,sample_ids=list(MICRO_CASES[case])))


def engine(tmp_path,name,c=None,model=None,evaluate=None,**kw):
    return StageEngine(model or ScalarV4(),c or config(),'fixture-data','fixture-commit',tmp_path/name,batches,
        evaluate or (lambda *_:{'pass_fixture':False,'full_set_no_regression':True}),
        loss_fn=loss,configure_fn=configure,gate_fn=lambda s,m:bool(m.get('pass_fixture')),**kw)


def roots(tmp_path,c):
    c['stop_on_initial_pass']=True
    ground=engine(tmp_path,'ground',c=c,evaluate=lambda *_:{'pass_fixture':True}).run('grounding')['checkpoint']
    checks=[]
    for case in MICRO_CASES:
        e=engine(tmp_path,case,c=c,evaluate=lambda *_,case=case:micro_metric(case))
        checks.append(e.run('micro_pose',parent=ground,micro_case=case)['checkpoint'])
    return ground,checks


def test_disabled_cli_and_strict_goal_configuration(tmp_path):
    c=read_config('configs/recovery_v4_execution.json')
    for stage in ('grounding','micro_pose','oracle_layout'):
        out=tmp_path/stage
        p=subprocess.run([sys.executable,'-m','knolling_model.arrangement.driver','--config','configs/recovery_v4_execution.json',
            '--stage',stage,'--output',str(out),'--approval','nonexistent'],capture_output=True,text=True)
        assert p.returncode!=0 and 'training disabled' in p.stderr and not out.exists()
    for field in ('goal_contract','data_v3','micro_pose'):
        bad=deepcopy(c);bad.pop(field);p=tmp_path/(field+'.json');p.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='v4 requires'):read_config(p)
    bad=deepcopy(c);bad['goal_contract']['vocabulary']['gap'].append(.021)
    p=tmp_path/'vocab.json';p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):read_config(p)


@pytest.mark.parametrize('stage',['grounding','micro_pose','oracle_layout'])
def test_v4_resume_optimizer_rng_cursor_and_goal_freeze(tmp_path,stage):
    c=config();parent=None;micro=[];case=None
    if stage!='grounding':
        parent,micro=roots(tmp_path,c);c['stop_on_initial_pass']=True
        # Keep the same config contract as the roots, but never pass this run.
    if stage=='micro_pose':case='translation'
    kwargs=dict(parent=parent,micro_case=case,micro_checkpoints=micro if stage=='oracle_layout' else None)
    torch.manual_seed(91);whole=engine(tmp_path,'whole',c=c).run(stage,**kwargs)
    torch.manual_seed(91);stop=threading.Event();calls=0
    def evaluate(*_):
        nonlocal calls
        calls+=1
        if calls==4:stop.set()
        return {'pass_fixture':False}
    part=engine(tmp_path,'part',c=c,evaluate=evaluate,stop_event=stop).run(stage,**kwargs)
    assert part['status']=='paused' and part['updates']==2
    resumed=engine(tmp_path,'resume',c=c).run(stage,resume=part['checkpoint'],micro_case=case)
    a=load_checkpoint(whole['checkpoint'],c,'fixture-data','fixture-commit')
    b=load_checkpoint(resumed['checkpoint'],c,'fixture-data','fixture-commit')
    for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],atol=0,rtol=0)
    for k in ('update','epoch','batch'):assert a['cursor'][k]==b['cursor'][k]
    for k in a['optimizer']['state'][0]:torch.testing.assert_close(a['optimizer']['state'][0][k],b['optimizer']['state'][0][k],atol=0,rtol=0)
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    if parent:
        p=load_checkpoint(parent,c,'fixture-data','fixture-commit')
        torch.testing.assert_close(b['model']['layout.goal_encoder.weight'],p['model']['layout.goal_encoder.weight'],atol=0,rtol=0)
    with pytest.raises(ValueError,match='interrupted'):
        engine(tmp_path,'denied',c=c).run(stage,resume=whole['checkpoint'],micro_case=case)
    bad=deepcopy(b);bad['goal_contract']['schema']='bad';p=tmp_path/'bad.pt';save_exclusive(p,bad)
    with pytest.raises(ValueError,match='goal'):load_checkpoint(p,c,'fixture-data','fixture-commit')


def test_micro_parent_isolation_production_gate_and_contracts(tmp_path):
    c=config();ground,checks=roots(tmp_path,c)
    parent=load_checkpoint(ground,c,'fixture-data','fixture-commit')
    # A diagnostic checkpoint must never become a production parent.
    with pytest.raises(ValueError,match='previous stage'):
        engine(tmp_path,'diagnostic-parent',c=c).run('oracle_layout',parent=checks[0],micro_checkpoints=checks)
    with pytest.raises(ValueError,match='three'):
        engine(tmp_path,'missing',c=c).run('oracle_layout',parent=ground,micro_checkpoints=checks[:2])
    for change in ('case','parent','status'):
        bad=load_checkpoint(checks[0],c,'fixture-data','fixture-commit')
        if change=='case':bad['micro_case']='axis'
        elif change=='parent':bad['goal_parent_sha256']='different'
        else:bad['status']='budget_exhausted'
        p=tmp_path/(change+'.pt');save_exclusive(p,bad)
        with pytest.raises(ValueError,match='diagnostic'):
            engine(tmp_path,change+'-out',c=c).run('oracle_layout',parent=ground,micro_checkpoints=[str(p),*checks[1:]])
    # Alter diagnostic weights to show production uses the grounding original.
    altered=load_checkpoint(checks[0],c,'fixture-data','fixture-commit')
    altered['model']['layout.pose.weight'].fill_(100)
    p=tmp_path/'altered.pt';save_exclusive(p,altered)
    e=engine(tmp_path,'oracle',c=c,evaluate=lambda *_:{'pass_fixture':True})
    result=e.run('oracle_layout',parent=ground,micro_checkpoints=[str(p),*checks[1:]])
    assert result['status']=='passed' and result['updates']==0
    saved=load_checkpoint(result['checkpoint'],c,'fixture-data','fixture-commit')
    assert len(saved['micro_evidence'])==3 and saved['goal_parent_sha256']==file_hash(ground)
    torch.testing.assert_close(saved['model']['layout.pose.weight'],parent['model']['layout.pose.weight'],atol=0,rtol=0)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(os.environ.get('RUN_RECOVERY_CUDA_TESTS')!='1',reason='explicit CUDA'))])
def test_v4_joint_timeout_final_evaluation_rollback(tmp_path,device):
    c=config();c['stages']['joint']['max_seconds']=1
    ground,checks=roots(tmp_path,c);parent=ground
    for stage in ('oracle_layout','perception','predicted_layout'):
        parent=engine(tmp_path,stage,c=c,evaluate=lambda *_:{'pass_fixture':True}).run(
            stage,parent=parent,micro_checkpoints=checks if stage=='oracle_layout' else None)['checkpoint']
    reference=load_checkpoint(parent,c,'fixture-data','fixture-commit')['model']
    now=[0.];calls=[0]
    def evaluation(*_):
        calls[0]+=1
        return {'pass_fixture':False,'full_set_no_regression':calls[0]<=2}
    e=engine(tmp_path,'timeout',c=c,model=ScalarV4().to(device),evaluate=evaluation,clock=lambda:now[0])
    def expire(*args,**kwargs):
        out=loss(*args,**kwargs);now[0]=2.;return out
    e.loss_fn=expire
    result=e.run('joint',parent=parent)
    assert result['status']=='rolled_back' and result['updates']==0 and calls[0]==3
    for k,v in e.model.state_dict().items():torch.testing.assert_close(v.cpu(),reference[k],atol=0,rtol=0)


def test_uniform_exposure_and_fixed_micro_subset(suite_v3):
    from knolling_model.arrangement.goal_labels import GoalDataset
    from knolling_model.arrangement.batches import stage_batches
    from knolling_model.arrangement.micro import MicroDataset
    ds=GoalDataset(suite_v3)
    for stage in ('grounding','oracle_layout','predicted_layout'):
        batches_=list(stage_batches(ds,lambda x:x,stage,seed=1203,epoch=0,batch_size=8,uniform=True))
        ids=[s['metadata']['sample_id'] for b in batches_ for s in b.batch]
        assert len(ids)==len(set(ids))==76
    for case,ids in MICRO_CASES.items():
        subset=MicroDataset(ds,case)
        assert [subset[i]['metadata']['sample_id'] for i in range(len(subset))]==list(ids)


def test_micro_coordinator_disabled_before_files(tmp_path):
    output=tmp_path/'never'
    p=subprocess.run([sys.executable,'-m','knolling_model.arrangement.micro','--grounding-checkpoint','absent',
                      '--approval','absent','--output',str(output)],capture_output=True,text=True)
    assert p.returncode!=0 and 'training disabled' in p.stderr and not output.exists()


@pytest.mark.parametrize('paused',[False,True])
def test_micro_coordinator_independent_roots_and_stop(tmp_path,monkeypatch,paused):
    import knolling_model.arrangement.driver as driver
    import knolling_model.arrangement.micro as micro
    c=config();c['allow_training']=True
    monkeypatch.setattr(driver,'read_config',lambda _:c)
    calls=[]
    def run(cmd,check):
        calls.append(cmd)
        folder=__import__('pathlib').Path(cmd[cmd.index('--output')+1]);folder.mkdir(parents=True)
        (folder/'result.json').write_text(json.dumps(dict(status='paused' if paused else 'passed',checkpoint=str(folder/'checkpoint.pt'))))
    monkeypatch.setattr(subprocess,'run',run)
    out=tmp_path/'micro'
    monkeypatch.setattr(sys,'argv',['micro','--grounding-checkpoint','same-grounding.pt','--approval','review.json','--output',str(out)])
    if paused:
        with pytest.raises(RuntimeError,match='stop'):micro.main()
        assert len(calls)==1 and not (out/'micro-results.json').exists()
    else:
        micro.main()
        assert len(calls)==3 and (out/'micro-results.json').exists()
        assert [cmd[cmd.index('--micro-case')+1] for cmd in calls]==list(MICRO_CASES)
    assert all(cmd[cmd.index('--parent-checkpoint')+1]=='same-grounding.pt' for cmd in calls)
