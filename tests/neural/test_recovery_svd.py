"""SVD calibration fixtures only: no real model training."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from knolling_dataset.schema import digest
from knolling_model.recovery.readout import ReadoutData, make_head_fitter
from knolling_model.recovery.svd_readout import SVDXYFitter
from knolling_model.recovery.driver import read_config, save_exclusive
from knolling_model.recovery.integrated import load_integrated, validate_config
from test_recovery_readout import DEVICES, Fixture
from test_recovery_integrated import setup, make, equal, metrics


def fixture(device, overdetermined=False):
    torch.manual_seed(8)
    n,d=(20,3) if overdetermined else (4,8)
    h=torch.randn(n,d)
    if not overdetermined:h[1:]=h[0]+h[1:]*.003
    linear=nn.Linear(d,38).to(device)
    model=nn.Module();model.layout=nn.Module();model.layout.pose=nn.Module()
    model.layout.pose.output=nn.Sequential(nn.Identity(),linear)
    data=ReadoutData(h.to(device),(.3+torch.randn(n,2)*.03).to(device),
        torch.zeros(n,dtype=torch.long,device=device),torch.full((n,),1/n,device=device),
        [[str(i),'o',0] for i in range(n)],'fixture','fixture')
    return model,data


@pytest.mark.parametrize('device',DEVICES)
def test_exact_fit_cast_and_frozen_heads(device):
    m,d=fixture(device);before=deepcopy(m.state_dict());f=SVDXYFitter(m,d)
    # Preparation must not update anything; solve coordinates are CPU FP64.
    equal(before,m.state_dict());assert f.z.device.type=='cpu' and f.z.dtype==torch.float64
    a=f.weights.sqrt()[:,None]*f.design
    torch.testing.assert_close(a.T@a,torch.eye(4,dtype=torch.float64),atol=1e-8,rtol=1e-8)
    loss,_=f.step(lambda:None)
    assert f.measurements()['value']<.01 and loss<1e-15
    for k,v in m.state_dict().items():
        torch.testing.assert_close(v[2:],before[k][2:],atol=0,rtol=0)
    raw=m.layout.pose.output(d.features)[:,:2]
    scale=raw.new_tensor([.3,.2])
    torch.testing.assert_close(raw*scale+scale,f.deployed_xy(f.weight,f.bias),atol=0,rtol=0)
    assert ((raw*scale+scale-d.xy).norm(dim=-1)*1000).max()<.01
    saved=f.state_dict();again=SVDXYFitter(m,d,saved);equal(saved,again.state_dict())


@pytest.mark.parametrize('device',DEVICES)
def test_huber_gradient_weighting_and_resume(device):
    m,d=fixture(device,True)
    # Unequal row weights must multiply Huber OUTSIDE the clipping threshold.
    d.weights=torch.arange(1,21,device=device,dtype=torch.float32);d.weights/=d.weights.sum()
    f=SVDXYFitter(m,d)
    def objective(z):
        q=f.offset+f.design@z
        return (torch.nn.functional.smooth_l1_loss(q,torch.zeros_like(q),reduction='none').sum(-1)*f.weights).sum()
    assert torch.autograd.gradcheck(objective,(f.z,),eps=1e-6,atol=1e-5)
    f.step(lambda:None);f.step(lambda:None)
    saved=deepcopy(f.state_dict());weights=deepcopy(m.state_dict())
    f.step(lambda:None);expected=deepcopy(f.state_dict());end=deepcopy(m.state_dict())
    m.load_state_dict(weights);again=SVDXYFitter(m,d,saved);again.step(lambda:None)
    equal(expected,again.state_dict());equal(end,m.state_dict())
    assert f.diagnostics['full_rank_ls_rms_mm']>1


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('warm',[False,True])
def test_abort_restores_solver_and_model(device,warm):
    m,d=fixture(device,True);f=SVDXYFitter(m,d)
    if warm:f.step(lambda:None)
    saved=deepcopy(f.state_dict());before=deepcopy(m.state_dict());calls=[0]
    def abort():
        calls[0]+=1
        if calls[0]==2:raise TimeoutError('bounded trial')
    with pytest.raises(TimeoutError):f.step(abort)
    equal(saved,f.state_dict());equal(before,m.state_dict())
    f.step(lambda:None)


def test_corrupt_factors_cache_and_algorithm_rejected():
    m,d=fixture('cpu');f=SVDXYFitter(m,d);f.step(lambda:None);state=f.state_dict()
    for key in ('base','projection','z'):
        bad=deepcopy(state);bad[key].flatten()[0]+=.1
        with pytest.raises(ValueError):SVDXYFitter(m,d,bad)
    bad=deepcopy(d);bad.features[0,0]+=.1
    with pytest.raises(ValueError):SVDXYFitter(m,bad,state)
    with pytest.raises(ValueError):make_head_fitter(m,d,'xy',state)
    legacy=make_head_fitter(m,d,'xy').state_dict()
    with pytest.raises(ValueError):make_head_fitter(m,d,'xy',legacy,xy_solver='svd_huber_v1')
    c=read_config('configs/recovery_v4_svd_review.json')
    assert not c['allow_training'];validate_config(c)
    c['integrated_pose']['xy_solver']='unknown'
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('device',DEVICES)
def test_huber_does_not_accept_worse_least_squares_outlier_fit(device):
    m,d=fixture(device,True)
    with torch.no_grad():
        m.layout.pose.output[1].weight[:2].zero_();m.layout.pose.output[1].bias[:2].zero_()
    d.features.zero_();d.xy[:]=d.xy.new_tensor([.3,.2]);d.xy[-1,0]+=.3
    f=SVDXYFitter(m,d);before=float(f.loss().detach())
    assert f.diagnostics['rank']==1
    value,_=f.step(lambda:None)
    assert value<before and not torch.equal(f.z.detach(),f.ls)
    assert f.warm_started and f.optimizer.state and not f.measurements()['passed']
    for p in m.parameters():assert p.grad is None


def svd_setup(tmp_path,device):
    c,m,parent=setup(tmp_path,device)
    c['integrated_pose']['xy_solver']='svd_huber_v1'
    raw=torch.load(parent,map_location='cpu',weights_only=True)
    raw['config']=deepcopy(c);raw['config_sha256']=digest(c)
    parent=tmp_path/'ground-svd.pt';save_exclusive(parent,raw)
    return c,m,parent


class Crash(BaseException):pass


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('outcome',['pass','regression','continue'])
@pytest.mark.parametrize('boundary',['prepared','accepted'])
def test_svd_journal_resume_and_rollback(tmp_path,monkeypatch,device,outcome,boundary):
    from knolling_model.recovery.integrated import IntegratedEngine
    c,m,parent=svd_setup(tmp_path,device)
    active={};original=IntegratedEngine.journal
    def runner(name,model,crash=False):
        active.update(phase='coarse',update=0)
        paths=[]
        def journal(engine):
            path=original(engine)
            at_boundary=(engine.state['pending_head'] is not None)==(boundary=='accepted')
            if crash and engine.state['phase']=='head_xy' and at_boundary:
                paths.append(path);raise Crash()
            return path
        monkeypatch.setattr(IntegratedEngine,'journal',journal)
        def observe(event,**r):active.update(phase=r['phase'],update=r['update'])
        def evaluate(*_):
            candidate=active['phase']=='head_xy' and active['update']>1
            return metrics(position=20 if active['update']==0 else 5,
                success=[] if candidate and outcome=='regression' else ['stable'],
                passed=candidate and outcome=='pass')
        e,events=make(tmp_path,name,c,model,observer=observe,evaluate=evaluate)
        return e,events,paths
    whole,_,_=runner('whole',m);expected=whole.run('micro_pose',parent=parent,micro_case='axis')
    assert expected['status']=={'pass':'passed','regression':'rolled_back','continue':'budget_exhausted'}[outcome]
    part,_,paths=runner('part',Fixture().to(device),True)
    with pytest.raises(Crash):part.run('micro_pose',parent=parent,micro_case='axis')
    again,events,_=runner('again',Fixture().to(device))
    result=again.run('micro_pose',resume=paths[0],micro_case='axis')
    assert result['status']==expected['status'] and result['updates']==expected['updates']
    a=load_integrated(expected['checkpoint'],c,'fixture','test');b=load_integrated(result['checkpoint'],c,'fixture','test')
    for key in ('model','optimizer','rng','cursor','metrics'):equal(a[key],b[key])
    for key in ('solver','phase','phase_update','pending_head','fine_resets'):equal(a['integrated'][key],b['integrated'][key])
    if boundary=='accepted' and outcome!='continue':assert not any(e=='update' for e,_ in events)
    if outcome=='continue':
        caches=[torch.load(p,weights_only=True) for p in (tmp_path/'whole').glob('features-*.pt')]
        assert len(caches)==2 and caches[0]['producer']!=caches[1]['producer']


def test_fp32_stall_restores_head_and_solver(monkeypatch):
    from knolling_model.recovery.svd_readout import SVDNumericalStall
    m,d=fixture('cpu',True);f=SVDXYFitter(m,d);f.step(lambda:None)
    before=deepcopy(m.state_dict());state=deepcopy(f.state_dict())
    baseline=float(f.deployed_loss(f.weight,f.bias));calls=[]
    def rounding(w,b):
        calls.append(1)
        return torch.tensor(baseline+(0 if len(calls)==1 else 1e-6))
    monkeypatch.setattr(f,'deployed_loss',rounding)
    with pytest.raises(SVDNumericalStall) as caught:f.step(lambda:None)
    assert caught.value.diagnostic['fp64_after']<=caught.value.diagnostic['fp64_before']
    equal(before,m.state_dict());equal(state,f.state_dict())


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_deployment_still_fails(monkeypatch,bad):
    m,d=fixture('cpu',True);f=SVDXYFitter(m,d);f.step(lambda:None)
    before=deepcopy(m.state_dict());state=deepcopy(f.state_dict());original=f.deployed_loss;calls=[]
    def nonfinite(w,b):
        calls.append(1);return original(w,b) if len(calls)==1 else torch.tensor(bad)
    monkeypatch.setattr(f,'deployed_loss',nonfinite)
    with pytest.raises(FloatingPointError):f.step(lambda:None)
    equal(before,m.state_dict());equal(state,f.state_dict())


@pytest.mark.parametrize('outcome',['pass','regression','continue'])
def test_stall_journal_resumes_decision_without_solver_retry(tmp_path,monkeypatch,outcome):
    from knolling_model.recovery.integrated import IntegratedEngine
    from knolling_model.recovery.svd_readout import SVDNumericalStall
    c,m,parent=svd_setup(tmp_path,'cpu');active={};original=IntegratedEngine.journal
    def step(self,*args):raise SVDNumericalStall({'fixture':True})
    monkeypatch.setattr(SVDXYFitter,'step',step)
    def runner(name,model,crash=False):
        active.update(phase='coarse',update=0);paths=[]
        def journal(engine):
            path=original(engine)
            if crash and (engine.state.get('head_diagnostic') or {}).get('reason')=='fp32_numerical_stall':
                paths.append(path);raise Crash()
            return path
        monkeypatch.setattr(IntegratedEngine,'journal',journal)
        def observe(event,**r):active.update(phase=r['phase'],update=r['update'])
        def evaluate(*_):
            stalled=active['phase']=='head_xy'
            return metrics(position=20 if active['update']==0 else 5,
                success=[] if stalled and outcome=='regression' else ['stable'],passed=stalled and outcome=='pass')
        e,events=make(tmp_path,name,c,model,observer=observe,evaluate=evaluate)
        return e,events,paths
    whole,_,_=runner('whole',m);a=whole.run('micro_pose',parent=parent,micro_case='axis')
    part,_,paths=runner('part',Fixture(),True)
    with pytest.raises(Crash):part.run('micro_pose',parent=parent,micro_case='axis')
    again,events,_=runner('again',Fixture());b=again.run('micro_pose',resume=paths[0],micro_case='axis')
    assert a['status']==b['status']=={'pass':'passed','regression':'rolled_back','continue':'budget_exhausted'}[outcome]
    assert a['updates']==b['updates']
    aa=load_integrated(a['checkpoint'],c,'fixture','test');bb=load_integrated(b['checkpoint'],c,'fixture','test')
    for k in ('model','optimizer','rng','cursor','metrics'):equal(aa[k],bb[k])
    assert not any(e=='update' and r['phase']=='head_xy' for e,r in events)


def test_real_coefficient_loss_regression_remains_fatal(monkeypatch):
    m,d=fixture('cpu',True);f=SVDXYFitter(m,d);f.step(lambda:None)
    before=deepcopy(m.state_dict());state=deepcopy(f.state_dict());calls=[];doubles=[]
    baseline=float(f.deployed_loss(f.weight,f.bias))
    def worse(w,b):
        calls.append(1);return torch.tensor(baseline+(0 if len(calls)==1 else 1e-6))
    def coefficient(w,b):
        doubles.append(1);return float(len(doubles))
    monkeypatch.setattr(f,'deployed_loss',worse);monkeypatch.setattr(f,'coefficient_loss64',coefficient)
    with pytest.raises(FloatingPointError,match='increases Huber'):f.step(lambda:None)
    equal(before,m.state_dict());equal(state,f.state_dict())
