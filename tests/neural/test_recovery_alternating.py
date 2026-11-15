"""CPU-only tensor fixtures; no real model/experiment feature updates."""
from copy import deepcopy
import json
import threading

import pytest
import torch

from knolling_model.arrangement import alternating as alt
from knolling_model.arrangement.driver import StageEngine, read_config, rng_state, save_exclusive, authorize
from knolling_model.arrangement.integrated import load_integrated
from knolling_model.arrangement.readout import HeadFitter, collect_readout
from knolling_model.arrangement.svd_readout import SVDXYFitter, SVDNumericalStall
from knolling_model.arrangement.stages import configure_stage, stage_loss
from test_recovery_readout import Fixture, batch_for
from test_recovery_integrated import equal
from test_recovery_svd import fixture as svd_fixture


class Crash(BaseException): pass


def diagnostic(rms=None, hv=30., angle=89., bins=100, keep=10.):
    return dict(xy_svd_rms_mm=alt.POLICY['reference_rms_mm'] if rms is None else rms,
        hv_relative_error_mm=hv, angle_p95_deg=angle, bin_correct=bins,
        keep_rms_mm=keep, keep_angle_rms_deg=0., keep={'o':dict(xy_mm=keep,angle_deg=0)})


def test_gate_requires_all_evidence_not_loss():
    base=diagnostic();now=diagnostic(rms=22.,hv=29.)
    assert alt.trial_gate(now,base)
    for changes in (dict(xy_svd_rms_mm=23.),dict(hv_relative_error_mm=30.),dict(angle_p95_deg=90.),
                    dict(bin_correct=99),dict(keep_rms_mm=11.),dict(keep_angle_rms_deg=.1),
                    dict(xy_svd_rms_mm=float('nan'))):
        assert not alt.trial_gate(dict(now,**changes),base)
    base['keep']['o']['xy_mm']=.5;now['keep']['o']['xy_mm']=1.1
    assert not alt.trial_gate(now,base)


def test_shared_update_freezes_every_head_and_goal():
    torch.set_num_threads(2);torch.manual_seed(12);m=Fixture();configure_stage(m,'oracle_layout')
    b=batch_for(m);before=deepcopy(m.state_dict());opt=alt.FeatureAdam(m,1e-4,1e-4)
    terms,out=stage_loss(m,b.inputs,b.supervision);sum(terms.values()).backward()
    assert all(p.grad is None for p in m.layout.pose.output[1].parameters())
    assert all(p.grad is None for p in m.layout.goal_encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in opt.parameters)
    opt.optimizer.step()
    changed=[k for k,v in m.state_dict().items() if not torch.equal(v,before[k])]
    assert changed and all(k.startswith('layout.pose.') and not k.startswith('layout.pose.output.1.') for k in changed)
    saved=deepcopy(opt.state_dict());again=alt.FeatureAdam(m,1e-4,1e-4);again.load_state_dict(saved)
    equal(saved,again.state_dict())


def test_huber_inner_converges_with_outlier_and_differs_from_ls():
    torch.set_num_threads(2);m,d=svd_fixture('cpu',True)
    with torch.no_grad():m.layout.pose.output[1].weight[:2].zero_();m.layout.pose.output[1].bias[:2].zero_()
    d.features.zero_();d.xy[:]=torch.tensor([.3,.2]);d.xy[-1,0]+=.3
    f=SVDXYFitter(m,d)
    for _ in range(50):
        try:loss,done=f.step(lambda:None)
        except SVDNumericalStall:break
        if done:break
    grad,=torch.autograd.grad(f.loss(),f.z)
    assert float(grad.abs().max()) < 1e-6
    # Huber optimum: 19 inliers balance one saturated outlier at .01/19 m.
    xy=f.deployed_xy(f.weight,f.bias)
    assert abs(float(xy[0,0])-(.3+.01/19))<1e-7
    assert abs(float(xy[0,0])-(.3+.3/20))>.01


def test_fixed_head_feature_gradient_matches_finite_difference():
    torch.set_num_threads(2);torch.manual_seed(4)
    # Double precision differentiates the unchanged Huber math, away from kinks.
    x=torch.randn(12,3,dtype=torch.float64);theta=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
    head=torch.randn(4,2,dtype=torch.float64)*.02;target=torch.randn(12,2,dtype=torch.float64)*.03
    weights=torch.arange(1,13,dtype=torch.float64);weights/=weights.sum()
    def loss(t):
        xy=torch.tanh(x@t)@head
        return (torch.nn.functional.smooth_l1_loss(xy/.01,target/.01,reduction='none').sum(-1)*weights).sum()
    assert torch.autograd.gradcheck(loss,(theta,),eps=1e-6,atol=1e-5,rtol=1e-4)
    assert not head.requires_grad


def test_huber_profile_envelope_gradient_finite_difference():
    """Re-solve convex Huber at +/- perturbations, not just a fixed-head check."""
    torch.set_num_threads(2);torch.manual_seed(27)
    x=torch.randn(16,2,dtype=torch.float64);y=torch.randn(16,2,dtype=torch.float64)*.04
    # One nonlinear feature parameter; full-column-rank design at all probes.
    def features(t):return torch.cat((torch.tanh(x*t),torch.ones(16,1,dtype=torch.float64)),1)
    def solve(t):
        f=features(t).detach();a=torch.linalg.lstsq(f,y).solution.detach().requires_grad_(True)
        opt=torch.optim.LBFGS([a],max_iter=200,tolerance_grad=1e-11,tolerance_change=1e-15,line_search_fn='strong_wolfe')
        def loss():return torch.nn.functional.smooth_l1_loss((f@a)/.01,y/.01,reduction='none').sum(-1).mean()
        def closure():opt.zero_grad();v=loss();v.backward();return v
        opt.step(closure);grad,=torch.autograd.grad(loss(),a)
        assert float(grad.abs().max())<1e-5
        return a.detach(),float(loss().detach())
    t=torch.tensor(.7,dtype=torch.float64,requires_grad=True);head,value=solve(t)
    loss=torch.nn.functional.smooth_l1_loss((features(t)@head)/.01,y/.01,reduction='none').sum(-1).mean()
    grad,=torch.autograd.grad(loss,t);eps=1e-4
    numerical=(solve(t.detach()+eps)[1]-solve(t.detach()-eps)[1])/(2*eps)
    assert float(grad)==pytest.approx(numerical,rel=2e-3,abs=2e-5)


def build(tmp_path,monkeypatch,outcome='fail'):
    torch.set_num_threads(2);torch.manual_seed(81)
    c=read_config('configs/recovery_v4_alternating_review.json');m=Fixture();b=batch_for(m)
    ids=[str(i) for i in range(76)];batches=[]
    for k in range(38):
        batch=deepcopy(b)
        for j,meta in enumerate(batch.metadata):meta['sample_id']=ids[2*k+j]
        batches.append(batch)
    metrics=dict(complete_set=True,integrated_evidence=dict(sample_ids=ids,successful_scenes=[],successful_pairs=[],
        keep_violations=0,max_position_mm=100,max_angle_deg=89,angle_errors_deg={'o':89.}))
    prior=dict(model=deepcopy(m.state_dict()),completed={'grounding':{'passed':True}},goal_parent_sha256='ground',
        micro_evidence={},cursor=dict(update=600,epoch=59,batch=10,elapsed_seconds=147.9),rng=rng_state(),metrics=metrics)
    source=tmp_path/'source.pt';save_exclusive(source,prior)
    monkeypatch.setattr(alt,'load_source',lambda *a:deepcopy(prior))
    real=HeadFitter.step
    def one_step(self,*a,**kw):v,_=real(self,*a,**kw);return v,True
    monkeypatch.setattr(HeadFitter,'step',one_step)
    def diag(self):
        if not hasattr(self,'cursor') or self.cursor.update==0:return diagnostic()
        if outcome=='pass':return diagnostic(rms=22.,hv=29.)
        if outcome=='regression':return diagnostic(angle=90.)
        return diagnostic()
    monkeypatch.setattr(alt.AlternatingEngine,'diagnostics',diag)
    def make(name,observer=None,clock=lambda:0.,stop=None):
        model=Fixture()
        return StageEngine(model,c,'fixture','test',tmp_path/name,lambda *_:[],lambda *_:metrics,
            readout_batches=lambda:batches,expected_ids=ids,pairs=[{}]*35,observer=observer,clock=clock,stop_event=stop)
    return c,source,make


@pytest.mark.parametrize('boundary',['head_candidate','feature_update','transition','initial_journal'])
@pytest.mark.parametrize('outcome',['pass','regression','fail'])
def test_crash_resume_exact_weights_optimizer_rng_and_counts(tmp_path,monkeypatch,boundary,outcome):
    c,source,make=build(tmp_path,monkeypatch,outcome)
    whole=make('whole').run('oracle_layout',parent=source)
    assert whole['status']=={'pass':'passed','regression':'rolled_back','fail':'budget_exhausted'}[outcome],whole
    paths=[]
    original=alt.AlternatingEngine.journal
    def journal(self):
        p=original(self)
        if boundary=='initial_journal' and self.e.output.name=='part':paths.append(p);raise Crash()
        return p
    monkeypatch.setattr(alt.AlternatingEngine,'journal',journal)
    def crash(event,**r):
        if (boundary=='head_candidate' and event=='head_candidate' or
            boundary=='feature_update' and event=='update' or
            boundary=='transition' and event=='phase_transition' and r['next_phase']=='feature'):
            if event=='head_candidate':paths.append(r['journal'])
            else:paths.append(sorted((tmp_path/'part').glob('checkpoint-*.pt'))[-1])
            raise Crash()
    with pytest.raises(Crash):make('part',observer=crash).run('oracle_layout',parent=source)
    resumed=make('resumed').run('oracle_layout',resume=paths[0])
    assert (resumed['status'],resumed['updates'])==(whole['status'],whole['updates'])
    a=load_integrated(whole['checkpoint'],c,'fixture','test');b=load_integrated(resumed['checkpoint'],c,'fixture','test')
    for key in ('model','optimizer','rng','cursor','metrics'):equal(a[key],b[key])
    for key in ('solver_updates','closure_evaluations','pending_feature','pending_head','last_trial','calibrations'):
        equal(a['integrated'].get(key),b['integrated'].get(key))
    assert whole['updates'] <= 10
    with pytest.raises(ValueError,match='terminal'):make('again').run('oracle_layout',resume=resumed['checkpoint'])


def test_deadline_rolls_back_full_state_and_never_passes_partial_calibration(tmp_path,monkeypatch):
    c,source,make=build(tmp_path,monkeypatch,'pass');now=[0.]
    def expire(event,**r):
        if event=='update':now[0]=120.
    end=make('expired',observer=expire,clock=lambda:now[0]).run('oracle_layout',parent=source)
    assert end['status']=='rolled_back' and end['updates']==1,end
    s=load_integrated(end['checkpoint'],c,'fixture','test')
    equal(s['optimizer'],s['integrated']['anchor_optimizer']);equal(s['rng'],s['integrated']['anchor_rng'])
    assert s['integrated']['rolled_back_to_update']==0


def test_time_and_source_config_locked():
    c=read_config('configs/recovery_v4_alternating_review.json');alt.validate_config(c)
    with pytest.raises(PermissionError):authorize(c,{},'oracle_layout','data','commit')
    for field,value in [('max_feature_updates',11),('max_seconds',121),('reference_rms_mm',24.)]:
        bad=deepcopy(c);bad['alternating_pose'][field]=value
        with pytest.raises(ValueError):alt.validate_config(bad)
    with pytest.raises(PermissionError):alt.load_source('missing.pt',c,'data',{})


def test_actual_pose_feature_gradient_numerical_and_forward_path():
    from knolling_model.arrangement.contracts import oracle_objects
    torch.set_num_threads(2);torch.manual_seed(32);m=Fixture();b=batch_for(m)
    configure_stage(m,'oracle_layout');opt=alt.FeatureAdam(m,1e-4,1e-4)
    terms,out=stage_loss(m,b.inputs,b.supervision);loss=sum(terms.values())
    parameter=m.layout.pose.blocks[0].film.weight
    gradient,=torch.autograd.grad(loss,parameter)
    direction=gradient/gradient.norm();analytic=float((gradient*direction).sum())
    original=parameter.detach().clone();values=[];eps=1e-3
    for sign in (-1,1):
        with torch.no_grad():parameter.copy_(original+sign*eps*direction)
        values.append(float(sum(stage_loss(m,b.inputs,b.supervision)[0].values()).detach()))
    with torch.no_grad():parameter.copy_(original)
    assert analytic==pytest.approx((values[1]-values[0])/(2*eps),rel=.01,abs=.01)
    terms,out=stage_loss(m,b.inputs,b.supervision)
    pred=m.layout(oracle_objects(b.supervision),b.inputs['input_ids'],b.inputs['attention_mask'])
    for key in ('xy','angle_logits','angle_residual'):
        torch.testing.assert_close(out['layout'][key] if 'layout' in out else out[key],pred[key],atol=0,rtol=0)


def test_measurements_use_actual_forward_pair_centering_and_no_target_leakage():
    torch.set_num_threads(2);torch.manual_seed(14);m=Fixture();b=batch_for(m)
    # Equal identities/count in the synthetic H/V pair.
    b.supervision[1]={k:v[:2] if isinstance(v,torch.Tensor) and v.ndim else v for k,v in b.supervision[1].items()}
    b.metadata[1]['object_ids']=['0','1']
    for meta in b.metadata:meta['record']['supervision_v3']={'copy_required':{'0':True,'1':False}}
    pairs=[dict(factor='axis_alignment',sample_ids=['0','1'])]
    result=alt.trial_measurements(m,[b],['0','1'],'fixture',pairs)
    assert result['hv_pairs']==1 and len(result['keep'])==2
    data=collect_readout(m,[b],expected_ids=['0','1'],fingerprint='fixture')
    shifted=deepcopy(b)
    for truth in shifted.supervision:truth['target_xy']+=.01
    other=collect_readout(m,[shifted],expected_ids=['0','1'],fingerprint='fixture')
    equal(data.features,other.features)
    # Pair relative difference is invariant to a common shift in labels.
    changed=alt.trial_measurements(m,[shifted],['0','1'],'fixture',pairs)
    assert changed['hv_relative_error_mm']==pytest.approx(result['hv_relative_error_mm'],abs=1e-4)


def test_rollback_restores_nonempty_optimizer_and_rng(tmp_path,monkeypatch):
    c,source,make=build(tmp_path,monkeypatch)
    monkeypatch.setattr(alt.AlternatingEngine,'diagnostics',lambda self:diagnostic(angle=90. if self.cursor.update>=2 else 89.))
    end=make('late-regression').run('oracle_layout',parent=source)
    assert end['status']=='rolled_back' and end['updates']==2
    s=load_integrated(end['checkpoint'],c,'fixture','test')
    equal(s['optimizer'],s['integrated']['anchor_optimizer']);equal(s['rng'],s['integrated']['anchor_rng'])
    assert s['integrated']['rolled_back_to_update']==1
    assert {float(v['step']) for v in s['optimizer']['optimizer']['state'].values()}=={1.}


@pytest.mark.parametrize('outcome',['pass','regression'])
def test_pending_final_calibration_resume_decides_before_any_new_update(tmp_path,monkeypatch,outcome):
    c,source,make=build(tmp_path,monkeypatch,outcome);paths=[]
    def crash(event,**r):
        if event=='head_candidate' and r['phase']=='head_angle' and r['update']==1:
            paths.append(r['journal']);raise Crash()
    with pytest.raises(Crash):make('part-final',observer=crash).run('oracle_layout',parent=source)
    state=load_integrated(paths[0],c,'fixture','test')
    assert state['integrated']['pending_head']['phase_done']
    events=[]
    end=make('resume-final',observer=lambda event,**r:events.append(event)).run('oracle_layout',resume=paths[0])
    assert end['status']=={'pass':'passed','regression':'rolled_back'}[outcome]
    assert 'update' not in events and 'head_candidate' not in events


def test_corrupt_cache_cannot_continue(tmp_path,monkeypatch):
    c,source,make=build(tmp_path,monkeypatch);paths=[]
    def crash(event,**r):
        if event=='head_candidate':paths.append(r['journal']);raise Crash()
    with pytest.raises(Crash):make('corrupt-part',observer=crash).run('oracle_layout',parent=source)
    s=load_integrated(paths[0],c,'fixture','test')
    from pathlib import Path
    Path(s['integrated']['cache_path']).write_bytes(b'corrupted test cache')
    end=make('corrupt-resume').run('oracle_layout',resume=paths[0])
    assert end['status']=='rolled_back' and 'cache artifact hash' in end['error']


def test_shared_gradient_full_scene_mean_independent_of_chunks():
    # The feature optimizer sees a scene mean, not a mean of unequal chunks.
    torch.set_num_threads(2);torch.manual_seed(36);m=Fixture();b=batch_for(m)
    configure_stage(m,'oracle_layout');opt=alt.FeatureAdam(m,1e-4,1e-4)
    sum(stage_loss(m,b.inputs,b.supervision)[0].values()).backward()
    expected=[p.grad.clone() for p in opt.parameters];opt.optimizer.zero_grad()
    for i in range(2):
        inputs={k:v[i:i+1] for k,v in b.inputs.items()}
        (sum(stage_loss(m,inputs,b.supervision[i:i+1])[0].values())*.5).backward()
    for p,g in zip(opt.parameters,expected):torch.testing.assert_close(p.grad,g,rtol=1e-4,atol=2e-5)
