from copy import deepcopy
import pytest
import torch
from knolling_model.arrangement.split_pose import SplitJointPose, load_model_state, SCHEMA, PREFIX
from knolling_model.arrangement.alternating import FeatureAdam, alternating_evaluation, trial_measurements
from knolling_model.arrangement.storage import CheckpointStore,load_artifact,tensor_digest
from knolling_model.arrangement.driver import rng_state,restore_rng
from knolling_model.arrangement.stages import configure_stage,stage_loss
from knolling_model.arrangement.contracts import oracle_objects
from test_recovery_readout import Fixture,batch_for
from test_recovery_integrated import equal


def setup():
    torch.set_num_threads(2);torch.manual_seed(35)
    original=Fixture();m=deepcopy(original);m.layout.pose=SplitJointPose(m.layout.pose)
    with torch.no_grad():m.layout.pose.input.weight.add_(.02)
    b=batch_for(m)
    return original,m,b


def test_shared_goal_real_forward_branch_identity_and_padding():
    original,m,b=setup();obj=oracle_objects(b.supervision);calls=[]
    hook=m.layout.goal_encoder.register_forward_hook(lambda *a:calls.append(1))
    m.eval();original.eval()
    with alternating_evaluation(m):out=m.layout(obj,b.inputs['input_ids'],b.inputs['attention_mask'])
    hook.remove()
    with alternating_evaluation(original):ref=original.layout(obj,b.inputs['input_ids'],b.inputs['attention_mask'])
    assert len(calls)==1
    for k in ('angle','angle_logits','angle_residual'):equal(out[k],ref[k])
    goal=out['goal']
    with alternating_evaluation(m):xy=super(SplitJointPose,m.layout.pose).forward(obj,goal['vector'],goal['selected'])
    equal(xy['xy'],out['xy'])
    assert torch.isfinite(out['xy']).all()
    assert not out['xy'][~obj.valid].count_nonzero()
    assert not out['angle_logits'][~obj.valid].count_nonzero()
    assert not any(p.requires_grad for p in m.layout.pose.orientation_pose.parameters())


def test_cpu_optimizer_rng_resume_and_frozen_branch(tmp_path):
    original,m,b=setup();configure_stage(m,'oracle_layout');opt=FeatureAdam(m,1e-4,1e-4)
    assert not any('orientation_pose' in n for n in opt.shared)
    def step(model,adam):
        adam.activate();sum(stage_loss(model,b.inputs,b.supervision)[0].values()).backward()
        assert all(p.grad is None for p in model.layout.pose.orientation_pose.parameters())
        assert all(p.grad is None for p in model.layout.goal_encoder.parameters())
        torch.nn.utils.clip_grad_norm_(adam.parameters,1);adam.optimizer.step()
    step(m,opt)
    spec=dict(schema=SCHEMA,source_sha256='a'*64,xy_sha256='b'*64,
      orientation_state_sha256=tensor_digest(m.layout.pose.orientation_pose.state_dict()))
    c=dict(split_pose=spec,alternating_pose={'diagnostic':'xy_feature_window_v1'},model={'architecture':'recovery_v4_goal_joint'})
    payload=dict(stage='oracle_layout',config=c,model=deepcopy(m.state_dict()),optimizer=deepcopy(opt.state_dict()),rng=rng_state())
    p=tmp_path/'split.pt';CheckpointStore(tmp_path/'shared').save_checkpoint(p,payload)
    step(m,opt);expected=deepcopy(m.state_dict());optim=deepcopy(opt.state_dict());random=torch.rand(7)
    saved=load_artifact(p);n=Fixture();load_model_state(n,saved);configure_stage(n,'oracle_layout');other=FeatureAdam(n,1e-4,1e-4)
    other.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);step(n,other)
    equal(n.state_dict(),expected);equal(other.state_dict(),optim);equal(torch.rand(7),random)
    for k,v in original.layout.pose.state_dict().items():equal(v,n.layout.pose.orientation_pose.state_dict()[k])
    saved['model'][PREFIX+'input.weight'][0,0]+=1
    with pytest.raises(ValueError,match='integrity'):load_model_state(Fixture(),saved)


def test_measurement_uses_actual_orientation_not_xy_feature_logits():
    original,m,b=setup();m.eval()
    with torch.no_grad():
        m.layout.pose.output[1].weight[2:20].zero_();m.layout.pose.output[1].bias[2:20].fill_(-10);m.layout.pose.output[1].bias[2]=10
    b.supervision[1]={k:v[:2] if isinstance(v,torch.Tensor) and v.ndim else v for k,v in b.supervision[1].items()}
    b.metadata[1]['object_ids']=['0','1']
    for x in b.metadata:x['record']['supervision_v3']={'copy_required':{'0':True,'1':False}}
    with alternating_evaluation(m):
        out=m.layout(oracle_objects(b.supervision),b.inputs['input_ids'],b.inputs['attention_mask'])
    expected=0
    for i,t in enumerate(b.supervision):
        bins,_=m.layout.orientation.encode(t['target_angle']);expected+=int((out['angle_logits'][i,:len(bins)].argmax(-1)==bins).sum())
    diag=trial_measurements(m,[b],['0','1'],'fixture',[dict(factor='axis_alignment',sample_ids=['0','1'])])
    assert diag['bin_correct']==expected


def test_target_labels_never_enter_split_forward():
    original,m,b=setup()
    def forward():
        with alternating_evaluation(m):
            return m.layout(oracle_objects(b.supervision),b.inputs['input_ids'],b.inputs['attention_mask'])
    before=forward()
    for t in b.supervision:t['target_xy'].add_(100);t['target_angle'].add_(1)
    after=forward()
    for k in ('xy','angle','angle_logits','angle_residual'):equal(before[k],after[k])


def test_split_permutation_equivariance():
    from dataclasses import fields,replace
    original,m,b=setup();obj=oracle_objects(b.supervision);order=torch.tensor([1,0,2]);inverse=order.argsort()
    shuffled=replace(obj,**{f.name:getattr(obj,f.name)[:,order] for f in fields(obj)})
    with alternating_evaluation(m):
        a=m.layout(obj,b.inputs['input_ids'],b.inputs['attention_mask'])
        v=m.layout(shuffled,b.inputs['input_ids'],b.inputs['attention_mask'])
    for k in ('xy','angle','angle_logits','angle_residual'):
        torch.testing.assert_close(a[k],v[k][:,inverse],rtol=1e-5,atol=1e-6)
