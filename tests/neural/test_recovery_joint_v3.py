from dataclasses import fields, replace
from itertools import permutations

import pytest
import torch

from knolling_model.coordinates import angle_delta
from knolling_model.arrangement.contracts import Objects
from knolling_model.arrangement.joint_layout import JointPoseDecoder, joint_pose_loss


def fixture(n=4, batch=1):
    torch.manual_seed(71)
    model=JointPoseDecoder(dimension=32,layers=1)
    obj=Objects(torch.rand(batch,n,2)*.3,torch.full((batch,n,2),.04),torch.rand(batch,n),
        torch.softmax(torch.randn(batch,n,3),-1),torch.softmax(torch.randn(batch,n,3),-1),
        torch.ones(batch,n,dtype=torch.bool),torch.arange(n).expand(batch,-1))
    text=torch.randn(batch,5,384)
    mask=torch.ones(batch,5,dtype=torch.bool)
    return model,obj,text,mask


@pytest.mark.parametrize('n',range(1,9))
def test_equivariance_all_small_permutations_and_twenty_large(n):
    model,obj,text,mask=fixture(n)
    model.eval()
    with torch.no_grad():
        a=model(obj,text,mask)
        orders=list(permutations(range(n))) if n<=4 else [torch.randperm(n).tolist() for _ in range(20)]
        for order in orders:
            p=torch.tensor(order); inv=p.argsort()
            other=Objects(**{f.name:getattr(obj,f.name)[:,p] for f in fields(Objects)})
            b=model(other,text,mask)
            torch.testing.assert_close(a['xy'],b['xy'][:,inv],atol=1e-5,rtol=0)
            assert angle_delta(a['angle'],b['angle'][:,inv]).abs().max()<1e-4
            torch.testing.assert_close(a['angle_logits'],b['angle_logits'][:,inv],atol=1e-5,rtol=1e-5)
        # IDs are bookkeeping, never features.
        b=model(replace(obj,query_index=obj.query_index+1000),text,mask)
        torch.testing.assert_close(a['xy'],b['xy'],rtol=0,atol=0)


def test_padding_empty_detection_and_text_padding_are_inert():
    model,obj,text,mask=fixture(3,2)
    with torch.no_grad():
        a=model(obj,text,mask)
        padded={}
        for f in fields(Objects):
            v=getattr(obj,f.name)
            extra=torch.full((2,2,*v.shape[2:]),False if f.name=='valid' else -1 if f.name=='query_index' else 999.,dtype=v.dtype)
            padded[f.name]=torch.cat((v,extra),1)
        b=model(Objects(**padded),torch.cat((text,torch.full((2,3,384),999.)),1),torch.cat((mask,torch.zeros(2,3,dtype=torch.bool)),1))
        torch.testing.assert_close(a['xy'],b['xy'][:,:3],atol=1e-6,rtol=1e-5)
        empty=replace(obj,valid=torch.zeros_like(obj.valid),query_index=torch.full_like(obj.query_index,-1))
        out=model(empty,text,mask)
        assert torch.isfinite(out['xy']).all() and out['xy'].count_nonzero()==0


def test_exact_three_final_losses_keep_supervision_and_refinement_gradients():
    model,obj,text,mask=fixture(3,2)
    text.requires_grad_();obj.xy.requires_grad_();obj.size.requires_grad_()
    output=model(obj,text,mask)
    targets=dict(xy=obj.xy.detach(),angle=obj.angle.detach(),move=torch.zeros(2,3,dtype=torch.bool))
    terms=joint_pose_loss(output,targets,model.orientation)
    assert set(terms)=={'position_huber','angle_ce','angle_residual_huber'}
    assert terms['position_huber']>0  # Even a keep-only batch must train poses.
    states=output['states']
    assert len(states)==3
    for state in states:
        for value in state.values():value.retain_grad()
    sum(terms.values()).backward()
    for state in states:
        for value in state.values():
            assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    for value in (text,obj.xy,obj.size):assert value.grad.abs().sum()>0
    assert not any('move' in name or 'selection' in name or 'axis' in name for name,_ in model.named_parameters())
    original={k:v.detach().clone() for k,v in terms.items()}
    # Diagnostic intermediate outputs are not additional loss targets.
    changed=dict(output,states=())
    for key,value in joint_pose_loss(changed,targets,model.orientation).items():torch.testing.assert_close(value.detach(),original[key])
    with pytest.raises(TypeError):model(obj,text,mask,target_xy=targets['xy'])


def test_loss_matches_formula_and_scene_normalization():
    model,obj,text,mask=fixture(3,2)
    out=model(obj,text,mask)
    out['valid']=torch.tensor([[True,False,False],[True,True,True]])
    target=dict(xy=out['xy'].detach()+torch.tensor([[[.02,0]],[[.01,0]]]),angle=obj.angle)
    losses=joint_pose_loss(out,target,model.orientation)
    torch.testing.assert_close(losses['position_huber'],torch.tensor((1.5+.5)/2),atol=1e-5,rtol=0)
    target['move']=torch.zeros_like(out['valid'])
    torch.testing.assert_close(joint_pose_loss(out,target,model.orientation)['position_huber'],losses['position_huber'])


def test_stage_wiring_and_train_inference_equality():
    from test_recovery_stages import TextFixture, inputs, selected_fixture
    from test_recovery_perception import tiny_detector
    from knolling_model.arrangement.model import RecoveryModel
    from knolling_model.arrangement.stages import configure_stage,stage_loss
    model=RecoveryModel(TextFixture(),tiny_detector(),architecture='recovery_v3_joint',dimension=32,layers=1,num_denoising=4)
    batch=inputs();configure_stage(model,'predicted_layout');truth=selected_fixture(model,batch)
    terms,training=stage_loss(model,batch,[truth])
    model.eval();inference=model(**batch)['layout']
    torch.testing.assert_close(training['layout']['xy'],inference['xy'],atol=1e-6,rtol=1e-5)
    assert len(terms)==3
    sum(terms.values()).backward()
    assert all(p.grad is None for p in model.text_encoder.parameters())
    with pytest.raises(ValueError,match='prefix'):stage_loss(model,batch,[truth],teacher_forcing=True)
    with pytest.raises(TypeError):model(**batch,truths=[truth])
