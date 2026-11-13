from copy import deepcopy
from dataclasses import fields
import pytest
import torch
from test_phase1_single_row import Reference, example, device
from knolling_model.recovery.contracts import Objects
from knolling_model.recovery.storage import CheckpointStore, load_artifact, tensor_digest
from knolling_model.experiments.neural_sequential import SequentialPose, position_loss


def test_disjoint_gradients_and_two_dimensional_output(device):
    m=SequentialPose(Reference()).to(device).eval();obj,g,s=example(device)
    out=m(obj,g,s);target=out['xy'].detach()+torch.tensor([.004,-.006],device=device)
    position_loss(out,target).backward()
    assert all(p.grad is None for p in m.selector.parameters())
    assert all(p.grad is None for p in m.reference.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.decoder.features.parameters())
    assert (m.decoder.output[-1].weight.grad.abs().sum(-1)>0).all()
    m.zero_grad(set_to_none=True)
    order=torch.tensor([[2,0,1]],device=device)
    loss=m.selector_loss(obj,g,s,out['move_mask'],order);loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.selector.parameters())
    assert all(p.grad is None for p in m.decoder.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_permutation_padding_keep_and_identity(device):
    m=SequentialPose(Reference()).to(device).eval();obj,g,s=example(device,2)
    s[:,1]=0
    out=m(obj,g,s)
    assert sorted(out['predicted_order'][0,:2].tolist())==[0,2]
    assert (out['predicted_order'][0,2:]==-1).all()
    assert torch.equal(out['xy'][:,1],obj.xy[:,1])
    p=torch.tensor([2,0,1,3,4],device=device)
    perm=Objects(**{f.name:getattr(obj,f.name)[:,p] for f in fields(obj)})
    changed=m(perm,g,s[:,p])
    torch.testing.assert_close(changed['xy'],out['xy'][:,p],atol=1e-6,rtol=0)
    assert torch.equal(perm.query_index.gather(1,changed['predicted_order'].clamp_min(0))[:,:2],
                       obj.query_index.gather(1,out['predicted_order'].clamp_min(0))[:,:2])
    short=Objects(**{f.name:getattr(obj,f.name)[:,:3] for f in fields(obj)})
    torch.testing.assert_close(m(short,g,s[:,:3])['xy'],out['xy'][:,:3],atol=1e-6,rtol=0)
    assert out['xy'][:,3:].count_nonzero()==0
    kept=m(obj,g,torch.zeros_like(s))
    torch.testing.assert_close(kept['xy'],obj.xy.masked_fill(~obj.valid[...,None],0))


def test_teacher_prefix_is_diagnostic_and_labels_do_not_enter_forward(device):
    m=SequentialPose(Reference()).to(device).eval();obj,g,s=example(device)
    order=torch.tensor([[2,0,1]],device=device)
    a=m.diagnose(obj,g,s,order);b=m.diagnose(obj,g,s,order,obj.xy)
    assert not torch.equal(a['xy'],b['xy'])
    actual=m(obj,g,s)
    m.diagnose(obj,g,s,order,obj.xy+1.)
    assert torch.equal(actual['xy'],m(obj,g,s)['xy'])
    m.train()
    with pytest.raises(ValueError):m.diagnose(obj,g,s,order,obj.xy)
    with pytest.raises(TypeError):m(obj,g,s,teacher_xy=obj.xy)
    with pytest.raises(ValueError):m.selector_loss(obj,g,s,obj.valid,torch.tensor([[0,0,2]],device=device))


def test_eight_objects_and_empty_scene(device):
    m=SequentialPose(Reference()).to(device).eval();obj,g,s=example(device,5)
    obj.valid[:]=True;obj.query_index[:]=torch.arange(8,device=device);s[:]=1
    out=m(obj,g,s)
    assert sorted(out['predicted_order'][0].tolist())==list(range(8))
    obj.valid[:]=False;obj.query_index[:]=-1;s[:]=0
    out=m(obj,g,s)
    assert torch.isfinite(out['xy']).all() and out['xy'].count_nonzero()==0
    assert (out['predicted_order']==-1).all()


def test_checkpoint_optimizer_resume(device,tmp_path):
    m=SequentialPose(Reference()).to(device).eval();obj,g,s=example(device)
    order=torch.tensor([[2,0,1]],device=device)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    def step(model,optimizer):
        optimizer.zero_grad(set_to_none=True);out=model(obj,g,s)
        loss=position_loss(out,obj.xy)+model.selector_loss(obj,g,s,out['move_mask'],order)
        loss.backward();optimizer.step()
    step(m,opt);path=tmp_path/'checkpoint.pt'
    CheckpointStore(tmp_path/'shared').save_checkpoint(path,dict(stage='oracle_layout',model=m.state_dict(),optimizer=opt.state_dict()))
    raw=load_artifact(path);restored=deepcopy(m);restored.load_state_dict(raw['model'],strict=True)
    other=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=1e-4);other.load_state_dict(raw['optimizer'])
    assert torch.equal(m(obj,g,s)['xy'],restored(obj,g,s)['xy'])
    step(m,opt);step(restored,other)
    assert tensor_digest(m.state_dict())==tensor_digest(restored.state_dict())
    for a,b in zip(opt.state.values(),other.state.values()):
        for k in a:assert torch.equal(a[k],b[k])
