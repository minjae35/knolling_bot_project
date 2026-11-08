from copy import deepcopy
from types import SimpleNamespace
import torch
from knolling_model.recovery.vertical_gap import loss, GAP_SCALE
from knolling_model.recovery.relative_task import configure, restore, SCHEMA
from knolling_model.recovery.storage import CheckpointStore, load_artifact
from test_recovery_relative_task import setup
from test_recovery_integrated import equal


def test_actual_edges_gradient_and_translation_invariance():
    xy=torch.tensor([[[.1,.15],[.1,.08]]],requires_grad=True)
    angle=torch.tensor([[0.,0.]],requires_grad=True)
    obj=SimpleNamespace(size=torch.tensor([[[.04,.02],[.04,.02]]]))
    out=dict(xy=xy,angle=angle)
    labels=[(0,[([0],[1],.02)])]
    value=loss(out,obj,labels)
    expected=torch.nn.functional.smooth_l1_loss(torch.tensor(.05/GAP_SCALE),torch.tensor(.02/GAP_SCALE))
    torch.testing.assert_close(value,expected)
    value.backward()
    assert angle.grad is None and not xy.grad[...,0].count_nonzero()
    assert xy.grad[0,0,1]>0 and xy.grad[0,1,1]<0
    shifted=dict(out,xy=xy.detach()+torch.tensor([.15,-.04]))
    torch.testing.assert_close(loss(shifted,obj,labels),value)
    eps=1e-5;hi=xy.detach().clone();lo=hi.clone();hi[0,0,1]+=eps;lo[0,0,1]-=eps
    numeric=(loss(dict(out,xy=hi),obj,labels)-loss(dict(out,xy=lo),obj,labels))/(2*eps)
    torch.testing.assert_close(numeric,xy.grad[0,0,1],atol=.1,rtol=.002)


def test_extrema_and_rotated_footprints():
    obj=SimpleNamespace(size=torch.tensor([[[.06,.02],[.04,.02],[.04,.02]]]))
    # Rotated upper row object is 60 mm tall; lower row top is .09.
    out=dict(xy=torch.tensor([[[.1,.15],[.2,.15],[.1,.08]]]),angle=torch.tensor([[torch.pi/2,0.,0.]]))
    labels=[(0,[([0,1],[2],.03)])]
    assert float(loss(out,obj,labels))<1e-9


def test_frozen_orientation_survives_xy_update_and_checkpoint(tmp_path):
    m,_,targets,forward=setup();groups=configure(m,orientation=False)
    frozen=deepcopy(m.layout.pose.orientation_pose.state_dict())
    opt=torch.optim.AdamW(groups['xy'],lr=1e-4)
    out=forward();orientation=out['angle'].detach().clone()
    obj=SimpleNamespace(size=torch.full((*out['angle'].shape,2),.02))
    loss(out,obj,[(0,[([0],[1],.02)])]).backward()
    assert all(p.grad is None for p in m.layout.pose.orientation_pose.parameters())
    opt.step();equal(m.layout.pose.orientation_pose.state_dict(),frozen)
    equal(forward()['angle'].detach(),orientation)
    payload=dict(schema=SCHEMA,stage='oracle_layout',config={'freeze_orientation':True},model=m.state_dict(),optimizers={'xy':opt.state_dict()})
    path=tmp_path/'gap.pt';CheckpointStore(tmp_path/'shared').save_checkpoint(path,payload)
    saved=load_artifact(path);restore(m,saved,{'xy':opt})
    assert not any(p.requires_grad for p in m.layout.pose.orientation_pose.parameters())
    equal(m.layout.pose.orientation_pose.state_dict(),frozen)


def test_no_adjacent_rows_is_zero_differentiable():
    xy=torch.zeros(1,1,2,requires_grad=True)
    value=loss(dict(xy=xy,angle=torch.zeros(1,1)),SimpleNamespace(size=torch.ones(1,1,2)),[])
    value.backward();assert value==0 and not xy.grad.count_nonzero()
