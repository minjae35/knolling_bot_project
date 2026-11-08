from copy import deepcopy
from dataclasses import fields, replace
import os

import pytest
import torch

from knolling_model.recovery.goal_joint_layout import GoalJointLayout, ARCHITECTURE
from knolling_model.recovery.goal_encoder import goal_loss
from knolling_model.recovery.goal_labels import GoalDataset, goal_labels, VOCAB
from knolling_model.recovery.joint_layout import joint_pose_loss
from knolling_model.recovery.contracts import Objects
from test_recovery_joint_v3 import fixture as old_fixture
from test_recovery_dataset_v3 import suite_v3


def fixture(n=4, batch=1):
    _, objects, text, mask = old_fixture(n, batch)
    return GoalJointLayout(dimension=32), objects, text, mask


@pytest.mark.parametrize('n', range(1, 9))
def test_equivariance_and_id_independence(n):
    model, obj, text, mask = fixture(n)
    with torch.no_grad():
        out = model(obj, text, mask)
        for _ in range(12):
            p = torch.randperm(n); inv = p.argsort()
            other = Objects(**{f.name:getattr(obj,f.name)[:,p] for f in fields(Objects)})
            pred = model(other,text,mask)
            for key in ('xy','angle_logits','angle_residual'):
                torch.testing.assert_close(out[key], pred[key][:,inv],atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(out['goal']['selected'],pred['goal']['selected'][:,inv])
            torch.testing.assert_close(out['goal']['vector'],pred['goal']['vector'],atol=0,rtol=0)
        other = model(replace(obj,query_index=obj.query_index+100),text,mask)
        torch.testing.assert_close(out['xy'],other['xy'],atol=0,rtol=0)


def test_padding_empty_and_no_hidden_text_bypass():
    model,obj,text,mask=fixture(3,2)
    a=model(obj,text,mask)
    padded={}
    for f in fields(Objects):
        v=getattr(obj,f.name)
        extra=torch.full((2,2,*v.shape[2:]),False if f.name=='valid' else -1 if f.name=='query_index' else 999.,dtype=v.dtype)
        padded[f.name]=torch.cat((v,extra),1)
    b=model(Objects(**padded),torch.cat((text,torch.full((2,2,384),999.)),1),torch.cat((mask,torch.zeros(2,2,dtype=torch.bool)),1))
    torch.testing.assert_close(a['xy'],b['xy'][:,:3],atol=1e-6,rtol=1e-5)
    assert b['xy'][:,3:].count_nonzero()==0 and b['goal']['selected'][:,3:].count_nonzero()==0
    empty=replace(obj,valid=torch.zeros_like(obj.valid),query_index=torch.full_like(obj.query_index,-1))
    e=model(empty,text,mask)
    assert torch.isfinite(e['xy']).all() and e['xy'].count_nonzero()==0
    manual=model.pose(obj,a['goal']['vector'],a['goal']['selected'])
    torch.testing.assert_close(manual['xy'],a['xy'],atol=0,rtol=0)
    assert len(model.pose.blocks)==2 and model.pose.blocks[0] is not model.pose.blocks[1]
    with pytest.raises(TypeError):model(obj,text,mask,goal_labels={})


def test_absolute_output_and_final_only_pose_loss_gradient():
    model,obj,text,mask=fixture(3,2)
    model.goal_encoder.requires_grad_(False)
    obj.xy.requires_grad_();obj.size.requires_grad_();text.requires_grad_()
    out=model(obj,text,mask)
    out['goal']['selected'].retain_grad();out['goal']['vector'].retain_grad()
    terms=joint_pose_loss(out,dict(xy=obj.xy.detach(),angle=obj.angle.detach()),model.orientation)
    assert set(terms)=={'position_huber','angle_ce','angle_residual_huber'}
    sum(terms.values()).backward()
    for t in (obj.xy,obj.size,text,out['goal']['selected'],out['goal']['vector']):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum()>0
    assert all(p.grad is None for p in model.goal_encoder.parameters())
    for block in model.pose.blocks:
        assert block.film.weight.grad.abs().sum()>0
    with torch.no_grad():
        model.pose.output[-1].weight.zero_();model.pose.output[-1].bias.zero_()
        model.pose.output[-1].bias[:2].fill_(2)
    pred=model(obj,text,mask)
    torch.testing.assert_close(pred['xy'],torch.tensor([.9,.6]).expand_as(pred['xy']))


def test_goal_labels_source_only_keep_vs_noop_and_strict_vocab(suite_v3):
    ds=GoalDataset(suite_v3)
    assert len(ds)==76 and len(ds.pairs)==35
    for i,r in enumerate(ds.records):
        sample=ds[i];ids=sample['metadata']['object_ids'];truth=sample['supervision']
        corrupt=deepcopy(r);corrupt.pop('target');corrupt.pop('supervision_v3');corrupt['mode']='anything'
        for k,v in goal_labels(corrupt,ids).items():torch.testing.assert_close(v,truth[k])
        assert bool(truth['goal_keep'])==r['instruction']['keep_all']
        if r['instruction']['keep_all']:
            assert not truth['goal_selected'].any()
            assert all(truth['goal_'+k]==-100 for k in VOCAB)
        elif r['mode']=='no_op':
            assert truth['goal_selected'].any() and truth['goal_axis']>=0
    r=deepcopy(next(r for r in ds.records if not r['instruction']['keep_all']))
    r['instruction']['gap_m']=.021
    with pytest.raises(ValueError,match='unsupported'):goal_labels(r,sorted(o['object_id'] for o in r['source']['objects']))


def test_all_keep_goal_loss_finite_and_all_slots_have_gradients():
    model,obj,text,mask=fixture(3,2)
    out=model.goal_encoder(obj,text,mask)
    truths=[dict(goal_keep=torch.tensor(1.),goal_selected=torch.zeros(3),**{'goal_'+k:torch.tensor(-100) for k in VOCAB}) for _ in range(2)]
    terms=goal_loss(out,truths,[torch.arange(3)]*2)
    assert terms['attributes_ce']==0 and all(torch.isfinite(v) for v in terms.values())
    sum(terms.values()).backward()
    assert model.goal_encoder.queries.grad.abs().sum(-1).gt(0).all()


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(os.environ.get('RUN_RECOVERY_CUDA_TESTS')!='1',reason='explicit CUDA'))])
def test_stage_gradients_gt_isolation_and_inference_identity(device):
    from test_recovery_stages import TextFixture,inputs,selected_fixture
    from test_recovery_perception import tiny_detector,truth
    from knolling_model.recovery.model import RecoveryModel
    from knolling_model.recovery.stages import configure_stage,stage_loss,parameter_groups
    m=RecoveryModel(TextFixture(),tiny_detector(),architecture=ARCHITECTURE,dimension=32,layers=2,num_denoising=4).to(device)
    x={k:v.to(device) for k,v in inputs().items()}
    before={k:v.detach().clone() for k,v in m.state_dict().items()}
    for stage in ('grounding','micro_pose','oracle_layout','perception','predicted_layout','joint'):
        configure_stage(m,stage)
        t=selected_fixture(m,x) if stage in ('predicted_layout','joint') else {k:v.to(device) for k,v in truth().items()}
        t.update(goal_keep=torch.tensor(0.,device=device),goal_selected=torch.ones(len(t['source_xy']),device=device),
                 **{'goal_'+k:torch.tensor(0,device=device) for k in VOCAB})
        terms,out=stage_loss(m,x,[t],replay=(x,[t]) if stage=='joint' else None)
        sum(terms.values()).backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        assert all(p.grad is None for p in m.text_encoder.parameters())
        groups=parameter_groups(m);params=[p for g in groups for p in g['params']]
        assert len({id(p) for p in params})==len(params)
        goalgrads=[p.grad for p in m.layout.goal_encoder.parameters() if p.grad is not None]
        assert bool(goalgrads)==(stage=='grounding')
        if stage not in ('grounding','perception'):
            assert len([k for k in terms if k.startswith('layout/')])==3
            corrupt={k:v.clone() for k,v in t.items()};corrupt['target_xy']+=.2
            for k in list(corrupt):
                if k.startswith('goal_'):corrupt[k].zero_()
            _,other=stage_loss(m,x,[corrupt],replay=(x,[t]) if stage=='joint' else None)
            torch.testing.assert_close(out['layout']['xy'],other['layout']['xy'],rtol=0,atol=0)
        if stage=='predicted_layout':
            m.eval();inference=m(**x)
            torch.testing.assert_close(out['layout']['xy'],inference['layout']['xy'],atol=1e-6,rtol=1e-5)
        m.zero_grad(set_to_none=True)
    configure_stage(m,'joint');x['images'].requires_grad_()
    output=m(**x)
    (output['layout']['xy'].square().sum()+output['layout']['angle_logits'].square().mean()).backward()
    assert x['images'].grad.abs().sum()>0
    assert sum(p.grad.abs().sum() for p in m.perception.physical.parameters() if p.grad is not None)>0
    for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
