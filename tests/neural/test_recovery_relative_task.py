from copy import deepcopy
import torch
import pytest
from knolling_model.recovery.relative_task import configure, losses, pair_indices, restore, SCHEMA
from knolling_model.recovery.contracts import oracle_objects
from knolling_model.recovery.stages import aligned_targets
from knolling_model.recovery.storage import CheckpointStore,load_artifact
from knolling_model.recovery.driver import rng_state,restore_rng
from test_recovery_readout import Fixture,batch_for
from test_recovery_integrated import equal


def setup():
    torch.set_num_threads(2);torch.manual_seed(1203)
    m=Fixture();groups=configure(m);b=batch_for(m);o=oracle_objects(b.supervision)
    targets=aligned_targets(b.supervision,[torch.arange(len(t['source_xy'])) for t in b.supervision],o)
    def forward():return m.layout(o,b.inputs['input_ids'],b.inputs['attention_mask'])
    return m,groups,targets,forward


def test_gradient_paths_are_separate_and_goal_frozen():
    m,groups,t,f=setup();out=f();terms=losses(out,t,m.layout.orientation,[])
    xy=terms['position_huber']+terms['relative_position_huber']
    xy.backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in groups['xy'])
    assert all(p.grad is None or not p.grad.count_nonzero() for p in groups['orientation'])
    m.zero_grad(set_to_none=True)
    (terms['angle_ce']+terms['angle_residual_huber']).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in groups['orientation'])
    assert all(p.grad is None or not p.grad.count_nonzero() for p in groups['xy'])
    assert all(p.grad is None for p in m.layout.goal_encoder.parameters())


def test_relative_loss_translation_invariance_and_contrast_signal():
    m,_,t,f=setup();out=f();out['xy']=t['xy'].clone().requires_grad_()
    # Both scenes' first two object identities correspond, third is padding in A.
    pair=[torch.tensor([[0,0,1,1,0,1]])]
    base=losses(out,t,m.layout.orientation,pair)
    assert base['relative_position_huber']==0 and base['counterfactual_relative_huber']==0
    moved=dict(out,xy=out['xy']+torch.tensor([.1,-.2]))
    translated=losses(moved,t,m.layout.orientation,pair)
    assert translated['relative_position_huber']<1e-10
    assert translated['counterfactual_relative_huber']<1e-10
    wrong=out['xy'].detach().clone();wrong[0,0,0]+=.005;wrong.requires_grad_()
    value=losses(dict(out,xy=wrong),t,m.layout.orientation,pair)['counterfactual_relative_huber']
    g=torch.autograd.grad(value,wrong)[0]
    assert value>0 and g[0,0,0]>0 and g[1,0,0]<0
    eps=1e-5;a=wrong.detach().clone();a[0,0,0]+=eps;z=wrong.detach().clone();z[0,0,0]-=eps
    numeric=(losses(dict(out,xy=a),t,m.layout.orientation,pair)['counterfactual_relative_huber']-losses(dict(out,xy=z),t,m.layout.orientation,pair)['counterfactual_relative_huber'])/(2*eps)
    torch.testing.assert_close(g[0,0,0],numeric,rtol=.002,atol=.002)


def test_counterfactual_joins_object_ids_not_array_positions():
    meta=[dict(sample_id='a',object_ids=['x','y'],record={'scene_family_id':'s'}),dict(sample_id='b',object_ids=['y','x'],record={'scene_family_id':'s'})]
    idx=pair_indices(meta,[{'sample_ids':['a','b']}],'cpu')
    assert idx[0].tolist()==[[0,0,1,1,1,0]]
    meta[1]['record']['scene_family_id']='different'
    with pytest.raises(ValueError,match='family'):pair_indices(meta,[{'sample_ids':['a','b']}],'cpu')


def test_cpu_checkpoint_optimizer_rng_next_step_reproducible(tmp_path):
    m,g,t,f=setup();opts={k:torch.optim.AdamW(v,lr=1e-4) for k,v in g.items()}
    def step():
        for o in opts.values():o.zero_grad(set_to_none=True)
        sum(losses(f(),t,m.layout.orientation,[]).values()).backward()
        for o in opts.values():o.step()
    step();payload=dict(schema=SCHEMA,stage='oracle_layout',model=deepcopy(m.state_dict()),optimizers={k:deepcopy(v.state_dict()) for k,v in opts.items()},rng=rng_state())
    p=tmp_path/'c.pt';CheckpointStore(tmp_path/'shared').save_checkpoint(p,payload)
    step();expected=deepcopy(m.state_dict());states={k:deepcopy(v.state_dict()) for k,v in opts.items()};r=torch.rand(5)
    saved=load_artifact(p);restore(m,saved,opts);restore_rng(saved['rng']);step()
    equal(m.state_dict(),expected);equal({k:v.state_dict() for k,v in opts.items()},states);equal(torch.rand(5),r)


@pytest.mark.parametrize('xy,angle',[(True,False),(False,True)])
def test_each_branch_can_be_trained_independently(xy,angle):
    m,_,_,_=setup();groups=configure(m,xy=xy,orientation=angle)
    assert bool(groups['xy'])==xy and bool(groups['orientation'])==angle
