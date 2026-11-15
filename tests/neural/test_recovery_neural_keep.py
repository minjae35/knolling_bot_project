from copy import deepcopy
from dataclasses import fields
import torch
from knolling_model.arrangement.neural_keep import attach, loss, labels, adopt, POLICY
from knolling_model.arrangement.relative_task import configure, restore, SCHEMA
from knolling_model.arrangement.contracts import Objects, oracle_objects
from knolling_model.arrangement.storage import CheckpointStore, load_artifact
from knolling_model.arrangement.driver import rng_state, restore_rng
from test_recovery_relative_task import setup
from test_recovery_readout import batch_for, Fixture
from test_recovery_integrated import equal


def prepared():
    m,_,_,forward=setup()
    with torch.no_grad():original=forward()
    attach(m);groups=configure(m,orientation=False)
    return m,groups,forward,original


def test_neutral_initialization_and_predicted_source_copy():
    m,g,f,old=prepared();out=f()
    for key in ('xy','angle','angle_logits','angle_residual'):
        torch.testing.assert_close(out[key],old[key],atol=1e-6,rtol=1e-6)
    assert out['move_mask'][out['valid']].all()
    with torch.no_grad():m.layout.pose.move_keep_head[-1].bias.fill_(-1)
    out=f();objects=oracle_objects(batch_for(m).supervision)
    equal(out['xy'],objects.xy.masked_fill(~objects.valid[...,None],0))
    equal(out['angle'],objects.angle.masked_fill(~objects.valid,0))
    assert not out['move_mask'].any()
    assert g['xy'] and not g['orientation']


def test_head_gradients_only_and_full_checkpoint_resume(tmp_path):
    m,g,f,_=prepared();target=torch.tensor([[0.,1.,0.],[1.,0.,1.]])
    original=deepcopy(m.state_dict());opt=torch.optim.AdamW(g['xy'],lr=1e-4)
    def step():
        opt.zero_grad(set_to_none=True);loss(f(),target).backward()
        assert all(p.grad is None for name,p in m.named_parameters() if 'move_keep_head' not in name)
        opt.step()
    step();step()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in g['xy'])
    for name,value in m.state_dict().items():
        if 'move_keep_head' not in name:equal(value,original[name])
    payload=dict(schema=SCHEMA,stage='oracle_layout',config={'keep_router':POLICY,'freeze_orientation':True},
        model=deepcopy(m.state_dict()),optimizers={'xy':deepcopy(opt.state_dict())},rng=rng_state())
    path=tmp_path/'keep.pt';CheckpointStore(tmp_path/'shared').save_checkpoint(path,payload)
    saved=load_artifact(path)
    fresh=Fixture();restore(fresh,saved);equal(fresh.state_dict(),payload['model'])
    assert all(not p.requires_grad for n,p in fresh.named_parameters() if 'move_keep_head' not in n)
    step();expected=deepcopy(m.state_dict());state=deepcopy(opt.state_dict());r=torch.rand(5)
    restore(m,saved,{'xy':opt});restore_rng(saved['rng']);step()
    equal(m.state_dict(),expected);equal(opt.state_dict(),state);equal(torch.rand(5),r)


def test_permutation_padding_and_no_target_argument():
    m,_,_,_=prepared();pose=m.layout.pose
    objects=oracle_objects(batch_for(m).supervision);goal=torch.randn(2,13);selected=torch.rand(2,3)
    with torch.no_grad():pose.move_keep_head[-1].weight.normal_(std=.1)
    out=pose(objects,goal,selected)
    idx=torch.tensor([[1,0,2],[2,0,1]])
    def permute(t):return torch.stack([row[index] for row,index in zip(t,idx)])
    changed=Objects(**{field.name:permute(getattr(objects,field.name)) for field in fields(objects)})
    p=pose(changed,goal,permute(selected))
    for k in ('xy','angle','move_logit'):
        torch.testing.assert_close(p[k],permute(out[k]),atol=1e-6,rtol=1e-5)
    assert not out['move_mask'][~objects.valid].any()
    assert not out['xy'][~objects.valid].count_nonzero()
    # Finite padding values must not change any real object's routing/output.
    padded=deepcopy(objects);padded.xy[~objects.valid]=100;padded.angle[~objects.valid]=50
    again=pose(padded,goal,selected)
    for k in ('xy','angle','move_logit'):equal(out[k],again[k])


def test_keep_labels_are_not_selection_and_padding_has_no_loss():
    metadata=[dict(object_ids=['a','b'],record={'supervision_v3':{'copy_required':{'a':True,'b':False}},'selected':['a','b']}),
              dict(object_ids=['a'],record={'supervision_v3':{'copy_required':{'a':False}},'selected':['a']})]
    target=labels(metadata,'cpu');equal(target,torch.tensor([[0.,1.],[1.,0.]]))
    logits=torch.zeros(2,2,requires_grad=True);valid=torch.tensor([[1,1],[1,0]],dtype=torch.bool)
    value=loss(dict(valid=valid,move_logit=logits),target);value.backward()
    assert logits.grad[1,1]==0 and logits.grad[0,0]>0 and logits.grad[0,1]<0
    modified=target.clone();modified[~valid]=1
    equal(value,loss(dict(valid=valid,move_logit=logits),modified))


def test_adoption_requires_actual_task_gain_and_no_prior_success_regression():
    b=dict(integrated_evidence=dict(successful_scenes=['old'],successful_pairs=['p'],angle_errors_deg={'o':.01}),
           validity={'keep_violations':1},target_angle_p95_deg=.02)
    c=deepcopy(b);c['integrated_evidence']['successful_scenes']+=['new'];c['integrated_evidence']['successful_pairs']+=['q']
    assert adopt(c,b)
    c['integrated_evidence']['successful_scenes'].remove('old');assert not adopt(c,b)
    c['integrated_evidence']['successful_scenes'].append('old');c['target_angle_p95_deg']=.03
    assert not adopt(c,b)
