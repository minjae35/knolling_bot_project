"""Synthetic tensor fixtures only; actual cached Knolling models never step here."""
from copy import deepcopy
from types import SimpleNamespace
import os

import pytest
import torch
from torch import nn

from knolling_model.recovery.readout import (HeadFitter, PoseAdam, ReadoutData, SLICES,
                                           collect_readout, feature_hash)
from knolling_model.recovery.stages import stage_loss, configure_stage
from knolling_model.recovery.goal_joint_layout import ARCHITECTURE
from test_recovery_goal_v4 import fixture

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    os.environ.get('RUN_RECOVERY_CUDA_TESTS') != '1', reason='explicit CUDA'))]


class Fixture(nn.Module):
    architecture = ARCHITECTURE
    max_objects = 8
    def __init__(self):
        super().__init__()
        self.layout, self.obj, self.text, self.mask = fixture(3, 2)
        self.perception = nn.Linear(1, 1)

    def encode_text(self, ids, mask): return ids


def batch_for(model, device='cpu'):
    obj = model.obj
    truths = []
    for i, n in enumerate((2, 3)):
        truths.append(dict(source_xy=obj.xy[i, :n].to(device), size=obj.size[i, :n].to(device),
            source_angle=obj.angle[i, :n].to(device), color=obj.color[i, :n].argmax(-1).to(device),
            shape=obj.shape[i, :n].argmax(-1).to(device), target_xy=obj.xy[i, :n].to(device).clone(),
            target_angle=obj.angle[i, :n].to(device).clone(), move=torch.ones(n, device=device),
            selected=torch.ones(n, device=device), axis=torch.tensor(0, device=device),region=torch.tensor(0, device=device)))
    return SimpleNamespace(inputs=dict(input_ids=model.text.to(device),attention_mask=model.mask.to(device)),
        supervision=truths,metadata=[dict(sample_id=str(i),object_ids=[str(j) for j in range(n)],record={'split':'train'}) for i,n in enumerate((2,3))])


@pytest.mark.parametrize('device',DEVICES)
def test_capture_scene_weighting_and_no_target_leakage(device):
    torch.manual_seed(9);m=Fixture().to(device);batch=batch_for(m,device)
    data=collect_readout(m,[batch],expected_ids=['0','1'],fingerprint='fixture')
    assert len(data.features)==5 and not data.features.requires_grad
    torch.testing.assert_close(data.weights,torch.tensor([.25,.25,1/6,1/6,1/6],device=device))
    bad=deepcopy(batch)
    for t in bad.supervision:t['target_xy']+=100;t['target_angle']+=.1
    other=collect_readout(m,[bad],expected_ids=['0','1'],fingerprint='fixture')
    torch.testing.assert_close(data.features,other.features,atol=0,rtol=0)
    assert data.key!=other.key
    configure_stage(m,'micro_pose');terms,_=stage_loss(m,batch.inputs,batch.supervision)
    for kind,term in [('xy','position_huber'),('angle','angle_ce')]:
        fitter=HeadFitter(m,data,kind)
        torch.testing.assert_close(fitter.loss(),terms['layout/'+term],rtol=2e-6,atol=1e-6)
    bad.metadata[0]['record']['split']='test'
    with pytest.raises(ValueError,match='Train-only'):collect_readout(m,[bad],expected_ids=['0','1'],fingerprint='fixture')
    with pytest.raises(ValueError):collect_readout(m,[batch,batch],expected_ids=['0','1'],fingerprint='fixture')
    with torch.no_grad():m.layout.pose.input.bias.add_(1)
    with pytest.raises(ValueError,match='stale'):data.validate(m,'fixture',['0','1'])


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kind',['xy','angle'])
def test_slice_only_updates_transactional_abort_and_solver_roundtrip(device,kind):
    torch.manual_seed(2);m=Fixture().to(device);batch=batch_for(m,device)
    data=collect_readout(m,[batch],expected_ids=['0','1'],fingerprint='fixture')
    m.requires_grad_(False);before=deepcopy(m.state_dict());f=HeadFitter(m,data,kind)
    f.step(lambda:None)
    for k,v in m.state_dict().items():
        if k.startswith('layout.pose.output.1.'):
            mask=torch.ones(38,dtype=torch.bool);mask[SLICES[kind]]=False
            torch.testing.assert_close(v[mask],before[k][mask],atol=0,rtol=0)
        else:torch.testing.assert_close(v,before[k],atol=0,rtol=0)
    assert all(p.grad is None for p in m.parameters())
    saved=deepcopy(f.state_dict());weights=deepcopy(m.state_dict())
    f.step(lambda:None);expected=deepcopy(m.state_dict())
    m.load_state_dict(weights);again=HeadFitter(m,data,kind,saved);again.step(lambda:None)
    for k,v in m.state_dict().items():torch.testing.assert_close(v,expected[k],atol=0,rtol=0)
    state=deepcopy(again.state_dict());w=again.weight.detach().clone();b=again.bias.detach().clone();calls=[0]
    def abort():
        calls[0]+=1
        if calls[0]==3:raise TimeoutError('trial deadline')
    with pytest.raises(TimeoutError):again.step(abort)
    torch.testing.assert_close(again.weight,w,atol=0,rtol=0);torch.testing.assert_close(again.bias,b,atol=0,rtol=0)
    assert again.optimizer.state_dict()['state'][0]['n_iter']==state['optimizer']['state'][0]['n_iter']
    for k,v in m.state_dict().items():torch.testing.assert_close(v,expected[k],atol=0,rtol=0)


@pytest.mark.parametrize('device',DEVICES)
def test_split_adam_matches_fused_then_preserves_only_appropriate_fine_state(device):
    torch.manual_seed(19);m=Fixture().to(device);reference=deepcopy(m);b=batch_for(m,device)
    configure_stage(m,'micro_pose');configure_stage(reference,'micro_pose')
    adam=PoseAdam(m,1e-4,1e-4)
    opt=torch.optim.AdamW(reference.layout.pose.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False)
    for _ in range(2):
        opt.zero_grad();sum(stage_loss(reference,b.inputs,b.supervision)[0].values()).backward()
        torch.nn.utils.clip_grad_norm_(reference.layout.pose.parameters(),1);opt.step()
        adam.optimizer.zero_grad()
        with adam.forward_context():sum(stage_loss(m,b.inputs,b.supervision)[0].values()).backward()
        torch.nn.utils.clip_grad_norm_(adam.parameters,1);adam.optimizer.step();adam.sync()
    for k,v in m.state_dict().items():torch.testing.assert_close(v,reference.state_dict()[k],atol=2e-7,rtol=2e-6)
    old=deepcopy(adam.state_dict())
    for p in adam.parameters:assert int(adam.optimizer.state[p]['step'])==2
    with torch.no_grad():m.layout.pose.output[1].bias[:20].add_(.1)
    adam.enter_fine()
    for name,p in zip(adam.names,adam.parameters):
        if name.startswith(('xy.','angle.')):assert p not in adam.optimizer.state
        else:
            state=adam.optimizer.state[p];idx=adam.names.index(name)
            assert int(state['step'])==2 and not state['exp_avg'].count_nonzero()
            torch.testing.assert_close(state['exp_avg_sq'],old['optimizer']['state'][idx]['exp_avg_sq'],atol=0,rtol=0)
    assert adam.optimizer.param_groups[0]['lr']==1e-6
    with adam.forward_context():train=m.layout(__import__('knolling_model.recovery.contracts',fromlist=['oracle_objects']).oracle_objects(b.supervision),b.inputs['input_ids'],b.inputs['attention_mask'])
    actual=m.layout(__import__('knolling_model.recovery.contracts',fromlist=['oracle_objects']).oracle_objects(b.supervision),b.inputs['input_ids'],b.inputs['attention_mask'])
    for key in ('xy','angle_logits','angle_residual'):torch.testing.assert_close(train[key],actual[key],atol=0,rtol=0)


@pytest.mark.skipif(os.environ.get('RUN_RECOVERY_CUDA_TESTS')!='1',reason='explicit real cached-model CUDA check')
def test_real_cached_model_gradients_and_readout_inference_no_optimizer_steps():
    from knolling_model.recovery.model import load_recovery
    from knolling_model.recovery.driver import read_config
    from knolling_model.recovery.dataset_v3 import RecoveryDatasetV3
    from knolling_model.recovery.goal_labels import GoalDataset
    from knolling_model.recovery.runtime import FrozenTextCache
    from knolling_model.recovery.contracts import oracle_objects
    from knolling_model.data import S0Dataset,Collator
    c=read_config('configs/recovery_v4_integrated_review.json')
    ds=GoalDataset(RecoveryDatasetV3(S0Dataset(c['dataset'],c['selection']),c['data_v3']['path'],c['data_v3']['sha256']))
    m,tok=load_recovery('configs/recovery_v4_review.json');m.to('cuda');m.encode_text=FrozenTextCache(m.encode_text)
    batch=Collator(tok)([ds[i] for i in range(4)]).to('cuda')
    before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    configure_stage(m,'micro_pose');adam=PoseAdam(m,1e-4,1e-4);torch.cuda.reset_peak_memory_stats()
    with adam.forward_context():terms,_=stage_loss(m,batch.inputs,batch.supervision)
    sum(terms.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adam.parameters)
    assert all(p.grad is None for p in m.layout.goal_encoder.parameters())
    assert all(p.grad is None for p in m.text_encoder.parameters())
    assert all(p.grad is None for p in m.perception.parameters())
    m.zero_grad(set_to_none=True);m.requires_grad_(False)
    data=collect_readout(m,[batch],expected_ids=[x['sample_id'] for x in batch.metadata],fingerprint=ds.fingerprint)
    for kind in ('xy','angle'):
        fit=HeadFitter(m,data,kind);fit.loss().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (fit.weight,fit.bias))
        assert all(p.grad is None for p in m.parameters())
        with torch.no_grad():
            out=m.layout(oracle_objects(batch.supervision),m.encode_text(batch.inputs['input_ids'],batch.inputs['attention_mask']),batch.inputs['attention_mask'])
            raw=torch.nn.functional.linear(fit.features,fit.weight,fit.bias)
            expected=raw*raw.new_tensor([.3,.2])+raw.new_tensor([.3,.2]) if kind=='xy' else raw
            torch.testing.assert_close(out['xy' if kind=='xy' else 'angle_logits'][out['valid']],expected,rtol=2e-6,atol=2e-6)
    m.eval()
    with torch.no_grad():
        output=m(**batch.inputs)
        assert torch.isfinite(output['layout']['xy']).all()
    for k,v in m.state_dict().items():torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
    print('integrated real CUDA forward/backward peak_MiB',round(torch.cuda.max_memory_allocated()/2**20,2),'optimizer_steps=0')
