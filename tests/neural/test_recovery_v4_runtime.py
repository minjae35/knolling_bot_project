import os

import pytest
import torch

from knolling_model.arrangement.driver import read_config,save_exclusive
from knolling_model.arrangement.goal_joint_layout import GoalJointLayout
from test_recovery_goal_v4 import fixture


def test_v4_state_roundtrip_and_legacy_rejection(tmp_path):
    model,obj,text,mask=fixture()
    out=model(obj,text,mask)
    path=tmp_path/'v4.pt';save_exclusive(path,model.state_dict())
    restored=GoalJointLayout(dimension=32)
    restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
    again=restored(obj,text,mask)
    for k in ('xy','angle_logits','angle_residual'):
        torch.testing.assert_close(out[k],again[k],atol=0,rtol=0)
    torch.testing.assert_close(out['goal']['vector'],again['goal']['vector'],atol=0,rtol=0)
    from knolling_model.arrangement.joint_layout import JointPoseDecoder
    from knolling_model.arrangement.layout import AutoregressiveLayout
    for legacy in (JointPoseDecoder(dimension=32),AutoregressiveLayout(dimension=32)):
        with pytest.raises(RuntimeError):restored.load_state_dict(legacy.state_dict(),strict=True)


@pytest.mark.skipif(os.environ.get('RUN_RECOVERY_CUDA_TESTS')!='1',reason='explicit CUDA real cached-model verification')
def test_cached_r18_minilm_v4_forward_backward_fp32_bf16_no_updates():
    from knolling_model.arrangement.model import load_recovery
    from knolling_model.arrangement.stages import configure_stage,stage_loss
    from knolling_model.arrangement.dataset_v3 import RecoveryDatasetV3
    from knolling_model.arrangement.goal_labels import GoalDataset
    from knolling_model.data import S0Dataset,Collator
    from knolling_model.arrangement.runtime import FrozenTextCache
    c=read_config('configs/recovery_v4_execution.json')
    ds=GoalDataset(RecoveryDatasetV3(S0Dataset(c['dataset'],c['selection']),c['data_v3']['path'],c['data_v3']['sha256']))
    m,tok=load_recovery('configs/recovery_v4_review.json');m.to('cuda');m.encode_text=FrozenTextCache(m.encode_text)
    batch=Collator(tok)([ds[i] for i in range(4)]).to('cuda')
    before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    for stage in ('grounding','micro_pose','oracle_layout','perception'):
        for bf16 in (False,True):
            configure_stage(m,stage);torch.cuda.reset_peak_memory_stats()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
                terms,out=stage_loss(m,batch.inputs,batch.supervision)
            sum(terms.values()).backward()
            grads=[p.grad for p in m.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert all(p.grad is None for p in m.text_encoder.parameters())
            assert any(p.grad is not None for p in m.layout.goal_encoder.parameters())==(stage=='grounding')
            print('v4',stage,'bf16' if bf16 else 'fp32','peak_MiB',round(torch.cuda.max_memory_allocated()/2**20,2),
                  'loss',float(sum(terms.values()).detach()))
            m.zero_grad(set_to_none=True)
    m.eval()
    with torch.no_grad():
        out=m(**batch.inputs)
        assert torch.isfinite(out['layout']['xy']).all() and torch.isfinite(out['goal']['vector']).all()
    for k,v in m.state_dict().items():torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
