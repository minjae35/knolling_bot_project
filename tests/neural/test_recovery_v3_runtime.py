import json
import os
import subprocess
import sys
import threading
from copy import deepcopy

import pytest
import torch
from torch import nn

from knolling_model.recovery.driver import read_config,StageEngine,load_checkpoint,save_exclusive
from knolling_model.recovery.joint_layout import JointPoseDecoder
from test_recovery_driver import ScalarFixture,engine


def test_v3_cli_remains_disabled_and_pins_data(tmp_path):
    c=read_config('configs/recovery_v3_execution.json')
    assert c['allow_training'] is False and c['execution']=='review_only'
    result=subprocess.run([sys.executable,'-m','knolling_model.recovery.driver','--config','configs/recovery_v3_execution.json',
        '--stage','oracle_layout','--approval','nonexistent','--output',str(tmp_path/'never')],capture_output=True,text=True)
    assert result.returncode!=0 and 'training disabled' in result.stderr and not (tmp_path/'never').exists()
    c.pop('data_v3');p=tmp_path/'bad.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='pinned'):read_config(p)


def test_v3_scalar_resume_preserves_optimizer_rng_cursor(tmp_path):
    # Optimizer updates are ONLY a scalar lifecycle fixture, not a Knolling model.
    from test_recovery_driver import config
    c=config();c['model']['architecture']='recovery_v3_joint'
    def create(name,**kw):
        m=ScalarFixture();m.architecture='recovery_v3_joint'
        return engine(tmp_path,name,model=m,c=c,**kw)
    torch.manual_seed(101);a=create('whole');whole=a.run('oracle_layout')
    torch.manual_seed(101);stop=threading.Event();seen=0
    def evaluation(*_):
        nonlocal seen
        seen+=1
        if seen==4:stop.set()
        return {'pass':False,'full_set_no_regression':True}
    part=create('part',event=stop,evaluator=evaluation).run('oracle_layout')
    b=create('resumed');end=b.run('oracle_layout',resume=part['checkpoint'])
    x=load_checkpoint(whole['checkpoint'],c,'fixture-data','fixture-commit')
    y=load_checkpoint(end['checkpoint'],c,'fixture-data','fixture-commit')
    for name in x['model']:torch.testing.assert_close(x['model'][name],y['model'][name],atol=0,rtol=0)
    for key in x['optimizer']['state'][0]:torch.testing.assert_close(x['optimizer']['state'][0][key],y['optimizer']['state'][0][key],atol=0,rtol=0)
    assert torch.equal(x['rng']['torch'],y['rng']['torch'])
    for key in ('update','epoch','batch'):assert x['cursor'][key]==y['cursor'][key]
    with pytest.raises(ValueError,match='contract'):load_checkpoint(end['checkpoint'],c,'other-data','fixture-commit')
    y['architecture']='recovery_v2_ar';bad=tmp_path/'ar.pt';save_exclusive(bad,y)
    with pytest.raises(ValueError,match='contract'):load_checkpoint(bad,c,'fixture-data','fixture-commit')
    y['architecture']='recovery_v3_joint';y['evaluation_schema']='incorrect_metrics'
    bad=tmp_path/'metrics.pt';save_exclusive(bad,y)
    with pytest.raises(ValueError,match='contract'):load_checkpoint(bad,c,'fixture-data','fixture-commit')


def test_joint_checkpoint_roundtrip_without_updates(tmp_path):
    from test_recovery_joint_v3 import fixture
    model,obj,text,mask=fixture()
    before=model(obj,text,mask)
    p=tmp_path/'joint.pt';save_exclusive(p,model.state_dict())
    restored=JointPoseDecoder(dimension=32,layers=1)
    restored.load_state_dict(torch.load(p,weights_only=True),strict=True)
    after=restored(obj,text,mask)
    torch.testing.assert_close(before['xy'],after['xy'],atol=0,rtol=0)
    torch.testing.assert_close(before['angle_logits'],after['angle_logits'],atol=0,rtol=0)
    from knolling_model.recovery.layout import AutoregressiveLayout
    with pytest.raises(RuntimeError):restored.load_state_dict(AutoregressiveLayout(dimension=32,layers=1).state_dict(),strict=True)


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(os.environ.get('RUN_RECOVERY_CUDA_TESTS')!='1',reason='explicit CUDA check'))])
def test_v3_rgb_gradient_stage_paths_and_gt_isolation(device):
    from test_recovery_stages import TextFixture,inputs,selected_fixture
    from test_recovery_perception import tiny_detector,truth
    from knolling_model.recovery.model import RecoveryModel
    from knolling_model.recovery.stages import configure_stage,stage_loss
    m=RecoveryModel(TextFixture(),tiny_detector(),architecture='recovery_v3_joint',dimension=32,layers=1,num_denoising=4).to(device)
    batch={k:v.to(device) for k,v in inputs().items()}
    before={k:v.detach().clone() for k,v in m.state_dict().items()}
    for stage in ('oracle_layout','perception','predicted_layout','joint'):
        configure_stage(m,stage)
        # Synthetic source-matched labels exercise loss wiring, not detector quality.
        t=selected_fixture(m,batch) if stage in ('predicted_layout','joint') else {k:v.to(device) for k,v in truth().items()}
        terms,out=stage_loss(m,batch,[t],replay=(batch,[t]) if stage=='joint' else None)
        sum(terms.values()).backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        assert all(p.grad is None for p in m.text_encoder.parameters())
        if stage!='perception':
            assert {k for k in terms if k.startswith('layout/')}=={'layout/position_huber','layout/angle_ce','layout/angle_residual_huber'}
            corrupt={k:v.clone() for k,v in t.items()};corrupt['target_xy']+=.1;corrupt['move'].logical_not_()
            _,other=stage_loss(m,batch,[corrupt],replay=(batch,[t]) if stage=='joint' else None)
            torch.testing.assert_close(out['layout']['xy'],other['layout']['xy'],rtol=0,atol=0)
        m.zero_grad(set_to_none=True)
    # Direct layout objective must reach visual attributes even without replay loss.
    configure_stage(m,'joint');batch['images'].requires_grad_()
    output=m(**batch)
    (output['layout']['xy'].square().sum()+output['layout']['angle_logits'].square().mean()).backward()
    assert batch['images'].grad.abs().sum()>0
    assert sum(p.grad.abs().sum() for p in m.perception.physical.parameters() if p.grad is not None)>0
    for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)


@pytest.mark.skipif(os.environ.get('RUN_RECOVERY_CUDA_TESTS')!='1',reason='explicit real-model CUDA check')
def test_cached_r18_minilm_v3_cuda_forward_backward_no_updates():
    from knolling_model.recovery.model import load_recovery
    from knolling_model.recovery.stages import configure_stage,stage_loss
    from knolling_model.recovery.dataset_v3 import RecoveryDatasetV3
    from knolling_model.data import S0Dataset,Collator
    from knolling_model.recovery.runtime import FrozenTextCache
    c=read_config('configs/recovery_v3_execution.json')
    ds=RecoveryDatasetV3(S0Dataset(c['dataset'],c['selection']),c['data_v3']['path'],c['data_v3']['sha256'])
    m,tok=load_recovery('configs/recovery_v3_review.json');m.to('cuda');m.encode_text=FrozenTextCache(m.encode_text)
    batch=Collator(tok)([ds[i] for i in range(4)]).to('cuda')
    before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    for stage in ('oracle_layout','perception'):
        configure_stage(m,stage);torch.cuda.reset_peak_memory_stats()
        terms,out=stage_loss(m,batch.inputs,batch.supervision)
        sum(terms.values()).backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        assert all(p.grad is None for p in m.text_encoder.parameters())
        print('v3',stage,'peak_MiB',torch.cuda.max_memory_allocated()/2**20)
        m.zero_grad(set_to_none=True)
    m.eval()
    with torch.no_grad():
        deployed=m(**batch.inputs)
        assert torch.isfinite(deployed['layout']['xy']).all()
    for k,v in m.state_dict().items():torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
