"""Opt-in real GPU forward/backward verification; zero optimizer updates."""
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import pytest
import torch


@pytest.mark.skipif(os.environ.get('RUN_RECOVERY_CUDA_TESTS') != '1', reason='explicit GPU verification only')
def test_real_r18_minilm_cuda_paths_gradients_and_memory(s0_dataset):
    from knolling_model.runner import seed_everything
    from knolling_model.data import Collator
    from knolling_model.recovery.data import RecoveryDataset
    from knolling_model.recovery.model import load_recovery
    from knolling_model.recovery.stages import configure_stage, stage_loss
    from knolling_model.recovery.selection import SourceSelectionError
    from test_recovery_stages import selected_fixture
    seed_everything(1203, 2)
    assert torch.cuda.is_available()
    model, tokenizer = load_recovery()
    model.to('cuda')
    def digest_weights():
        h=hashlib.sha256()
        for name,value in model.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=digest_weights()
    batch=Collator(tokenizer)([RecoveryDataset(s0_dataset)[0]]).to('cuda')
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    checked=[]
    for stage in ('perception','oracle_layout'):
        configure_stage(model,stage)
        terms,out=stage_loss(model,batch.inputs,batch.supervision)
        assert all(torch.isfinite(v) for v in terms.values())
        sum(terms.values()).backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
        checked.append(stage)
        del terms,out,grads
        model.zero_grad(set_to_none=True)
    configure_stage(model,'predicted_layout')
    # Fresh task foreground head must not replace missing detections with GT rows.
    with pytest.raises(SourceSelectionError):
        stage_loss(model,batch.inputs,batch.supervision)
    original=model.perception.forward
    labels=selected_fixture(model,batch.inputs)
    try:
        for stage in ('predicted_layout','joint'):
            configure_stage(model,stage)
            terms,out=stage_loss(model,batch.inputs,[labels],
                replay=(batch.inputs,[labels]) if stage=='joint' else None)
            assert all(torch.isfinite(v) for v in terms.values())
            assert out['objects'].query_index.tolist()==[[0,1]]
            if stage=='predicted_layout':
                with torch.no_grad():
                    deployed=model(**batch.inputs)['layout']
                torch.testing.assert_close(deployed['xy'],out['layout']['xy'],atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(deployed['angle'],out['layout']['angle'],atol=1e-6,rtol=1e-5)
            # Layout-only gradient must reach unfrozen perception in the joint stage.
            sum(v for k,v in terms.items() if k.startswith('layout/')).backward(retain_graph=stage=='joint')
            if stage=='joint':
                grads=[p.grad for p in model.perception.parameters() if p.grad is not None]
                assert grads and sum(g.abs().sum() for g in grads)>0
                model.zero_grad(set_to_none=True)
                sum(terms.values()).backward()  # Combined arrangement + detection replay.
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert all(p.grad is None for p in model.text_encoder.parameters())
            checked.append(stage+' (synthetic selection fixture)')
            del terms,out
            model.zero_grad(set_to_none=True)
    finally:
        model.perception.forward=original
    torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    assert before==digest_weights()
    report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        stages=checked,joint_replay_backward=True,optimizer_updates=0,weights_and_buffers_unchanged=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,seconds=elapsed,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if os.environ.get('RECOVERY_CUDA_REPORT'):
        with Path(os.environ['RECOVERY_CUDA_REPORT']).open('x') as stream:
            json.dump(report,stream,indent=2)
    print(json.dumps(report))
    del model
    gc.collect()
    torch.cuda.empty_cache()
