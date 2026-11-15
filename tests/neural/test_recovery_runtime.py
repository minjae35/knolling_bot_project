import json
import os

import pytest
import torch

from knolling_model.data import Collator, S0Dataset
from knolling_model.arrangement.batches import stage_batches
from knolling_model.arrangement.counterfactuals import RecoverySuite
from knolling_model.arrangement.driver import read_config
from knolling_model.arrangement.runtime import CachedDataset, FrozenTextCache, ProgressLog, precision_context, warm_text_cache


def test_frozen_text_cache_respects_tokens_and_padding():
    calls=[]
    def encode(ids, mask):
        calls.append(ids.clone())
        return ids.float()[..., None].expand(-1,-1,4)
    cache=FrozenTextCache(encode)
    a=cache(torch.tensor([[1,2,0]]), torch.tensor([[True,True,False]]))
    b=cache(torch.tensor([[1,2],[3,4]]), torch.ones(2,2,dtype=torch.bool))
    torch.testing.assert_close(a[0,:2], b[0])
    assert a[0,2].sum()==0 and len(calls)==2 and calls[-1].tolist()==[[3,4]]
    cache(torch.tensor([[3,4]]), torch.ones(1,2,dtype=torch.bool))
    assert len(calls)==2
    with pytest.raises(ValueError):
        cache(torch.tensor([[0,1]]), torch.tensor([[False,True]]))


def test_dataloader_does_not_consume_model_rng(s0_dataset):
    class DummyCollator:
        def __call__(self, samples):
            return [s['metadata']['sample_id'] for s in samples]
    before=torch.get_rng_state().clone()
    first=list(stage_batches(s0_dataset,DummyCollator(),'joint',seed=12,epoch=0,batch_size=4))
    second=list(stage_batches(s0_dataset,DummyCollator(),'joint',seed=12,epoch=0,batch_size=4))
    assert torch.equal(before,torch.get_rng_state())
    assert [b.batch for b in first]==[b.batch for b in second]
    assert set(x for b in first for x in b.batch)=={r['sample_id'] for r in s0_dataset.records}
    assert len(set(x for b in first for x in b.replay))==25


def test_continuation_cache_uses_initial_evaluation_order_and_shapes():
    from types import SimpleNamespace
    calls=[]
    def encode(ids,mask):
        calls.append((ids.clone(),mask.clone()))
        return ids.float()[...,None].expand(-1,-1,4)
    model=SimpleNamespace(encode_text=FrozenTextCache(encode))
    def collate(samples):
        n=max(len(s) for s in samples)
        ids=torch.tensor([s+[0]*(n-len(s)) for s in samples])
        return SimpleNamespace(inputs={'input_ids':ids,'attention_mask':ids!=0})
    dataset=[[1,2],[3],[4,5,6],[1,2]]
    warm_text_cache(model,dataset,collate,2,'cpu')
    assert [ids.tolist() for ids,_ in calls]==[[[1,2]],[[3,0]],[[4,5,6]]]
    expected={k:v.clone() for k,v in model.encode_text.values.items()}
    calls.clear()
    warm_text_cache(model,dataset,collate,2,'cpu')
    assert len(calls)==3
    for key in expected:
        torch.testing.assert_close(expected[key],model.encode_text.values[key],rtol=0,atol=0)


def test_progress_records_measured_loss_and_evaluations(tmp_path):
    from test_recovery_driver import engine
    observer=ProgressLog(tmp_path/'observed')
    try:
        result=engine(tmp_path,'observed',observer=observer).run('oracle_layout')
    finally:
        observer.close()
    history=[json.loads(line) for line in (tmp_path/'observed'/'history.jsonl').read_text().splitlines()]
    updates=[r for r in history if r['event']=='update']
    assert [r['update'] for r in updates]==[1,2,3,4]
    assert all(isinstance(r['loss'],float) and r['losses'] for r in updates)
    assert any(r['event']=='evaluation' and 'metrics' in r for r in history)
    assert history[-1]['status']==result['status']=='budget_exhausted'


@pytest.mark.skipif(os.environ.get('RUN_RECOVERY_CUDA_TESTS')!='1', reason='explicit GPU verification only')
def test_bf16_arrangement_and_fp32_perception_batch_smoke():
    from knolling_model.runner import seed_everything
    from knolling_model.arrangement.model import load_recovery
    from knolling_model.arrangement.stages import configure_stage, stage_loss
    seed_everything(1203,2)
    c=read_config('configs/recovery_gpu_initial.json')
    dataset=RecoverySuite(S0Dataset(c['dataset'],c['selection']),c['counterfactuals']['path'],c['counterfactuals']['sha256'])
    cached=CachedDataset(dataset)
    assert cached.fingerprint==dataset.fingerprint
    model,tokenizer=load_recovery()
    model.to('cuda')
    model.encode_text=FrozenTextCache(model.encode_text)
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    collator=Collator(tokenizer)
    for stage in ('oracle_layout','perception'):
        configure_stage(model,stage)
        batch=next(stage_batches(cached,collator,stage,seed=1203,epoch=0,
            batch_size=c['batch_sizes'][stage],pin_memory=True)).to('cuda',non_blocking=True)
        inputs,truth,replay=batch.loss_arguments()
        torch.cuda.reset_peak_memory_stats()
        with precision_context(c,stage):
            terms,out=stage_loss(model,inputs,truth)
        assert all(torch.isfinite(v) for v in terms.values())
        sum(terms.values()).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.text_encoder.parameters())
        if stage=='oracle_layout':
            assert out['layout']['xy'].dtype==torch.float32
            model.eval()
            with torch.no_grad(),precision_context(c,stage):
                from knolling_model.arrangement.contracts import oracle_objects
                deployed=model.layout(oracle_objects(truth),model.encode_text(inputs['input_ids'],inputs['attention_mask']),inputs['attention_mask'])
            torch.testing.assert_close(deployed['xy'],out['layout']['xy'],atol=1e-5,rtol=1e-5)
        print(stage,'batch',len(truth),'peak_MiB',torch.cuda.max_memory_allocated()/2**20)
        model.zero_grad(set_to_none=True)
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
