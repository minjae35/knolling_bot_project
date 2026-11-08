#!/usr/bin/env python3
"""Existing-cache SVD/Forward validation; no training driver or iterative updates.

One analytic CPU candidate is applied in memory. Optimizer.step is forbidden on
actual-model fixtures. Model checkpoints, dataset and cache inputs are read-only.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

from knolling_dataset.builder import file_hash
from knolling_model.data import S0Dataset, Collator
from knolling_model.recovery.dataset_v3 import RecoveryDatasetV3
from knolling_model.recovery.goal_labels import GoalDataset
from knolling_model.recovery.evaluation_v4 import evaluate_v4, MICRO_CASES
from knolling_model.recovery.model import load_recovery
from knolling_model.recovery.readout import ReadoutData
from knolling_model.recovery.svd_readout import SVDXYFitter
from knolling_model.recovery.storage import load_artifact
from knolling_model.recovery.stages import configure_stage, stage_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint','cache','output'):parser.add_argument('--'+flag,required=True)
    parser.add_argument('--cuda',action='store_true');args=parser.parse_args()
    if Path(sys.prefix).name!='knolling':raise RuntimeError('knolling environment required')
    torch.set_num_threads(2)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    sources={p:file_hash(p) for p in (args.checkpoint,args.cache)}
    raw=load_artifact(args.checkpoint);data=ReadoutData(**torch.load(args.cache,map_location='cpu',weights_only=True))
    c=raw['config'];model,tokenizer=load_recovery('configs/recovery_v4_review.json')
    ds=GoalDataset(RecoveryDatasetV3(S0Dataset(c['dataset'],c['selection']),c['data_v3']['path'],c['data_v3']['sha256']))
    records=[ds[i] for i,r in enumerate(ds.records) if r['sample_id'] in MICRO_CASES['axis']]
    batch=Collator(tokenizer)(records)
    report=dict(source_sha256=sources,source_status=raw['status'],training_driver_runs=0,iterative_optimizer_updates=0,devices={})
    for device in (['cpu','cuda'] if args.cuda else ['cpu']):
        model.to(device);model.load_state_dict(raw['model'],strict=True);model.requires_grad_(False);model.eval()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        data.validate(model,raw['dataset_fingerprint'],list(MICRO_CASES['axis']))
        fitter=SVDXYFitter(model,data)
        def forbidden(*a,**k):raise AssertionError('iterative optimizer updates forbidden in real-model validation')
        fitter.optimizer.step=forbidden
        loss,converged=fitter.step(lambda:None)
        state=fitter.state_dict();resumed=SVDXYFitter(model,data,state)
        assert torch.equal(resumed.z,fitter.z) and torch.equal(resumed.weight,fitter.weight)
        actual=batch.to(device)
        metrics,_=evaluate_v4(model,[actual],stage='micro_pose',expected_ids=list(MICRO_CASES['axis']),micro_case='axis')
        assert metrics['micro']['passed'],metrics['micro']
        # Actual backward graph, without any optimizer construction/update.
        configure_stage(model,'micro_pose');terms,_=stage_loss(model,actual.inputs,actual.supervision)
        sum(terms.values()).backward()
        grads={k:p.grad for k,p in model.named_parameters() if p.grad is not None}
        assert grads and all(k.startswith('layout.pose.') and torch.isfinite(v).all() for k,v in grads.items())
        for k,v in model.state_dict().items():
            if k in ('layout.pose.output.1.weight','layout.pose.output.1.bias'):
                assert torch.equal(v[2:].cpu(),raw['model'][k][2:])
            else:assert torch.equal(v.cpu(),raw['model'][k]),k
        report['devices'][device]=dict(micro=metrics['micro'],validity=metrics['validity'],
            cached_readout=resumed.measurements(),huber=loss,converged=converged,gradient_tensors=len(grads),
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    assert all(file_hash(p)==h for p,h in sources.items())
    report['sources_unchanged']=True
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
