#!/usr/bin/env python3
"""Offline storage validation of existing weights; NEVER executes optimizer.step.

Writes exclusively into a new validation directory and the immutable common store.
Existing sources, experiment results, and datasets are never modified.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

from knolling_dataset.builder import file_hash
from knolling_model.arrangement.driver import load_checkpoint, save_exclusive
from knolling_model.arrangement.storage import CheckpointStore, common_store, load_artifact


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a.cpu().contiguous().reshape(-1).view(torch.uint8),
                           b.cpu().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) == type(b) and len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else: assert a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grounding', required=True); p.add_argument('--pose', required=True)
    p.add_argument('--output', required=True); p.add_argument('--cuda', action='store_true')
    args = p.parse_args()
    if Path(sys.prefix).name != 'knolling': raise RuntimeError('knolling environment required')
    torch.set_num_threads(2)
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    store = CheckpointStore(common_store(out))
    results = {'optimizer_steps': 0, 'files': [], 'environment': sys.prefix}
    sources = {}
    for kind, source in [('grounding', args.grounding), ('pose', args.pose)]:
        source = Path(source).resolve(); original_hash = file_hash(source)
        raw = load_artifact(source); sources[kind] = raw
        target = out/(kind + '.pt'); started = time.monotonic()
        store.save_checkpoint(target, raw); seconds = time.monotonic()-started
        restored = load_checkpoint(target, raw['config'], raw['dataset_fingerprint'], raw['commit'])
        equal(raw, restored)
        assert file_hash(source) == original_hash
        results['files'].append(dict(kind=kind, source=str(source), source_sha256=original_hash,
            legacy_bytes=source.stat().st_size, split_bytes=target.stat().st_size,
            split_sha256=file_hash(target), save_seconds=seconds, all_state_bitwise_equal=True))
    raw = sources['pose']; store.save_anchor(out/'anchor.pt', raw['model'], raw['stage'])
    equal(raw['model'], load_artifact(out/'anchor.pt'))
    # Complete standalone export, verified then removed only as a new scratch probe.
    probe = out/'export-probe.pt'; save_exclusive(probe, load_artifact(out/'pose.pt'))
    equal(raw, load_artifact(probe))
    results['standalone_export'] = dict(bytes=probe.stat().st_size, sha256=file_hash(probe),
        exact_state_restored=True, removed_own_probe=True)
    probe.unlink()
    results['anchor_bytes'] = (out/'anchor.pt').stat().st_size
    results['shared_files'] = [{'path': str(f), 'bytes': f.stat().st_size, 'sha256': file_hash(f)}
                               for f in sorted(store.root.glob('*.pt'))]
    if args.cuda:
        from knolling_model.arrangement.model import load_recovery
        from knolling_model.arrangement.stages import configure_stage, stage_loss
        from knolling_model.arrangement.runtime import FrozenTextCache
        from knolling_model.arrangement.dataset_v3 import RecoveryDatasetV3
        from knolling_model.arrangement.goal_labels import GoalDataset
        from knolling_model.data import S0Dataset, Collator
        model, tokenizer = load_recovery('configs/recovery_v4_review.json')
        config = raw['config']
        ds = GoalDataset(RecoveryDatasetV3(S0Dataset(config['dataset'], config['selection']),
                                          config['data_v3']['path'], config['data_v3']['sha256']))
        samples = [ds[i] for i in range(2)]
        model.to('cuda'); model.encode_text = FrozenTextCache(model.encode_text)
        batch = Collator(tokenizer)(samples).to('cuda')
        results['cuda'] = []
        for label, value in [('legacy', raw), ('split', load_artifact(out/'pose.pt'))]:
            model.load_state_dict(value['model'], strict=True); configure_stage(model, 'micro_pose')
            terms, output = stage_loss(model, batch.inputs, batch.supervision)
            sum(terms.values()).backward()
            grads = {k: v.grad.detach().cpu().clone() for k, v in model.named_parameters() if v.grad is not None}
            assert grads and all(torch.isfinite(g).all() for g in grads.values())
            assert all(k.startswith('layout.pose.') for k in grads)
            predictions = {k: output['layout'][k].detach().cpu().clone() for k in ('xy', 'angle_logits')}
            if label == 'legacy': reference_grads, reference_predictions = grads, predictions
            else: equal(reference_grads, grads); equal(reference_predictions, predictions)
            equal(value['model'], {k: v.detach().cpu() for k, v in model.state_dict().items()})
            results['cuda'].append(dict(source=label, gradient_tensors=len(grads), optimizer_steps=0))
        results['cuda_forward_backward_bitwise_equal'] = True
    (out/'result.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__': main()
