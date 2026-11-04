"""One budgeted repair candidate, or the conditionally admitted joint pilot.

The immutable root protocol names the two candidates for each bottleneck. A
ledger directory prevents accidentally rerunning one candidate from scratch.
Every micro candidate uses the same original model weights and fresh AdamW.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from knolling_model import runner
from validate_s0_fixes import (
    make_cache, evaluate_cached, cached_output, optimizer_step, slice_output,
)
from s0_validation_gates import (
    DETECTION_SAMPLES, LANGUAGE_SAMPLES, detection_pass, language_pass, stable,
    family_groups, pilot_pass, repair_admitted,
)


def configure(model, candidate):
    model.requires_grad_(False).eval()
    if candidate == 'detection1':
        model.perception.detector.class_embed[-1].requires_grad_(True)
    elif candidate == 'detection2':
        model.perception.requires_grad_(True)
        model.source_heads.requires_grad_(True)
        model.perception.detector.model.denoising_class_embed.requires_grad_(False)
    elif candidate.startswith('language'):
        model.text_projection.requires_grad_(True)
        model.arrangement.requires_grad_(True)
    elif candidate == 'joint':
        model.requires_grad_(True)
        model.text_encoder.requires_grad_(False)
        model.perception.detector.model.denoising_class_embed.requires_grad_(False)
    else:
        raise ValueError('unknown candidate')


def weight_load(model, state):
    missing = model.load_state_dict(state, strict=False)
    assert not missing.unexpected_keys
    assert all(n.startswith('text_encoder.') for n in missing.missing_keys)
    return missing


def save_candidate(path, model, optimizer, updates, evidence):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {n: p for n, p in model.state_dict().items()
             if not n.startswith('text_encoder.') or n in trainable}
    temporary = path.with_suffix('.tmp')
    torch.save({'format': 's0_repair_candidate_v1', 'diagnostic_only': True,
                'model': state, 'optimizer': optimizer.state_dict(),
                'cursor': {'updates': updates}, 'provenance': evidence,
                'rng': runner.rng_state()}, temporary)
    temporary.replace(path)


def run(args):
    root, name = args.root, args.candidate
    plan = json.loads((root/'protocol.json').read_text())
    assert plan['schema'] == 's0_repair_protocol_v1'
    completed = {p.parent.name: json.loads(p.read_text())['passed']
                 for p in root.glob('*/result.json')}
    if not repair_admitted(name, completed):
        raise ValueError('repair admission denied; inspect prior metric gates and stop rules')
    digest = hashlib.sha256()
    with Path(plan['base_checkpoint']).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == plan['base_checkpoint_sha256']
    if name.endswith('2'):
        first = json.loads((root/(name[:-1]+'1')/'result.json').read_text())
        assert first['passed'] is False, 'a second candidate requires first-candidate failure'
    if name == 'joint':
        winners = json.loads((root/'winners.json').read_text())
        assert set(winners) == {'detection', 'language'}
        for winner in winners.values():
            assert json.loads((root/winner/'result.json').read_text())['passed']
    folder = root/name
    folder.mkdir(exist_ok=False)
    cfg = runner.read_config('configs/s0_rtdetr.json')
    cfg = {**cfg, 'objectness_normalization': 'balanced_groups'}
    ds, model, collator = runner.prepare(cfg, 'cuda', True)
    saved = torch.load(plan['base_checkpoint'], map_location='cpu', weights_only=True)
    weight_load(model, saved['model'])
    if name == 'joint':
        d = torch.load(root/winners['detection']/'checkpoint.pt', map_location='cpu', weights_only=True)
        l = torch.load(root/winners['language']/'checkpoint.pt', map_location='cpu', weights_only=True)
        weight_load(model, d['model'])
        state = model.state_dict()
        for n, v in l['model'].items():
            if n.startswith(('text_encoder.', 'text_projection.', 'arrangement.')):
                state[n].copy_(v)
    configure(model, name)
    optimizer = runner.make_optimizer(model, cfg)
    criterion = runner.make_criterion(cfg)
    torch.cuda.reset_peak_memory_stats()
    initial_cache = make_cache(ds, model, collator)
    baseline = evaluate_cached(model, initial_cache)
    runner.write_json(folder/'initial-all32.json', baseline)
    original = json.loads((root/'baseline-all32.json').read_text()) if (root/'baseline-all32.json').exists() else baseline
    if not (root/'baseline-all32.json').exists():
        runner.write_json(root/'baseline-all32.json', original)
    cached = name in ('detection1', 'language1')
    selected = DETECTION_SAMPLES if name.startswith('detection') else LANGUAGE_SAMPLES
    ids = {r['sample_id']: i for i, r in enumerate(ds.records)}
    trainable = {n for n, p in model.named_parameters(remove_duplicate=False) if p.requires_grad}
    frozen = {n: p.detach().cpu().clone() for n, p in model.state_dict().items() if n not in trainable}
    evidence = runner.provenance(cfg)
    runner.write_json(folder/'provenance.json', {**evidence,
        'initialization': 'weights_only_from_original_checkpoint_fresh_adam',
        'base_checkpoint_sha256': plan['base_checkpoint_sha256'],
        'protocol_sha256': hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
        'trainable_names': sorted(trainable)})
    limit = 200 if name == 'joint' else 100
    history, updates, epoch = [], 0, 0
    start = time.monotonic()
    while updates < limit:
        if name == 'joint':
            families = sorted({r['scene_family_id'] for r in ds.records})
            g = torch.Generator().manual_seed(cfg['seed']+epoch)
            names = [families[i] for i in torch.randperm(len(families), generator=g).tolist()]
            groups = list(family_groups(ds.records, names))
        else:
            groups = [[(ids[sid], 1/len(selected)) for sid in selected]]
        for group in groups:
            if updates == limit:
                break
            optimizer.zero_grad(set_to_none=True)
            objective = 0.
            for start_micro in range(0, len(group), 2):
                micro = group[start_micro:start_micro+2]
                if cached:
                    for i, w in micro:
                        c = initial_cache[ds.records[i]['sample_id']]
                        p = cached_output(model, c, targets=name.startswith('language'))
                        terms, _ = criterion(p, c['batch'].supervision)
                        value = terms['detection' if name.startswith('detection') else 'target']*w
                        if not torch.isfinite(value):
                            raise FloatingPointError('nonfinite candidate loss')
                        value.backward()
                        objective += float(value.detach())
                else:
                    b = collator([ds[i] for i, _ in micro]).to('cuda')
                    p = model(**b.inputs)
                    key = 'loss' if name == 'joint' else 'detection' if name.startswith('detection') else 'target'
                    value = sum(criterion(slice_output(p, j), [b.supervision[j]])[0][key]*w
                                for j, (_, w) in enumerate(micro))
                    if not torch.isfinite(value):
                        raise FloatingPointError('nonfinite candidate loss')
                    value.backward()
                    objective += float(value.detach())
            norm = optimizer_step(model, optimizer, updates, limit)
            updates += 1
            runner.append_event(folder/'optimizer.jsonl', {'update': updates, 'loss': objective, 'gradient_norm': norm})
            interval = 20 if name == 'joint' else 10
            if updates % interval == 0:
                evaluation_cache = initial_cache if cached else make_cache(ds, model, collator)
                result = evaluate_cached(model, evaluation_cache)
                if name == 'joint':
                    gate = pilot_pass(result, original)
                    history.append({'passed': gate})
                else:
                    selected_result = evaluate_cached(model, evaluation_cache, list(selected))
                    values = selected_result['detection' if name.startswith('detection') else 'language']
                    gate = (detection_pass if name.startswith('detection') else language_pass)(values)
                    history.append(values)
                    result['selected'] = selected_result
                result.update(update=updates, gate_pass=gate)
                runner.write_json(folder/f'eval-{updates:03d}.json', result)
                print(name, updates, 'gate', gate,
                      json.dumps(result['selected']['detection' if name.startswith('detection') else 'language'])
                      if name != 'joint' else '', flush=True)
                # Immutable snapshots support interruption review without
                # permitting more than 100 calls for an attempted candidate.
                if updates % 50 == 0 or updates == limit:
                    save_candidate(folder/f'checkpoint-{updates:03d}.pt', model, optimizer,
                                   updates, evidence)
                if not cached:
                    del evaluation_cache
        epoch += 1
    final_cache = initial_cache if cached else make_cache(ds, model, collator)
    final = evaluate_cached(model, final_cache)
    frozen_ok = all(torch.equal(v, model.state_dict()[n].cpu()) for n, v in frozen.items())
    assert frozen_ok
    if name == 'joint':
        passed = len(history) >= 3 and all(v['passed'] for v in history[-3:])
    else:
        passed = stable(history, detection_pass if name.startswith('detection') else language_pass)
        if name.startswith('detection'):
            passed = passed and final['detection']['count_correct'] >= original['detection']['count_correct']
    # Loss-policy changes are explicit weight-only warm starts, never a silent
    # official resume. Serialized optimizer/cursor/provenance use production IO.
    save_candidate(folder/'checkpoint.pt', model, optimizer, updates, evidence)
    restored = torch.load(folder/'checkpoint.pt', weights_only=True, map_location='cpu')
    assert all(torch.equal(v.cpu(), restored['model'][n]) for n, v in model.state_dict().items()
               if n in restored['model'])
    result = {'candidate': name, 'passed': bool(passed), 'optimizer_updates': updates,
              'elapsed_training_seconds': time.monotonic()-start,
              'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
              'frozen_state_unchanged': frozen_ok,
              'checkpoint_roundtrip_equal': True, 'final': final}
    runner.write_json(folder/'result.json', result)
    return 0 if passed else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', choices=['detection1', 'detection2', 'language1', 'joint'], required=True)
    raise SystemExit(run(parser.parse_args()))
