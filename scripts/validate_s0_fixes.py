"""Explicit bounded validations. Never invokes the official S0 training loop.

Caches only actual predicted Source/features when perception is frozen in a
micro-test. The conditional pilot uses full image->prediction forwards/backwards.
Ground truth is consumed only by matching, losses and diagnostic evaluation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from knolling_model import runner
from knolling_model.coordinates import to_meters, angle_delta
from knolling_model.evaluation import evaluate_batch, summarize
from knolling_model.losses import S0Criterion, match_source
from verify_s0_data import verify
from s0_validation_gates import (
    MICRO_LIMIT, PILOT_LIMIT, DETECTION_SAMPLES, LANGUAGE_SAMPLES,
    detection_pass, language_pass, pilot_pass, stable, step_allowed,
    family_groups, protocol, decision_exit_code,
)


def write(path, value):
    runner.write_json(path, value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def hashes(paths):
    cache, result = {}, {}
    for p in sorted(set(map(Path, paths))):
        s = p.stat()
        key = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if key not in cache:
            cache[key] = sha(p)
        result[str(p.resolve())] = cache[key]
    return result


def detach(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach(v) for k, v in value.items()}
    if isinstance(value, list):
        return [detach(v) for v in value]
    return value


def slice_output(value, i):
    if torch.is_tensor(value):
        return value[i:i + 1]
    if isinstance(value, dict):
        return {k: slice_output(v, i) for k, v in value.items()}
    if isinstance(value, list):
        return [slice_output(v, i) for v in value]
    raise TypeError(type(value))


def rectangle(xy, ab, theta):
    c, s = math.cos(theta), math.sin(theta)
    a, b = ab / 2
    return Polygon([(xy[0] + c*x - s*y, xy[1] + s*x + c*y)
                    for x, y in [(-a, -b), (a, -b), (a, b), (-a, b)]])


def metrics(output, batch):
    """Independent active identity counts plus explicitly conditional pose errors."""
    source, truth = output['source'], batch.supervision[0]
    q, g = match_source(source, [truth])[0]
    scores = source['objectness_logits'][0].sigmoid().detach().cpu().numpy()
    xy = to_meters(source['uv'][0]).detach().cpu().numpy()
    ab = source['size'][0].detach().cpu().numpy()
    theta = source['angle'][0].detach().cpu().numpy()
    gtxy, gtab, gtangle = (truth[k].cpu().numpy() for k in ['source_xy', 'size', 'source_angle'])
    active = np.flatnonzero(scores >= .5)
    valid = np.zeros((len(active), len(gtxy)), bool)
    for j, qi in enumerate(active):
        p = rectangle(xy[qi], ab[qi], theta[qi])
        for gi in range(len(gtxy)):
            r = rectangle(gtxy[gi], gtab[gi], gtangle[gi])
            iou = p.intersection(r).area / max(p.union(r).area, 1e-12)
            valid[j, gi] = np.linalg.norm(xy[qi] - gtxy[gi]) <= .01 and iou >= .5
    if len(active):
        a, b = linear_sum_assignment(~valid)
        used = {int(i) for i, j in zip(a, b) if valid[i, j]}
    else:
        used = set()
    tp = len(used)
    negative = np.ones(300, bool)
    negative[q.cpu().numpy()] = False
    final = output['targets'][-1]
    conditional = {
        'source_position_mm': (to_meters(source['uv'][0, q])-truth['source_xy'][g]).norm(dim=-1)*1000,
        'source_size_mm': (source['size'][0, q]-truth['size'][g]).abs().max(-1).values*1000,
        'source_angle_deg': angle_delta(source['angle'][0, q], truth['source_angle'][g]).abs()*180/math.pi,
        'target_position_mm': (to_meters(final['uv'][0, q])-truth['target_xy'][g]).norm(dim=-1)*1000,
        'target_angle_deg': angle_delta(final['angle'][0, q], truth['target_angle'][g]).abs()*180/math.pi,
    }
    angles = final['angle'][0, q[g.argsort()]].detach().cpu().tolist()
    return {
        'sample_id': batch.metadata[0]['sample_id'], 'gt_count': len(gtxy),
        'predicted_count': len(active), 'count_correct': int(len(active) == len(gtxy)),
        'tp': tp, 'fn': len(gtxy)-tp, 'fp': len(active)-tp,
        'duplicates': sum(bool(valid[i].any()) for i in range(len(active)) if i not in used),
        'min_matched_confidence': float(scores[q.cpu().numpy()].min()),
        'max_unmatched_confidence': float(scores[negative].max()),
        'matched_scores': scores[q.cpu().numpy()].tolist(),
        'unmatched_scores': scores[negative].tolist(),
        'conditional_pose': {k: v.detach().cpu().tolist() for k, v in conditional.items()},
        'predicted_target_angles_rad_gt_order': angles,
        'predicted_target_xy_m_gt_order': to_meters(final['uv'][0, q[g.argsort()]]).detach().cpu().tolist(),
        'finite': all(bool(torch.isfinite(v).all()) for v in conditional.values())
                  and bool(np.isfinite(scores).all()),
    }


def aggregate(rows):
    return {**{k: sum(r[k] for r in rows) for k in
               ('count_correct', 'tp', 'fn', 'fp', 'duplicates')},
            'min_matched_confidence': min(r['min_matched_confidence'] for r in rows),
            'max_unmatched_confidence': max(r['max_unmatched_confidence'] for r in rows),
            'finite': all(r['finite'] for r in rows)}


def language_metrics(rows):
    by_id = {r['sample_id']: r for r in rows}
    pair = [by_id[k] for k in LANGUAGE_SAMPLES]
    a, b = [torch.tensor(r['predicted_target_angles_rad_gt_order']) for r in pair]
    changes = angle_delta(a, b).abs() * 180/math.pi
    return aggregate(pair) | {
        'instructions': [{'sample_id': r['sample_id'],
                          'target_position_p95_mm': float(np.percentile(r['conditional_pose']['target_position_mm'], 95)),
                          'target_angle_p95_deg': float(np.percentile(r['conditional_pose']['target_angle_deg'], 95))}
                         for r in pair],
        'pair_angle_change_deg': changes.tolist(),
        'minimum_pair_angle_change_deg': float(changes.min()),
    }


def evaluate_cached(model, cache, ids=None):
    diagnostics, official, records = [], [], []
    ids = ids or list(cache)
    with torch.no_grad():
        for sid in ids:
            c = cache[sid]
            p = cached_output(model, c)
            diagnostics.append(metrics(p, c['batch']))
            official.extend(evaluate_batch(p, c['batch']))
            records.append(c['batch'].metadata[0]['record'])
    result = {'detection': aggregate(diagnostics), 'rows': diagnostics,
              'official': summarize(official, records), 'finite': all(r['finite'] for r in diagnostics)}
    if set(LANGUAGE_SAMPLES).issubset(ids):
        result['language'] = language_metrics(diagnostics)
    result['conditional_pose'] = {
        k: {'p95': float(np.percentile([x for r in diagnostics for x in r['conditional_pose'][k]], 95))}
        for k in diagnostics[0]['conditional_pose']}
    return result


def cached_output(model, cache, targets=True):
    p = dict(cache['output'])
    p['source'] = dict(p['source'])
    logits = model.perception.detector.class_embed[-1](p['object_tokens'])
    p['source']['objectness_logits'] = logits.max(-1).values
    if targets:
        text = model.text_projection(cache['text_features'])
        p['targets'] = model.arrangement(p['object_tokens'], p['source'], text,
                                        cache['batch'].inputs['attention_mask'])
    return p


def configure(model, phase):
    model.requires_grad_(False).eval()
    if phase == 'detection':
        model.perception.detector.class_embed[-1].requires_grad_(True)
    elif phase == 'language':
        model.text_projection.requires_grad_(True)
        model.arrangement.requires_grad_(True)
    else:
        model.requires_grad_(True)
        model.text_encoder.requires_grad_(False)
        model.perception.detector.model.denoising_class_embed.requires_grad_(False)


def save_probe(path, model, optimizer, phase, updates, base_sha):
    torch.save({'format': 's0_bounded_validation_v1', 'not_official_s0': True,
                'phase': phase, 'updates': updates, 'base_sha256': base_sha,
                'model': {n: p.detach().cpu() for n, p in model.state_dict().items()
                          if not n.startswith('text_encoder.')},
                'optimizer': optimizer.state_dict(), 'rng': runner.rng_state(),
                'trainable_names': [n for n, p in model.named_parameters() if p.requires_grad]}, path)


def optimizer_step(model, optimizer, completed, limit):
    if not step_allowed(completed, limit):
        raise RuntimeError('optimizer update budget exhausted')
    params = [p for p in model.parameters() if p.requires_grad]
    if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
        raise FloatingPointError('nonfinite gradient; no optimizer update')
    norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in params):
        raise FloatingPointError('nonfinite optimizer output')
    return norm


def zero_update(model, cache, ds, cfg):
    criterion = S0Criterion()
    gaps, derivative_errors = [], []
    for c in cache.values():
        p = cached_output(model, c, targets=False)
        truth = c['batch'].supervision[0]
        q, g = match_source(p['source'], [truth])[0]
        s = p['source']
        xy, ab = to_meters(s['uv'][0]).cpu().numpy(), s['size'][0].cpu().numpy()
        angle = s['angle'][0].cpu().numpy()
        score = s['objectness_logits'][0].detach().sigmoid().cpu().numpy()
        delta = (angle[:, None]-truth['source_angle'].cpu().numpy()[None, :]+math.pi/2)%math.pi-math.pi/2
        cost = (-score[:, None] + np.abs(xy[:, None]-truth['source_xy'].cpu().numpy()).sum(-1)/.020
                + np.abs(ab[:, None]-truth['size'].cpu().numpy()).sum(-1)/.010 + np.abs(delta)/(math.pi/18))
        iq, ig = linear_sum_assignment(cost)
        gaps.append(float(cost[q.cpu(), g.cpu()].sum()-cost[iq, ig].sum()))
        terms, _ = criterion(p, [truth])
        logit = s['objectness_logits']
        actual = torch.autograd.grad(terms['detection'], logit)[0][0]
        labels = torch.zeros_like(logit[0]); labels[q] = 1
        weights = torch.where(labels.bool(), 1., .1)
        expected = (logit[0].detach().sigmoid()-labels)*weights/weights.sum()/4
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
        derivative_errors.append(float((actual-expected).abs().max()))
    assert max(abs(x) for x in gaps) < 1e-4
    # Target derivatives on each actual instruction, including perception.
    gradient_groups = {
        'text_projection': list(model.text_projection.parameters()),
        'arrangement': list(model.arrangement.parameters()),
        'perception_decoder': list(model.perception.detector.model.decoder.parameters()),
    }
    model.requires_grad_(True)
    model.text_encoder.requires_grad_(False)
    gradients, losses = {}, []
    for sid in LANGUAGE_SAMPLES:
        b = cache[sid]['batch']
        p = model(**b.inputs)
        terms, _ = criterion(p, b.supervision)
        losses.append(float(terms['target'].detach()))
        for name, params in gradient_groups.items():
            values = torch.autograd.grad(terms['target'], params, retain_graph=True, allow_unused=True)
            vector = torch.cat([(v if v is not None else torch.zeros_like(p)).detach().flatten()
                                for p, v in zip(params, values)])
            assert torch.isfinite(vector).all() and vector.norm() > 0
            gradients[sid, name] = vector
    conflict = {}
    for name in gradient_groups:
        a, b = [gradients[sid, name] for sid in LANGUAGE_SAMPLES]
        conflict[name] = {'cosine': float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                          'paired_norm_over_individual_norm_sum': float((a+b).norm()/(a.norm()+b.norm()))}
    return {'optimizer_updates': 0, 'matching_cost_gaps_all32': gaps,
            'max_bce_derivative_error': max(derivative_errors), 'target_losses': losses,
            'gradient_conflict': conflict, 'finite_nonzero_language_to_arrangement_to_perception_gradients': True,
            'positive_weight': 1., 'negative_weight': .1, 'queries': 300,
            'direct_final_classification_coefficient': .25,
            'native_classifier': 'pretrained80 classes retained; max class binary objectness, not native VFL/GIoU/denoising',
            'existing_sampler': 'one unit per family, one member/epoch; paired members not co-batched',
            'proposed_pair_weighting': 'both existing members, each half family weight; no claim cancellation is solved',
            'paired_sampler_weight_test': 'tests/neural/test_validation_gates.py'}


def micro(model, cache, cfg, phase, folder, base_sha):
    folder.mkdir()
    runner.seed_everything(cfg['seed'], cfg['cpu_threads'])
    configure(model, phase)
    trainable = {n for n, p in model.named_parameters(remove_duplicate=False) if p.requires_grad}
    frozen = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()
              if n not in trainable}
    optimizer = runner.make_optimizer(model, cfg)
    criterion = S0Criterion()
    ids = DETECTION_SAMPLES if phase == 'detection' else LANGUAGE_SAMPLES
    initial = evaluate_cached(model, cache, list(ids))
    write(folder/'initial.json', initial)
    history, updates = [], 0
    start = time.monotonic()
    while step_allowed(updates, MICRO_LIMIT):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for sid in ids:
            c = cache[sid]
            p = cached_output(model, c, targets=(phase == 'language'))
            terms, _ = criterion(p, c['batch'].supervision)
            loss = terms['detection' if phase == 'detection' else 'target'] / len(ids)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite micro loss')
            loss.backward()
            total_loss += float(loss.detach())
        norm = optimizer_step(model, optimizer, updates, MICRO_LIMIT)
        updates += 1
        runner.append_event(folder/'optimizer.jsonl', {'update': updates, 'loss': total_loss, 'gradient_norm': norm})
        if updates % 10 == 0:
            result = evaluate_cached(model, cache, list(ids))
            gate = result['detection'] if phase == 'detection' else result['language']
            history.append(gate)
            result.update(update=updates, gate_pass=(detection_pass if phase == 'detection' else language_pass)(gate))
            write(folder/f'eval-{updates:03d}.json', result)
            print(phase, updates, 'gate', result['gate_pass'], json.dumps(gate), flush=True)
    predicate = detection_pass if phase == 'detection' else language_pass
    state = model.state_dict()
    immutable = all(torch.equal(v, state[n].cpu()) for n, v in frozen.items())
    assert immutable, 'a frozen parameter or running statistic changed'
    result = {'phase': phase, 'optimizer_updates': updates, 'passed': stable(history, predicate),
              'frozen_state_unchanged': immutable, 'elapsed_seconds': time.monotonic()-start,
              'final_gate_metrics': history[-1], 'retained_evaluations': history[-3:],
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    save_probe(folder/'checkpoint.pt', model, optimizer, phase, updates, base_sha)
    write(folder/'result.json', result)
    return result


def make_cache(ds, model, collator):
    cache = {}
    with torch.no_grad():
        for i in range(len(ds)):
            b = collator([ds[i]]).to('cuda')
            p = model(**b.inputs)
            features = model.text_encoder(input_ids=b.inputs['input_ids'],
                                          attention_mask=b.inputs['attention_mask']).last_hidden_state
            sid = b.metadata[0]['sample_id']
            cache[sid] = {'output': detach(p), 'text_features': features.detach(), 'batch': b}
            replay = cached_output(model, cache[sid])
            torch.testing.assert_close(replay['source']['objectness_logits'], p['source']['objectness_logits'], atol=1e-6, rtol=1e-6)
            for a, z in zip(replay['targets'], p['targets']):
                for key in ['uv', 'angle']:
                    torch.testing.assert_close(a[key], z[key], atol=1e-6, rtol=1e-6)
    return cache


def pilot(model, ds, collator, cfg, baseline, folder, base_sha):
    folder.mkdir()
    runner.seed_everything(cfg['seed'], cfg['cpu_threads'])
    configure(model, 'pilot')
    optimizer, criterion = runner.make_optimizer(model, cfg), S0Criterion()
    updates, epoch, retained = 0, 0, []
    names = sorted({r['scene_family_id'] for r in ds.records})
    while updates < PILOT_LIMIT:
        generator = torch.Generator().manual_seed(cfg['seed']+epoch)
        shuffled = [names[i] for i in torch.randperm(len(names), generator=generator).tolist()]
        for group in family_groups(ds.records, shuffled):
            if not step_allowed(updates, PILOT_LIMIT):
                break
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(group), 2):
                selected = group[start:start+2]
                b = collator([ds[i] for i, _ in selected]).to('cuda')
                p = model(**b.inputs)  # Full end-to-end graph; no cached tokens.
                loss = sum(criterion(slice_output(p, j), [b.supervision[j]])[0]['loss']*weight
                           for j, (_, weight) in enumerate(selected))
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite pilot loss')
                loss.backward()
            norm = optimizer_step(model, optimizer, updates, PILOT_LIMIT)
            updates += 1
            runner.append_event(folder/'optimizer.jsonl', {'update': updates, 'gradient_norm': norm})
            if updates % 20 == 0:
                cache = make_cache(ds, model, collator)
                result = evaluate_cached(model, cache)
                result.update(update=updates, gate_pass=pilot_pass(result, baseline))
                retained.append(result)
                retained = retained[-3:]
                write(folder/f'eval-{updates:03d}.json', result)
                print('joint pilot', updates, 'gate', result['gate_pass'], flush=True)
                del cache
        epoch += 1
    result = {'optimizer_updates': updates,
              'passed': len(retained) == 3 and all(r['gate_pass'] for r in retained),
              'last_evaluation': retained[-1], 'full_s0_started': False}
    save_probe(folder/'checkpoint.pt', model, optimizer, 'pilot', updates, base_sha)
    write(folder/'result.json', result)
    return result


def register_terminal_run(out):
    """Use the existing notify.py watcher registration; never send directly."""
    path = Path('/opt/knolling-notify/watch.json')
    previous = json.loads(path.read_text())
    proc = Path(f"/proc/{previous['pid']}/stat")
    if proc.exists():
        fields = proc.read_text().rsplit(') ', 1)[1].split()
        if fields[0] != 'Z' and fields[19] == previous['start_ticks']:
            raise RuntimeError('another live experiment owns notification registration')
    write(out/'previous-notification-registration.json', previous)
    fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[1].split()
    registration = {'run_dir': str(out.resolve()), 'pid': os.getpid(),
                    'start_ticks': fields[19],
                    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    'total': 400}
    temp = path.with_suffix('.validation.tmp')
    temp.write_text(json.dumps(registration)+'\n')
    temp.chmod(0o600)
    temp.replace(path)
    write(out/'notification-registration.json', registration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--register-terminal-notifications', action='store_true')
    args = parser.parse_args()
    cfg = runner.read_config('configs/s0_rtdetr.json')
    runner.protect_dataset_output(args.output, cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    plan = protocol() | {'created_at': datetime.now(ZoneInfo('America/New_York')).isoformat(),
                         'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': sha(args.checkpoint),
                         'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    write(out/'protocol.json', plan)
    protocol_sha = sha(out/'protocol.json')
    files = [p for root in [Path(cfg['dataset']), Path('artifacts/s0/runs'), Path('artifacts/s0_diagnosis')]
             for p in root.rglob('*') if p.is_file()]
    before = hashes(files)
    write(out/'preserved-hashes.json', before)
    final = {'decision': 'NO-GO', 'full_s0_started': False, 's1_started': False,
             'not_official_s0': True, 's0_gate': False, 'optimizer_updates': {},
             'phase3': 'not_admitted', 'protocol_sha256': protocol_sha}
    try:
        if args.register_terminal_notifications:
            register_terminal_run(out)
        print('Phase1: protocol recorded; read-only dataset audit...', flush=True)
        write(out/'dataset-verification.json', verify(Path(cfg['dataset']), Path(cfg['selection'])))
        assert torch.__version__ == '2.9.1+cu128' and '4090' in torch.cuda.get_device_name(0)
        ds, model, collator = runner.prepare(cfg, 'cuda', True)
        saved = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
        assert saved['provenance']['source_sha256'] == runner.provenance(cfg)['source_sha256']
        mismatch = model.load_state_dict(saved['model'], strict=False)
        assert not mismatch.unexpected_keys and all(n.startswith('text_encoder.') for n in mismatch.missing_keys)
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        cache = make_cache(ds, model, collator)
        configure(model, 'detection')
        zero = zero_update(model, cache, ds, cfg)
        assert all(torch.equal(p.cpu(), saved['model'][n]) for n, p in model.state_dict().items()
                   if n in saved['model'])
        write(out/'phase1.json', zero)
        baseline = evaluate_cached(model, cache)
        write(out/'baseline-all32.json', baseline)
        print('Phase1 passed; micro A: existing final classifier only.', flush=True)
        a = micro(model, cache, cfg, 'detection', out/'detection', plan['checkpoint_sha256'])
        final['optimizer_updates']['detection'] = a['optimizer_updates']
        all32a = evaluate_cached(model, cache)
        write(out/'detection'/'all32-after.json', all32a)
        a['all32_count_retained'] = all32a['detection']['count_correct'] >= baseline['detection']['count_correct']
        a['passed'] = a['passed'] and a['all32_count_retained']
        write(out/'detection'/'result.json', a)
        if not a['passed']:
            final['blocker'] = 'Detection micro gate failed: later phases not executed.'
        else:
            a_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()
                       if 'decoder.class_embed.2.' in n}
            model.load_state_dict(saved['model'], strict=False)
            print('Micro A passed; micro B: predicted Source, paired Target loss.', flush=True)
            b = micro(model, cache, cfg, 'language', out/'language', plan['checkpoint_sha256'])
            final['optimizer_updates']['language'] = b['optimizer_updates']
            write(out/'language'/'all32-after.json', evaluate_cached(model, cache))
            if not b['passed']:
                final['blocker'] = 'Language/Target micro gate failed: joint pilot not executed.'
            else:
                # Compose only verified micro changes; never load micro optimizer states.
                state = model.state_dict()
                assert a_state
                for n, v in a_state.items():
                    state[n].copy_(v)
                del cache
                torch.cuda.empty_cache()
                final['phase3'] = 'admitted_after_both_micro_gates'
                c = pilot(model, ds, collator, cfg, baseline, out/'joint', plan['checkpoint_sha256'])
                final['optimizer_updates']['joint'] = c['optimizer_updates']
                final['decision'] = 'GO' if c['passed'] else 'NO-GO'
                final['blocker'] = None if c['passed'] else 'Joint pilot metric gate failed.'
        final['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        final['stop_reason'] = 'update_budget'
    except Exception:
        (out/'error.log').write_text(traceback.format_exc())
        final['stop_reason'] = 'validation_error'
        final['blocker'] = 'Validation execution failed; inspect error.log; no later phase admitted.'
        raise
    finally:
        after = hashes(before)
        changed = [p for p in before if before[p] != after[p]]
        final['preservation'] = {'files_checked': len(before), 'changed': changed,
                                  'protocol_unchanged': protocol_sha == sha(out/'protocol.json')}
        if changed or not final['preservation']['protocol_unchanged']:
            final.update(decision='NO-GO', stop_reason='validation_error', blocker='Immutable evidence changed.')
        write(out/'result.json', final)
        print(json.dumps(final, indent=2), flush=True)
    return decision_exit_code(final['decision'])


if __name__ == '__main__':
    raise SystemExit(main())
