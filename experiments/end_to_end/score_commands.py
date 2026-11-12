"""Score full, partial, no-op, and counterfactual commands. Does not open the test set."""
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'experiments' / 'end_to_end')]
if os.environ.get('CUDA_VISIBLE_DEVICES', None) == '':
    del os.environ['CUDA_VISIBLE_DEVICES']
os.environ.update(dict(
    HF_HOME='/workspace/cache/huggingface',
    HF_HUB_CACHE='/workspace/cache/huggingface/hub',
    HUGGINGFACE_HUB_CACHE='/workspace/cache/huggingface/hub',
    TRANSFORMERS_CACHE='/workspace/cache/huggingface/hub',
    TORCH_HOME='/workspace/cache/torch',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    CUBLAS_WORKSPACE_CONFIG=':4096:8'))

import torch
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_dataset.schema import angle_error
from knolling_model.coordinates import angle_delta
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.partial_arrangement import arrange_held
from knolling_model.experiments.size_refinement import encode_instruction, poses_from_detection
from knolling_model.recovery.data import recovery_supervision
from score_end_to_end import DATA
from train_selector_full import blank, judge_scene, load_adopted


def _match(objects, truth):
    count = int(objects.valid[0].sum())
    distance = torch.cdist(objects.xy[0, :count].detach().cpu(), truth['source_xy'])
    used_query, used_gt, pairs = set(), set(), []
    for item in distance.flatten().argsort().tolist():
        query, gt = divmod(item, distance.shape[1])
        if query in used_query or gt in used_gt or float(distance[query, gt]) > .010:
            continue
        used_query.add(query)
        used_gt.add(gt)
        pairs.append((query, gt))
    return count, pairs


def _decisions(output, objects, record, identities, truth):
    instruction = record['instruction']
    gt_move = set() if instruction['keep_all'] else set(instruction['selected_ids'])
    count, pairs = _match(objects, truth)
    keep = float(output.get('keep_probability', torch.zeros(1))[0])
    chosen = []
    pose_mm, pose_deg = [], []
    for query, gt in pairs:
        pred_move = keep <= 0.5 and float(output['selected'][0, query]) >= 0.5
        identity = identities[gt]
        chosen.append(pred_move == (identity in gt_move))
        if identity not in gt_move:
            delta = output['xy'][0, query].detach().cpu() - truth['source_xy'][gt]
            pose_mm.append(float(delta.norm()) * 1000)
            pose_deg.append(math.degrees(angle_error(
                float(output['angle'][0, query].detach().cpu()), float(truth['source_angle'][gt]))))
    return dict(
        selection_correct=len(pairs) == count == len(identities) and all(chosen),
        keep_decision=(keep > 0.5) == bool(instruction['keep_all']),
        pose_mm=pose_mm, pose_deg=pose_deg)


def main():
    runner.setup()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    data = ExpandedDataset(DATA, 'validation')
    system, relation, model, head, _saved = load_adopted()
    rows = []
    for index, record in enumerate(data.records):
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        image = data[index]['image'][None].to(stable.DEVICE)
        try:
            output, objects, visual = poses_from_detection(
                system, relation, model, image, [record['instruction']['text']], head)
        except ValueError:
            row = blank(record, ['count'], len(identities), 0, '')
            row.update(mode=record['mode'], route='', selection_correct=False, keep_decision=False,
                       pose_mm=[], pose_deg=[], pose=None, pose_angle=None)
            rows.append(row)
            continue
        judged = judge_scene(
            output, objects, visual, record, identities, truth, 0, system.threshold, detail=False)
        count = int(objects.valid[0].sum())
        judged.update(
            mode=record['mode'], route=str(output.get('route', '')),
            pose=output['xy'][0, :count].detach().cpu(),
            pose_angle=output['angle'][0, :count].detach().cpu(),
            **_decisions(output, objects, record, identities, truth))
        if record['mode'] == 'no_op' and output.get('route') == 'noop':
            judged['copy_xy'] = float((output['xy'][0, :count] - objects.xy[0, :count]).abs().max())
            judged['copy_angle'] = float((output['angle'][0, :count] - objects.angle[0, :count]).abs().max())
            signed = []
            count_m, pairs = _match(objects, truth)
            for query, gt in pairs:
                signed.append(float(angle_delta(
                    output['angle'][0, query].detach().cpu(), truth['source_angle'][gt])) * 180 / math.pi)
            judged['signed_deg'] = signed
        if record['mode'] == 'partial' and output.get('route'):
            text, mask = encode_instruction(system, [record['instruction']['text']], image.device)
            encoded = system.layout.goal_encoder(objects, text, mask)
            gt_move = set() if record['instruction']['keep_all'] else set(record['instruction']['selected_ids'])
            count_m, pairs = _match(objects, truth)
            if len(pairs) == count_m == len(identities):
                chosen = torch.zeros_like(encoded['selected'])
                for query, gt in pairs:
                    chosen[0, query] = 1. if identities[gt] in gt_move else 0.
                replaced = dict(encoded)
                replaced['selected'] = chosen
                replaced['keep_logit'] = encoded['keep_logit'].new_full(encoded['keep_logit'].shape, -8.)
                oracle = arrange_held(relation, model, objects, replaced)
                judged['oracle_selection'] = judge_scene(
                    oracle, objects, visual, record, identities, truth, 0, system.threshold, detail=False)['task']
                patched = dict(output)
                patched['xy'] = output['xy'].clone()
                patched['angle'] = output['angle'].clone()
                for query, gt in pairs:
                    if identities[gt] not in gt_move:
                        patched['xy'][0, query] = truth['source_xy'][gt].to(patched['xy'].device)
                        patched['angle'][0, query] = truth['source_angle'][gt].to(patched['angle'].device)
                judged['oracle_pose'] = judge_scene(
                    patched, objects, visual, record, identities, truth, 0, system.threshold, detail=False)['task']
        rows.append(judged)
        if (index + 1) % 250 == 0:
            print('progress', index + 1, 'task', sum(row['task'] for row in rows), flush=True)
    by_id = {}
    summary = {}
    for mode in ('full', 'partial', 'no_op'):
        chosen = [row for row, record in zip(rows, data.records) if record['mode'] == mode and (
            mode != 'full' or is_full(record))]
        errors = Counter(name for row in chosen if not row['task'] for name in row['errors'])
        summary[mode] = dict(
            task=sum(row['task'] for row in chosen), total=len(chosen),
            collision=errors['collision'], boundary=errors['boundary'], clearance=errors['clearance'],
            keep=errors['keep'],
            selection_correct=sum(row['selection_correct'] for row in chosen),
            keep_decision=sum(row['keep_decision'] for row in chosen),
            route=dict(Counter(row['route'] for row in chosen)),
            errors=dict(errors.most_common(8)))
        held = [value for row in chosen for value in row['pose_mm']]
        held_deg = [value for row in chosen for value in row['pose_deg']]
        summary[mode]['held_mm_median'] = None if not held else sorted(held)[len(held) // 2]
        summary[mode]['held_deg_median'] = None if not held_deg else sorted(held_deg)[len(held_deg) // 2]
        summary[mode]['held_within_1mm_1deg'] = (
            None if not held else sum(mm <= 1 and deg <= 1 for mm, deg in zip(held, held_deg)) / len(held))
        for row in chosen:
            by_id[row['sample_id']] = row
    contrasts = Counter()
    contrast_pass = Counter()
    for pair in data.pairs:
        ids = pair['sample_ids']
        if any(sample not in by_id for sample in ids):
            continue
        left, right = by_id[ids[0]], by_id[ids[1]]
        group = 'selection' if pair['factor'] == 'selection' else 'arrangement'
        contrasts[group] += 1
        same_shape = left['pose'] is not None and right['pose'] is not None and left['pose'].shape == right['pose'].shape
        moved = (not same_shape) or float((left['pose'] - right['pose']).abs().mean()) > 1e-3 or float(
            (left['pose_angle'] - right['pose_angle']).abs().mean()) > 1e-3
        if left['task'] and right['task'] and moved:
            contrast_pass[group] += 1
    summary['counterfactual'] = {
        group: dict(passed=contrast_pass[group], total=contrasts[group]) for group in ('arrangement', 'selection')}
    # Selection-correct partials still failing are geometry, not the selector.
    partial = [row for row, record in zip(rows, data.records) if record['mode'] == 'partial']
    wrong = [row for row in partial if not row['selection_correct']]
    right = [row for row in partial if row['selection_correct'] and not row['task']]
    noop = [row for row in rows if row['mode'] == 'no_op' and row.get('route') == 'noop']
    signed = [value for row in noop for value in row.get('signed_deg', [])]
    summary['noop_angle'] = dict(
        copy_exact=sum(row.get('copy_xy', 1) < 1e-6 and row.get('copy_angle', 1) < 1e-6 for row in noop),
        copy_n=len(noop),
        signed_median=None if not signed else sorted(signed)[len(signed) // 2],
        abs_median=None if not signed else sorted(abs(v) for v in signed)[len(signed) // 2])
    partial_rows = [row for row in rows if row['mode'] == 'partial']
    summary['oracle'] = dict(
        selection=sum(row.get('oracle_selection', False) for row in partial_rows),
        pose=sum(row.get('oracle_pose', False) for row in partial_rows),
        scored=sum('oracle_selection' in row for row in partial_rows))
    summary['partial_split'] = dict(
        selection_wrong=len(wrong),
        selection_wrong_errors=dict(Counter(name for row in wrong if not row['task'] for name in row['errors']).most_common(6)),
        selection_right_but_failed=len(right),
        geometry_errors=dict(Counter(name for row in right for name in row['errors']).most_common(8)))
    print('RESULT', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
