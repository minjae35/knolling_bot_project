"""Score full, partial, no-op, and counterfactual commands on validation, dev, and a reserved family slice.

Does not open the test set. Ground truth is used only to judge.
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict
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
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.partial_arrangement import HELD_BELOW
from knolling_model.experiments.size_refinement import poses_from_detection
from knolling_model.recovery.data import recovery_supervision
from dev_splits import families, fresh_split
from score_end_to_end import DATA, blank, judge_scene, load_adopted


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
    chosen, pose_mm, pose_deg = [], [], []
    for query, gt in pairs:
        pred_move = keep <= 0.5 and float(output['selected'][0, query]) >= HELD_BELOW
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


def score(system, relation, model, head, data, indices):
    rows = []
    chosen = [data.records[index] for index in indices]
    for number, (index, record) in enumerate(zip(indices, chosen)):
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        image = data[index]['image'][None].to(stable.DEVICE)
        try:
            output, objects, visual = poses_from_detection(
                system, relation, model, image, [record['instruction']['text']], head)
        except ValueError:
            row = blank(record, ['count'], len(identities), 0, '')
            row.update(mode=record['mode'], route='', selection_correct=False, keep_decision=False,
                       pose_mm=[], pose_deg=[], pose=None, pose_angle=None, copy_xy=None, copy_angle=None)
            rows.append(row)
            continue
        judged = judge_scene(
            output, objects, visual, record, identities, truth, 0, system.threshold, detail=False)
        count = int(objects.valid[0].sum())
        copy_xy = copy_angle = None
        if output.get('route') == 'noop':
            copy_xy = float((output['xy'][0, :count] - objects.xy[0, :count]).abs().max())
            copy_angle = float((output['angle'][0, :count] - objects.angle[0, :count]).abs().max())
        judged.update(
            mode=record['mode'], route=str(output.get('route', '')),
            pose=output['xy'][0, :count].detach().cpu(),
            pose_angle=output['angle'][0, :count].detach().cpu(),
            copy_xy=copy_xy, copy_angle=copy_angle,
            **_decisions(output, objects, record, identities, truth))
        rows.append(judged)
        if (number + 1) % 400 == 0:
            print('progress', number + 1, 'task', sum(row['task'] for row in rows), flush=True)
    summary = {}
    by_id = {}
    for mode in ('full', 'partial', 'no_op'):
        picked = [row for row, record in zip(rows, chosen) if record['mode'] == mode and (
            mode != 'full' or is_full(record))]
        errors = Counter(name for row in picked if not row['task'] for name in row['errors'])
        held = [value for row in picked for value in row['pose_mm']]
        held_deg = [value for row in picked for value in row['pose_deg']]
        copies = [row for row in picked if row.get('route') == 'noop']
        summary[mode] = dict(
            task=sum(row['task'] for row in picked), total=len(picked),
            collision=errors['collision'], boundary=errors['boundary'], clearance=errors['clearance'],
            keep=errors['keep'],
            selection_correct=sum(row['selection_correct'] for row in picked),
            keep_decision=sum(row['keep_decision'] for row in picked),
            route=dict(Counter(row['route'] for row in picked)),
            errors=dict(errors.most_common(6)),
            held_mm_median=None if not held else sorted(held)[len(held) // 2],
            held_deg_median=None if not held_deg else sorted(held_deg)[len(held_deg) // 2],
            held_within=None if not held else sum(mm <= 1 and deg <= 1 for mm, deg in zip(held, held_deg)) / len(held),
            xy_exact=sum(row.get('copy_xy') is not None and row['copy_xy'] < 1e-6 for row in copies),
            angle_copy_median=None if not copies or copies[0].get('copy_angle') is None else sorted(
                row['copy_angle'] for row in copies if row.get('copy_angle') is not None)[len(copies) // 2],
            copy_n=len(copies))
        for row in picked:
            by_id[row['sample_id']] = row
    contrasts, passed = Counter(), Counter()
    for pair in data.pairs:
        ids = pair['sample_ids']
        if any(sample not in by_id for sample in ids):
            continue
        left, right = by_id[ids[0]], by_id[ids[1]]
        group = 'selection' if pair['factor'] == 'selection' else 'arrangement'
        contrasts[group] += 1
        same = left['pose'] is not None and right['pose'] is not None and left['pose'].shape == right['pose'].shape
        moved = (not same) or float((left['pose'] - right['pose']).abs().mean()) > 1e-3 or float(
            (left['pose_angle'] - right['pose_angle']).abs().mean()) > 1e-3
        if left['task'] and right['task'] and moved:
            passed[group] += 1
    summary['counterfactual'] = {
        group: dict(passed=passed[group], total=contrasts[group]) for group in ('arrangement', 'selection')}
    return summary


def main():
    runner.setup()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    system, relation, model, head, _saved = load_adopted()
    validation = ExpandedDataset(DATA, 'validation')
    wanted = set(sys.argv[1:]) or {'validation', 'dev', 'reserved'}
    print('SPLIT validation', flush=True)
    print('RESULT validation', json.dumps(score(
        system, relation, model, head, validation, list(range(len(validation))))), flush=True)
    if not ({'dev', 'reserved'} & wanted):
        return
    train = ExpandedDataset(DATA, 'train')
    grouped = defaultdict(list)
    for index, record in enumerate(train.records):
        grouped[record['scene_family_id']].append(index)
    _grouped, trainable, reserved, _blocked = families(train.records)
    _fit, dev512, _blocked_names = fresh_split(train.records)
    dev_names = {train.records[index]['scene_family_id'] for index in dev512}
    if dev_names & set(reserved) or dev_names & set(trainable):
        raise RuntimeError('dev families overlap the training or reserved slice')
    for label, names in (('dev', dev_names), ('reserved', reserved)):
        if label not in wanted:
            continue
        indices = [index for name in sorted(names) for index in grouped[name]]
        print('SPLIT', label, 'records', len(indices), flush=True)
        print('RESULT', label, json.dumps(score(system, relation, model, head, train, indices)), flush=True)


if __name__ == '__main__':
    main()
