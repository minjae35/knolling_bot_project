"""Classify requested-region failures and test one translation into the region.

Does not open the test set and does not read a target pose during inference.
"""
import json
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
from knolling_dataset.schema import REGIONS
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.geometry_decoder import _instruction, apply_geometry, extent_offsets
from knolling_model.recovery.data import recovery_supervision
from dev_splits import family_splits, fresh_split, heldout_validation
from score_end_to_end import DATA
from train_selector_full import blank, judge_scene, load_adopted, perceive
from confirm_axis_alignment import forward

TOL = 0.0001
INSETS = (0.0, 0.001)


def _boxes(xy, angle, major, minor):
    boxes = []
    for index in range(len(xy)):
        xmin, xmax, ymin, ymax = extent_offsets(angle[index], major[index], minor[index])
        boxes.append((
            float(xy[index, 0] + xmin), float(xy[index, 1] + ymin),
            float(xy[index, 0] + xmax), float(xy[index, 1] + ymax)))
    return boxes


def _limits(record):
    left, bottom, right, top = REGIONS[record['instruction']['region']]
    return left - TOL, bottom - TOL, right + TOL, top + TOL


def _shift(boxes, limits, inset):
    left, bottom, right, top = limits
    left, bottom, right, top = left + inset, bottom + inset, right - inset, top - inset
    if left > right or bottom > top:
        return None
    dx_low = max(left - xmin for xmin, _ymin, _xmax, _ymax in boxes)
    dx_high = min(right - xmax for _xmin, _ymin, xmax, _ymax in boxes)
    dy_low = max(bottom - ymin for _xmin, ymin, _xmax, _ymax in boxes)
    dy_high = min(top - ymax for _xmin, _ymin, _xmax, ymax in boxes)
    if dx_low > dx_high + 1e-9 or dy_low > dy_high + 1e-9:
        return None

    def pick(low, high):
        if low <= 0 <= high:
            return 0.0
        return low if low > 0 else high

    return pick(dx_low, dx_high), pick(dy_low, dy_high)


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


def _moved(base, dx, dy, count):
    shifted = dict(base)
    xy = base['xy'].clone()
    xy[0, :count, 0] += dx
    xy[0, :count, 1] += dy
    shifted['xy'] = xy
    return shifted


def _summarize(rows):
    task = sum(row['task'] for row in rows)
    errors = Counter(name for row in rows if not row['task'] for name in row['errors'])
    sole = Counter(row['errors'][0] for row in rows if not row['task'] and len(row['errors']) == 1)
    return dict(task=task, total=len(rows),
                collision=errors['collision'], boundary=errors['boundary'],
                requested_region=errors['requested_region'],
                sole_region=sole['requested_region'])


def score_split(system, relation, model, head, data, indices, name):
    base_rows = []
    moved_rows = {inset: [] for inset in INSETS}
    kind = Counter()
    overflow = []
    status = Counter()
    regions = Counter()
    for done, index in enumerate(indices, start=1):
        record = data.records[index]
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        image = data[index]['image'][None].to(stable.DEVICE)
        try:
            visual = perceive(system, head, image)
            output, objects, goal = forward(
                system, relation, model, visual, [record['instruction']['text']])
        except ValueError:
            row = blank(record, ['count'], len(identities), 0, '')
            base_rows.append(row)
            for inset in INSETS:
                moved_rows[inset].append(row)
            continue
        current = apply_geometry(output, objects, goal)
        judged = judge_scene(
            current, objects, visual, record, identities, truth, 0, system.threshold, detail=False)
        base_rows.append(judged)
        count, pairs = _match(objects, truth)
        angle = current['angle'][0, :count].detach().cpu()
        xy = current['xy'][0, :count].detach().cpu()
        pred_boxes = _boxes(
            xy, angle, objects.size[0, :count, 0].detach().cpu(), objects.size[0, :count, 1].detach().cpu())
        limits = _limits(record)
        proposals = {}
        for inset in INSETS:
            shift = _shift(pred_boxes, limits, inset)
            if shift is None or (abs(shift[0]) < 1e-9 and abs(shift[1]) < 1e-9):
                moved_rows[inset].append(judged)
            else:
                proposals[inset] = _moved(current, *shift, count)
        for inset, proposal in proposals.items():
            moved_rows[inset].append(judge_scene(
                proposal, objects, visual, record, identities, truth, 0, system.threshold, detail=False))
        sole = (not judged['task']) and judged['errors'] == ['requested_region']
        if sole and len(pairs) == count:
            gt_boxes = _boxes(
                xy, angle,
                torch.stack([truth['size'][gt][0] for _query, gt in sorted(pairs)]),
                torch.stack([truth['size'][gt][1] for _query, gt in sorted(pairs)]))
            # pairs are not in query order. Rebuild by query index.
            major = torch.zeros(count)
            minor = torch.zeros(count)
            for query, gt in pairs:
                major[query] = truth['size'][gt][0]
                minor[query] = truth['size'][gt][1]
            gt_boxes = _boxes(xy, angle, major, minor)
            gt_shift = _shift(gt_boxes, limits, 0.0)
            pred_shift = _shift(pred_boxes, limits, 0.0)
            left, bottom, right, top = limits
            stick = 0.0
            for xmin, ymin, xmax, ymax in gt_boxes:
                stick = max(stick, left - xmin, bottom - ymin, xmax - right, ymax - top, 0.0)
            overflow.append(stick * 1000)
            status[str(current['geometry_status'][0])] += 1
            regions[record['instruction']['region']] += 1
            if gt_shift is None:
                kind['impossible'] += 1
            elif pred_shift is None:
                kind['gt_fits_pred_does_not'] += 1
            elif abs(pred_shift[0]) < 1e-9 and abs(pred_shift[1]) < 1e-9:
                kind['pred_inside_gt_sticks_out'] += 1
            else:
                kind['pred_outside_and_movable'] += 1
        if done % 200 == 0:
            print(name, done, 'base', sum(row['task'] for row in base_rows),
                  {f'{inset:.3f}': sum(row['task'] for row in rows) for inset, rows in moved_rows.items()},
                  flush=True)
    base_ids = {row['sample_id'] for row in base_rows if row['task']}
    report = dict(base=_summarize(base_rows), kind=dict(kind), status=dict(status), regions=dict(regions),
                  overflow_mm=dict(n=len(overflow),
                                   median=None if not overflow else sorted(overflow)[len(overflow) // 2],
                                   p90=None if not overflow else sorted(overflow)[int(len(overflow) * .9)]))
    for inset, rows in moved_rows.items():
        ok = {row['sample_id'] for row in rows if row['task']}
        lost = [row for row in rows if row['sample_id'] in base_ids and row['sample_id'] not in ok]
        report[f'inset_{inset:.3f}'] = dict(
            **_summarize(rows), gained=len(ok - base_ids), lost=len(base_ids - ok),
            lost_errors=dict(Counter(name for row in lost for name in row['errors'])))
    return report


def main():
    runner.setup()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    validation = ExpandedDataset(DATA, 'validation')
    train = ExpandedDataset(DATA, 'train')
    val_full = [index for index, record in enumerate(validation.records) if is_full(record)]
    fit, dev, _blocked = fresh_split(train.records)
    _fit, _dev, hold = family_splits(train.records)
    splits = (
        ('validation', validation, val_full),
        ('dev512', train, dev),
        ('holdout', train, hold))
    print('SPLITS', {name: len(indices) for name, _data, indices in splits}, flush=True)
    system, relation, model, head, _saved = load_adopted()
    report = {}
    for name, data, indices in splits:
        report[name] = score_split(system, relation, model, head, data, indices, name)
        print(name, json.dumps(report[name]), flush=True)
    print('RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
