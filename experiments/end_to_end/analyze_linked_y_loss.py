"""Why the linked-Y pilot broke successes, and whether Y alone can add any. No training."""
import json
from collections import Counter
import torch

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from counterfactual_color_pose import legal_row_set
from knolling_dataset.schema import Pose, REGIONS, WIDTH_M, HEIGHT_M
from knolling_dataset.validator import geometry, inferred_rows, polygon
from knolling_model.data import Batch
from knolling_model.diagnostics import positioned_objects
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import poses_from_detection
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
from score_end_to_end import load_adopted
from train_linked_y import VAL_COUNT, VAL_SEED
from train_set_context import fresh_split

UNFIXABLE = {'count', 'source_identity', 'horizontal_gap', 'mixed_color_row', 'size_order',
             'axis_alignment', 'noncontiguous_color_group', 'row_alignment'}


def collect(system, relation, model, head, data, indices):
    found = []
    for index in indices:
        record = data.records[index]
        image = data[index]['image'][None].to(stable.DEVICE)
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        try:
            output, objects, visual = poses_from_detection(
                system, relation, model, image, [record['instruction']['text']], head)
        except ValueError:
            found.append(dict(sample_id=record['sample_id'], task=False, errors=['count'], poses=None))
            continue
        source = {key: value.detach().cpu() for key, value in visual['source'].items()}
        layout = dict(xy=output['xy'].detach().cpu(), angle=output['angle'].detach().cpu(),
                      valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                      rollout='free_rollout')
        judged = evaluate_batch(legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5)),
                                Batch({}, [truth], [dict(sample_id=record['sample_id'], object_ids=identities,
                                                         record=record)]),
                                semantic_check=semantic_errors_v3)[0]
        poses = judged.get('predicted_targets')
        found.append(dict(sample_id=record['sample_id'], task=bool(accepted_success(record, judged['task_success'])),
                          errors=list(judged['errors']), poses=poses, record=record))
    return found


def openings(rows, record):
    values = []
    gap = record['instruction']['gap_m']
    for upper, lower in zip(rows, rows[1:]):
        bottom = min(polygon(obj, obj.target_pose).bounds[1] for obj in upper)
        top = max(polygon(obj, obj.target_pose).bounds[3] for obj in lower)
        factor = 2 if record['instruction']['group_by'] == 'color' and upper[0].color != lower[0].color else 1
        values.append((bottom-top, factor*gap))
    return values


def solve_y(record, poses):
    """Existential Y: exact edge gaps, X and angle fixed, anchor only if it fits."""
    objects = positioned_objects(record, poses)
    current = inferred_rows(objects)
    legal, _sets = legal_row_set(record)
    key = tuple(tuple(obj.object_id for obj in row) for row in current)
    if key not in legal:
        return None, 'illegal_row'
    if any(name in UNFIXABLE for name in arrangement_errors(record, poses)['errors']):
        return None, 'not_y'
    local = []
    for row in current:
        extents = []
        for obj in row:
            shifted = polygon(obj, Pose(obj.target_pose.x_m, 0., obj.target_pose.theta_rad))
            extents.append((shifted.bounds[1], shifted.bounds[3]))
        local.append(extents)
    gap = record['instruction']['gap_m']
    deltas = [0.]
    for index, (upper, lower) in enumerate(zip(current, current[1:])):
        factor = 2 if record['instruction']['group_by'] == 'color' and upper[0].color != lower[0].color else 1
        upper_min = min(item[0] for item in local[index])
        lower_max = max(item[1] for item in local[index+1])
        deltas.append(deltas[-1]-(factor*gap-upper_min+lower_max))
    region = REGIONS[record['instruction']['region']]
    low, high = -1., 1.
    for row, extents, delta in zip(current, local, deltas):
        for obj, (ymin, ymax) in zip(row, extents):
            low = max(low, region[1]-delta-ymin, .010-delta-ymin)
            high = min(high, region[3]-delta-ymax, HEIGHT_M-.010-delta-ymax)
            if obj.target_pose.x_m+polygon(obj, Pose(0., 0., obj.target_pose.theta_rad)).bounds[0] < .010:
                return None, 'x_boundary'
            if obj.target_pose.x_m+polygon(obj, Pose(0., 0., obj.target_pose.theta_rad)).bounds[2] > WIDTH_M-.010:
                return None, 'x_boundary'
    if low > high:
        return None, 'no_vertical_room'
    current_anchor = current[0][0].target_pose.y_m-deltas[0]
    anchor = min(max(current_anchor, low), high)
    updated = dict(poses)
    for row, delta in zip(current, deltas):
        for obj in row:
            pose = updated[obj.object_id]
            updated[obj.object_id] = dict(x_m=pose['x_m'], y_m=anchor+delta, theta_rad=pose['theta_rad'])
    detail = arrangement_errors(record, updated)
    return (updated, 'pass') if detail['task'] else (None, 'still:'+','.join(detail['errors']))


def hinge_probe(record, poses):
    objects = positioned_objects(record, poses)
    rows = inferred_rows(objects)
    if len(rows) < 2:
        return None
    upper, lower = rows[0], rows[1]
    gap = record['instruction']['gap_m']
    factor = 2 if record['instruction']['group_by'] == 'color' and upper[0].color != lower[0].color else 1
    target = factor*gap
    y = torch.tensor([upper[0].target_pose.y_m, lower[0].target_pose.y_m], dtype=torch.float64, requires_grad=True)
    upper_min = min(polygon(obj, Pose(obj.target_pose.x_m, 0., obj.target_pose.theta_rad)).bounds[1] for obj in upper)
    lower_max = max(polygon(obj, Pose(obj.target_pose.x_m, 0., obj.target_pose.theta_rad)).bounds[3] for obj in lower)
    opening = (y[0]+upper_min)-(y[1]+lower_max)
    violation = torch.relu((opening-target).abs()-0.002)
    violation.backward()
    return dict(violation_mm=float(violation)*1000, grad_upper=float(y.grad[0]), grad_lower=float(y.grad[1]),
                opening_mm=float(opening)*1000, target_mm=target*1000)


def main():
    runner.setup()
    data = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    fit, dev, _blocked = fresh_split(data.records)
    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(VAL_SEED)).tolist()
    indices = [fit[index] for index in order[:VAL_COUNT]]
    system, relation, model, head, _saved = load_adopted()
    before = collect(system, relation, model, head, data, indices)
    saved = torch.load('artifacts/experiments/linked_y/checkpoint.pt', map_location=stable.DEVICE, weights_only=False)
    model.placement.row_anchor.load_state_dict(saved['row_anchor'])
    model.placement.row_gap.load_state_dict(saved['row_gap'])
    after = collect(system, relation, model, head, data, indices)
    lost_cause = Counter()
    gained_fixed = Counter()
    examples = []
    for left, right in zip(before, after):
        if left['task'] and not right['task']:
            added = sorted(set(right['errors'])-set(left['errors']))
            lost_cause[','.join(added) or 'same_errors'] += 1
            if left['poses'] and right['poses']:
                shift = [abs(right['poses'][key]['y_m']-left['poses'][key]['y_m'])*1000
                         for key in left['poses'] if key in right['poses']]
                xshift = [abs(right['poses'][key]['x_m']-left['poses'][key]['x_m'])*1000
                          for key in left['poses'] if key in right['poses']]
                before_gap = openings(inferred_rows(positioned_objects(left['record'], left['poses'])), left['record'])
                after_gap = openings(inferred_rows(positioned_objects(right['record'], right['poses'])), right['record'])
                examples.append(dict(kind='lost', added=added, y_mm=max(shift) if shift else None,
                                     x_mm=max(xshift) if xshift else None,
                                     before_gap_error_mm=[abs(value-target)*1000 for value, target in before_gap],
                                     after_gap_error_mm=[abs(value-target)*1000 for value, target in after_gap]))
        elif right['task'] and not left['task']:
            gained_fixed[','.join(sorted(set(left['errors'])-set(right['errors']))) or 'none'] += 1
    gains = Counter()
    hinge = None
    kept = sum(item['task'] for item in before)
    for item in before:
        if item['task'] or not item['poses']:
            if hinge is None and item['task'] and item['poses']:
                probe = hinge_probe(item['record'], item['poses'])
                if probe is not None:
                    hinge = dict(success=probe)
            continue
        _poses, reason = solve_y(item['record'], item['poses'])
        gains[reason] += 1
        if hinge is None and reason == 'pass':
            hinge = dict(failure=hinge_probe(item['record'], item['poses']))
    if 'failure' not in (hinge or {}) :
        for item in before:
            if item['poses'] and 'vertical_gap' in item['errors']:
                hinge = dict(hinge or {}, failure=hinge_probe(item['record'], item['poses']))
                break
    print(json.dumps(dict(
        before=kept, lost=sum(lost_cause.values()), gained=sum(gained_fixed.values()),
        lost_new_errors=dict(lost_cause), gained_cleared_errors=dict(gained_fixed),
        lost_examples=examples[:8],
        y_only_new_successes=gains['pass'], y_only_outcomes=dict(gains),
        hinge=hinge), indent=2))


if __name__ == '__main__':
    main()
