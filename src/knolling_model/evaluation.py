"""Read-only S0 prediction evaluation; GT is never passed into the model."""
from collections import Counter
from dataclasses import replace
import math
import numpy as np
import torch

from knolling_dataset.schema import scene_from_dict, Pose, REGIONS, angle_error
from knolling_dataset.validator import polygon, geometry, inferred_rows
from .coordinates import to_meters, angle_delta
from .losses import match_source
from .diagnostics import condition_labels, success_summary, summarize_diagnostics


def semantic_errors(objects, record):
    """Prediction tolerances, intentionally separate from exact dataset checks."""
    ins = record['instruction']
    by_id = {o.object_id: o for o in objects}
    errors = []
    locks = list(by_id) if record['mode'] == 'no_op' else ins['locked_ids']
    for oid in locks:
        obj = by_id[oid]
        if (math.hypot(obj.target_pose.x_m-obj.source_pose.x_m, obj.target_pose.y_m-obj.source_pose.y_m) > .003
                or angle_error(obj.target_pose.theta_rad, obj.source_pose.theta_rad) > math.radians(3)):
            errors.append('keep')
    if ins['keep_all']:
        return sorted(set(errors))
    selected = [by_id[oid] for oid in ins['selected_ids']]
    expected = 0. if ins['axis_alignment'] == 'horizontal' else math.pi/2
    l, b, r, t = REGIONS[ins['region']]
    for obj in selected:
        if angle_error(obj.target_pose.theta_rad, expected) > math.radians(5):
            errors.append('axis_alignment')
        xmin, ymin, xmax, ymax = polygon(obj, obj.target_pose).bounds
        if xmin < l-.0001 or ymin < b-.0001 or xmax > r+.0001 or ymax > t+.0001:
            errors.append('requested_region')
    rows = inferred_rows(selected)
    colors = []
    for row in rows:
        representative = (max(o.target_pose.y_m for o in row)+min(o.target_pose.y_m for o in row))/2
        if any(abs(o.target_pose.y_m-representative) > .0051 for o in row):
            errors.append('row_alignment')
        if ins['group_by'] == 'color':
            if len({o.color for o in row}) != 1:
                errors.append('mixed_color_row')
            colors.append(row[0].color)
        for left, right in zip(row, row[1:]):
            gap = polygon(right, right.target_pose).bounds[0]-polygon(left, left.target_pose).bounds[2]
            if abs(gap-ins['gap_m']) > .0051:
                errors.append('horizontal_gap')
    runs = [c for i, c in enumerate(colors) if i == 0 or c != colors[i-1]]
    if len(runs) != len(set(runs)):
        errors.append('noncontiguous_color_group')
    for up, down in zip(rows, rows[1:]):
        gap = min(polygon(o, o.target_pose).bounds[1] for o in up)-max(polygon(o, o.target_pose).bounds[3] for o in down)
        multiplier = 2 if ins['group_by'] == 'color' and up[0].color != down[0].color else 1
        if abs(gap-multiplier*ins['gap_m']) > .0051:
            errors.append('vertical_gap')
    flat = [o for row in rows for o in row]
    groups = [[o for o in flat if o.color == c] for c in ('red', 'green', 'blue')] if ins['group_by']=='color' else [flat]
    for group in groups:
        areas = [o.asset.a_m*o.asset.b_m for o in group]
        for a, b in zip(areas, areas[1:]):
            if (ins['sort_direction']=='asc' and a>1.05*b+1e-12 or ins['sort_direction']=='desc' and b>1.05*a+1e-12):
                errors.append('size_order')
    return sorted(set(errors))


@torch.no_grad()
def evaluate_batch(output, batch):
    rows = []
    for b, (truth, meta) in enumerate(zip(batch.supervision, batch.metadata)):
        source = {k: v[b:b+1] for k, v in output['source'].items()}
        active = torch.where(source['objectness_logits'][0].sigmoid() >= .5)[0]
        n = len(truth['source_xy'])
        row = {'sample_id': meta['sample_id'], 'mode': meta['record']['mode'], 'gt_count': n,
               'conditions': condition_labels(meta['record']), 'keep_violation': None,
               'predicted_count': len(active), 'count_correct': len(active)==n, 'matched_objects': 0,
               'source_position_mm': [], 'source_size_mm': [], 'source_angle_deg': [],
               'target_position_mm': [], 'target_angle_deg': [], 'stages': [],
               'raw_geometry_success': False, 'task_success': False, 'errors': []}
        if len(active) != n:
            row['errors'] = ['count']
            rows.append(row)
            continue
        limited = {k: v[:, active] for k, v in source.items()}
        q, gt = match_source(limited, [truth])[0]
        q = active[q]
        pairs = sorted(zip(q.tolist(), gt.tolist()), key=lambda pair: pair[1])
        objects = {o.object_id: o for o in scene_from_dict(meta['record']['source']).objects}
        good, mapped = [], []
        for query, i in pairs:
            original = objects[meta['object_ids'][i]]
            xy = to_meters(source['uv'][0, query])
            size = source['size'][0, query]
            angle = source['angle'][0, query]
            # Predicted source envelope, with predicted dimensions, for identity gating.
            # Asset schema forbids out-of-range predictions; compute this rectangle directly.
            from shapely.geometry import Polygon
            a, bb = float(size[0])/2, float(size[1])/2
            c, s = math.cos(float(angle)), math.sin(float(angle))
            p = Polygon([(float(xy[0])+c*x-s*y, float(xy[1])+s*x+c*y) for x,y in ((-a,-bb),(a,-bb),(a,bb),(-a,bb))])
            g = polygon(original, original.source_pose)
            iou = p.intersection(g).area / max(p.union(g).area, 1e-12)
            distance = float(torch.linalg.vector_norm(xy-truth['source_xy'][i]))
            valid = distance <= .010 and iou >= .5
            good.append(valid)
            mapped.append((query, i, original))
            if valid:
                row['source_position_mm'].append(distance*1000)
                row['source_size_mm'].append(float((size-truth['size'][i]).abs().max())*1000)
                row['source_angle_deg'].append(math.degrees(float(angle_delta(angle, truth['source_angle'][i]).abs())))
        row['matched_objects'] = sum(good)
        if not all(good):
            row['errors'].append('source_identity')
        for stage in output['targets']:
            positioned, positions, angles = [], [], []
            for valid, (query, i, original) in zip(good, mapped):
                xy = to_meters(stage['uv'][b, query]); angle = stage['angle'][b, query]
                positioned.append(replace(original, target_pose=Pose(float(xy[0]), float(xy[1]), float(angle))))
                if valid:
                    positions.append(float(torch.linalg.vector_norm(xy-truth['target_xy'][i]))*1000)
                    angles.append(math.degrees(float(angle_delta(angle, truth['target_angle'][i]).abs())))
            safety = geometry(positioned, 'target_pose', .010, .010)
            semantics = semantic_errors(positioned, meta['record'])
            row['stages'].append({'position_mm': positions, 'angle_deg': angles,
                                  'geometry_success': all(good) and safety.valid,
                                  'geometry_errors': safety.errors,
                                  'errors': sorted(set(safety.errors+semantics))})
        final = row['stages'][-1]
        row['target_position_mm'], row['target_angle_deg'] = final['position_mm'], final['angle_deg']
        row['raw_geometry_success'] = final['geometry_success']
        row['errors'] = sorted(set(row['errors']+final['errors']))
        row['task_success'] = all(good) and not row['errors']
        if all(good):
            row['predicted_targets'] = {o.object_id: {'x_m': o.target_pose.x_m, 'y_m': o.target_pose.y_m,
                                                      'theta_rad': o.target_pose.theta_rad} for o in positioned}
            row['keep_violation'] = 'keep' in final['errors']
        rows.append(row)
    return rows


def summarize(rows, records=(), *, stage_count=3):
    result = {'samples': len(rows), 'count_accuracy': sum(r['count_correct'] for r in rows)/len(rows),
              'correspondence_coverage': sum(r['matched_objects'] for r in rows)/sum(r['gt_count'] for r in rows),
              'raw_geometry_success': sum(r['raw_geometry_success'] for r in rows)/len(rows),
              'task_success': sum(r['task_success'] for r in rows)/len(rows),
              'failure_counts': dict(Counter(e for r in rows for e in r['errors']))}
    for key in ('source_position_mm', 'source_size_mm', 'source_angle_deg', 'target_position_mm', 'target_angle_deg'):
        values = [v for r in rows for v in r[key]]
        result[key] = {'median': float(np.median(values)) if values else None,
                       'p95': float(np.percentile(values, 95)) if values else None, 'n': len(values)}
    result['by_mode'] = {mode: success_summary(group)
                         for mode in ('full', 'partial', 'no_op') if (group := [r for r in rows if r['mode']==mode])}
    result['s0_gate'] = (result['count_accuracy']==1 and result['correspondence_coverage']==1 and
                        all(result[k]['p95'] is not None and result[k]['p95'] <= limit for k,limit in
                            [('source_position_mm',1),('target_position_mm',1),('source_size_mm',1),
                             ('source_angle_deg',2),('target_angle_deg',2)]))
    # Safety/semantic residuals remain explicit even when the memorization gate passes.
    result.update(summarize_diagnostics(rows, records, semantic_errors, stage_count=stage_count))
    return result


@torch.no_grad()
def evaluate(model, batches, device):
    was_training = model.training
    model.eval()
    rows, records = [], []
    try:
        for batch in batches:
            batch = batch.to(device)
            records.extend(meta['record'] for meta in batch.metadata)
            rows.extend(evaluate_batch(model(**batch.inputs), batch))
        return summarize(rows, records), rows
    finally:
        model.train(was_training)
