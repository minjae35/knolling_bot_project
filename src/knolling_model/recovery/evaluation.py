"""Measured, complete-set stage metrics; no GT in predicted inference inputs."""
import math
from collections import Counter

import torch
from scipy.optimize import linear_sum_assignment

from knolling_dataset.schema import scene_from_dict
from knolling_dataset.validator import polygon
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch, summarize, semantic_errors
from knolling_model.diagnostics import positioned_objects
from .contracts import oracle_objects
from .data import recovery_supervision
from .model import CapacityError, legacy_evaluation_view
from .selection import select_queries

METRICS_SCHEMA = 'recovery_metrics_v2'


def correspondences(source, truth, record, threshold=.5):
    """Independent TP/FP/FN even when cardinality is wrong; source geometry only."""
    from shapely.geometry import Polygon
    indices = select_queries(source, threshold)[0].tolist()
    originals = {o.object_id: o for o in scene_from_dict(record['source']).objects}
    originals = [originals[k] for k in sorted(originals)]
    costs, valid = [], []
    for q in indices:
        xy = source['uv'][0, q].detach().cpu()*torch.tensor([.6, .4])
        a, b = (source['size'][0, q].detach().cpu()/2).tolist()
        theta = float(source['angle'][0, q])
        c, s = math.cos(theta), math.sin(theta)
        envelope = Polygon([(float(xy[0])+c*x-s*y, float(xy[1])+s*x+c*y)
                            for x, y in ((-a,-b),(a,-b),(a,b),(-a,b))])
        row, flags = [], []
        for obj in originals:
            gt = polygon(obj, obj.source_pose)
            distance = math.hypot(float(xy[0])-obj.source_pose.x_m, float(xy[1])-obj.source_pose.y_m)
            iou = envelope.intersection(gt).area/max(envelope.union(gt).area, 1e-12)
            good = distance <= .010 and iou >= .5
            row.append(distance+(0 if good else 1e6))
            flags.append(good)
        costs.append(row)
        valid.append(flags)
    if not indices:
        return [], 0, len(originals)
    p, g = linear_sum_assignment(costs)
    pairs = [(indices[i], int(j)) for i, j in zip(p, g) if valid[i][j]]
    return pairs, len(indices)-len(pairs), len(originals)-len(pairs)


def _oracle_view(objects, layout):
    source = dict(uv=objects.xy/objects.xy.new_tensor([.6,.4]), size=objects.size, angle=objects.angle,
                  color_logits=objects.color*20, shape_logits=objects.shape*20,
                  objectness_logits=torch.where(objects.valid, 20., -20.))
    return legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5))


def _pure_axis(rows, records):
    from .counterfactuals import verify_pair
    by_id = {r['sample_id']: r for r in rows}
    groups = {}
    for r in records:
        if r.get('counterfactual', {}).get('factor') == 'axis_alignment':
            groups.setdefault(r['counterfactual']['pair_id'], []).append(r)
    passed = 0
    for members in groups.values():
        if len(members) != 2:
            continue
        a, b = members
        verify_pair(a, b)
        ra, rb = by_id[a['sample_id']], by_id[b['sample_id']]
        if not (ra['task_success'] and rb['task_success']):
            continue
        if (semantic_errors(positioned_objects(a, ra['predicted_targets']), b)
                and semantic_errors(positioned_objects(b, rb['predicted_targets']), a)):
            passed += 1
    return dict(pairs=len(groups), passed=passed, success=passed/len(groups) if groups else None)


def regression(current, baseline):
    """Compare complete measurements of the same samples and all observed modes."""
    if not baseline or current['sample_ids'] != baseline.get('sample_ids'):
        return False
    if not current['complete_set'] or not baseline.get('complete_set'):
        return False
    higher = ('count_accuracy', 'coverage', 'task_success', 'pure_axis_success', 'move_accuracy',
              'color_accuracy', 'shape_accuracy')
    lower = ('false_positives', 'false_negatives', 'source_position_p95_mm', 'source_size_p95_mm',
             'source_angle_p95_deg', 'target_position_p95_mm', 'target_angle_p95_deg')
    for key in higher+lower:
        a, b = current.get(key), baseline.get(key)
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            return False
        if (key in higher and a < b-1e-9) or (key in lower and a > b+1e-6):
            return False
    return all(mode in current['by_mode'] and current['by_mode'][mode]['task_success'] >= values['task_success']
               for mode, values in baseline['by_mode'].items())


@torch.no_grad()
def evaluate_recovery(model, batches, *, stage='predicted_layout', expected_ids=None, baseline=None):
    from .stages import STAGES
    if stage not in STAGES or model.threshold != .5:
        raise ValueError('invalid stage or nonstandard evaluation threshold')
    was_training = model.training
    model.eval()
    rows, records = [], []
    fp = fn = move_correct = color_correct = shape_correct = objects_total = 0
    try:
        for batch in batches:
            for b, meta in enumerate(batch.metadata):
                inputs = {k: v[b:b+1] for k, v in batch.inputs.items()}
                truth = recovery_supervision(meta['record'])[1]
                device = next(iter(inputs.values())).device
                truth = {k: v.to(device) for k, v in truth.items()}
                one = Batch(inputs, [truth], [meta])
                layout, overflow = None, False
                if stage == 'oracle_layout':
                    objects = oracle_objects([truth])
                    text = model.encode_text(inputs['input_ids'], inputs['attention_mask'])
                    layout = model.layout(objects, text, inputs['attention_mask'])
                    view = _oracle_view(objects, layout)
                else:
                    visual = model.perception(inputs['images'])
                    source = visual['source']
                    if stage == 'perception':
                        view = dict(source=source, targets=[dict(uv=source['uv'], angle=source['angle'])])
                    else:
                        try:
                            output = model.layout_from_perception(visual, inputs['input_ids'], inputs['attention_mask'])
                            layout = output['layout']
                            view = legacy_evaluation_view(output)
                        except CapacityError:
                            if len(truth['source_xy']) > model.max_objects:
                                raise ValueError('evaluation truth exceeds reviewed capacity') from None
                            view = dict(source=source, targets=[dict(uv=source['uv'], angle=source['angle'])])
                            overflow = True
                row = evaluate_batch(view, one)[0]
                if overflow:
                    row['errors'].append('layout_capacity')
                pairs, extra, missing = correspondences(view['source'], truth, meta['record'])
                fp += extra
                fn += missing
                objects_total += len(truth['source_xy'])
                for q, gt in pairs:
                    color_correct += int(view['source']['color_logits'][0, q].argmax() == truth['color'][gt])
                    shape_correct += int(view['source']['shape_logits'][0, q].argmax() == truth['shape'][gt])
                    if layout is not None:
                        index = torch.where(layout['valid'][0] & (layout['query_index'][0] == q))[0]
                        if len(index) == 1:
                            move_correct += int((layout['move_logits'][0, index[0]] >= 0) == truth['move'][gt])
                rows.append(row)
                records.append(meta['record'])
        if not rows:
            raise ValueError('empty evaluation set')
        result = summarize(rows, records, stage_count=1)
        ids = [r['sample_id'] for r in rows]
        axis = _pure_axis(rows, records) if stage != 'perception' else dict(pairs=0, passed=0, success=None)
        legacy = [(r, record) for r, record in zip(rows, records) if 'counterfactual' not in record]
        legacy_gate = summarize([r for r, _ in legacy], [record for _, record in legacy], stage_count=1)['s0_gate'] if legacy else None
        result.update(schema=METRICS_SCHEMA, stage=stage, evaluation_mode='free_rollout',
                      source_mode='oracle' if stage == 'oracle_layout' else 'predicted',
                      sample_ids=sorted(ids), complete_set=(expected_ids is not None and len(ids) == len(set(ids))
                        and len(expected_ids) == len(set(expected_ids)) and set(ids) == set(expected_ids)),
                      coverage=result['correspondence_coverage'], false_positives=fp, false_negatives=fn,
                      color_accuracy=color_correct/objects_total, shape_accuracy=shape_correct/objects_total,
                      move_accuracy=move_correct/objects_total if stage != 'perception' else None,
                      pure_axis_success=axis['success'], pure_axis=axis, legacy_s0_gate=legacy_gate)
        for name in ('source_position', 'source_size', 'source_angle', 'target_position', 'target_angle'):
            unit = 'deg' if 'angle' in name else 'mm'
            result[f'{name}_p95_{unit}'] = result[f'{name}_{unit}']['p95']
        mode_ok = bool(stage != 'perception' and baseline and result['complete_set'] and baseline.get('complete_set')
                       and result['sample_ids'] == baseline.get('sample_ids')
                       and all(mode in result['by_mode'] and result['by_mode'][mode]['task_success'] >= values['task_success']
                               for mode, values in baseline['by_mode'].items()))
        result.update(per_mode_no_regression=mode_ok,
                      full_set_no_regression=stage != 'perception' and regression(result, baseline),
                      target_evaluated=stage != 'perception')
        if stage == 'perception':
            result.update(task_success=None, legacy_s0_gate=None, s0_gate=False,
                          target_position_p95_mm=None, target_angle_p95_deg=None,
                          target_position_mm=dict(n=0, median=None, p95=None),
                          target_angle_deg=dict(n=0, median=None, p95=None),
                          raw_geometry_success=None, refinement=[],
                          keep_violation=dict(evaluated=False), language_sensitivity=dict(evaluated=False))
            for summary in list(result['by_mode'].values()) + [s for values in result['by_condition'].values() for s in values.values()]:
                summary.update(successes=None, task_success=None, raw_geometry_success=None,
                               failure_counts={}, evaluated=False)
            for row in rows:
                row.update(task_success=None, raw_geometry_success=None, stages=[],
                           target_position_mm=[], target_angle_deg=[], keep_violation=None)
                row.pop('predicted_targets', None)
                row['errors'] = [e for e in row['errors'] if e in ('count', 'source_identity')]
            result['failure_counts'] = dict(Counter(e for row in rows for e in row['errors']))
        return result, rows
    finally:
        model.train(was_training)
