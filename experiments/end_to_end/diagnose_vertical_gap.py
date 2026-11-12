"""Vertical-gap causes after the axis correction. Diagnosis only: no training and no test set."""
import json
import os
import sys
from collections import Counter
from dataclasses import replace
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
from knolling_dataset.schema import Pose
from knolling_dataset.validator import inferred_rows, polygon
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full, selected_objects
from knolling_model.experiments.geometry_decoder import (
    _anchor, _bands, _feasible, _instruction, _members, _openings, apply_geometry, decode_batch,
    decode_scene, extent_offsets)
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import encode_instruction, pack_sources, select_queries
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from score_end_to_end import DATA
from train_selector_full import load_adopted, perceive, select_layout_of

OUT = Path('artifacts/experiments/selector_full/vertical_gap_diagnosis.json')
TOL_MM = 2.0


def forward(system, relation, model, visual, texts):
    queries = select_queries(visual['source'], system.threshold)
    if any(len(chosen) < 1 or len(chosen) > 8 for chosen in queries):
        raise ValueError('detection count is outside 1..8')
    objects = pack_sources(visual['source'], queries)
    text, mask = encode_instruction(system, texts, objects.xy.device)
    goal = system.layout.goal_encoder(objects, text, mask)['vector']
    hidden = relation._hidden(objects.features())
    output = model(
        hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
        None, None, True, colors=objects.color)
    output = select_layout_of(model, hidden, objects, goal, output)
    return output, objects, goal


def replace_xy(output, decoded, status):
    replaced = dict(output)
    replaced['xy'] = decoded.to(dtype=output['xy'].dtype, device=output['xy'].device)
    replaced['geometry_status'] = status
    return replaced


def without_axis(output, objects, goal):
    decoded, status = decode_batch(
        output['xy'], output['angle'], output['row'], output['slot'],
        objects.size[..., 0], objects.size[..., 1], objects.color, goal, objects.valid)
    return replace_xy(output, decoded, status)


def decode_vertical_target(xy, angle, row, slot, major, minor, color, goal, tight_only):
    """Same placement as the decoder, but a vertical opening aims at the requested gap."""
    if len(xy) == 0:
        return xy, 'empty'
    gap, grouped, limits = _instruction(goal)
    offsets = [extent_offsets(angle[index], major[index], minor[index]) for index in range(len(xy))]
    groups = _members(row, slot)
    if groups is None:
        return xy, 'infeasible:duplicate_slot'
    colors = [int(color[index]) for index in range(len(color))]
    horizontal, vertical = _openings(xy, groups, offsets, gap, colors, grouped)
    desired_h = [_bands(opening, required) for opening, required in horizontal]
    desired_v = []
    for opening, required in vertical:
        if tight_only and float((opening - required).detach()) >= -1e-6:
            desired_v.append(_bands(opening, required))
        else:
            desired_v.append(required)
    placed = xy.clone()
    cursor = 0
    for members in groups:
        deltas = [xy.new_zeros(())]
        for left, right, desired in zip(members, members[1:], desired_h[cursor:cursor + len(members) - 1]):
            deltas.append(deltas[-1] + offsets[left][1] - offsets[right][0] + desired)
        cursor += max(len(members) - 1, 0)
        prefer = torch.stack([xy[index, 0] - delta for index, delta in zip(members, deltas)]).mean()
        anchor = _anchor(prefer, deltas, [offsets[index][0] for index in members],
                         [offsets[index][1] for index in members], limits[0], limits[2])
        if anchor is None:
            return xy, 'infeasible:horizontal_room'
        for index, delta in zip(members, deltas):
            placed[index, 0] = anchor + delta
    if vertical:
        deltas = [xy.new_zeros(())]
        for upper, lower, desired in zip(groups, groups[1:], desired_v):
            upper_drop = torch.stack([offsets[index][2] for index in upper]).min()
            lower_rise = torch.stack([offsets[index][3] for index in lower]).max()
            deltas.append(deltas[-1] + upper_drop - lower_rise - desired)
        prefer = []
        for members, delta in zip(groups, deltas):
            prefer.extend(xy[index, 1] - delta for index in members)
        anchor = _anchor(
            torch.stack(prefer).mean(),
            [delta for members, delta in zip(groups, deltas) for _ in members],
            [offsets[index][2] for members in groups for index in members],
            [offsets[index][3] for members in groups for index in members], limits[1], limits[3])
        if anchor is None:
            return xy, 'infeasible:vertical_room'
        for members, delta in zip(groups, deltas):
            for index in members:
                placed[index, 1] = anchor + delta
    if not _feasible(placed, groups, offsets, gap, colors, grouped, limits):
        return xy, 'infeasible:constraint'
    return placed, 'placed'


def retarget_batch(output, objects, goal, tight_only):
    angle = output['angle']
    xy, row, slot = output['xy'], output['row'], output['slot']
    major, minor = objects.size[..., 0], objects.size[..., 1]
    color, valid = objects.color, objects.valid
    decoded = torch.zeros_like(xy)
    status = []
    for index in range(len(xy)):
        count = int(valid[index].sum())
        if count == 0 or not bool(valid[index, :count].all()):
            status.append('empty')
            continue
        args = (xy[index, :count], angle[index, :count], row[index, :count], slot[index, :count],
                major[index, :count], minor[index, :count], color[index, :count].argmax(-1), goal[index])
        current, current_name = decode_scene(*args)
        proposed, proposed_name = decode_vertical_target(*args, tight_only)
        if proposed_name.startswith('infeasible') or proposed_name == 'empty':
            decoded[index, :count] = current
            status.append(current_name)
        else:
            decoded[index, :count] = proposed
            status.append(proposed_name)
    return replace_xy(output, decoded, status)


def with_gt_size(output, objects, goal, queries, sizes):
    cloned = objects.size.clone()
    cloned[0, queries] = sizes.to(cloned.device)
    return apply_geometry(output, replace(objects, size=cloned), goal)


def assess(output, objects, visual, record, identities, truth):
    source = {key: value[0:1].detach().cpu() for key, value in visual['source'].items()}
    layout = dict(
        xy=output['xy'][0:1].detach().cpu(), angle=output['angle'][0:1].detach().cpu(),
        valid=objects.valid[0:1].detach().cpu(), query_index=objects.query_index[0:1].detach().cpu(),
        rollout='free_rollout')
    row = evaluate_batch(
        legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5)),
        Batch({}, [truth], [dict(sample_id=record['sample_id'], object_ids=identities, record=record)]),
        semantic_check=semantic_errors_v3)[0]
    status = '' if not output.get('geometry_status') else str(output['geometry_status'][0])
    return dict(
        sample_id=record['sample_id'], objects=len(identities),
        task=bool(accepted_success(record, row['task_success'])), errors=list(row['errors']),
        targets=row.get('predicted_targets'), geometry=status)


def match_queries(xy, truth_xy):
    if xy.shape[0] != truth_xy.shape[0] or xy.shape[0] == 0:
        return None
    distance = torch.cdist(xy, truth_xy.to(device=xy.device, dtype=xy.dtype))
    used_query, used_gt, pairs = set(), set(), []
    for item in distance.flatten().argsort().tolist():
        query, gt = divmod(item, distance.shape[1])
        if query in used_query or gt in used_gt or float(distance[query, gt]) > .010:
            continue
        used_query.add(query)
        used_gt.add(gt)
        pairs.append((query, gt))
    if len(pairs) != xy.shape[0]:
        return None
    return pairs


def stats(values):
    if not values:
        return dict(n=0, median=None, p25=None, p75=None)
    tensor = torch.tensor(values, dtype=torch.float)
    return dict(n=len(values), median=round(float(tensor.median()), 3),
                p25=round(float(tensor.quantile(.25)), 3), p75=round(float(tensor.quantile(.75)), 3))


def group_stats(buckets):
    return {str(key): stats(value) for key, value in sorted(buckets.items(), key=lambda item: str(item[0]))}


def edge_span(obj, pose, vertical):
    bounds = polygon(obj, pose).bounds
    return (bounds[3] - bounds[1]) if vertical else (bounds[2] - bounds[0])


def pair_record(record, targets, output, objects, truth, identities):
    """Evaluator gap, predicted-extent gap, and the row the decoder actually spaced."""
    if not targets:
        return None
    posed = selected_objects(record, targets)
    rows = inferred_rows(posed)
    by_id = {obj.object_id: obj for obj in posed}
    count = int(objects.valid[0].sum())
    xy = output['xy'][0, :count].detach().cpu()
    used = set()
    query_of = {}
    for obj in posed:
        point = torch.tensor([obj.target_pose.x_m, obj.target_pose.y_m])
        distance = (xy - point).norm(dim=-1)
        for index in used:
            distance[index] = 1e9
        query = int(distance.argmin())
        if float(distance[query]) > 1e-4:
            return None
        used.add(query)
        query_of[obj.object_id] = query
    gt_of = {identities[gt]: gt for gt in range(len(identities))}
    pred_row = {}
    pred_color = {}
    pred_offset = {}
    gt_span = {}
    pred_span = {}
    for obj in posed:
        query = query_of[obj.object_id]
        pred_row[obj.object_id] = int(output['row'][0, query])
        pred_color[obj.object_id] = int(objects.color[0, query].argmax())
        major = objects.size[0, query, 0]
        minor = objects.size[0, query, 1]
        xmin, xmax, ymin, ymax = extent_offsets(output['angle'][0, query].cpu(), major.cpu(), minor.cpu())
        pred_offset[obj.object_id] = (float(xmin.detach()), float(xmax.detach()), float(ymin.detach()), float(ymax.detach()))
        gt_index = gt_of[obj.object_id]
        gt_major = float(truth['size'][gt_index, 0])
        gt_minor = float(truth['size'][gt_index, 1])
        pred_span[obj.object_id] = (float(major) - gt_major, float(minor) - gt_minor)
        gt_span[obj.object_id] = edge_span(obj, obj.target_pose, True)
    decoder_rows = []
    for row_id in sorted(set(pred_row.values())):
        decoder_rows.append({oid for oid, value in pred_row.items() if value == row_id})
    inferred = [{obj.object_id for obj in row} for row in rows]
    partition_diff = sorted(tuple(sorted(row)) for row in decoder_rows) != sorted(tuple(sorted(row)) for row in inferred)
    instruction = record['instruction']
    vertical, horizontal = [], []

    def one_vertical(upper, lower):
        upper_id = min(upper, key=lambda obj: polygon(obj, obj.target_pose).bounds[1]).object_id
        lower_id = max(lower, key=lambda obj: polygon(obj, obj.target_pose).bounds[3]).object_id
        gt_open = min(polygon(obj, obj.target_pose).bounds[1] for obj in upper)
        gt_open -= max(polygon(obj, obj.target_pose).bounds[3] for obj in lower)
        pred_open = min(obj.target_pose.y_m + pred_offset[obj.object_id][2] for obj in upper)
        pred_open -= max(obj.target_pose.y_m + pred_offset[obj.object_id][3] for obj in lower)
        gt_colors = {obj.color for obj in upper}, {obj.color for obj in lower}
        eval_factor = 2 if instruction['group_by'] == 'color' and upper[0].color != lower[0].color else 1
        pred_factor = 2 if instruction['group_by'] == 'color' and pred_color[upper[0].object_id] != pred_color[lower[0].object_id] else 1
        required = eval_factor * instruction['gap_m']
        own = pred_factor * instruction['gap_m']
        contact = gt_span[upper_id] + gt_span[lower_id]
        return dict(
            eval_mm=(gt_open - required) * 1000, pred_mm=(pred_open - required) * 1000,
            own_mm=(pred_open - own) * 1000, residue_mm=(gt_open - pred_open) * 1000,
            contact_mm=contact * 1000, rows=len(rows),
            wide=max(len(row) for row in rows), color_boundary=eval_factor == 2,
            color_mismatch=eval_factor != pred_factor, partition_diff=partition_diff,
            gap_mm=instruction['gap_m'] * 1000,
            major_mm=(pred_span[upper_id][0] + pred_span[lower_id][0]) * 500,
            minor_mm=(pred_span[upper_id][1] + pred_span[lower_id][1]) * 500)

    for upper, lower in zip(rows, rows[1:]):
        vertical.append(one_vertical(upper, lower))
    for row in rows:
        for left, right in zip(row, row[1:]):
            gt_open = polygon(right, right.target_pose).bounds[0] - polygon(left, left.target_pose).bounds[2]
            pred_open = (right.target_pose.x_m + pred_offset[right.object_id][0])
            pred_open -= left.target_pose.x_m + pred_offset[left.object_id][1]
            required = instruction['gap_m']
            horizontal.append(dict(
                eval_mm=(gt_open - required) * 1000, pred_mm=(pred_open - required) * 1000,
                residue_mm=(gt_open - pred_open) * 1000))
    return dict(vertical=vertical, horizontal=horizontal, rows=len(rows),
                wide=max((len(row) for row in rows), default=0), partition_diff=partition_diff)


def scene_cause(status, pairs):
    if str(status).startswith('infeasible'):
        return 'space'
    if any(pair['partition_diff'] for pair in pairs):
        return 'row'
    worst = max(pairs, key=lambda pair: abs(pair['eval_mm']))
    if worst['color_mismatch'] and abs(worst['own_mm']) <= TOL_MM + 0.15:
        return 'color_factor'
    if abs(worst['pred_mm']) >= abs(worst['residue_mm']):
        return 'geometry'
    return 'size'


def recovery(before, after):
    before_ok = {row['sample_id'] for row in before if row['task']}
    after_ok = {row['sample_id'] for row in after if row['task']}
    lost = [row for row in after if row['sample_id'] in before_ok and row['sample_id'] not in after_ok]
    return dict(
        before=len(before_ok), after=len(after_ok), gained=len(after_ok - before_ok), lost=len(before_ok - after_ok),
        lost_errors=dict(Counter(name for row in lost for name in row['errors'])),
        lost_status=dict(Counter(row['geometry'] or 'none' for row in lost)))


def failing_values(detail, axis, key):
    if detail is None:
        return None
    return [pair[key] for pair in detail[axis] if abs(pair['eval_mm']) > TOL_MM]


@torch.no_grad()
def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    runner.setup()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    validation = ExpandedDataset(DATA, 'validation')
    indices = [index for index, record in enumerate(validation.records) if is_full(record)]
    if limit:
        indices = indices[:limit]
    system, relation, model, head, _saved = load_adopted()
    current_rows, presnap_rows, oracle_rows, tight_rows, exact_rows = [], [], [], [], []
    fail_pairs, pass_pairs = [], []
    cause = Counter()
    status_count = Counter()
    by_count, by_rows, by_gap, by_contact, by_color, by_wide = {}, {}, {}, {}, {}, {}
    new_horizontal, fixed_horizontal = [], []
    collisions = []
    size_lost_bias, size_gain_bias = [], []
    oracle_lost_reason = Counter()
    lost_major, lost_minor = [], []
    for done, index in enumerate(indices, start=1):
        record = validation.records[index]
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        image = validation[index]['image'][None].to(stable.DEVICE)
        try:
            visual = perceive(system, head, image)
            raw, objects, goal = forward(system, relation, model, visual, [record['instruction']['text']])
        except ValueError:
            empty = dict(sample_id=record['sample_id'], objects=len(identities), task=False,
                         errors=['count'], targets=None, geometry='')
            for bucket in (current_rows, presnap_rows, oracle_rows, tight_rows, exact_rows):
                bucket.append(empty)
            continue
        current = apply_geometry(raw, objects, goal)
        previous = without_axis(raw, objects, goal)
        snapped = current
        pairs = match_queries(objects.xy[0, :int(objects.valid[0].sum())].detach().cpu(), truth['source_xy'])
        if pairs is None:
            oracle = snapped
        else:
            order = sorted(pairs)
            queries = torch.tensor([query for query, _gt in order])
            sizes = torch.stack([truth['size'][gt] for _query, gt in order])
            oracle = with_gt_size(raw, objects, goal, queries, sizes)
        tight = retarget_batch(snapped, objects, goal, True)
        exact = retarget_batch(snapped, objects, goal, False)
        judged = [assess(item, objects, visual, record, identities, truth)
                  for item in (snapped, previous, oracle, tight, exact)]
        current_rows.append(judged[0])
        presnap_rows.append(judged[1])
        oracle_rows.append(judged[2])
        tight_rows.append(judged[3])
        exact_rows.append(judged[4])
        detail = pair_record(record, judged[0]['targets'], snapped, objects, truth, identities)
        previous_detail = pair_record(record, judged[1]['targets'], previous, objects, truth, identities)
        if detail is not None:
            for pair in detail['vertical']:
                bucket = fail_pairs if abs(pair['eval_mm']) > TOL_MM else pass_pairs
                bucket.append(pair)
                if abs(pair['eval_mm']) > TOL_MM:
                    by_count.setdefault(judged[0]['objects'], []).append(pair['eval_mm'])
                    by_rows.setdefault(pair['rows'], []).append(pair['eval_mm'])
                    by_gap.setdefault(int(pair['gap_mm']), []).append(pair['eval_mm'])
                    by_color.setdefault('color' if pair['color_boundary'] else 'plain', []).append(pair['eval_mm'])
                    by_wide.setdefault('wide' if pair['wide'] >= 3 else 'narrow', []).append(pair['eval_mm'])
                    by_contact.setdefault(int(pair['contact_mm'] // 20) * 20, []).append(pair['eval_mm'])
            if 'vertical_gap' in judged[0]['errors'] and detail['vertical']:
                label = scene_cause(judged[0]['geometry'], detail['vertical'])
                cause[label] += 1
                status_count[judged[0]['geometry'] or 'none'] += 1
        if 'horizontal_gap' in judged[0]['errors'] and 'horizontal_gap' not in judged[1]['errors']:
            new_horizontal.append(dict(
                snapped=float((snapped['angle'] - previous['angle']).abs().max()) > 1e-3,
                axis_before='axis_alignment' in judged[1]['errors'],
                status=judged[0]['geometry'], status_before=judged[1]['geometry'],
                after=failing_values(detail, 'horizontal', 'eval_mm'),
                pred=failing_values(detail, 'horizontal', 'pred_mm'),
                residue=failing_values(detail, 'horizontal', 'residue_mm'),
                before=failing_values(previous_detail, 'horizontal', 'eval_mm')))
        if 'horizontal_gap' in judged[1]['errors'] and 'horizontal_gap' not in judged[0]['errors']:
            fixed_horizontal.append(1)
        if 'collision' in judged[0]['errors'] and detail is not None:
            posed = selected_objects(record, judged[0]['targets'])
            overlaps = []
            for left_i, left in enumerate(posed):
                for right in posed[left_i + 1:]:
                    area = polygon(left, left.target_pose).intersection(polygon(right, right.target_pose)).area
                    if area > 1e-12:
                        overlaps.append(round(area * 1e6, 3))
            collisions.append(dict(
                sample_id=record['sample_id'], areas_mm2=overlaps, status=judged[0]['geometry'],
                errors=judged[0]['errors'], before_errors=judged[1]['errors'],
                snapped=float((snapped['angle'] - previous['angle']).abs().max()) > 1e-3,
                residue_h=None if detail is None else [round(pair['residue_mm'], 3) for pair in detail['horizontal']],
                residue_v=None if detail is None else [round(pair['residue_mm'], 3) for pair in detail['vertical']],
                pred_h=None if detail is None else [round(pair['pred_mm'], 3) for pair in detail['horizontal']]))
        if judged[0]['task'] and not judged[2]['task']:
            reason = judged[2]['geometry'] if str(judged[2]['geometry']).startswith('infeasible') else ''
            if not reason:
                for name in ('requested_region', 'boundary', 'collision', 'clearance', 'horizontal_gap', 'vertical_gap'):
                    if name in judged[2]['errors']:
                        reason = name
                        break
            oracle_lost_reason[reason or 'other'] += 1
            if pairs is not None:
                lost_major.extend((float(objects.size[0, query, 0]) - float(truth['size'][gt, 0])) * 1000 for query, gt in pairs)
                lost_minor.extend((float(objects.size[0, query, 1]) - float(truth['size'][gt, 1])) * 1000 for query, gt in pairs)
            if detail is not None and detail['vertical']:
                size_lost_bias.append(detail['vertical'][0]['residue_mm'])
        if judged[2]['task'] and not judged[0]['task'] and detail is not None and detail['vertical']:
            size_gain_bias.append(detail['vertical'][0]['residue_mm'])
        if done % 100 == 0:
            print(done, sum(row['task'] for row in current_rows), sum(row['task'] for row in presnap_rows), flush=True)

    def component(pairs, name):
        failing = [pair for pair in pairs if abs(pair['eval_mm']) > TOL_MM] if name == 'fail' else pairs
        return dict(eval=stats([pair['eval_mm'] for pair in failing]),
                    pred=stats([pair['pred_mm'] for pair in failing]),
                    residue=stats([pair['residue_mm'] for pair in failing]),
                    own=stats([pair['own_mm'] for pair in failing]),
                    major=stats([pair['major_mm'] for pair in failing]),
                    minor=stats([pair['minor_mm'] for pair in failing]))

    def flat(rows, key):
        values = []
        for row in rows:
            if row.get(key):
                values.extend(row[key])
        return stats(values)

    vertical_fail_scenes = [row for row in current_rows if 'vertical_gap' in row['errors']]
    only_vertical = [row for row in vertical_fail_scenes if row['errors'] == ['vertical_gap']]
    band = [pair for pair in fail_pairs if abs(abs(pair['pred_mm']) - TOL_MM) <= 0.25]
    report = dict(
        scenes=len(indices),
        current_task=sum(row['task'] for row in current_rows),
        presnap_task=sum(row['task'] for row in presnap_rows),
        vertical_scenes=len(vertical_fail_scenes),
        vertical_only=len(only_vertical),
        cause=dict(cause),
        status=dict(status_count),
        fail_components=component(fail_pairs, 'fail'),
        pass_components=component(pass_pairs, 'pass'),
        band_edge_pairs=len(band),
        fail_pairs=len(fail_pairs),
        by_objects=group_stats(by_count),
        by_rows=group_stats(by_rows),
        by_requested_gap_mm=group_stats(by_gap),
        by_contact_mm=group_stats(by_contact),
        by_color=group_stats(by_color),
        by_width=group_stats(by_wide),
        cooccur=dict(Counter(name for row in vertical_fail_scenes for name in row['errors'])),
        oracle=recovery(current_rows, oracle_rows),
        tight_target=recovery(current_rows, tight_rows),
        exact_target=recovery(current_rows, exact_rows),
        oracle_lost_reason=dict(oracle_lost_reason),
        size_bias_mm=dict(lost_gap=stats(size_lost_bias), gained_gap=stats(size_gain_bias),
                          lost_major=stats(lost_major), lost_minor=stats(lost_minor)),
        horizontal=dict(
            before=sum('horizontal_gap' in row['errors'] for row in presnap_rows),
            after=sum('horizontal_gap' in row['errors'] for row in current_rows),
            new=len(new_horizontal), fixed=len(fixed_horizontal),
            new_snapped=sum(row['snapped'] for row in new_horizontal),
            new_axis_before=sum(row['axis_before'] for row in new_horizontal),
            new_status=dict(Counter(row['status'] for row in new_horizontal)),
            new_eval=flat(new_horizontal, 'after'), new_pred=flat(new_horizontal, 'pred'),
            new_residue=flat(new_horizontal, 'residue'), new_before=flat(new_horizontal, 'before')),
        collisions=dict(
            after=sum('collision' in row['errors'] for row in current_rows),
            before=sum('collision' in row['errors'] for row in presnap_rows),
            scenes=collisions),
        tight_vertical_scenes=sum('vertical_gap' in row['errors'] for row in tight_rows),
        exact_vertical_scenes=sum('vertical_gap' in row['errors'] for row in exact_rows),
        test_scored=False)
    if not limit:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
