"""Geometry bottleneck on official validation. No training and no checkpoint writes."""
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
from knolling_dataset.schema import angle_error
from knolling_dataset.validator import inferred_rows, polygon
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full, selected_objects
from knolling_model.recovery.contracts import Objects
from knolling_model.recovery.data import recovery_supervision
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from score_end_to_end import DATA
from train_selector_full import (
    align_teacher, apply_geometry_of, blank, decode_layout, judge_scene, load_adopted, perceive, select_layout_of)

OLD = Path('artifacts/experiments/selector_only/checkpoint-best.pt')
OUT = Path('artifacts/experiments/selector_full/geometry_diagnosis.json')
FOCUS = ('vertical_gap', 'horizontal_gap', 'axis_alignment')
AXES = ('horizontal', 'vertical')
REGIONS = ('full', 'left', 'right')
GAPS = (0.015, 0.020, 0.030)
GROUPS = ('none', 'color')
SORTS = ('asc', 'desc')


def load_old():
    saved = torch.load(OLD, map_location='cpu', weights_only=False)
    return saved


def use(model, saved):
    model.context.load_state_dict(saved['context'])
    model.placement.load_state_dict(saved['placement'])
    model.eval()


def match_indices(xy, truth_xy):
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


def instruction_goal(record, like):
    ins = record['instruction']
    goal = torch.zeros(like.shape, dtype=like.dtype)

    def choose(value, options):
        for index, option in enumerate(options):
            if value == option:
                return index
            try:
                if abs(float(value) - float(option)) < 1e-9:
                    return index
            except (TypeError, ValueError):
                continue
        raise ValueError(value)

    goal[1 + choose(ins['axis_alignment'], AXES)] = 1
    goal[3 + choose(ins['region'], REGIONS)] = 1
    goal[6 + choose(ins['gap_m'], GAPS)] = 1
    goal[9 + choose(ins['group_by'], GROUPS)] = 1
    goal[11 + choose(ins['sort_direction'], SORTS)] = 1
    return goal


def goal_agreement(goal, record):
    ins = record['instruction']
    values = goal.detach().cpu()
    return dict(
        axis=AXES[int(values[1:3].argmax())] == ins['axis_alignment'],
        region=REGIONS[int(values[3:6].argmax())] == ins['region'],
        gap=abs(GAPS[int(values[6:9].argmax())] - float(ins['gap_m'])) < 1e-9,
        group=GROUPS[int(values[9:11].argmax())] == ins['group_by'])


def signed_gaps(record, poses):
    if not poses:
        return [], []
    objects = selected_objects(record, poses)
    rows = inferred_rows(objects)
    instruction = record['instruction']
    horizontal, vertical = [], []
    for row in rows:
        for left, right in zip(row, row[1:]):
            opening = polygon(right, right.target_pose).bounds[0] - polygon(left, left.target_pose).bounds[2]
            horizontal.append((opening - instruction['gap_m']) * 1000)
    for upper, lower in zip(rows, rows[1:]):
        opening = min(polygon(obj, obj.target_pose).bounds[1] for obj in upper)
        opening -= max(polygon(obj, obj.target_pose).bounds[3] for obj in lower)
        factor = 2 if instruction['group_by'] == 'color' and upper[0].color != lower[0].color else 1
        vertical.append((opening - factor * instruction['gap_m']) * 1000)
    return horizontal, vertical


def pack_cpu(objects):
    names = ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')
    return {name: getattr(objects, name).detach().cpu() for name in names}


def move_objects(stored, device):
    return Objects(**{name: value.to(device) for name, value in stored.items()})


def clone_output(output):
    copied = {}
    for key, value in output.items():
        copied[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return copied


def to_device_output(output, device):
    moved = {}
    for key, value in output.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def forward_cached(model, item, teacher, search):
    device = stable.DEVICE
    objects = move_objects(item['objects'], device)
    hidden = item['hidden'].to(device)
    goal = item['goal'].to(device)
    row = slot = None
    if teacher:
        if item['teacher_row'] is None:
            return None
        row = item['teacher_row'][None].to(device)
        slot = item['teacher_slot'][None].to(device)
    output = model(
        hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
        row, slot, True, colors=objects.color)
    if search:
        output = select_layout_of(model, hidden, objects, goal, output)
    raw = clone_output(output)
    decoded = apply_geometry_of(output, objects, goal)
    return raw, clone_output(decoded), objects


def with_inputs(raw, item, size=None, angle=None, color=None, goal=None):
    device = stable.DEVICE
    objects = move_objects(item['objects'], device)
    output = to_device_output(raw, device)
    if size is not None:
        objects.size = objects.size.clone()
        objects.size[0, :size.shape[0]] = size.to(device)
    if angle is not None:
        output['angle'] = output['angle'].clone()
        output['angle'][0, :angle.shape[0]] = angle.to(device)
    if color is not None:
        objects.color = objects.color.clone()
        objects.color[0, :color.shape[0]] = 0
        objects.color[0, torch.arange(color.shape[0], device=device), color.to(device)] = 1
    used_goal = item['goal'].to(device) if goal is None else goal[None].to(device)
    return apply_geometry_of(output, objects, used_goal), objects


def brief(scene):
    return dict(task=bool(scene['task']), errors=list(scene['errors']), geometry=scene['geometry'],
                sample_id=scene['sample_id'])


def focus_key(errors):
    return '+'.join(name for name in FOCUS if name in errors) or 'none'


def strict_key(errors):
    chosen = [name for name in errors if name in FOCUS]
    others = [name for name in errors if name not in FOCUS]
    if len(chosen) == 1 and not others:
        return chosen[0]
    return None


def median(values):
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float).median())


def summarize_signed(values):
    if not values:
        return dict(n=0, median_mm=None, abs_median_mm=None, too_tight=0, too_loose=0)
    tensor = torch.tensor(values, dtype=torch.float)
    return dict(n=len(values), median_mm=float(tensor.median()),
                abs_median_mm=float(tensor.abs().median()),
                too_tight=int((tensor < -2).sum()), too_loose=int((tensor > 2).sum()))


def recovery(base, oracle):
    before = {row['sample_id'] for row in base if row['task']}
    after = {row['sample_id'] for row in oracle if row['task']}
    return dict(task=len(after), total=len(base), gained=len(after - before), lost=len(before - after),
                net=len(after) - len(before))


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    runner.setup()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the diagnosis forward pass')
    validation = ExpandedDataset(DATA, 'validation')
    indices = [index for index, record in enumerate(validation.records) if is_full(record)]
    if limit:
        indices = indices[:limit]
    print('LOAD', len(indices), flush=True)
    system, relation, model, head, new_saved = load_adopted()
    old_saved = load_old()
    use(model, old_saved)
    cached = []
    old_rows, failed = [], []
    for done, index in enumerate(indices, start=1):
        record = validation.records[index]
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        image = validation[index]['image'][None].to(stable.DEVICE)
        try:
            visual = perceive(system, head, image)
            output, objects = decode_layout(
                system, relation, model, visual, [record['instruction']['text']], None, None, True)
        except ValueError:
            failed.append(index)
            old_rows.append(blank(record, ['count'], len(identities), 0, ''))
            cached.append(None)
            continue
        count = int(objects.valid[0].sum())
        pairs = match_indices(objects.xy[0, :count], truth['source_xy'])
        teacher = align_teacher(objects.xy[0, :count], truth['source_xy'], record, identities) if pairs else None
        text, mask = None, None
        from knolling_model.experiments.size_refinement import encode_instruction
        text, mask = encode_instruction(system, [record['instruction']['text']], stable.DEVICE)
        with torch.no_grad():
            goal = system.layout.goal_encoder(objects, text, mask)['vector'].detach().cpu()
            hidden = relation._hidden(objects.features()).detach().cpu()
        gt_size = gt_color = None
        if pairs:
            order = sorted(pairs)
            gt_size = torch.stack([truth['size'][gt] for _query, gt in order])
            gt_color = torch.tensor([int(truth['color'][gt]) for _query, gt in order], dtype=torch.long)
        expected = 0. if record['instruction']['axis_alignment'] == 'horizontal' else math.pi / 2
        cached.append(dict(
            objects=pack_cpu(objects), goal=goal, hidden=hidden,
            visual={'source': {key: value.detach().cpu() for key, value in visual['source'].items()}},
            teacher_row=None if teacher is None else teacher[0],
            teacher_slot=None if teacher is None else teacher[1],
            gt_size=gt_size, gt_color=gt_color, axis=expected, record=record,
            identities=identities, truth=truth, goal_ok=goal_agreement(goal[0], record)))
        old_rows.append(judge_scene(output, objects, visual, record, identities, truth, 0, system.threshold, False))
        if done % 100 == 0 or done == len(indices):
            print(f'OLD {done}/{len(indices)} task={sum(row["task"] for row in old_rows)}', flush=True)

    use(model, new_saved)
    new_rows, teacher_rows, raws = [], [], []
    size_rows, angle_rows, color_rows, goal_rows = [], [], [], []
    vertical_signed, horizontal_signed = [], []
    size_gap, size_ok, angle_bad, angle_ok = [], [], [], []
    for item, old in zip(cached, old_rows):
        if item is None:
            new_rows.append(old)
            teacher_rows.append(old)
            for bucket in (size_rows, angle_rows, color_rows, goal_rows):
                bucket.append(old)
            raws.append(None)
            continue
        forwarded = forward_cached(model, item, False, True)
        raw, decoded, objects = forwarded
        raws.append(raw)
        judged = judge_scene(decoded, objects, item['visual'], item['record'], item['identities'], item['truth'],
                             0, system.threshold, False)
        new_rows.append(judged)
        if judged.get('errors') is not None and 'count' not in judged['errors']:
            from knolling_model.data import Batch
            from knolling_model.evaluation import evaluate_batch
            from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
            from knolling_model.recovery.model import legacy_evaluation_view
            source = {key: value[0:1].detach().cpu() for key, value in item['visual']['source'].items()}
            layout = dict(xy=decoded['xy'][0:1], angle=decoded['angle'][0:1], valid=objects.valid[0:1].cpu(),
                          query_index=objects.query_index[0:1].cpu(), rollout='free_rollout')
            detail = evaluate_batch(
                legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5)),
                Batch({}, [item['truth']], [dict(sample_id=item['record']['sample_id'],
                                                 object_ids=item['identities'], record=item['record'])]),
                semantic_check=semantic_errors_v3)[0]
            horizontal, vertical = signed_gaps(item['record'], detail.get('predicted_targets'))
            if 'horizontal_gap' in judged['errors']:
                horizontal_signed.extend(horizontal)
            if 'vertical_gap' in judged['errors']:
                vertical_signed.extend(vertical)
            count = int(item['objects']['valid'][0].sum())
            predicted_angle = decoded['angle'][0, :count]
            degrees = [math.degrees(angle_error(float(value), item['axis'])) for value in predicted_angle]
            (angle_bad if 'axis_alignment' in judged['errors'] else angle_ok).extend(degrees)
            if item['gt_size'] is not None:
                delta = (item['objects']['size'][0, :count] - item['gt_size']) * 1000
                (size_gap if 'horizontal_gap' in judged['errors'] or 'vertical_gap' in judged['errors'] else size_ok).extend(
                    delta.abs().max(-1).values.tolist())
        teacher = forward_cached(model, item, True, False)
        if teacher is None:
            teacher_rows.append(blank(item['record'], ['alignment'], len(item['identities']), 0, ''))
        else:
            _raw, decoded_teacher, objects_teacher = teacher
            teacher_rows.append(judge_scene(
                decoded_teacher, objects_teacher, item['visual'], item['record'], item['identities'], item['truth'],
                0, system.threshold, False))
        if item['gt_size'] is None:
            size_rows.append(judged)
            color_rows.append(judged)
        else:
            decoded_size, objects_size = with_inputs(raw, item, size=item['gt_size'])
            size_rows.append(judge_scene(decoded_size, objects_size, item['visual'], item['record'], item['identities'],
                                         item['truth'], 0, system.threshold, False))
            decoded_color, objects_color = with_inputs(raw, item, color=item['gt_color'])
            color_rows.append(judge_scene(decoded_color, objects_color, item['visual'], item['record'],
                                          item['identities'], item['truth'], 0, system.threshold, False))
        axis = torch.full((int(item['objects']['valid'][0].sum()),), item['axis'])
        decoded_angle, objects_angle = with_inputs(raw, item, angle=axis)
        angle_rows.append(judge_scene(decoded_angle, objects_angle, item['visual'], item['record'], item['identities'],
                                      item['truth'], 0, system.threshold, False))
        oracle_goal = instruction_goal(item['record'], item['goal'][0])
        decoded_goal, objects_goal = with_inputs(raw, item, goal=oracle_goal)
        goal_rows.append(judge_scene(decoded_goal, objects_goal, item['visual'], item['record'], item['identities'],
                                     item['truth'], 0, system.threshold, False))
        if len(new_rows) % 100 == 0:
            print(f'NEW {len(new_rows)}/{len(indices)} task={sum(row["task"] for row in new_rows)}', flush=True)

    def tasks(rows):
        return sum(row['task'] for row in rows)

    overlap = Counter(focus_key(row['errors']) for row in new_rows if not row['task'])
    isolated = Counter(strict_key(row['errors']) for row in new_rows if not row['task'])
    lost = [new for old, new in zip(old_rows, new_rows) if old['task'] and not new['task']]
    lost_errors = Counter(name for row in lost for name in row['errors'])
    goal_counter = Counter()
    for item in cached:
        if item is None:
            continue
        for key, ok in item['goal_ok'].items():
            goal_counter[key] += int(not ok)
    applicable = sum(item is not None and item['gt_size'] is not None for item in cached)
    report = dict(
        scenes=len(indices), count_failures=len(failed), aligned=applicable,
        old_task=tasks(old_rows), new_task=tasks(new_rows), teacher_task=tasks(teacher_rows),
        overlap=dict(overlap), isolated={str(key): value for key, value in isolated.items()},
        vertical_signed=summarize_signed(vertical_signed),
        horizontal_signed=summarize_signed(horizontal_signed),
        size_abs_max_mm=dict(gap_fail=median(size_gap), other=median(size_ok),
                             gap_fail_n=len(size_gap), other_n=len(size_ok)),
        angle_deg=dict(axis_fail=median(angle_bad), other=median(angle_ok),
                       axis_fail_n=len(angle_bad), other_n=len(angle_ok)),
        goal_disagreements=dict(goal_counter), goal_scenes=sum(item is not None for item in cached),
        oracle=dict(size=recovery(new_rows, size_rows), angle=recovery(new_rows, angle_rows),
                    color=recovery(new_rows, color_rows), goal=recovery(new_rows, goal_rows)),
        lost=dict(scenes=len(lost), errors=dict(lost_errors),
                  size_would_save=sum(after['task'] for before, after in zip(new_rows, size_rows)
                                      if before['sample_id'] in {row['sample_id'] for row in lost} and after['task']),
                  angle_would_save=sum(after['task'] for before, after in zip(new_rows, angle_rows)
                                       if before['sample_id'] in {row['sample_id'] for row in lost} and after['task'])),
        test_scored=False)
    if not limit:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
