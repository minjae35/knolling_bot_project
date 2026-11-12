"""Limited neural beam over an illegal selector output. No training and no test split.

The argmax row and order stay in place when they already satisfy the predicted
command. Other candidates are ranked only by the selector's own row and order
scores. Ground truth never chooses a layout.
"""
import json
import time
from collections import Counter

import torch
from torch.nn import functional as F

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_dataset.schema import REGIONS
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.geometry_decoder import (
    BAND_M, GAP_VALUES, REGION_NAMES, REGION_SLACK, WORK_LOW, WORK_X_HIGH, decode_scene, extent_offsets)
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import encode_instruction, install_refined_size
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
from knolling_model.recovery.selection import select_queries
from score_end_to_end import load_adopted
from train_linked_y import VAL_COUNT, VAL_SEED
from train_set_context import fresh_split

K = 8
ACTIONABLE = {'mixed_color', 'size_order', 'color_order', 'duplicate', 'missing', 'horizontal'}


def judge(record, identities, truth, source, objects, xy, angle):
    layout = dict(xy=xy.detach().cpu(), angle=angle.detach().cpu(),
                  valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                  rollout='free_rollout')
    judged = evaluate_batch(
        legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5)),
        Batch({}, [truth], [dict(sample_id=record['sample_id'], object_ids=identities, record=record)]),
        semantic_check=semantic_errors_v3)[0]
    return bool(accepted_success(record, judged['task_success'])), list(judged['errors'])


def top_rows(log_probs, k):
    beams = [([], 0.)]
    for index in range(log_probs.shape[0]):
        expanded = []
        for assignment, score in beams:
            for row_id, value in enumerate(log_probs[index].tolist()):
                expanded.append((assignment+[row_id], score+value))
        expanded.sort(key=lambda item: item[1], reverse=True)
        beams = expanded[:k]
    return beams


def score_slot(placement, hidden, goal, row, slot):
    """Sum the order log-probabilities of one complete slot assignment."""
    total = hidden.new_zeros(())
    width = hidden.shape[-1]
    for row_id in sorted({int(item) for item in row.tolist() if int(item) >= 0}):
        members = [index for index, value in enumerate(row.tolist()) if int(value) == row_id]
        members.sort(key=lambda index: int(slot[index]))
        previous = placement.order_begin
        for index in members:
            mask = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
            mask[members] = True
            # Remaining objects are this pick and every later slot in the row.
            mask_index = members[members.index(index):]
            mask = torch.zeros_like(mask)
            mask[mask_index] = True
            context = hidden[mask].mean(0)
            query = placement.order_query(torch.cat((context, previous, goal), -1))
            scores = (hidden*query).sum(-1)/(width**.5)
            scores = scores.masked_fill(~mask, -1e9)
            total = total+torch.log_softmax(scores, dim=0)[index]
            previous = hidden[index]
    return total


def beam_orders(placement, hidden, goal, row, k):
    """Top orderings of one scene under the autoregressive order head."""
    hidden, goal, row = hidden[0], goal[0], row[0]
    n = hidden.shape[0]
    sequences = [({}, 0.)]
    for row_id in sorted({int(item) for item in row.tolist()}):
        members = [index for index, value in enumerate(row.tolist()) if int(value) == row_id]
        beams = [([], 0., placement.order_begin)]
        for _step in members:
            expanded = []
            contexts, masks, previous = [], [], []
            for chosen, _score, prev in beams:
                remaining = [index for index in members if index not in chosen]
                mask = torch.zeros(n, dtype=torch.bool, device=hidden.device)
                mask[remaining] = True
                masks.append(mask)
                contexts.append(hidden[mask].mean(0))
                previous.append(prev)
            query = placement.order_query(torch.cat((
                torch.stack(contexts), torch.stack(previous), goal.expand(len(beams), -1)), -1))
            scores = torch.matmul(query, hidden.T)/(hidden.shape[-1]**.5)
            scores = scores.masked_fill(~torch.stack(masks), -1e9)
            logp = torch.log_softmax(scores, dim=-1)
            for local, (chosen, score, _prev) in enumerate(beams):
                for index in members:
                    if index in chosen:
                        continue
                    expanded.append((chosen+[index], score+float(logp[local, index]), hidden[index]))
            expanded.sort(key=lambda item: item[1], reverse=True)
            beams = expanded[:k]
        combined = []
        for base, base_score in sequences:
            for chosen, score, _prev in beams:
                merged = dict(base)
                merged.update({index: step for step, index in enumerate(chosen)})
                combined.append((merged, base_score+score))
        combined.sort(key=lambda item: item[1], reverse=True)
        sequences = combined[:k]
    found = []
    for mapping, score in sequences:
        slot = torch.full((n,), -1, dtype=torch.long, device=hidden.device)
        for index, step in mapping.items():
            slot[index] = step
        found.append((slot, score))
    return found


def structure_issues(row, slot, major, minor, angle, color, goal):
    row_ids = [int(value) for value in row.tolist()]
    slots = [int(value) for value in slot.tolist()]
    colors = [int(value) for value in color.tolist()]
    issues = set()
    if any(value < 0 for value in row_ids+slots):
        issues.add('missing')
    groups = {}
    for index, row_id in enumerate(row_ids):
        groups.setdefault(row_id, []).append(index)
    ordered = []
    for row_id in sorted(groups):
        members = groups[row_id]
        taken = [slots[index] for index in members]
        if len(taken) != len(set(taken)) or set(taken) != set(range(len(members))):
            issues.add('duplicate')
        ordered.append(sorted(members, key=lambda index: slots[index]))
    ids = sorted(row_id for row_id in groups if row_id >= 0)
    if ids and ids != list(range(ids[0], ids[-1]+1)):
        issues.add('row_gap')
    grouped = bool(goal[10] > goal[9])
    descending = bool(goal[12] > goal[11])
    if grouped:
        row_colors = []
        for members in ordered:
            if len({colors[index] for index in members}) > 1:
                issues.add('mixed_color')
            row_colors.append(colors[members[0]])
        runs = [name for index, name in enumerate(row_colors) if index == 0 or name != row_colors[index-1]]
        if len(runs) != len(set(runs)):
            issues.add('color_order')
    flat = [index for members in ordered for index in members]
    area = major*minor
    sequences = [[index for index in flat if colors[index] == name] for name in range(3)] if grouped else [flat]
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            before, after = float(area[left]), float(area[right])
            if (not descending and before > 1.05*after+1e-12) or (descending and after > 1.05*before+1e-12):
                issues.add('size_order')
    region = REGIONS[REGION_NAMES[int(goal[3:6].argmax())]]
    room = min(region[2]+REGION_SLACK, WORK_X_HIGH)-max(region[0]-REGION_SLACK, WORK_LOW)
    gap = float((goal[6:9]*goal.new_tensor(GAP_VALUES)).sum())
    for members in ordered:
        span = 0.
        for index in members:
            xmin, xmax, _ymin, _ymax = extent_offsets(angle[index], major[index], minor[index])
            span += float(xmax-xmin)
        if len(members) > 1:
            span += (len(members)-1)*max(gap-BAND_M, 0.)
        if span > room+1e-6:
            issues.add('horizontal')
    return issues


def decode(xy, angle, row, slot, major, minor, color, goal):
    return decode_scene(xy.detach().cpu(), angle.detach().cpu(), row.detach().cpu(), slot.detach().cpu(),
                        major.detach().cpu(), minor.detach().cpu(), color.detach().cpu(), goal.detach().cpu())


def consider(placement, mixed, goal, row_logits, base_row, base_slot, xy, angle, major, minor, color):
    """Return a replacement row and slot, or the original pair when none is legal."""
    count = base_row.shape[0]
    log_row = F.log_softmax(row_logits[:count], dim=-1)
    original = [int(value) for value in base_row.tolist()]
    options = top_rows(log_row, K)
    if original not in [item[0] for item in options]:
        options.append((original, sum(float(log_row[index, row_id]) for index, row_id in enumerate(original))))
    pool = []
    seen = set()
    for assignment, row_score in options:
        row = torch.tensor(assignment, device=base_row.device)
        if assignment == original:
            orders = beam_orders(placement, mixed, goal, row[None], K)
        else:
            predicted, _logits, _targets = placement._order(
                mixed, torch.ones(mixed.shape[:2], dtype=torch.bool, device=mixed.device), goal, row[None], None)
            greedy = predicted[0]
            orders = [(greedy, float(score_slot(placement, mixed[0], goal[0], row, greedy)))]
        for slot, order_score in orders:
            key = (tuple(assignment), tuple(int(value) for value in slot.tolist()))
            if key in seen:
                continue
            seen.add(key)
            pool.append((row_score+float(order_score), row, slot))
    pool.sort(key=lambda item: item[0], reverse=True)
    checked = 0
    for _score, row, slot in pool:
        if checked >= K:
            break
        checked += 1
        if structure_issues(row, slot, major, minor, angle, color, goal[0]):
            continue
        _pose, status = decode(xy, angle, row, slot, major, minor, color, goal[0])
        if str(status).startswith('infeasible'):
            continue
        return row, slot, 'replaced', checked
    return base_row, base_slot, 'kept_illegal', checked


def main():
    runner.setup()
    data = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    fit, dev, _blocked = fresh_split(data.records)
    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(VAL_SEED)).tolist()
    splits = {'dev512': dev, 'dev256': [fit[index] for index in order[:VAL_COUNT]]}
    system, relation, model, head, _saved = load_adopted()
    placement = model.placement
    report = {}
    for name, indices in splits.items():
        rows = []
        flagged = Counter()
        seconds_base, seconds_extra = [], []
        for seen, index in enumerate(indices, start=1):
            record = data.records[index]
            image = data[index]['image'][None].to(stable.DEVICE)
            identities, truth = recovery_supervision(record)
            truth = {key: value.cpu() for key, value in truth.items()}
            started = time.perf_counter()
            try:
                with torch.no_grad():
                    visual = system.perception(image)
                    visual = install_refined_size(head, image, visual, system.threshold)
                    queries = select_queries(visual['source'], system.threshold)
                    if any(len(chosen) < 1 or len(chosen) > 8 for chosen in queries):
                        raise ValueError('count')
                    objects = pack_sources(visual['source'], queries)
                    text, mask = encode_instruction(system, [record['instruction']['text']], image.device)
                    goal = system.layout.goal_encoder(objects, text, mask)['vector']
                    hidden = relation._hidden(objects.features())
                    output = model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                                   None, None, True, colors=objects.color)
                    mixed = model.context(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1])
            except ValueError:
                seconds_base.append(time.perf_counter()-started)
                seconds_extra.append(0.)
                rows.append(dict(before=False, after=False, before_errors=['count'], after_errors=['count'],
                                 action='count', status='count', id=record['sample_id']))
                continue
            count = int(objects.valid[0].sum())
            source = {key: value.detach().cpu() for key, value in visual['source'].items()}
            neural = output['xy'][0, :count]
            angle = output['angle'][0, :count]
            base_row = output['row'][0, :count]
            base_slot = output['slot'][0, :count]
            major = objects.size[0, :count, 0]
            minor = objects.size[0, :count, 1]
            color = objects.color[0, :count].argmax(-1)
            goal_row = goal[0]
            base_pose, base_status = decode(neural, angle, base_row, base_slot, major, minor, color, goal_row)
            pose = neural.detach().cpu() if str(base_status).startswith('infeasible') else base_pose
            before, before_errors = judge(record, identities, truth, source, objects, pose[None], angle[None].cpu())
            issues = structure_issues(base_row, base_slot, major, minor, angle, color, goal_row)
            if str(base_status).startswith('infeasible'):
                issues.add(base_status)
            base_elapsed = time.perf_counter()-started
            action = 'legal'
            chosen_row, chosen_slot = base_row, base_slot
            actionable = bool(issues & ACTIONABLE) or any(str(item).startswith('infeasible') for item in issues)
            if actionable:
                if before:
                    flagged.update(issues)
                search_started = time.perf_counter()
                with torch.no_grad():
                    chosen_row, chosen_slot, action, _checked = consider(
                        placement, mixed[:, :count], goal, output['row_logits'][0], base_row, base_slot,
                        neural, angle, major, minor, color)
                seconds_extra.append(time.perf_counter()-search_started)
            else:
                seconds_extra.append(0.)
            seconds_base.append(base_elapsed)
            if action == 'replaced':
                new_pose, new_status = decode(neural, angle, chosen_row, chosen_slot, major, minor, color, goal_row)
                scored = neural.detach().cpu() if str(new_status).startswith('infeasible') else new_pose
                after, after_errors = judge(record, identities, truth, source, objects, scored[None], angle[None].cpu())
                status = new_status
            else:
                after, after_errors, status = before, before_errors, base_status
            rows.append(dict(before=before, after=after, before_errors=before_errors, after_errors=after_errors,
                             action=action, status=status, id=record['sample_id']))
            if seen % 64 == 0:
                print(name, seen, sum(row['before'] for row in rows), sum(row['after'] for row in rows), flush=True)
        before_ids = {row['id'] for row in rows if row.get('before')}
        after_ids = {row['id'] for row in rows if row.get('after')}
        failures = Counter(name for row in rows if not row['after'] for name in row.get('after_errors', []))
        base_failures = Counter(name for row in rows if not row['before'] for name in row.get('before_errors', []))
        report[name] = dict(
            before=len(before_ids), after=len(after_ids), total=len(rows),
            preserved=len(before_ids & after_ids), lost=len(before_ids-after_ids),
            gained=len(after_ids-before_ids),
            searched=sum(row.get('action') not in (None, 'legal', 'count') and row.get('action') != 'legal' for row in rows),
            replaced=sum(row.get('action') == 'replaced' for row in rows),
            success_flags=dict(flagged),
            actions=dict(Counter(row.get('action', 'count') for row in rows)),
            before_failures={key: base_failures.get(key, 0) for key in ('mixed_color_row', 'size_order')},
            after_failures={key: failures.get(key, 0) for key in
                            ('mixed_color_row', 'size_order', 'horizontal_gap', 'vertical_gap',
                             'requested_region', 'collision', 'clearance', 'boundary')},
            horizontal_room=sum(str(row.get('status', '')).find('horizontal_room') >= 0 for row in rows),
            seconds_base=float(sum(seconds_base)/len(seconds_base)),
            seconds_extra=float(sum(seconds_extra)/len(seconds_extra)))
        print(name, report[name]['before'], report[name]['after'], 'lost', report[name]['lost'],
              'gained', report[name]['gained'], 'extra_s', round(report[name]['seconds_extra'], 4), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
