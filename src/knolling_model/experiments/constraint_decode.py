"""Pick a legal row and order from the selector's own scores.

The argmax stands when it already satisfies the predicted command. A search
runs only for an illegal output, and only the selector's row and order
log-probabilities rank the candidates. No teacher or ground-truth layout is read.
"""
import torch
from torch.nn import functional as F

from knolling_dataset.schema import REGIONS
from knolling_model.experiments.geometry_decoder import (
    BAND_M, GAP_VALUES, REGION_NAMES, REGION_SLACK, WORK_LOW, WORK_X_HIGH, decode_scene, extent_offsets)

K = 8
ACTIONABLE = {'mixed_color', 'size_order', 'color_order', 'duplicate', 'missing', 'horizontal'}


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
            mask_index = members[members.index(index):]
            mask = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
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


def _decode(xy, angle, row, slot, major, minor, color, goal):
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
        _pose, status = _decode(xy, angle, row, slot, major, minor, color, goal[0])
        if str(status).startswith('infeasible'):
            continue
        return row, slot, 'replaced'
    return base_row, base_slot, 'kept_illegal'


def select_layout(model, hidden, objects, goal, output):
    """Leave a legal selector output unchanged. Search only an illegal one."""
    if not hasattr(model, 'context') or not hasattr(model, 'placement'):
        return output
    with torch.no_grad():
        widths = objects.size[..., 0].detach()
        heights = objects.size[..., 1].detach()
        mixed = model.context(hidden.detach(), objects.valid, goal.detach(), widths, heights)
        row = output['row'].clone()
        slot = output['slot'].clone()
        status = []
        for index in range(row.shape[0]):
            count = int(objects.valid[index].sum())
            if count == 0 or not bool(objects.valid[index, :count].all()):
                status.append('empty')
                continue
            base_row, base_slot = row[index, :count], slot[index, :count]
            major, minor = widths[index, :count], heights[index, :count]
            angle = output['angle'][index, :count].detach()
            color = objects.color[index, :count].detach().argmax(-1)
            goal_row = goal[index].detach()
            issues = structure_issues(base_row, base_slot, major, minor, angle, color, goal_row)
            _pose, decoded = _decode(output['xy'][index, :count], angle, base_row, base_slot, major, minor, color, goal_row)
            if str(decoded).startswith('infeasible'):
                issues.add(decoded)
            actionable = bool(issues & ACTIONABLE) or any(str(item).startswith('infeasible') for item in issues)
            if not actionable:
                status.append('legal')
                continue
            new_row, new_slot, action = consider(
                model.placement, mixed[index:index+1, :count], goal[index:index+1], output['row_logits'][index],
                base_row, base_slot, output['xy'][index, :count], angle, major, minor, color)
            row[index, :count] = new_row
            slot[index, :count] = new_slot
            status.append(action)
    replaced = dict(output)
    replaced['row'] = row
    replaced['slot'] = slot
    replaced['constraint_status'] = status
    return replaced
