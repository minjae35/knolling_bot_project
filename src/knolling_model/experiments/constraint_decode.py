"""Pick a legal row and order from the selector's own scores.

The argmax stands when it already satisfies the predicted command. A search
runs only for an illegal output, and only the selector's row and order
log-probabilities rank the candidates. No teacher or ground-truth layout is read.
"""
import torch
from torch.nn import functional as F

from knolling_dataset.schema import REGIONS
from knolling_model.experiments.geometry_decoder import (
    BAND_M, GAP_VALUES, REGION_NAMES, REGION_SLACK, WORK_LOW, WORK_X_HIGH, apply_geometry,
    decode_scene, extent_offsets)

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


def order_by_area(row, slot, area, color, goal):
    """Move objects into the seats already chosen, ordered by predicted area.

    The legality check allows a 5% area tie, so a near-tie can sit in either
    order. The evaluator then reads the true asset areas in that seat order.
    Seats, rows, and colors stay where they are. An incomplete slot assignment
    is left untouched.
    """
    if row.ndim != 1 or slot.shape != row.shape or area.shape != row.shape or color.shape != row.shape:
        raise ValueError('size order expects one scene')
    if goal.shape[-1] != 13:
        raise ValueError('size order expects a 13-d goal')
    count = int(row.shape[0])
    if count == 0:
        return row, slot
    groups = {}
    for index in range(count):
        groups.setdefault(int(row[index]), []).append(index)
    for members in groups.values():
        taken = sorted(int(slot[index]) for index in members)
        if int(row[members[0]]) < 0 or taken != list(range(len(members))):
            return row, slot
    descending = bool(goal[12] > goal[11])
    grouped = bool(goal[10] > goal[9])
    order = []
    for row_id in sorted(groups):
        order.extend(sorted(groups[row_id], key=lambda index: int(slot[index])))
    sequences = [[index for index in order if int(color[index]) == name] for name in range(3)] if grouped else [order]
    new_row, new_slot = row.clone(), slot.clone()
    for sequence in sequences:
        ranked = sorted(sequence, key=lambda index: float(area[index]), reverse=descending)
        seats = [(row[index].clone(), slot[index].clone()) for index in sequence]
        for index, (row_id, slot_id) in zip(ranked, seats):
            new_row[index] = row_id
            new_slot[index] = slot_id
    return new_row, new_slot


def apply_area_order(output, objects, goal):
    """Reorder one batch by predicted area. Coordinates are unchanged here."""
    row = output['row'].clone()
    slot = output['slot'].clone()
    area = (objects.size[..., 0] * objects.size[..., 1]).detach()
    color = objects.color.detach().argmax(-1)
    for index in range(row.shape[0]):
        count = int(objects.valid[index].sum())
        if count == 0 or not bool(objects.valid[index, :count].all()):
            continue
        new_row, new_slot = order_by_area(
            row[index, :count], slot[index, :count], area[index, :count],
            color[index, :count], goal[index].detach())
        row[index, :count] = new_row
        slot[index, :count] = new_slot
    replaced = dict(output)
    replaced['row'] = row
    replaced['slot'] = slot
    return replaced


REPAIR_LIMIT = 8


def _set_partitions(items):
    if len(items) <= 1:
        yield [list(items)]
        return
    first, *rest = items
    for part in _set_partitions(rest):
        for index, group in enumerate(part):
            nxt = [list(group) for group in part]
            nxt[index] = [first, *group]
            yield nxt
        yield [[first], *[list(group) for group in part]]


def _row_groups(color, grouped):
    indices = list(range(len(color)))
    if not grouped:
        yield from _set_partitions(indices)
        return
    blocks = [[] for _ in range(3)]
    for index in indices:
        blocks[int(color[index])].append(index)
    keys = [key for key, members in enumerate(blocks) if members]
    from itertools import permutations, product
    for order in permutations(keys):
        parts = [list(_set_partitions(blocks[key])) for key in order]
        for combo in product(*parts):
            groups = []
            for block in combo:
                groups.extend(block)
            yield groups


def _too_wide(groups, spans, gap, room):
    allowance = max(gap-BAND_M, 0.)
    for members in groups:
        span = sum(spans[index] for index in members)
        if len(members) > 1:
            span += (len(members)-1)*allowance
        if span > room+1e-6:
            return True
    return False


def _rank_groups(groups, log_probability):
    matrix = torch.stack([log_probability[members].sum(0) for members in groups])
    choice = matrix.argmax(-1)
    if len({int(item) for item in choice}) < len(groups):
        order = matrix.max(-1).values.argsort(descending=True).tolist()
        used, mapped = set(), {}
        for group_index in order:
            available = [class_id for class_id in range(matrix.shape[1]) if class_id not in used]
            class_id = max(available, key=lambda item, group_index=group_index: float(matrix[group_index, item]))
            used.add(class_id)
            mapped[group_index] = class_id
    else:
        mapped = {group_index: int(class_id) for group_index, class_id in enumerate(choice.tolist())}
    score = sum(float(matrix[group_index, class_id]) for group_index, class_id in mapped.items())
    return score, mapped


def repair_rows(row_logits, angle, major, minor, color, goal, limit=REPAIR_LIMIT):
    """Row splits that fit the region width, highest selector score first.

    Only the selector's row scores rank the splits. Color groups, size order,
    and the requested gap stay in force. At most `limit` splits are returned.
    """
    if row_logits.ndim != 2 or angle.ndim != 1 or goal.shape[-1] != 13:
        raise ValueError('row repair expects one scene')
    if not (row_logits.shape[0] == angle.shape[0] == major.shape[0] == minor.shape[0] == color.shape[0]):
        raise ValueError('row repair scene shapes do not match')
    count = int(angle.shape[0])
    if count == 0 or limit < 1:
        return []
    grouped = bool(goal[10] >= goal[9])
    gap = float((goal[6:9]*goal.new_tensor(GAP_VALUES)).sum())
    region = REGIONS[REGION_NAMES[int(goal[3:6].argmax())]]
    room = min(region[2]+REGION_SLACK, WORK_X_HIGH)-max(region[0]-REGION_SLACK, WORK_LOW)
    spans = []
    for index in range(count):
        xmin, xmax, _ymin, _ymax = extent_offsets(angle[index], major[index], minor[index])
        spans.append(float(xmax-xmin))
    log_probability = F.log_softmax(row_logits.detach().float(), dim=-1)
    ranked = []
    for groups in _row_groups(color.detach().cpu(), grouped):
        if _too_wide(groups, spans, gap, room):
            continue
        score, mapped = _rank_groups(groups, log_probability)
        ranked.append((score, groups, mapped))
    ranked.sort(key=lambda item: item[0], reverse=True)
    area = (major*minor).detach()
    found = []
    for _score, groups, mapped in ranked:
        renumber = {old: new for new, old in enumerate(sorted(set(mapped.values())))}
        row = torch.empty(count, dtype=torch.long, device=angle.device)
        slot = torch.empty(count, dtype=torch.long, device=angle.device)
        for group_index, members in enumerate(groups):
            for seat, member in enumerate(members):
                row[member] = renumber[mapped[group_index]]
                slot[member] = seat
        row, slot = order_by_area(row, slot, area, color, goal)
        if structure_issues(row, slot, major, minor, angle, color, goal):
            continue
        found.append((row, slot))
        if len(found) >= limit:
            break
    return found


def apply_row_repair(output, objects, goal):
    """Replace a row the region cannot hold. A row the decoder accepted stays put."""
    if 'geometry_status' not in output or 'row_logits' not in output:
        return output
    if all(not str(name).startswith('infeasible') for name in output['geometry_status']):
        return output
    row = output['row'].clone()
    slot = output['slot'].clone()
    xy = output['xy'].clone()
    angle = output['angle'].clone()
    status = list(output['geometry_status'])
    notes = []
    changed = False
    for index, name in enumerate(status):
        if not str(name).startswith('infeasible'):
            notes.append('kept')
            continue
        count = int(objects.valid[index].sum())
        found = False
        proposals = repair_rows(
            output['row_logits'][index, :count], angle[index, :count],
            objects.size[index, :count, 0], objects.size[index, :count, 1],
            objects.color[index, :count].argmax(-1), goal[index])
        for new_row, new_slot in proposals:
            trial_row, trial_slot = row.clone(), slot.clone()
            trial_row[index, :count] = new_row
            trial_slot[index, :count] = new_slot
            candidate = apply_geometry(dict(output, row=trial_row, slot=trial_slot, xy=xy, angle=angle), objects, goal)
            if str(candidate['geometry_status'][index]).startswith('infeasible'):
                continue
            xy[index] = candidate['xy'][index]
            angle[index] = candidate['angle'][index]
            row[index] = candidate['row'][index]
            slot[index] = candidate['slot'][index]
            status[index] = candidate['geometry_status'][index]
            notes.append('replaced')
            found = True
            changed = True
            break
        if not found:
            notes.append('unresolved')
    if not changed and 'unresolved' not in notes:
        return output
    replaced = dict(output)
    replaced['xy'] = xy
    replaced['angle'] = angle
    replaced['row'] = row
    replaced['slot'] = slot
    replaced['geometry_status'] = status
    replaced['row_repair'] = notes
    return replaced


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
