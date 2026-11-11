"""Check that another row's height reaches the row-origin input."""
from collections import defaultdict

import torch


def _scene(record):
    by_id = {obj['object_id']: obj for obj in record['source']['objects']}
    widths, heights, row_ids, slots = [], [], [], []
    upper = None
    for row_id, members in enumerate(record['supervision_v3']['rows']):
        member_heights = []
        for slot_id, oid in enumerate(members):
            asset = by_id[oid]['asset']
            widths.append(float(asset['a_m']))
            heights.append(float(asset['b_m']))
            row_ids.append(row_id)
            slots.append(slot_id)
            member_heights.append(float(asset['b_m']))
        if row_id == 0:
            upper = max(member_heights)
    count = len(widths)
    return dict(upper=upper, hidden=torch.zeros(1, count, 128), valid=torch.ones(1, count, dtype=torch.bool),
                goal=torch.zeros(1, 13), widths=torch.tensor(widths)[None], heights=torch.tensor(heights)[None],
                row=torch.tensor(row_ids)[None], slot=torch.tensor(slots)[None])


def conflicting_upper_rows(records):
    """Groups whose second row matches but whose first-row height does not."""
    groups = defaultdict(list)
    for record in records:
        rows = record['supervision_v3']['rows']
        if len(rows) < 2:
            continue
        by_id = {obj['object_id']: obj for obj in record['source']['objects']}
        dims = [(by_id[oid]['asset']['a_m'], by_id[oid]['asset']['b_m']) for oid in rows[1]]
        key = (round(float(record['instruction']['gap_m']), 3),
               tuple(round(width, 5) for width, _ in dims), round(max(height for _, height in dims), 5))
        groups[key].append(record)
    conflicts = []
    for records_in_group in groups.values():
        heights = {round(_scene(record)['upper'], 5) for record in records_in_group}
        if len(heights) > 1:
            conflicts.append(records_in_group)
    return conflicts


def cross_row_height_reaches_origin(model, records):
    conflicts = conflicting_upper_rows(records)
    checked = misses = 0
    own_misses = 0
    with torch.no_grad():
        for records_in_group in conflicts:
            scenes = [( _scene(record)['upper'], _scene(record)) for record in records_in_group]
            for upper, scene in scenes:
                partner = next(other for other_upper, other in scenes if abs(other_upper-upper) > 1e-6)
                left = model.origin_inputs(scene['hidden'], scene['valid'], scene['goal'], scene['row'],
                                           scene['widths'], scene['heights'], scene['slot'])
                right = model.origin_inputs(partner['hidden'], partner['valid'], partner['goal'], partner['row'],
                                            partner['widths'], partner['heights'], partner['slot'])
                height_delta = upper-partner['upper']
                input_delta = float(left[0, 1, model.above_height_channel]-right[0, 1, model.above_height_channel])
                own_delta = float(left[0, 1, model.own_height_channel]-right[0, 1, model.own_height_channel])
                checked += 1
                if abs(input_delta-height_delta) > 1e-5:
                    misses += 1
                if abs(own_delta) > 1e-4:
                    own_misses += 1
    return dict(groups=len(conflicts), checked=checked, misses=misses, own_misses=own_misses,
                reached=len(conflicts) == 112 and checked > 0 and misses == 0 and own_misses == 0)
