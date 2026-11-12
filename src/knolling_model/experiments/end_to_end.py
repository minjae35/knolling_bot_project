"""RGB and natural language to a target pose, through the adopted row layout.

Inference reads an image and an instruction. Target poses, rows, and slots are
supervision only. The relation head's anchor and neighbor chain are not on
this path.
"""
import torch
from torch.nn import functional as F

from knolling_model.coordinates import angle_delta
from knolling_model.experiments.row_layout import assignment_loss, order_loss, step_placement_loss
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.selection import SourceSelectionError, match_selected, select_queries


def arrange_free(row_model, hidden, valid, goal, widths, heights):
    """Predicted row and order. No teacher row, slot, or target pose."""
    return row_model(hidden, valid, goal, widths, heights, None, None, True)


def arrangement_supervision_loss(output, target_xy, target_angle, valid, teacher_row, teacher_slot):
    """GT pose, row, and slot supervise the adopted placement losses."""
    placed = step_placement_loss(output, target_xy, target_angle, valid, teacher_row, teacher_slot)
    loss = placed['loss']+assignment_loss(output, teacher_row, valid)+order_loss(output)
    return dict(loss=loss, xy=placed['x']+placed['y'], dx=placed['dx'], angle=placed['angle'])


def encode_instruction(system, texts, device):
    tokens = system.tokenizer(texts, return_tensors='pt', padding=True, truncation=False)
    mask = tokens['attention_mask'].to(device).bool()
    text = system.arrangement.text_encoder(
        input_ids=tokens['input_ids'].to(device), attention_mask=mask).last_hidden_state
    return text, mask


def predict_poses(system, relation, row_model, images, texts):
    """One pose per detected object. Records and target tensors are not arguments."""
    if images.ndim != 4 or len(images) != len(texts):
        raise ValueError('one instruction per image required')
    visual = system.perception(images)
    queries = select_queries(visual['source'], system.threshold)
    if any(len(chosen) < 1 or len(chosen) > 8 for chosen in queries):
        raise ValueError('detection count is outside 1..8')
    objects = pack_sources(visual['source'], queries)
    text, mask = encode_instruction(system, texts, images.device)
    goal = system.layout.goal_encoder(objects, text, mask)['vector']
    hidden = relation._hidden(objects.features())
    output = arrange_free(row_model, hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1])
    return output, objects, visual


def _isolate(visual, index):
    def cut(value):
        return value[index:index+1] if torch.is_tensor(value) else value
    return dict(source={key: cut(value) for key, value in visual['source'].items()},
                stages=[{key: cut(value) for key, value in stage.items()} for stage in visual['stages']])


def _labels(record, ids, width, device):
    row = torch.full((width,), -1, dtype=torch.long, device=device)
    slot = torch.full((width,), -1, dtype=torch.long, device=device)
    location = {oid: index for index, oid in enumerate(ids)}
    for row_id, members in enumerate(record['supervision_v3']['rows']):
        for slot_id, oid in enumerate(members):
            row[location[oid]] = row_id
            slot[location[oid]] = slot_id
    return row, slot


def supervise_arrangement(system, relation, row_model, images, records):
    """Match detections, then apply GT pose, row, and slot as labels only."""
    visual = system.perception(images)
    kept, queries, truths, identities = [], [], [], []
    for index, record in enumerate(records):
        obj_ids, truth = recovery_supervision(record)
        truth = {key: value.to(images.device) for key, value in truth.items()}
        try:
            selected, gt = match_selected(system, _isolate(visual, index), [truth])[0]
        except (SourceSelectionError, ValueError):
            continue
        if len(selected) != len(obj_ids) or set(gt.tolist()) != set(range(len(obj_ids))):
            continue
        order = torch.empty(len(obj_ids), dtype=torch.long, device=images.device)
        order[gt] = selected
        kept.append(index)
        queries.append(order)
        truths.append(truth)
        identities.append(obj_ids)
    if not kept:
        return None
    source = {key: value[kept] for key, value in visual['source'].items()}
    objects = pack_sources(source, queries)
    chosen = [records[index] for index in kept]
    text, mask = encode_instruction(system, [record['instruction']['text'] for record in chosen], images.device)
    goal = system.layout.goal_encoder(objects, text, mask)['vector']
    hidden = relation._hidden(objects.features())
    width = objects.valid.shape[1]
    teacher_row = torch.stack([_labels(record, ids, width, images.device)[0]
                              for record, ids in zip(chosen, identities)])
    teacher_slot = torch.stack([_labels(record, ids, width, images.device)[1]
                               for record, ids in zip(chosen, identities)])
    output = row_model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                       teacher_row, teacher_slot, True)
    target_xy = objects.xy.new_zeros(len(chosen), width, 2)
    target_angle = objects.angle.new_zeros(len(chosen), width)
    for index, truth in enumerate(truths):
        count = len(truth['target_xy'])
        target_xy[index, :count] = truth['target_xy']
        target_angle[index, :count] = truth['target_angle']
    terms = arrangement_supervision_loss(output, target_xy, target_angle, objects.valid, teacher_row, teacher_slot)
    return dict(terms=terms, kept=len(chosen), output=output)


def consistency_loss(live, reference, valid):
    """Match the frozen Best pose. The reference is a prediction, not a GT pose."""
    xy = F.smooth_l1_loss(live['xy'][valid], reference['xy'][valid], beta=.010)
    angle = angle_delta(live['angle'][valid], reference['angle'][valid]).abs().mean()
    return xy+angle
