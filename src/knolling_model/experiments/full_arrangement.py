"""Variable-length full arrangements on the existing relation head.

The head is unchanged. A later object is still the previous object plus one
predicted displacement, so a new row is the displacement from the end of the
previous row to the start of the next. Nothing here adds a solver or reads a
target pose during inference.
"""
import torch

from knolling_dataset.schema import scene_from_dict
from knolling_dataset.validator import geometry, inferred_rows, polygon
from knolling_model.diagnostics import positioned_objects
from knolling_model.arrangement.evaluation_v3 import semantic_errors_v3

MAX_OBJECTS = 8


def is_full(record):
    instruction = record['instruction']
    identities = {obj['object_id'] for obj in record['source']['objects']}
    return (record['mode'] == 'full' and not instruction['keep_all'] and not instruction['locked_ids']
            and set(instruction['selected_ids']) == identities and 2 <= len(identities) <= MAX_OBJECTS)


def selected_objects(record, poses=None):
    objects = positioned_objects(record, poses) if poses is not None else scene_from_dict(record['target']).objects
    chosen = set(record['instruction']['selected_ids'])
    return [obj for obj in objects if obj.object_id in chosen]


def row_ids(objects):
    return [[obj.object_id for obj in row] for row in inferred_rows(objects)]


def gap_errors(record, objects):
    instruction = record['instruction']
    rows = inferred_rows(objects)
    horizontal, vertical = [], []
    for row in rows:
        for left, right in zip(row, row[1:]):
            opening = polygon(right, right.target_pose).bounds[0]-polygon(left, left.target_pose).bounds[2]
            horizontal.append(abs(opening-instruction['gap_m'])*1000)
    for upper, lower in zip(rows, rows[1:]):
        opening = min(polygon(obj, obj.target_pose).bounds[1] for obj in upper)
        opening -= max(polygon(obj, obj.target_pose).bounds[3] for obj in lower)
        multiplier = 2 if instruction['group_by'] == 'color' and upper[0].color != lower[0].color else 1
        vertical.append(abs(opening-multiplier*instruction['gap_m'])*1000)
    return horizontal, vertical


def arrangement_errors(record, poses=None):
    """Score a pose assignment. Dataset targets are allowed here; model outputs are too."""
    objects = selected_objects(record, poses)
    source_ids = {obj['object_id'] for obj in record['source']['objects']}
    target_ids = {obj['object_id'] for obj in record['target']['objects']}
    pose_ids = {obj.object_id for obj in objects}
    semantics = semantic_errors_v3(objects, record)
    safety = geometry(objects, 'target_pose', .010, .010)
    rows = row_ids(objects)
    teacher = record['supervision_v3']['rows']
    horizontal, vertical = gap_errors(record, objects)
    errors = sorted(set(semantics+safety.errors))
    if source_ids != target_ids or pose_ids != set(record['instruction']['selected_ids']):
        errors = sorted(set(errors+['object_identity']))
    if [oid for row in rows for oid in row] != [oid for row in teacher for oid in row] and poses is None:
        errors = sorted(set(errors+['teacher_row_mismatch']))
    return dict(task=not errors, errors=errors, rows=rows, teacher_rows=teacher, row_match=rows == teacher,
                collision='collision' in safety.errors, boundary='boundary' in safety.errors,
                clearance='clearance' in safety.errors, horizontal_mm=horizontal, vertical_mm=vertical,
                row_transition=len(rows) > 1)


def collate(items, device):
    """Pad one variable-length scene list. Valid objects stay in front."""
    if not items or any(item['valid'].shape[0] > MAX_OBJECTS for item in items):
        raise ValueError('full arrangement batch must contain 1 to 8 objects')
    width = max(item['valid'].shape[0] for item in items)
    batch = {}
    for key, fill in (('features', 0.), ('xy', 0.), ('size', 0.), ('angle', 0.), ('color', 0.), ('shape', 0.),
                      ('valid', False), ('query_index', -1), ('target_xy', 0.), ('target_angle', 0.), ('order', -1)):
        rows = []
        for item in items:
            value = item[key]
            missing = width-value.shape[0]
            tail = value.new_full((missing, *value.shape[1:]), fill)
            rows.append(torch.cat((value, tail)) if missing else value)
        batch[key] = torch.stack(rows).to(device)
    batch['goal'] = torch.stack([item['goal'] for item in items]).to(device)
    if 'break' in items[0]:
        rows = []
        for item in items:
            value = item['break']
            missing = width-value.shape[0]
            rows.append(torch.cat((value, value.new_zeros(missing))) if missing else value)
        batch['break'] = torch.stack(rows).to(device)
    return batch
