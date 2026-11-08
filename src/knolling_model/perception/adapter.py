"""Optional bridge to existing Arrangement Objects; no architecture imports in core.

Changing Arrangement only requires replacing this adapter. GT use is an explicit
separate constructor and requires source_mode='oracle_gt' at the bridge call.
"""
import math

import torch
from torchvision.ops import box_convert

from knolling_model.recovery.contracts import detector_labels
from .representation import ObjectFrame, ObjectState


def oracle_frame(truth, object_ids, frame_id):
    """Diagnostic oracle only. This function is never used by predict()."""
    if len(object_ids) != len(truth['source_xy']):
        raise ValueError('oracle IDs/labels mismatch')
    boxes = box_convert(detector_labels([truth])[0]['boxes'], 'cxcywh', 'xyxy')
    objects = []
    for i, oid in enumerate(object_ids):
        color = tuple(float(j == int(truth['color'][i])) for j in range(3))
        shape = tuple(float(j == int(truth['shape'][i])) for j in range(3))
        theta = (float(truth['source_angle'][i])+math.pi/2) % math.pi-math.pi/2
        objects.append(ObjectState(oid, i, tuple(truth['source_xy'][i].tolist()),
            tuple(truth['size'][i].tolist()), theta, 1., tuple(boxes[i].tolist()), color, shape))
    return ObjectFrame(frame_id, 'oracle_gt', tuple(objects)).validate()


def to_arrangement(frames, *, source_mode='predicted_rgb', device='cpu', max_objects=None):
    """Return existing padded Objects plus ID/provenance sidecar; never truncate.

    Empty frames retain one invalid padding slot. An explicit legacy capacity
    overflow raises instead of taking GT count, top-k, or inventing an object.
    """
    from knolling_model.recovery.contracts import Objects
    if not frames or source_mode not in ('predicted_rgb', 'oracle_gt'):
        raise ValueError('frames and explicit supported source mode required')
    for frame in frames:
        frame.validate()
        if frame.source_mode != source_mode:
            raise ValueError('predicted/oracle provenance mismatch')
    n = max(1, max(len(f.objects) for f in frames))
    if max_objects is not None and n > max_objects:
        raise ValueError('Arrangement capacity exceeded; detection truncation forbidden')
    b = len(frames)
    xy = torch.zeros(b, n, 2, device=device)
    size = torch.zeros_like(xy)
    angle = torch.zeros(b, n, device=device)
    color, shape = (torch.zeros(b, n, 3, device=device) for _ in range(2))
    valid = torch.zeros(b, n, dtype=torch.bool, device=device)
    query = torch.full((b, n), -1, dtype=torch.long, device=device)
    for row, frame in enumerate(frames):
        for i, obj in enumerate(frame.objects):
            xy[row, i] = xy.new_tensor(obj.xy_m)
            size[row, i] = size.new_tensor(obj.size_ab_m)
            angle[row, i] = obj.theta_rad
            color[row, i] = color.new_tensor(obj.color_probability)
            shape[row, i] = shape.new_tensor(obj.shape_probability)
            valid[row, i], query[row, i] = True, obj.query_index
    objects = Objects(xy, size, angle, color, shape, valid, query).validate()
    sidecar = [dict(frame_id=f.frame_id, source_mode=f.source_mode,
                    object_ids=[o.object_id for o in f.objects],
                    confidence=[o.confidence for o in f.objects]) for f in frames]
    return objects, sidecar
