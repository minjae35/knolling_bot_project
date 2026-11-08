"""Reuse the reviewed RT-DETR task detector and supervised physical heads.

The inherited forward accepts RGB only; the training-only supervised path adds
isolated denoising labels. No Arrangement model is instantiated or checkpointed.
"""
import math

import torch
from torchvision.ops import box_convert

from knolling_model.coordinates import wrap_pi
from knolling_model.recovery.perception import RecoveryPerception
from knolling_model.rtdetr import load_detector
from .representation import ObjectFrame, ObjectState

ARCHITECTURE = 'standalone_rtdetr_r18_physical_v1'


class ObjectPerception(RecoveryPerception):
    architecture = ARCHITECTURE

    @torch.no_grad()
    def predict(self, images, frame_ids, *, threshold=.5):
        if self.training:
            raise ValueError('predict requires eval mode')
        return decode(self(images), frame_ids, threshold=threshold)


def build_model(pins, *, num_denoising=100):
    return ObjectPerception(load_detector(pins, local_files_only=True), num_denoising=num_denoising)


def decode(output, frame_ids, *, threshold=.5):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('invalid fixed confidence threshold')
    source = output['source']
    scores = source['objectness_logits'].sigmoid()
    if len(frame_ids) != len(scores):
        raise ValueError('one frame ID required per image')
    boxes = box_convert(output['stages'][-1]['boxes'], 'cxcywh', 'xyxy')
    xy = source['uv']*source['uv'].new_tensor([.6, .4])
    colors, shapes = source['color_logits'].softmax(-1), source['shape_logits'].softmax(-1)
    angles = wrap_pi(source['angle'])
    # Canonicalize in Python double precision too (float32 -pi/2 lies below -pi/2).
    canonical = lambda x: (float(x)+math.pi/2) % math.pi-math.pi/2
    frames = []
    for b, fid in enumerate(frame_ids):
        objects = []
        for q in torch.where(scores[b] >= threshold)[0].tolist():
            objects.append(ObjectState(f'query_{q:03d}', q, tuple(xy[b, q].tolist()),
                tuple(source['size'][b, q].tolist()), canonical(angles[b, q]), float(scores[b, q]),
                tuple(boxes[b, q].tolist()), tuple(colors[b, q].tolist()), tuple(shapes[b, q].tolist())))
        frames.append(ObjectFrame(fid, 'predicted_rgb', tuple(objects)).validate())
    return frames
