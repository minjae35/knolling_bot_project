"""Versioned public interface, established from dataset-contract and contracts.Objects.

IDs identify detections within one frame, not persistent tracks or dataset GT IDs.
Height, depth and roll/pitch are intentionally absent: this dataset/interface
supervises planar state. Box coordinates describe the physical envelope, not a mask.
"""
from dataclasses import asdict, dataclass
import math

SCHEMA = 'knolling_objects_v1'
COORDINATES = dict(frame='workspace', origin='bottom_left', x_axis='right', y_axis='up',
                   length_unit='m', workspace_m=[.6, .4], image_wh=[576, 384],
                   angle_unit='rad', angle_axis='major', angle_positive='CCW_from_positive_x',
                   angle_period=math.pi, angle_interval='[-pi/2,pi/2)',
                   box_format='xyxy_normalized_image_y_down_physical_envelope',
                   color_probability_order=['red', 'green', 'blue'],
                   shape_probability_order=['box', 'chamfer', 'capsule'],
                   identity_scope='frame_local_not_tracking',
                   pixel_centers='px=576*x/0.6-0.5; py=384*(1-y/0.4)-0.5')


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    query_index: int
    xy_m: tuple[float, float]
    size_ab_m: tuple[float, float]
    theta_rad: float
    confidence: float
    box_xyxy_image: tuple[float, float, float, float]
    color_probability: tuple[float, float, float]
    shape_probability: tuple[float, float, float]

    def validate(self):
        if not self.object_id or self.query_index < 0:
            raise ValueError('invalid frame-local object identity')
        if any(len(v) != n for v, n in ((self.xy_m, 2), (self.size_ab_m, 2),
                (self.box_xyxy_image, 4), (self.color_probability, 3), (self.shape_probability, 3))):
            raise ValueError('invalid object field dimensions')
        values = (*self.xy_m, *self.size_ab_m, self.theta_rad, self.confidence,
                  *self.box_xyxy_image, *self.color_probability, *self.shape_probability)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite object state')
        a, b = self.size_ab_m
        if not a >= b > 0 or not 0 <= self.confidence <= 1:
            raise ValueError('invalid dimensions or confidence')
        if not -math.pi/2 <= self.theta_rad < math.pi/2:
            raise ValueError('angle must be canonical pi-periodic major-axis orientation')
        x1, y1, x2, y2 = self.box_xyxy_image
        if x2 <= x1 or y2 <= y1:
            raise ValueError('invalid envelope box')
        for p in (self.color_probability, self.shape_probability):
            if min(p) < 0 or abs(sum(p)-1) > 1e-5:
                raise ValueError('semantic features must be probabilities')
        return self


@dataclass(frozen=True)
class ObjectFrame:
    frame_id: str
    source_mode: str  # predicted_rgb or oracle_gt; cannot silently mix
    objects: tuple[ObjectState, ...]

    def validate(self):
        if not self.frame_id or self.source_mode not in ('predicted_rgb', 'oracle_gt'):
            raise ValueError('frame identity and explicit source mode required')
        for obj in self.objects:
            obj.validate()
        if len({o.object_id for o in self.objects}) != len(self.objects) or len(
                {o.query_index for o in self.objects}) != len(self.objects):
            raise ValueError('duplicate frame-local object identity')
        return self

    def to_dict(self):
        self.validate()
        return dict(schema=SCHEMA, coordinates=COORDINATES, **asdict(self))

    @classmethod
    def from_dict(cls, value):
        if value.get('schema') != SCHEMA or value.get('coordinates') != COORDINATES:
            raise ValueError('unsupported object representation or coordinate convention')
        return cls(value['frame_id'], value['source_mode'], tuple(ObjectState(**o)
                   for o in value['objects'])).validate()
