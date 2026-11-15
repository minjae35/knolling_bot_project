"""Pinned RGB + language inference, without truth/teacher/count inputs.

Object identity is the detector's frame-local Query ID, not a dataset ID.
Ground-truth association belongs exclusively to the external evaluator.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn

from knolling_dataset.builder import file_hash
from knolling_model.current import load_current
from knolling_model.perception.adapter import to_arrangement
from knolling_model.perception.model import decode
from knolling_model.arrangement.alternating import alternating_evaluation
from knolling_model.arrangement.model import CapacityError
from knolling_model.arrangement.storage import load_artifact
from .architectures import AppearancePerception, StructurePose


class IntegratedSystem(nn.Module):
    def __init__(self, perception, arrangement, tokenizer):
        super().__init__()
        self.perception, self.arrangement = perception, arrangement
        self.tokenizer = tokenizer
        self.capture = None
        self.eval()

    @property
    def layout(self): return self.arrangement.layout

    @property
    def threshold(self): return self.arrangement.threshold

    @property
    def max_objects(self): return self.arrangement.max_objects

    @property
    def architecture(self): return self.arrangement.architecture

    def encode_text(self, input_ids, attention_mask):
        return self.arrangement.encode_text(input_ids, attention_mask)

    @torch.no_grad()
    def layout_from_perception(self, visual, input_ids, attention_mask, frame_ids=None):
        frame_ids = frame_ids or [f'frame_{i}' for i in range(len(input_ids))]
        frames = decode(visual, frame_ids, threshold=self.threshold)
        counts = [len(f.objects) for f in frames]
        if max(counts) > self.max_objects:
            if self.capture is not None:
                self.capture.extend(dict(status='layout_capacity',source=f.to_dict(),targets=None) for f in frames)
            raise CapacityError(counts, self.max_objects)
        objects, provenance = to_arrangement(frames, device=input_ids.device, max_objects=self.max_objects)
        with alternating_evaluation(self.arrangement):
            layout = self.layout(objects, self.encode_text(input_ids, attention_mask), attention_mask)
        if not torch.equal(layout['query_index'], objects.query_index) or not torch.equal(layout['valid'],objects.valid):
            raise ValueError('Arrangement changed detection identity or padding')
        if not torch.isfinite(layout['xy']).all() or not torch.isfinite(layout['angle']).all():
            raise FloatingPointError('nonfinite target pose')
        predictions=[]
        for b,frame in enumerate(frames):
            targets=[]
            for j,obj in enumerate(frame.objects):
                targets.append(dict(object_id=obj.object_id, query_index=obj.query_index,
                    x_m=float(layout['xy'][b,j,0]),y_m=float(layout['xy'][b,j,1]),
                    theta_rad=(float(layout['angle'][b,j])+math.pi/2)%math.pi-math.pi/2))
            predictions.append(dict(status='predicted',source=frame.to_dict(),targets=targets))
        if self.capture is not None:self.capture.extend(predictions)
        return dict(source=visual['source'],layout=layout,goal=layout['goal'],threshold=self.threshold,
                    objects=objects,provenance=provenance,predictions=predictions)

    @torch.no_grad()
    def forward(self, images, input_ids, attention_mask, frame_ids=None):
        if self.training:raise ValueError('integration is inference-only; call eval()')
        if (images.ndim!=4 or images.shape[1:]!=(3,384,576) or
                not torch.isfinite(images).all() or images.min()<0 or images.max()>1):
            raise ValueError('expected finite [B,3,384,576] RGB in [0,1]')
        if len(images)!=len(input_ids):raise ValueError('image/text batch mismatch')
        return self.layout_from_perception(self.perception(images),input_ids,attention_mask,frame_ids)

    @torch.no_grad()
    def predict(self, images, commands, frame_ids):
        if len(images)!=len(commands) or len(images)!=len(frame_ids):
            raise ValueError('one natural-language command and frame ID per image required')
        tokens=self.tokenizer(commands,return_tensors='pt',padding=True,truncation=False)
        return self(images,tokens['input_ids'].to(images.device),
                    tokens['attention_mask'].to(images.device).bool(),frame_ids)


def load_integrated(manifest='configs/structural_integration.json',device='cpu'):
    spec=json.loads(Path(manifest).read_text())
    if spec['schema']!='structural_integration_v1':raise ValueError('unsupported integration manifest')
    for name in ('arrangement','perception'):
        p=spec[name]
        if file_hash(p['path'])!=p['sha256']:raise ValueError(name+' checkpoint hash mismatch')
    baseline=load_current(device=device)
    baseline.arrangement.layout.pose=StructurePose(baseline.arrangement.layout.pose).to(device)
    baseline.perception=AppearancePerception.from_baseline(baseline.perception).to(device)
    for name in ('arrangement','perception'):
        saved=load_artifact(spec[name]['path'])
        if saved['update']!=spec[name]['update']:raise ValueError('unexpected selected update')
        getattr(baseline,name).load_state_dict(saved['model'],strict=True)
    return IntegratedSystem(baseline.perception,baseline.arrangement,baseline.tokenizer).to(device).eval()
