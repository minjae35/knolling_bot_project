"""Read-only composition of pinned Perception and Neural Keep checkpoints.

This adapter supplies predicted physical objects to the existing layout graph.
There is no teacher, target, GT count/mask, planning, or training in this path.
"""
import json
from pathlib import Path

import torch

from knolling_dataset.builder import file_hash
from .perception import checkpoint as perception_checkpoint
from .perception.adapter import to_arrangement
from .perception.model import build_model
from .arrangement.alternating import alternating_evaluation
from .arrangement.model import load_recovery
from .arrangement.relative_task import restore
from .arrangement.runtime import FrozenTextCache
from .arrangement.storage import load_artifact


class CurrentSystem:
    def __init__(self, perception, arrangement, tokenizer):
        self.perception = perception.eval()
        self.arrangement = arrangement.eval()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def predict(self, images, commands, frame_ids):
        if len(images) != len(commands) or len(images) != len(frame_ids):
            raise ValueError('one command and frame ID required per image')
        frames = self.perception.predict(images, frame_ids, threshold=self.arrangement.threshold)
        objects, provenance = to_arrangement(frames, device=images.device,
            max_objects=self.arrangement.max_objects)
        tokens = self.tokenizer(commands, return_tensors='pt', padding=True, truncation=False)
        ids = tokens['input_ids'].to(images.device)
        mask = tokens['attention_mask'].to(images.device).bool()
        with alternating_evaluation(self.arrangement):
            text = self.arrangement.encode_text(ids, mask)
            layout = self.arrangement.layout(objects, text, mask)
        return dict(objects=objects, frames=frames, provenance=provenance, layout=layout)


def load_current(manifest='configs/current_models.json', *, device='cpu'):
    spec = json.loads(Path(manifest).read_text())
    if spec.get('schema') != 'knolling_current_models_v1':
        raise ValueError('unsupported current model manifest')
    for key in ('arrangement', 'perception'):
        item = spec[key]
        if file_hash(item['checkpoint']) != item['sha256']:
            raise ValueError(f'{key} checkpoint hash mismatch')
    saved = load_artifact(spec['arrangement']['checkpoint'])
    if saved['config'].get('keep_router') != 'neural_keep_router_v1':
        raise ValueError('current Arrangement must be the Neural Keep model')
    arrangement, tokenizer = load_recovery(spec['arrangement']['model_config'])
    restore(arrangement, saved)
    arrangement.encode_text = FrozenTextCache(arrangement.encode_text)
    payload = perception_checkpoint.read(spec['perception']['checkpoint'])
    perception = build_model(payload['pins'], num_denoising=payload['config']['num_denoising'])
    perception_checkpoint.restore(payload, perception)
    return CurrentSystem(perception.to(device), arrangement.to(device), tokenizer)
