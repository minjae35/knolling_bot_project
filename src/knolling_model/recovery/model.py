"""RGB + tokenized natural language -> source-associated target poses.

No annotations, teacher, rule-based placement, NMS, or GT count in forward().
"""
import torch
from torch import nn

from .contracts import pack_sources
from .layout import AutoregressiveLayout
from .perception import RecoveryPerception


class CapacityError(ValueError):
    def __init__(self, counts, capacity):
        self.counts, self.capacity = counts, capacity
        super().__init__(f'predicted counts {counts} exceed layout capacity {capacity}; no truncation applied')


class RecoveryModel(nn.Module):
    def __init__(self, text_encoder, detector, *, dimension=128, layers=2, bins=18,
                 max_objects=8, threshold=.5, num_denoising=100):
        super().__init__()
        if max_objects < 1 or not 0 < threshold < 1:
            raise ValueError('invalid object capacity or foreground threshold')
        self.text_encoder = text_encoder.requires_grad_(False).eval()
        self.perception = RecoveryPerception(detector, num_denoising)
        self.layout = AutoregressiveLayout(dimension=dimension, layers=layers, bins=bins)
        self.max_objects, self.threshold = max_objects, threshold
        self.stage = None

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()
        if self.stage in ('oracle_layout', 'predicted_layout'):
            self.perception.eval()
        if self.stage == 'perception':
            self.layout.eval()
        return self

    def encode_text(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or input_ids.shape[1] > 512:
            raise ValueError('invalid text shape or instruction exceeds 512 tokens')
        if attention_mask.dtype != torch.bool or not attention_mask.any(-1).all():
            raise ValueError('invalid text attention mask')
        with torch.no_grad():
            return self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(self, images, input_ids, attention_mask):
        visual = self.perception(images)
        return self.layout_from_perception(visual, input_ids, attention_mask)

    def layout_from_perception(self, visual, input_ids, attention_mask):
        """Also used to evaluate a precomputed, label-free perception output."""
        source = visual['source']
        # Stable decoder-row order, never target sorting or confidence weighting.
        indices = [torch.where(row.sigmoid() >= self.threshold)[0]
                   for row in source['objectness_logits']]
        counts = [len(i) for i in indices]
        if max(counts) > self.max_objects:
            raise CapacityError(counts, self.max_objects)
        objects = pack_sources(source, indices)
        text = self.encode_text(input_ids, attention_mask)
        target = self.layout(objects, text, attention_mask)
        return dict(source=source, layout=target, perception=visual, threshold=self.threshold)


def load_recovery(config_path='configs/recovery_review.json'):
    """Pinned, cached base weights only; legacy trained checkpoints are not loaded."""
    from knolling_model.model import load_pretrained
    from .stages import read_review_config
    config = read_review_config(config_path)
    base, tokenizer = load_pretrained(config['pretrained'], local_files_only=True)
    model = RecoveryModel(base.text_encoder, base.perception.detector, **config['model'])
    return model, tokenizer


def legacy_evaluation_view(output):
    """Reuse strict S0 metrics, with exactly ONE final target stage.

    All detector rows and confidence values are preserved. Unselected rows are
    immaterial to the legacy evaluator, but retain their source pose for shape.
    """
    layout, source = output['layout'], output['source']
    if output.get('threshold') != .5:
        raise ValueError('legacy S0 comparison requires its fixed 0.5 threshold')
    if layout['rollout'] != 'free_rollout':
        raise ValueError('official evaluation requires free rollout')
    uv, angle = source['uv'].clone(), source['angle'].clone()
    for b, mask in enumerate(layout['valid']):
        indices = layout['query_index'][b, mask]
        uv[b, indices] = layout['xy'][b, mask]/uv.new_tensor([.6, .4])
        angle[b, indices] = layout['angle'][b, mask]
    return dict(source=source, targets=[dict(uv=uv, angle=angle)])
