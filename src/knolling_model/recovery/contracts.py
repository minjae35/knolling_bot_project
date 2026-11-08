"""Physical object interface shared by oracle diagnostics and RGB inference."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class Objects:
    xy: torch.Tensor                  # B,N,2; metres, workspace y-up
    size: torch.Tensor                # B,N,2; major/minor metres
    angle: torch.Tensor               # B,N; pi periodic
    color: torch.Tensor               # B,N,3; probabilities
    shape: torch.Tensor               # B,N,3; probabilities
    valid: torch.Tensor               # B,N; padding mask, not confidence
    query_index: torch.Tensor         # B,N; original detector row, padding=-1

    def validate(self):
        b, n = self.valid.shape
        expected = {'xy': (b, n, 2), 'size': (b, n, 2), 'angle': (b, n),
                    'color': (b, n, 3), 'shape': (b, n, 3), 'query_index': (b, n)}
        if n < 1 or self.valid.dtype != torch.bool:
            raise ValueError('objects require at least one padded slot and a Boolean mask')
        for key, shape in expected.items():
            value = getattr(self, key)
            if value.shape != shape or value.device != self.xy.device:
                raise ValueError(f'invalid object field {key}')
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f'nonfinite object field {key}')
        if self.valid.device != self.xy.device or self.query_index.dtype != torch.long:
            raise ValueError('invalid mask device or query identity dtype')
        if (self.size[self.valid] <= 0).any():
            raise ValueError('physical dimensions must be positive')
        for probability in (self.color, self.shape):
            p = probability[self.valid]
            if (p < 0).any() or not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5):
                raise ValueError('semantic features must be probabilities')
        if ((self.query_index < 0) & self.valid).any():
            raise ValueError('valid objects require a source identity')
        for row, mask in zip(self.query_index, self.valid):
            if row[mask].unique().numel() != mask.sum():
                raise ValueError('duplicate object identity')
        # Packed prefixes let autoregressive decoding ignore trailing padding.
        if ((~self.valid[:, :-1]) & self.valid[:, 1:]).any():
            raise ValueError('valid objects must precede padding')
        return self

    def features(self):
        self.validate()
        scale = self.xy.new_tensor([.6, .4])
        return torch.cat((self.xy / scale, self.size / scale,
                          torch.stack(((2*self.angle).cos(), (2*self.angle).sin()), -1),
                          self.color, self.shape), -1).masked_fill(~self.valid[..., None], 0)


def pack_sources(source, indices):
    """Gather selected continuous attributes without detaching their gradients.

    Indices are discrete. Inference uses foreground predictions; training may
    explicitly use source-only matching. Neither path sorts by target pose.
    """
    b = source['uv'].shape[0]
    if len(indices) != b:
        raise ValueError('one index vector per image required')
    n = max(1, max(i.numel() for i in indices))
    fields = {'xy': source['uv'] * source['uv'].new_tensor([.6, .4]),
              'size': source['size'], 'angle': source['angle'],
              'color': source['color_logits'].softmax(-1),
              'shape': source['shape_logits'].softmax(-1)}
    packed = {}
    for name, values in fields.items():
        rows = []
        for row, idx in enumerate(indices):
            selected = values[row, idx]
            rows.append(torch.cat((selected, values.new_zeros((n-len(idx), *values.shape[2:])))))
        packed[name] = torch.stack(rows)
    mask = torch.zeros((b, n), dtype=torch.bool, device=source['uv'].device)
    identity = torch.full((b, n), -1, dtype=torch.long, device=mask.device)
    for row, idx in enumerate(indices):
        mask[row, :len(idx)] = True
        identity[row, :len(idx)] = idx
    return Objects(**packed, valid=mask, query_index=identity).validate()


def oracle_objects(truths):
    """Explicit diagnostic/training helper; never called by RGB inference."""
    n = max(1, max(len(t['source_xy']) for t in truths))
    device = truths[0]['source_xy'].device
    def pad(key, dimensions):
        return torch.stack([torch.cat((t[key], t[key].new_zeros((n-len(t[key]), *dimensions))))
                            for t in truths])
    mask = torch.arange(n, device=device)[None] < torch.tensor(
        [len(t['source_xy']) for t in truths], device=device)[:, None]
    identity = torch.arange(n, device=device).expand(len(truths), -1).masked_fill(~mask, -1)
    probs = {}
    for key in ('color', 'shape'):
        probs[key] = F.one_hot(pad(key, ()), 3).to(truths[0]['source_xy'].dtype)
    return Objects(pad('source_xy', (2,)), pad('size', (2,)), pad('source_angle', ()),
                   **probs, valid=mask, query_index=identity).validate()


def detector_labels(truths):
    """Rotated physical rectangle -> axis-aligned normalized image box.

    These are envelope labels (also for chamfer/capsule); y flips only once.
    No target pose or instruction information enters detector supervision.
    """
    labels = []
    for t in truths:
        c, s = t['source_angle'].cos().abs(), t['source_angle'].sin().abs()
        a, b = t['size'].unbind(-1)
        wh = torch.stack((a*c + b*s, a*s + b*c), -1) / t['size'].new_tensor([.6, .4])
        uv = t['source_xy'] / t['size'].new_tensor([.6, .4])
        center = torch.stack((uv[:, 0], 1-uv[:, 1]), -1)
        boxes = torch.cat((center, wh), -1)
        if not torch.isfinite(boxes).all() or (wh <= 0).any():
            raise ValueError('invalid source box labels')
        labels.append({'boxes': boxes, 'class_labels': torch.zeros(len(boxes), dtype=torch.long,
                                                                   device=boxes.device)})
    return labels
