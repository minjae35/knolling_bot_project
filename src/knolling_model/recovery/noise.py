"""Optional source-geometry noise; scales must come from measured detector error.

No default noise budget is invented. This diagnostic transform is never applied
inside inference and never changes object identity, target labels or cardinality.
"""
from dataclasses import replace
import math

import torch

from knolling_model.coordinates import wrap_pi


def perturb_sources(objects, *, xy_std_m, size_std_m, angle_std_rad, generator):
    scales = (xy_std_m, size_std_m, angle_std_rad)
    if any(not math.isfinite(s) or s < 0 for s in scales):
        raise ValueError('noise scales must be finite, nonnegative measured values')
    def noise(value, scale):
        return value + torch.randn(value.shape, dtype=value.dtype, device=value.device,
                                   generator=generator)*scale
    xy = noise(objects.xy, xy_std_m)
    size = noise(objects.size, size_std_m).clamp_min(1e-4).sort(-1, descending=True).values
    angle = wrap_pi(noise(objects.angle, angle_std_rad))
    return replace(objects,
                   xy=torch.where(objects.valid[..., None], xy, objects.xy),
                   size=torch.where(objects.valid[..., None], size, objects.size),
                   angle=torch.where(objects.valid, angle, objects.angle)).validate()
