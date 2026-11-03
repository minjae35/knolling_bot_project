"""Zero-padded bilinear sampling for RT-DETR, with deterministic CUDA backward.

Equivalent to grid_sample(mode='bilinear', padding_mode='zeros',
align_corners=False). Gather backward uses PyTorch's deterministic reduction
when torch.use_deterministic_algorithms(True) is enabled. No global patches or
changes to PyTorch's determinism policy are made.
"""
import torch
from torch import nn


def bilinear_sample(value, grid):
    """NCHW values and N×Hout×Wout×2 grids in [-1, 1]."""
    n, channels, height, width = value.shape
    x = ((grid[..., 0] + 1) * width - 1) / 2
    y = ((grid[..., 1] + 1) * height - 1) / 2
    x0, y0 = x.floor(), y.floor()
    dx, dy = x - x0, y - y0
    flat = value.flatten(2)
    result = value.new_zeros(n, channels, *grid.shape[1:3])
    for ix, iy, weight in ((x0, y0, (1-dx)*(1-dy)),
                           (x0+1, y0, dx*(1-dy)),
                           (x0, y0+1, (1-dx)*dy),
                           (x0+1, y0+1, dx*dy)):
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        index = (iy.clamp(0, height-1).long()*width + ix.clamp(0, width-1).long())
        sampled = flat.gather(2, index.flatten(1).unsqueeze(1).expand(-1, channels, -1))
        result = result + sampled.reshape_as(result) * (weight*valid).unsqueeze(1)
    return result


class DeterministicDeformableAttention(nn.Module):
    """Drop-in sampling core for transformers 4.57.3 RT-DETR attention."""
    def forward(self, value, spatial_shapes, spatial_shapes_list, level_start_index,
                sampling_locations, attention_weights, im2col_step):
        batch, _, heads, channels = value.shape
        queries, points = sampling_locations.shape[1], sampling_locations.shape[4]
        levels = value.split([h*w for h, w in spatial_shapes_list], dim=1)
        sampled = []
        for level, ((height, width), feature) in enumerate(zip(spatial_shapes_list, levels)):
            feature = feature.permute(0, 2, 3, 1).reshape(batch*heads, channels, height, width)
            grid = sampling_locations[:, :, :, level].permute(0, 2, 1, 3, 4)
            grid = grid.reshape(batch*heads, queries, points, 2)*2-1
            sampled.append(bilinear_sample(feature, grid))
        weights = attention_weights.permute(0, 2, 1, 3, 4)
        weights = weights.reshape(batch*heads, 1, queries, -1)
        output = (torch.stack(sampled, -2).flatten(-2)*weights).sum(-1)
        return output.reshape(batch, heads*channels, queries).transpose(1, 2).contiguous()
