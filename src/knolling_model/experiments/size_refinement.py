"""Object-wise residual correction of the frozen detector's size head.

The detector, language encoder, goal encoder, relation head, and row layout
stay fixed. Ground-truth size is a training target only and is not an input.
"""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from knolling_model.experiments.constraint_decode import apply_area_order, select_layout
from knolling_model.experiments.geometry_decoder import apply_geometry
from knolling_model.experiments.partial_arrangement import (
    arrange_held, command_is_full, text_requests_every_object)
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.selection import select_queries


class ObjectSizeRefinement(nn.Module):
    """Add a residual to the predicted major and minor axes.

    The last readout starts at zero, so the frozen size prediction is unchanged
    until this module is trained. Inputs are the detection query and the RGB
    crop of its predicted box.
    """

    def __init__(self, token_dim=256, resolution=8):
        super().__init__()
        self.resolution = resolution
        self.crop = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.readout = nn.Sequential(
            nn.Linear(token_dim+16, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, tokens, images, boxes, base_size):
        if tokens.ndim != 3 or images.ndim != 4 or boxes.shape != (*tokens.shape[:2], 4):
            raise ValueError('one token and one predicted box per query')
        if base_size.shape != (*tokens.shape[:2], 2) or images.shape[0] != tokens.shape[0]:
            raise ValueError('base size and RGB do not match the queries')
        region = _region_features(self.crop, images, boxes, self.resolution)
        residual = self.readout(torch.cat((tokens, region), -1))
        return (base_size+residual).clamp_min(1e-4)


def matched_queries(tokens, images, boxes, base_size, pairs):
    """Stack matched queries as a batch of single-object crops.

    `pairs` holds (image index, query index). Ground-truth size is not read.
    """
    if not pairs:
        raise ValueError('size refinement needs at least one matched query')
    index = torch.tensor([item[0] for item in pairs], device=tokens.device, dtype=torch.long)
    query = torch.tensor([item[1] for item in pairs], device=tokens.device, dtype=torch.long)
    chosen_tokens = tokens[index, query][:, None].detach()
    chosen_boxes = boxes[index, query][:, None].detach()
    chosen_size = base_size[index, query][:, None].detach()
    return images[index].detach(), chosen_tokens, chosen_boxes, chosen_size


def scatter_refined(base, pairs, refined):
    """Write refined sizes back without dropping the residual gradient."""
    if not pairs:
        return base.detach()
    index = torch.tensor([item[0] for item in pairs], device=base.device, dtype=torch.long)
    query = torch.tensor([item[1] for item in pairs], device=base.device, dtype=torch.long)
    flat = base.detach().reshape(-1, 2)
    slot = index*base.shape[1]+query
    delta = torch.zeros_like(flat)
    delta = delta.index_add(0, slot, refined-flat[slot])
    return (flat+delta).view_as(base)


def install_refined_size(head, images, visual, threshold):
    """Replace selected query sizes. Centers, angles, and scores stay put."""
    source = visual['source']
    tokens = visual['stages'][-1]['tokens']
    boxes = visual['stages'][-1]['boxes']
    chosen = select_queries(source, threshold)
    pairs = [(image, int(query)) for image, queries in enumerate(chosen) for query in queries.tolist()]
    if not pairs:
        size = source['size'].detach().clone()
    else:
        rgb, query_tokens, query_boxes, query_size = matched_queries(
            tokens, images, boxes, source['size'], pairs)
        refined = head(query_tokens, rgb, query_boxes, query_size)[:, 0]
        size = scatter_refined(source['size'], pairs, refined)
    replaced = dict(source)
    replaced['size'] = size
    return dict(visual, source=replaced)


def encode_instruction(system, texts, device):
    tokens = system.tokenizer(texts, return_tensors='pt', padding=True, truncation=False)
    mask = tokens['attention_mask'].to(device).bool()
    text = system.arrangement.text_encoder(
        input_ids=tokens['input_ids'].to(device), attention_mask=mask).last_hidden_state
    return text, mask


def live_poses(system, relation, row_model, head, images, texts, visual=None, text=None, mask=None):
    """Same RGB path as inference, with a gradient only through the size residual.

    Perception, text, and the base size are detached. The goal encoder, relation
    hidden, and row layout stay in the graph so a pose loss can reach the residual,
    but this function does not read a target pose, row, or order.
    """
    if images.ndim != 4 or len(images) != len(texts):
        raise ValueError('one instruction per image required')
    if visual is None or text is None or mask is None:
        with torch.no_grad():
            visual = system.perception(images)
            text, mask = encode_instruction(system, texts, images.device)
    source = visual['source']
    chosen = select_queries(source, system.threshold)
    pairs = [(image, int(query)) for image, queries in enumerate(chosen) for query in queries.tolist()]
    if not pairs or any(len(queries) < 1 or len(queries) > 8 for queries in chosen):
        raise ValueError('detection count is outside 1..8')
    rgb, query_tokens, query_boxes, query_size = matched_queries(
        visual['stages'][-1]['tokens'], images, visual['stages'][-1]['boxes'], source['size'], pairs)
    refined = head(query_tokens, rgb, query_boxes, query_size)[:, 0]
    size = scatter_refined(source['size'], pairs, refined)
    replaced = dict(source)
    replaced['size'] = size
    objects = pack_sources(replaced, chosen)
    goal = system.layout.goal_encoder(objects, text, mask)['vector']
    hidden = relation._hidden(objects.features())
    output = row_model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                       None, None, True, colors=objects.color)
    output = select_layout(row_model, hidden, objects, goal, output)
    return apply_geometry(output, objects, goal), objects, refined, pairs


_KEEP_HEAD = None
_KEEP_TRIED = False
_CHROMA = 25
_WINDOW_M = 0.08


def _direction_angle(module, tokens):
    raw = module(tokens)
    near_zero = raw.square().sum(-1) < 1e-12
    horizontal = torch.where(near_zero, raw[..., 0] + 1e-6, raw[..., 0])
    return .5 * torch.atan2(raw[..., 1], horizontal)


def silhouette_angle(image, xy):
    """Major axis of the colored pixels around one detected center.

    Rectangles repeat every half turn, so the result is in [-pi/2, pi/2).
    A missing silhouette returns None and the caller keeps the direction head.
    """
    import math
    import numpy as np
    from scipy import ndimage

    plane = image.detach().cpu()
    if plane.ndim != 3:
        raise ValueError('one RGB image required')
    if plane.shape[0] == 3:
        plane = plane.permute(1, 2, 0)
    height, width = plane.shape[:2]
    pixels = plane.numpy()
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)
    chroma = pixels.max(-1).astype(np.int16) - pixels.min(-1).astype(np.int16)
    x_m, y_m = (float(item) for item in xy)
    center_x = width * x_m / 0.6 - 0.5
    center_y = height * (1.0 - y_m / 0.4) - 0.5
    radius = _WINDOW_M / 0.6 * width
    x0, x1 = int(max(0, math.floor(center_x - radius))), int(min(width, math.ceil(center_x + radius)))
    y0, y1 = int(max(0, math.floor(center_y - radius))), int(min(height, math.ceil(center_y + radius)))
    labeled, _count = ndimage.label(chroma[y0:y1, x0:x1] > _CHROMA)
    if labeled.size == 0 or not labeled.any():
        return None
    local_x = min(max(int(round(center_x - x0)), 0), labeled.shape[1] - 1)
    local_y = min(max(int(round(center_y - y0)), 0), labeled.shape[0] - 1)
    chosen = int(labeled[local_y, local_x])
    if chosen == 0:
        ys, xs = np.nonzero(labeled)
        if len(xs) == 0:
            return None
        nearest = int(np.argmin((xs - local_x) ** 2 + (ys - local_y) ** 2))
        if (int(xs[nearest]) - local_x) ** 2 + (int(ys[nearest]) - local_y) ** 2 > 12 ** 2:
            return None
        chosen = int(labeled[int(ys[nearest]), int(xs[nearest])])
    ys, xs = np.nonzero(labeled == chosen)
    if len(xs) < 40:
        return None
    coords = np.vstack((xs.astype(np.float64), ys.astype(np.float64)))
    coords -= coords.mean(1, keepdims=True)
    values, vectors = np.linalg.eigh(np.cov(coords))
    horizontal, vertical = vectors[:, int(np.argmax(values))]
    return (math.atan2(-vertical, horizontal) + math.pi / 2) % math.pi - math.pi / 2


def direction_keep_angles(system, visual, queries):
    """Direction-head angle for objects that stay put. None when that checkpoint is absent."""
    global _KEEP_HEAD, _KEEP_TRIED
    if not _KEEP_TRIED:
        _KEEP_TRIED = True
        path = Path(__file__).resolve().parents[3] / 'artifacts' / 'experiments' / 'selector_full' / 'direction-head.pt'
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            head = deepcopy(system.perception.physical.direction).cpu()
            head.load_state_dict(saved['direction'])
            head.eval()
            head.requires_grad_(False)
            _KEEP_HEAD = head
    if _KEEP_HEAD is None:
        return None
    tokens = visual['stages'][-1]['tokens']
    module = _KEEP_HEAD.to(tokens.device)
    packed = tokens.new_zeros(len(queries), max(int(item.numel()) for item in queries))
    for row, chosen in enumerate(queries):
        if int(chosen.numel()) == 0:
            continue
        packed[row, :int(chosen.numel())] = _direction_angle(module, tokens[row, chosen])
    return packed


def keep_angles(system, visual, queries, images):
    """Source angle for objects that stay put. The layout solver does not read it."""
    neural = direction_keep_angles(system, visual, queries)
    width = max(int(item.numel()) for item in queries)
    if neural is None:
        packed = visual['source']['angle'].new_zeros(len(queries), width)
        source = visual['source']['angle']
        for row, chosen in enumerate(queries):
            if int(chosen.numel()) == 0:
                continue
            packed[row, :int(chosen.numel())] = source[row, chosen]
    else:
        packed = neural.clone()
    scale = visual['source']['uv'].new_tensor([.6, .4])
    for row, chosen in enumerate(queries):
        for slot, query in enumerate(chosen.tolist()):
            angle = silhouette_angle(images[row], visual['source']['uv'][row, query] * scale)
            if angle is not None:
                packed[row, slot] = packed.new_tensor(angle)
    return packed


DUPLICATE_M = 0.010


def suppress_duplicate_detections(source, threshold=0.5, radius=DUPLICATE_M):
    """Ignore a lower-scoring detection that repeats one already kept.

    The detector weights are not changed. A detection farther than `radius`
    stays, including one that the threshold itself rejected.
    """
    logits = source['objectness_logits']
    centers = source['uv']
    if logits.ndim != 2 or centers.shape[:2] != logits.shape:
        raise ValueError('duplicate suppression expects one score row per image')
    scores = logits.detach()
    xy = centers.detach() * centers.new_tensor([0.6, 0.4])
    updated = logits.clone()
    for row in range(logits.shape[0]):
        chosen = torch.where(scores[row].sigmoid() >= threshold)[0]
        order = chosen[scores[row, chosen].argsort(descending=True)]
        kept = []
        for query in order.tolist():
            repeated = any(
                float(torch.linalg.vector_norm(xy[row, query] - xy[row, other])) < radius
                for other in kept)
            if repeated:
                updated[row, query] = -20
            else:
                kept.append(query)
    replaced = dict(source)
    replaced['objectness_logits'] = updated
    return replaced


def install_selection_head(system):
    """Use the trained selection layer when its checkpoint is present."""
    if getattr(system, '_selection_head_ready', False):
        return
    path = Path(__file__).resolve().parents[3] / 'artifacts' / 'experiments' / 'selector_full' / 'selection-head.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        system.layout.goal_encoder.selection.load_state_dict(saved['selection'])
    system._selection_head_ready = True


@torch.no_grad()
def poses_from_detection(system, relation, row_model, images, texts, head=None):
    """RGB and text to target poses. No ground-truth size, pose, row, or order."""
    if images.ndim != 4 or len(images) != len(texts):
        raise ValueError('one instruction per image required')
    install_selection_head(system)
    visual = system.perception(images)
    if head is not None:
        visual = install_refined_size(head, images, visual, system.threshold)
    visual = dict(visual, source=suppress_duplicate_detections(visual['source'], system.threshold))
    queries = select_queries(visual['source'], system.threshold)
    if any(len(chosen) < 1 or len(chosen) > 8 for chosen in queries):
        raise ValueError('detection count is outside 1..8')
    objects = pack_sources(visual['source'], queries)
    text, mask = encode_instruction(system, texts, images.device)
    encoded = system.layout.goal_encoder(objects, text, mask)
    goal = encoded['vector']
    if not text_requests_every_object(texts) and not command_is_full(
            encoded['keep_logit'], encoded['selected'], objects.valid):
        return arrange_held(
            relation, row_model, objects, encoded, keep_angles(system, visual, queries, images), texts), objects, visual
    hidden = relation._hidden(objects.features())
    output = row_model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                       None, None, True, colors=objects.color)
    output = select_layout(row_model, hidden, objects, goal, output)
    ordered = apply_area_order(output, objects, goal)
    placed = apply_geometry(ordered, objects, goal)
    undone = [index for index, name in enumerate(placed['geometry_status']) if str(name).startswith('infeasible')]
    if undone:
        fallback = apply_geometry(output, objects, goal)
        for index in undone:
            placed['xy'][index] = fallback['xy'][index]
            placed['angle'][index] = fallback['angle'][index]
            placed['row'][index] = fallback['row'][index]
            placed['slot'][index] = fallback['slot'][index]
            placed['geometry_status'][index] = fallback['geometry_status'][index]
    placed['route'] = 'full'
    placed['selected'] = encoded['selected']
    placed['keep_probability'] = encoded['keep_logit'].sigmoid()
    return placed, objects, visual


def horizontal_extent(size, angle):
    """Width of the rotated rectangle along workspace x, the row direction."""
    return size[..., 0]*angle.cos().abs()+size[..., 1]*angle.sin().abs()


def _region_features(crop, images, boxes, resolution):
    batch, queries, _ = boxes.shape
    axis = torch.linspace(-.5, .5, resolution, device=boxes.device, dtype=boxes.dtype)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing='ij')
    center_x, center_y, width, height = boxes.unbind(-1)
    x = center_x[..., None, None]+grid_x*width[..., None, None]
    y = center_y[..., None, None]+grid_y*height[..., None, None]
    grid = torch.stack((x*2-1, y*2-1), -1).reshape(batch*queries, resolution, resolution, 2)
    image = images[:, None].expand(-1, queries, -1, -1, -1).reshape(
        batch*queries, images.shape[1], images.shape[-2], images.shape[-1])
    patch = F.grid_sample(image, grid, align_corners=False, padding_mode='border')
    return crop(patch).flatten(1).reshape(batch, queries, -1)
