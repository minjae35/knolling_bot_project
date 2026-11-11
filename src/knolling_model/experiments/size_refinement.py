"""Object-wise residual correction of the frozen detector's size head.

The detector, language encoder, goal encoder, relation head, and row layout
stay fixed. Ground-truth size is a training target only and is not an input.
"""
import torch
from torch import nn
from torch.nn import functional as F

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
                       None, None, True)
    return output, objects, refined, pairs


@torch.no_grad()
def poses_from_detection(system, relation, row_model, images, texts, head=None):
    """RGB and text to target poses. No ground-truth size, pose, row, or order."""
    if images.ndim != 4 or len(images) != len(texts):
        raise ValueError('one instruction per image required')
    visual = system.perception(images)
    if head is not None:
        visual = install_refined_size(head, images, visual, system.threshold)
    queries = select_queries(visual['source'], system.threshold)
    if any(len(chosen) < 1 or len(chosen) > 8 for chosen in queries):
        raise ValueError('detection count is outside 1..8')
    objects = pack_sources(visual['source'], queries)
    text, mask = encode_instruction(system, texts, images.device)
    goal = system.layout.goal_encoder(objects, text, mask)['vector']
    hidden = relation._hidden(objects.features())
    output = row_model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                       None, None, True)
    return output, objects, visual


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
