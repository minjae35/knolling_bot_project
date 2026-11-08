"""One foreground selection policy for layout training and deployment."""
import torch

from .contracts import detector_labels


class SourceSelectionError(ValueError):
    """A layout update cannot silently replace bad detections with GT objects."""


def select_queries(source, threshold=.5):
    scores = source['objectness_logits']
    if not torch.isfinite(scores).all():
        raise SourceSelectionError('nonfinite foreground scores')
    return [torch.where(row.sigmoid() >= threshold)[0] for row in scores]


@torch.no_grad()
def match_selected(model, visual, truths):
    """Attach source labels AFTER inference selection; never change its rows.

    Layout warmup requires correct cardinality and source correspondence. A bad
    scene aborts the update; joint execution restores its retention checkpoint.
    Detection training itself still supervises all queries through its own loss.
    """
    from transformers.loss.loss_rt_detr import RTDetrHungarianMatcher
    selected = select_queries(visual['source'], model.threshold)
    labels = detector_labels(truths)
    final = visual['stages'][-1]
    matcher = RTDetrHungarianMatcher(model.perception.detector.config)
    matches = []
    for b, (indices, label) in enumerate(zip(selected, labels)):
        if len(indices) != len(label['boxes']) or len(indices) > model.max_objects:
            raise SourceSelectionError(f'scene {b}: predicted count {len(indices)} != supported GT count {len(label["boxes"])}')
        if not len(indices):
            raise SourceSelectionError('empty source scene is outside the training contract')
        prediction = dict(logits=final['logits'][b:b+1, indices], pred_boxes=final['boxes'][b:b+1, indices])
        pred, gt = matcher(prediction, [label])[0]
        order = pred.argsort()
        gt = gt[order].to(indices.device)
        # The matcher is a label join, not an object selector or target sorter.
        xy = visual['source']['uv'][b, indices]*final['boxes'].new_tensor([.6, .4])
        if ((xy-truths[b]['source_xy'][gt]).norm(dim=-1) > .010).any():
            raise SourceSelectionError(f'scene {b}: selected source correspondence exceeds 10mm')
        matches.append((indices, gt))
    return matches
