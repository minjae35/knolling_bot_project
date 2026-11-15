"""Training-only adjacent-row edge gap targets; neural forward is unchanged."""
import torch
from torch.nn import functional as F
from knolling_dataset.schema import scene_from_dict
from knolling_dataset.validator import inferred_rows, polygon

POLICY = 'adjacent_vertical_gap_v1'
GAP_SCALE = .002  # The existing independent task evaluator's 2 mm tolerance.


def supervision(metadata):
    """Teacher rows are labels, not inferred/passed to the prediction network."""
    result = []
    for sample, meta in enumerate(metadata):
        record = meta['record']; ins = record['instruction']
        if ins['keep_all']: continue
        ids = meta['object_ids']; selected = set(ins['selected_ids'])
        rows = inferred_rows([o for o in scene_from_dict(record['target']).objects if o.object_id in selected])
        adjacent = []
        for upper, lower in zip(rows, rows[1:]):
            target = (min(polygon(o, o.target_pose).bounds[1] for o in upper)
                      - max(polygon(o, o.target_pose).bounds[3] for o in lower))
            adjacent.append(([ids.index(o.object_id) for o in upper],
                             [ids.index(o.object_id) for o in lower], target))
        if adjacent: result.append((sample, adjacent))
    return result


def loss(output, objects, labels):
    """Actual rectangle edges, including frozen predicted orientation/size.

    Min/max matches the independent vertical-gap evaluator. A fixed Teacher
    topology avoids zero supervision when an incorrect predicted layout merges
    two rows. Only supervision knows that topology. Scene and edge averages
    prevent more crowded scenes receiving an arbitrary multiplicity weight.
    """
    angle = output['angle'].detach()  # Orientation is protected, no angle gradient.
    half_y = .5*(objects.size[...,0]*angle.sin().abs()+objects.size[...,1]*angle.cos().abs())
    y = output['xy'][...,1]
    values = []
    for sample, adjacent in labels:
        gaps = []
        for up, down, target in adjacent:
            gap = (y[sample,up]-half_y[sample,up]).min()-(y[sample,down]+half_y[sample,down]).max()
            gaps.append(F.smooth_l1_loss(gap/GAP_SCALE, gap.new_tensor(target/GAP_SCALE)))
        values.append(torch.stack(gaps).mean())
    return torch.stack(values).mean() if values else y.sum()*0
