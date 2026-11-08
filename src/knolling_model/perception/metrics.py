"""Class-agnostic envelope AP and confidence-thresholded count/pose metrics.

AP: all 300 queries/image, 101 recall points, IoU .50:.05:.95, no area
stratification or COCO crowd handling (none in these sources). This is explicitly
not the official COCO evaluator. Pose errors use only actual thresholded IoU>=.5
true positives; zero matches yields null errors rather than fabricated accuracy.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, box_iou

from knolling_model.coordinates import angle_delta
from knolling_model.recovery.contracts import detector_labels
from .data import collate
from .model import decode


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                p50=float(np.quantile(a, .5)) if len(a) else None,
                p95=float(np.quantile(a, .95)) if len(a) else None,
                maximum=float(a.max()) if len(a) else None)


def average_precision(rows, threshold):
    detections, total = [], 0
    for image, row in enumerate(rows):
        total += len(row['gt_boxes'])
        for q in range(len(row['scores'])):
            detections.append((float(row['scores'][q]), image, q))
    # Deterministic tie order: source/image order, then detector query index.
    detections.sort(key=lambda r: (-r[0], r[1], r[2]))
    used = [set() for _ in rows]
    tp = []
    ious = [box_iou(r['boxes'], r['gt_boxes']).numpy() for r in rows]
    for _, image, q in detections:
        candidates = [g for g in range(ious[image].shape[1]) if g not in used[image]]
        best = max(candidates, key=lambda g: ious[image][q, g], default=None)
        good = best is not None and ious[image][q, best] >= threshold
        if good:
            used[image].add(best)
        tp.append(int(good))
    if not total:
        raise ValueError('AP requires annotated objects')
    cumulative = np.cumsum(tp)
    recall = cumulative/total
    precision = cumulative/np.arange(1, len(tp)+1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    values = [float(precision[recall >= r].max()) if np.any(recall >= r) else 0.
              for r in np.linspace(0, 1, 101)]
    return sum(values)/101


def summarize(rows, *, threshold=.5):
    if not rows or len({r['family_id'] for r in rows}) != len(rows):
        raise ValueError('evaluation requires unique nonempty source families')
    tp = fp = fn = exact = color = shape = total = 0
    errors = {k: [] for k in ('position_mm', 'major_mm', 'minor_mm', 'angle_deg')}
    diagnostic = {k: [] for k in errors}
    counts, pose_covered, detail = [], 0, []
    for row in rows:
        truth = row['truth']
        selected = torch.where(row['scores'] >= threshold)[0]
        n = len(row['gt_boxes'])
        exact += int(len(selected) == n)
        counts.append(abs(len(selected)-n))
        total += n
        matrix = box_iou(row['boxes'], row['gt_boxes'])
        if len(selected):
            ious = matrix[selected].numpy()
            # Maximize valid match cardinality first, IoU second.
            p, g = linear_sum_assignment(-((ious >= .5).astype(float)*1000+ious))
            pairs = [(int(selected[i]), int(j)) for i, j in zip(p, g) if ious[i, j] >= .5]
        else:
            pairs = []
        tp += len(pairs)
        fp += len(selected)-len(pairs)
        fn += n-len(pairs)
        def measure(pairs, target):
            for q, g in pairs:
                target['position_mm'].append(float((row['xy'][q]-truth['source_xy'][g]).norm()*1000))
                target['major_mm'].append(float((row['size'][q, 0]-truth['size'][g, 0]).abs()*1000))
                target['minor_mm'].append(float((row['size'][q, 1]-truth['size'][g, 1]).abs()*1000))
                target['angle_deg'].append(float(angle_delta(row['angle'][q], truth['source_angle'][g]).abs()*180/torch.pi))
        measure(pairs, errors)
        for q, g in pairs:
            color += int(row['color'][q] == truth['color'][g])
            shape += int(row['shape'][q] == truth['shape'][g])
        pose_covered += len(pairs)
        # Explicit label-assisted diagnostic; never used for operational metrics,
        # inference, adapter IDs, checkpoint ranking, or selecting query count.
        p, g = linear_sum_assignment(-matrix.numpy())
        measure(list(zip(p, g)), diagnostic)
        detail.append(dict(family_id=row['family_id'], predicted_count=len(selected), gt_count=n,
                           tp=len(pairs), fp=len(selected)-len(pairs), fn=n-len(pairs)))
    aps = {f'{v:.2f}': average_precision(rows, v) for v in np.linspace(.5, .95, 10)}
    return dict(schema='perception_metrics_v1', source_mode='predicted_rgb', sources=len(rows),
        gt_objects=total, confidence_threshold=threshold, count_accuracy=exact/len(rows),
        count_mae=sum(counts)/len(counts), true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=tp/(tp+fp) if tp+fp else 0., recall=tp/total,
        ap50=aps['0.50'], ap75=aps['0.75'], map_50_95=sum(aps.values())/len(aps), ap_by_iou=aps,
        detection_box='annotated physical envelope; not mask silhouette', max_detections_per_image=len(rows[0]['scores']),
        ap_protocol='101 recall points; single object class; no area/crowd stratification',
        pose_matched_objects=pose_covered, pose_coverage=pose_covered/total,
        pose_error_scope='only predicted confidence>=threshold with envelope IoU>=0.5; missing detections excluded',
        errors={k: distribution(v) for k, v in errors.items()},
        color_accuracy_on_matches=color/pose_covered if pose_covered else None,
        shape_accuracy_on_matches=shape/pose_covered if pose_covered else None,
        unthresholded_gt_assignment_diagnostic=dict(
            warning='GT-assisted query assignment, NOT detection or deployed pose accuracy',
            errors={k: distribution(v) for k, v in diagnostic.items()}), rows=detail)


@torch.no_grad()
def evaluate(model, dataset, *, device='cpu', batch_size=2, threshold=.5):
    was_training = model.training
    model.eval()
    rows, frames = [], []
    try:
        for start in range(0, len(dataset), batch_size):
            batch = collate([dataset[i] for i in range(start, min(start+batch_size, len(dataset)))])
            # Only RGB reaches forward. All labels remain on CPU, outside model inputs.
            output = model(batch['images'].to(device))
            source = {k: v.detach().cpu() for k, v in output['source'].items()}
            boxes = output['stages'][-1]['boxes'].detach().cpu()
            frames.extend(f.to_dict() for f in decode(output, batch['family_ids'], threshold=threshold))
            for b, truth in enumerate(batch['truths']):
                rows.append(dict(family_id=batch['family_ids'][b], truth=truth,
                    gt_boxes=box_convert(detector_labels([truth])[0]['boxes'], 'cxcywh', 'xyxy'),
                    boxes=box_convert(boxes[b], 'cxcywh', 'xyxy'),
                    scores=source['objectness_logits'][b].sigmoid(), xy=source['uv'][b]*torch.tensor([.6, .4]),
                    size=source['size'][b], angle=source['angle'][b],
                    color=source['color_logits'][b].argmax(-1), shape=source['shape_logits'][b].argmax(-1)))
        metrics = summarize(rows, threshold=threshold)
        metrics.update(split=dataset.split, dataset_fingerprint=dataset.fingerprint)
        return metrics, frames
    finally:
        model.train(was_training)
