"""Read-only Source diagnostics and disposable Target-gradient checks."""
import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from shapely.geometry import box
from torch.nn import functional as F

from knolling_dataset.schema import scene_from_dict
from knolling_dataset.validator import polygon
from knolling_model.coordinates import to_meters
from knolling_model.losses import S0Criterion, match_source
from knolling_model import runner


def stock_sampling_check():
    value = torch.ones(1, 1, 2, 2, device='cuda', requires_grad=True)
    grid = torch.zeros(1, 1, 1, 2, device='cuda', requires_grad=True)
    try:
        F.grid_sample(value, grid, align_corners=False).sum().backward()
    except RuntimeError as exc:
        if 'deterministic' not in str(exc):
            raise
        return {'stock_backward_supported': False, 'error': str(exc),
                'strict_determinism_preserved': torch.are_deterministic_algorithms_enabled()}
    return {'stock_backward_supported': True,
            'strict_determinism_preserved': torch.are_deterministic_algorithms_enabled()}


@torch.no_grad()
def initial_detection(model, dataset, collator, report_dir):
    from transformers.models.rt_detr.modeling_rt_detr import (
        RTDetrMultiscaleDeformableAttention, MultiScaleDeformableAttention)
    model.eval()
    batch = collator([dataset[0], dataset[3]]).to('cuda')
    # Compare against the unmodified sampling implementation, not against our
    # replacement called through the native wrapper.
    cores = [(m, m.attn) for m in model.perception.detector.modules()
             if isinstance(m, RTDetrMultiscaleDeformableAttention)]
    try:
        for module, _ in cores:
            module.attn = MultiScaleDeformableAttention()
        native = model.perception.detector(pixel_values=F.interpolate(batch.inputs['images'],
            size=(640, 640), mode='bilinear', align_corners=False))
    finally:
        for module, core in cores:
            module.attn = core
    actual = model.perception(batch.inputs['images'])[-1]
    parity = {}
    for key, expected in (('boxes', native.pred_boxes), ('logits', native.logits),
                          ('tokens', native.last_hidden_state)):
        torch.testing.assert_close(actual[key], expected, atol=5e-5, rtol=5e-5)
        parity[key+'_max_abs_difference'] = float((actual[key]-expected).abs().max())
    rows, errors, matched_scores, unmatched_scores = [], [], [], []
    gt_total = active_total = true_positive_iou = correct_counts = 0
    for start in range(0, len(dataset), 2):
        batch = collator([dataset[i] for i in range(start, min(start+2, len(dataset)))]).to('cuda')
        output = model(**batch.inputs)
        assignments = match_source(output['source'], batch.supervision)
        for b, (truth, meta, (q, gt)) in enumerate(zip(batch.supervision, batch.metadata, assignments)):
            scores = output['source']['objectness_logits'][b].sigmoid()
            active = torch.where(scores >= .5)[0]
            xy = to_meters(output['source']['uv'][b])
            error = torch.linalg.vector_norm(xy[q]-truth['source_xy'][gt], dim=-1)*1000
            errors.extend(error.cpu().tolist())
            matched_scores.extend(scores[q].cpu().tolist())
            unmatched = torch.ones_like(scores, dtype=torch.bool)
            unmatched[q] = False
            unmatched_scores.extend(scores[unmatched].cpu().tolist())
            gt_objects = {o.object_id: o for o in scene_from_dict(meta['record']['source']).objects}
            gt_boxes = [box(*polygon(gt_objects[oid], gt_objects[oid].source_pose, envelope=False).bounds)
                        for oid in meta['object_ids']]
            native_boxes = output['detector_stages'][-1]['boxes'][b].cpu()
            active_boxes = []
            for query in active.tolist():
                cx, cy, width, height = native_boxes[query].tolist()
                active_boxes.append(box((cx-width/2)*.6, (1-cy-height/2)*.4,
                                        (cx+width/2)*.6, (1-cy+height/2)*.4))
            iou = np.array([[p.intersection(g).area/max(p.union(g).area, 1e-12)
                            for g in gt_boxes] for p in active_boxes])
            matched_iou = []
            if len(active_boxes):
                pq, gq = linear_sum_assignment(1-iou)
                matched_iou = iou[pq, gq].tolist()
            true_positive_iou += sum(v >= .5 for v in matched_iou)
            n = len(truth['source_xy'])
            gt_total += n
            active_total += len(active)
            correct_counts += int(len(active) == n)
            rows.append({'sample_id': meta['sample_id'], 'gt_count': n,
                'predicted_count': len(active), 'scores': scores.cpu().tolist(),
                'native_boxes_image_cxcywh': native_boxes.tolist(),
                'source_assignment': {'query': q.cpu().tolist(), 'gt': gt.cpu().tolist()},
                'conditional_source_center_mm': error.cpu().tolist(),
                'active_matched_native_aabb_iou': matched_iou})
    result = {'samples': len(dataset), 'gt_objects': gt_total, 'query_capacity': 300,
        'active_queries': active_total, 'count_correct_scenes': correct_counts,
        'count_accuracy': correct_counts/len(dataset),
        'native_aabb_true_positives_iou05': true_positive_iou,
        'native_aabb_recall_iou05': true_positive_iou/gt_total,
        'native_aabb_precision_iou05': true_positive_iou/active_total if active_total else None,
        'conditional_source_center_p95_mm': float(np.percentile(errors, 95)),
        'matched_score_median': float(np.median(matched_scores)),
        'unmatched_score_median': float(np.median(unmatched_scores)),
        'native_forward_parity': parity,
        'loading': model.perception.detector.pretrained_loading_info,
        'official_s0_success_claim': False,
        'note': 'Zero-shot. Native AABB IoU and GT-assisted center errors are diagnostics, '
                'not physical-pose identity gates or official S0 scores.'}
    runner.write_json(report_dir/'source-detection.json', {'summary': result, 'samples': rows})
    model.train()
    return result


def target_gradients_after_update(model, dataset, collator, digest):
    batch = collator([dataset[0], dataset[3]]).to('cuda')
    model.train()
    def probe():
        model.zero_grad(set_to_none=True)
        output = model(**batch.inputs)
        watched = [s[k] for s in output['detector_stages'] for k in ('tokens', 'boxes')]
        watched.append(output['detector_stages'][-1]['logits'])
        for tensor in watched:
            tensor.retain_grad()
        losses, _ = S0Criterion()(output, batch.supervision)
        losses['target'].backward()  # No Detection contribution can hide a detach.
        assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
                   for t in watched)
        assert all((s['tokens'].grad.abs().sum(-1) > 0).all() for s in output['detector_stages'])
        modules = {'backbone': model.perception.detector.model.backbone,
            'encoder': model.perception.detector.model.encoder,
            'proposal_features': model.perception.detector.model.enc_output,
            'proposal_boxes': model.perception.detector.model.enc_bbox_head,
            'decoder': model.perception.detector.model.decoder,
            'physical_heads': model.source_heads, 'arrangement': model.arrangement,
            'text_projection': model.text_projection}
        norms = {}
        for name, module in modules.items():
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients), name
            norm = math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
            assert norm > 0, name
            norms[name] = norm
        assert all(p.grad is None for p in model.text_encoder.parameters())
        fingerprint = digest({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
        return float(losses['target'].detach()), norms, fingerprint
    first, norms, a = probe()
    second, _, b = probe()
    assert first == second and a == b
    model.zero_grad(set_to_none=True)
    return {'target_loss_only': first, 'module_gradient_l2': norms,
        'all_300_token_rows_receive_target_gradients': True,
        'proposal_and_refinement_gradients_connected': True,
        'repeated_cuda_backward_bitwise_equal': True, 'frozen_text_has_no_gradients': True,
        'gradient_sha256': a}
