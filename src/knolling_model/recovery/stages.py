"""Stage-specific loss construction and freeze policy; contains no optimizer step.

The review config cannot launch a training run. Stage functions expose forward /
backward graphs for CPU tests and later independent review of training wiring.
"""
import json
from pathlib import Path

import torch

from .contracts import detector_labels, oracle_objects, pack_sources
from .layout import layout_loss

STAGES = ('oracle_layout', 'perception', 'predicted_layout', 'joint')


def read_review_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('schema') != 'neural_recovery_review_v1' or config.get('execution') != 'review_only':
        raise ValueError('only the review-only recovery configuration is accepted')
    if config.get('allow_training') is not False or config.get('device') != 'cpu':
        raise ValueError('GPU and training execution are disabled pending review')
    if tuple(config['stages']) != STAGES:
        raise ValueError('recovery stage order changed')
    return config


def configure_stage(model, stage):
    if stage not in STAGES:
        raise ValueError('unknown recovery stage')
    model.requires_grad_(False)
    if stage in ('oracle_layout', 'predicted_layout', 'joint'):
        model.layout.requires_grad_(True)
    if stage == 'perception':
        model.perception.requires_grad_(True)
    elif stage == 'joint':
        detector = model.perception.detector
        # Bounded joint tuning: final decoder and final task/geometry heads only.
        for module in (detector.model.decoder.layers[-1], detector.bbox_embed[-1],
                       detector.class_embed[-1], model.perception.physical):
            module.requires_grad_(True)
    model.stage = stage
    model.train()
    return [name for name, p in model.named_parameters() if p.requires_grad]


def parameter_groups(model, layout_lr=1e-4, perception_lr=1e-5):
    """Construct groups without creating an optimizer or modifying parameters."""
    groups = []
    for name, module, lr in (('layout', model.layout, layout_lr), ('perception', model.perception, perception_lr)):
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            groups.append(dict(name=name, params=params, lr=lr))
    return groups


def aligned_targets(truths, gt_indices, objects):
    n = objects.valid.shape[1]
    output = {}
    for destination, origin in (('xy', 'target_xy'), ('angle', 'target_angle'),
                                ('move', 'move'), ('selected', 'selected')):
        rows = []
        for truth, index in zip(truths, gt_indices):
            value = truth[origin][index]
            rows.append(torch.cat((value, value.new_zeros((n-len(value), *value.shape[1:])))))
        output[destination] = torch.stack(rows)
    for key in ('axis', 'region'):
        output[key] = torch.stack([t[key] for t in truths])
    return output


def source_matches(model, visual, truths):
    from transformers.loss.loss_rt_detr import RTDetrHungarianMatcher
    final = visual['stages'][-1]
    matches = RTDetrHungarianMatcher(model.perception.detector.config)(
        dict(logits=final['logits'], pred_boxes=final['boxes']), detector_labels(truths))
    result = []
    for pred, gt in matches:
        order = pred.argsort()
        result.append((pred[order].to(final['boxes'].device), gt[order].to(final['boxes'].device)))
    return result


def stage_loss(model, inputs, truths, *, teacher_forcing=False, replay=None, source_noise=None):
    """Return losses and diagnostic outputs; never calls backward/step/save.

    Predicted-layout training uses source-only matching even below threshold,
    explicitly conditioning on correct cardinality. Deployment still thresholds
    predicted foreground, so a separate strict inference gate is mandatory.

    Joint detection replay must use the full unique-source stream, not only the
    instruction-balanced layout batch. The caller supplies that separate batch.
    """
    stage = model.stage
    if stage not in STAGES or not model.training:
        raise ValueError('configure a training stage before constructing its loss')
    if teacher_forcing and stage not in ('oracle_layout', 'predicted_layout'):
        raise ValueError('teacher forcing is limited to layout warmup stages')
    if source_noise is not None and stage not in ('oracle_layout', 'predicted_layout'):
        raise ValueError('source noise is limited to layout warmup stages')
    if stage == 'perception':
        visual = model.perception.supervised(inputs['images'], truths)
        terms, _ = model.perception.detection_loss(visual, truths)
        return terms, dict(perception=visual, diagnostic='perception_only')
    if stage == 'oracle_layout':
        objects = oracle_objects(truths)
        gt_indices = [torch.arange(len(t['source_xy']), device=t['source_xy'].device) for t in truths]
        diagnostic = 'oracle_source'
    else:
        visual = model.perception(inputs['images'])
        matches = source_matches(model, visual, truths)
        if any(len(pred) != len(t['source_xy']) for (pred, _), t in zip(matches, truths)):
            raise ValueError('not enough detector queries for training correspondence')
        objects = pack_sources(visual['source'], [pred for pred, _ in matches])
        gt_indices = [gt for _, gt in matches]
        diagnostic = 'matched_predicted_source'
    if objects.valid.shape[1] > model.max_objects:
        raise ValueError('training source count exceeds reviewed object capacity')
    if source_noise is not None:
        from .noise import perturb_sources
        objects = perturb_sources(objects, **source_noise)
    targets = aligned_targets(truths, gt_indices, objects)
    text = model.encode_text(inputs['input_ids'], inputs['attention_mask'])
    if teacher_forcing:
        output = model.layout.teacher_forced(objects, text, inputs['attention_mask'], targets['xy'], targets['angle'])
    else:
        output = model.layout(objects, text, inputs['attention_mask'])
    terms = {'layout/'+k: v for k, v in layout_loss(output, targets, model.layout.orientation).items()}
    if stage == 'joint':
        if replay is None:
            raise ValueError('joint stage requires a separate unique-source detection replay batch')
        replay_inputs, replay_truths = replay
        detection = model.perception.supervised(replay_inputs['images'], replay_truths)
        detection_terms, _ = model.perception.detection_loss(detection, replay_truths)
        terms.update({'replay/'+k: v for k, v in detection_terms.items()})
    return terms, dict(layout=output, objects=objects, diagnostic=diagnostic)


def stage_gate(stage, metrics):
    """Fail closed on absent, conditional or non-free-rollout results.

    Thresholds are proposed acceptance criteria, not evidence of achieved skill.
    Joint promotion additionally requires an external full-set regression check.
    """
    if stage not in STAGES:
        raise ValueError('unknown recovery stage')
    if metrics.get('evaluation_mode') != 'free_rollout' or metrics.get('complete_set') is not True:
        return False
    def at_most(key, limit):
        value = metrics.get(key)
        return isinstance(value, (int, float)) and 0 <= value <= limit
    if stage == 'oracle_layout':
        return (metrics.get('source_mode') == 'oracle' and at_most('target_position_p95_mm', 1)
                and at_most('target_angle_p95_deg', 2) and metrics.get('move_accuracy') == 1
                and metrics.get('pure_axis_success') == 1)
    if metrics.get('source_mode') != 'predicted' or metrics.get('count_accuracy') != 1 or metrics.get('coverage') != 1:
        return False
    source_ok = (at_most('source_position_p95_mm', 1) and at_most('source_size_p95_mm', 1)
                 and at_most('source_angle_p95_deg', 2) and metrics.get('false_positives') == 0
                 and metrics.get('false_negatives') == 0)
    if stage == 'perception':
        return source_ok
    return (source_ok and metrics.get('legacy_s0_gate') is True and metrics.get('pure_axis_success') == 1
            and metrics.get('task_success') == 1 and metrics.get('per_mode_no_regression') is True
            and (stage != 'joint' or metrics.get('full_set_no_regression') is True))
