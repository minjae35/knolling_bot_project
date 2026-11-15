"""Stage-specific loss construction and freeze policy; contains no optimizer step.

The review config cannot launch a training run. Stage functions expose forward /
backward graphs for CPU tests and later independent review of training wiring.
"""
import json
from pathlib import Path

import torch

from .contracts import oracle_objects, pack_sources
from .layout import layout_loss
from .selection import match_selected

STAGES = ('oracle_layout', 'perception', 'predicted_layout', 'joint')
V4_STAGES = ('grounding', 'oracle_layout', 'perception', 'predicted_layout', 'joint')
ALL_STAGES = ('grounding', 'micro_pose', *STAGES)


def stage_order(architecture):
    return V4_STAGES if architecture == 'recovery_v4_goal_joint' else STAGES


def read_review_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('schema') != 'neural_recovery_review_v1' or config.get('execution') != 'review_only':
        raise ValueError('only the review-only recovery configuration is accepted')
    if config.get('allow_training') is not False or config.get('device') != 'cpu':
        raise ValueError('GPU and training execution are disabled pending review')
    if tuple(config['stages']) != stage_order(config.get('model', {}).get('architecture')):
        raise ValueError('recovery stage order changed')
    return config


def configure_stage(model, stage):
    v4 = getattr(model, 'architecture', None) == 'recovery_v4_goal_joint'
    if stage not in (ALL_STAGES if v4 else STAGES):
        raise ValueError('unknown recovery stage')
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    if stage in ('micro_pose', 'oracle_layout', 'predicted_layout', 'joint'):
        (model.layout.pose if v4 else model.layout).requires_grad_(True)
    if stage == 'grounding':
        model.layout.goal_encoder.requires_grad_(True)
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


def parameter_groups(model, layout_lr=1e-4, perception_lr=1e-5, goal_lr=3e-4):
    """Construct groups without creating an optimizer or modifying parameters."""
    groups = []
    modules = [('layout', model.layout, layout_lr), ('perception', model.perception, perception_lr)]
    if getattr(model, 'architecture', None) == 'recovery_v4_goal_joint':
        modules = [('goal', model.layout.goal_encoder, goal_lr), ('layout', model.layout.pose, layout_lr),
                   ('perception', model.perception, perception_lr)]
    for name, module, lr in modules:
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
    return match_selected(model, visual, truths)


def stage_loss(model, inputs, truths, *, teacher_forcing=False, replay=None, source_noise=None):
    """Return losses and diagnostic outputs; never calls backward/step/save.

    Predicted-layout training uses exactly the inference foreground rows.
    Source matching attaches labels only; bad count/correspondence aborts rather
    than substituting GT cardinality or selecting below-threshold queries.

    Joint detection replay must use the full unique-source stream, not only the
    instruction-balanced layout batch. The caller supplies that separate batch.
    """
    stage = model.stage
    v4 = getattr(model, 'architecture', None) == 'recovery_v4_goal_joint'
    joint_v3 = getattr(model, 'architecture', None) == 'recovery_v3_joint'
    if (joint_v3 or v4) and (teacher_forcing or source_noise is not None):
        raise ValueError('joint models have no teacher-prefix or unreviewed source-noise path')
    if stage not in (ALL_STAGES if v4 else STAGES) or not model.training:
        raise ValueError('configure a training stage before constructing its loss')
    if teacher_forcing and stage not in ('oracle_layout', 'predicted_layout'):
        raise ValueError('teacher forcing is limited to layout warmup stages')
    if source_noise is not None and stage not in ('oracle_layout', 'predicted_layout'):
        raise ValueError('source noise is limited to layout warmup stages')
    if stage == 'perception':
        visual = model.perception.supervised(inputs['images'], truths)
        terms, _ = model.perception.detection_loss(visual, truths)
        return terms, dict(perception=visual, diagnostic='perception_only')
    if stage in ('grounding', 'micro_pose', 'oracle_layout'):
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
        diagnostic = 'inference_selected_source'
    if objects.valid.shape[1] > model.max_objects:
        raise ValueError('training source count exceeds reviewed object capacity')
    if source_noise is not None:
        from .noise import perturb_sources
        objects = perturb_sources(objects, **source_noise)
    text = model.encode_text(inputs['input_ids'], inputs['attention_mask'])
    if stage == 'grounding':
        from .goal_encoder import goal_loss
        output = model.layout.goal_encoder(objects, text, inputs['attention_mask'])
        return {'goal/'+k: v for k, v in goal_loss(output, truths, gt_indices).items()}, dict(goal=output, objects=objects)
    targets = aligned_targets(truths, gt_indices, objects)
    if teacher_forcing:
        output = model.layout.teacher_forced(objects, text, inputs['attention_mask'], targets['xy'], targets['angle'])
    else:
        output = model.layout(objects, text, inputs['attention_mask'])
    if joint_v3 or v4:
        from .joint_layout import joint_pose_loss
        losses = joint_pose_loss(output, targets, model.layout.orientation)
    else:
        losses = layout_loss(output, targets, model.layout.orientation)
    terms = {'layout/'+k: v for k, v in losses.items()}
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
    if metrics.get('schema') == 'recovery_metrics_v4':
        from .evaluation_v4 import gate
        return gate(stage, metrics)
    if metrics.get('schema') == 'recovery_metrics_v3':
        from .evaluation_v3 import gate
        return gate(stage, metrics)
    if stage not in STAGES:
        raise ValueError('unknown recovery stage')
    from .evaluation import METRICS_SCHEMA
    if metrics.get('schema') != METRICS_SCHEMA or metrics.get('stage') != stage:
        return False
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
                 and metrics.get('false_negatives') == 0 and metrics.get('color_accuracy') == 1
                 and metrics.get('shape_accuracy') == 1)
    if stage == 'perception':
        return source_ok
    return (source_ok and metrics.get('legacy_s0_gate') is True and metrics.get('pure_axis_success') == 1
            and metrics.get('task_success') == 1 and metrics.get('per_mode_no_regression') is True
            and (stage != 'joint' or metrics.get('full_set_no_regression') is True))
