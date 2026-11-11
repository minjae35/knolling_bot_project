"""One limited run: supervise the existing X step with GT neighbor Δx.

The row origin and the shared Y keep absolute coordinates. Later absolute X
is not a target, so the step is not trained twice. This is not the rejected
offset loss: that expression cancels to absolute X on a sequential decoder.
"""
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_layout import (
    NeuralRowPlacement, assignment_loss, order_loss, step_placement_loss)
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from run_full_arrangement import cache_records, score, stamp, subset
from run_relation_joint import install_live_text
from run_row_layout import attach_rows, collate_rows, hidden_of
from run_row_transition import choose

ROOT = Path('artifacts/row_layout_step')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
PROBE = Path('artifacts/row_layout/checkpoint-probe.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
COORDINATE_STEPS = 1500
SELECTOR_STEPS = 800
EVAL_EVERY = 25
# Measured on this same initialization under absolute XY, before any update.
BASELINE_GRAD = dict(step=0.00576431793018575, row_layout=0.07183844484100946)


def rejected_offset_matches():
    """The rejected loss subtracts one previous x from both sides."""
    xy = torch.tensor([[[.30, .20], [.40, .20]]])
    delta = torch.tensor([[0., .05]])
    target = torch.tensor([[[.10, .20], [.19, .20]]])
    valid = torch.tensor([[True, True]])
    row = torch.tensor([[0, 0]])
    slot = torch.tensor([[0, 1]])
    angle = torch.zeros(1, 2)
    losses = step_placement_loss(dict(xy=xy, delta_x=delta, angle=angle), target, angle, valid, row, slot)
    previous = target[0, 0, 0]
    cancelled = F.smooth_l1_loss(xy[0, 1, 0]-previous, target[0, 1, 0]-previous, beta=.010)
    absolute = F.smooth_l1_loss(xy[0, 1, 0], target[0, 1, 0], beta=.010)
    return bool(torch.allclose(cancelled, absolute) and torch.allclose(losses['dx'], cancelled))


def train_loss(model, head, packed):
    hidden = hidden_of(head, packed['features'])
    output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['row'], packed['slot'], False)
    losses = step_placement_loss(output, packed['target_xy'], packed['target_angle'], packed['valid'],
                                 packed['row'], packed['slot'])
    return losses, output


def gaps_closed(point):
    return point['failures'].get('horizontal_gap', 0) == 0


def main():
    if rejected_offset_matches():
        raise RuntimeError('step Δx loss matches the rejected offset loss')
    if (ROOT/'result.json').exists():
        raise RuntimeError('step result already exists')
    ROOT.mkdir(parents=True, exist_ok=True)
    held = stamp(MODULAR)
    probe_held = stamp(PROBE)
    runner.setup()
    train_data = ExpandedDataset(DATA, 'train')
    val_data = ExpandedDataset(DATA, 'validation')
    records = [record for record in train_data.records if multi_row(record)]
    groups = {}
    for count in (2, 4, 6, 8):
        groups[count] = [index for index, record in enumerate(records) if len(record['source']['objects']) == count]
        groups[count].sort(key=lambda index: records[index]['sample_id'])
    probe_ids = {record['sample_id'] for record in choose(records)}
    probe_index = [index for index, record in enumerate(records) if record['sample_id'] in probe_ids]
    fresh_index = [index for count in (2, 4, 6, 8) for index in groups[count][8:16]]
    if len(probe_index) != 8 or len(fresh_index) != 32 or set(probe_index) & set(fresh_index):
        raise RuntimeError('the 8 and 32 scene pools are not disjoint')
    record_40 = [records[index] for index in probe_index+fresh_index]
    multi_val = [record for record in val_data.records if multi_row(record)]
    single_val = [record for record in val_data.records if is_full(record) and len(record['supervision_v3']['rows']) == 1]
    phase_val = subset(val_data, 'validation', 94)
    if len(multi_val) != 152 or len(single_val) != 104:
        raise RuntimeError('validation pools changed')
    body = load_integrated(device=stable.DEVICE)
    install_live_text(body)
    body.eval()
    modular = torch.load(MODULAR, map_location='cpu', weights_only=False)
    head = cap.load_head(128, modular['model'])
    head.requires_grad_(False)
    head.eval()
    before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    phase_cache = stable.cache_oracle(body, phase_val)
    stored = attach_rows(record_40, cache_records(body, record_40))
    if len(stored) != 40:
        raise RuntimeError('the limited training cache has the wrong size')
    probe_records, fresh_records = record_40[:8], record_40[8:]
    probe_stored, fresh_stored = stored[:8], stored[8:]
    val_stored = attach_rows(multi_val, cache_records(body, multi_val))
    single_stored = cache_records(body, single_val)
    del body
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    print('START', json.dumps(dict(oracle=oracle, single=single, offset_same=False)), flush=True)
    if (oracle, single) != (60, 62):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='initialization', oracle=oracle, single=single,
                                              modular_checkpoint_unchanged=stamp(MODULAR) == held))
        print('STOP initialization mismatch', flush=True)
        return
    model = NeuralRowPlacement().to(stable.DEVICE)
    saved = torch.load(PROBE, map_location=stable.DEVICE, weights_only=False)['model']
    own = model.state_dict()
    own.update({key: value for key, value in saved.items() if key in own and own[key].shape == value.shape})
    model.load_state_dict(own)
    for parameter in model.selector_parameters():
        parameter.requires_grad_(False)
    packed = collate_rows(stored, stable.DEVICE)
    history = (ROOT/'history.jsonl').open('w')
    started = time.perf_counter()

    def teacher_of(store, rows):
        return evaluate(model, head, store, rows, 'teacher')

    def log(stage, step, point, extra):
        payload = dict(stage=stage, update=step, task=point['task'], by_count={
            key: value['success'] for key, value in point['by_count'].items()},
            position_mm=point['position_mm'], horizontal_gap_mm=point['horizontal_gap_mm'],
            vertical_gap_mm=point['vertical_gap_mm'], boundary=point['boundary'], collision=point['collision'],
            failures=point['failures'], seconds=time.perf_counter()-started, **extra)
        history.write(json.dumps(payload)+'\n')
        history.flush()
        print('EVAL', json.dumps(dict(stage=stage, update=step, task=point['task'], by_count=payload['by_count'],
                                     mm=point['position_mm']['median'], gap=point['horizontal_gap_mm']['median'],
                                     failures=point['failures'])), flush=True)
        return payload

    probe_now = teacher_of(probe_stored, probe_records)
    fresh_now = teacher_of(fresh_stored, fresh_records)
    log('teacher_probe', 0, probe_now, {})
    log('teacher_fresh', 0, fresh_now, {})
    optimizer = torch.optim.AdamW(model.coordinate_parameters(), lr=LR, weight_decay=0., foreach=False)
    reason = 'coordinate_failed'
    best_gap = fresh_now['horizontal_gap_mm']['median']
    stale = 0
    grad_report = None
    for step in range(1, COORDINATE_STEPS+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, _ = train_loss(model, head, packed)
        losses['loss'].backward()
        if step == 1:
            def norm(parameters):
                return sum(parameter.grad.detach().pow(2).sum().item()
                           for parameter in parameters if parameter.grad is not None)**.5
            grad_report = dict(step=norm(model.step.parameters()),
                               row_layout=norm(model.row_layout.parameters()),
                               angle=norm(model.angle.parameters()),
                               dx=float(losses['dx'].detach()), x=float(losses['x'].detach()),
                               y=float(losses['y'].detach()))
            head_grad = any(parameter.grad is not None for parameter in head.parameters())
            print('PATH', json.dumps(dict(**grad_report, head=head_grad,
                                         baseline_step=BASELINE_GRAD['step'],
                                         baseline_row=BASELINE_GRAD['row_layout'])), flush=True)
            if grad_report['step'] == 0 or head_grad:
                reason = 'path_blocked'
                break
        torch.nn.utils.clip_grad_norm_(model.coordinate_parameters(), 10.)
        optimizer.step()
        if step % EVAL_EVERY and step != COORDINATE_STEPS:
            continue
        probe_now = teacher_of(probe_stored, probe_records)
        fresh_now = teacher_of(fresh_stored, fresh_records)
        log('teacher_probe', step, probe_now, dict(dx=float(losses['dx'].detach())))
        log('teacher_fresh', step, fresh_now, dict(dx=float(losses['dx'].detach())))
        if gaps_closed(probe_now) and gaps_closed(fresh_now):
            reason = 'teacher_x_passed'
            break
        median = fresh_now['horizontal_gap_mm']['median']
        if median is not None and (best_gap is None or median < best_gap):
            best_gap = median
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            reason = 'coordinate_plateau'
            break
    passed_x = reason == 'teacher_x_passed'
    free_probe = free_fresh = validation = validation_teacher = None
    if passed_x:
        for parameter in model.coordinate_parameters():
            parameter.requires_grad_(False)
        for parameter in model.selector_parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.selector_parameters(), lr=LR, weight_decay=0., foreach=False)
        reason = 'selector_failed'
        for step in range(1, SELECTOR_STEPS+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            hidden = hidden_of(head, packed['features'])
            output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0],
                           packed['row'], packed['slot'], True)
            loss = assignment_loss(output, packed['row'], packed['valid'])+order_loss(output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.selector_parameters(), 1.)
            optimizer.step()
            if step % EVAL_EVERY and step != SELECTOR_STEPS:
                continue
            held_probe = teacher_of(probe_stored, probe_records)
            held_fresh = teacher_of(fresh_stored, fresh_records)
            if not gaps_closed(held_probe) or not gaps_closed(held_fresh):
                reason = 'coordinates_moved'
                probe_now, fresh_now = held_probe, held_fresh
                break
            free_probe = evaluate(model, head, probe_stored, probe_records, 'free')
            free_fresh = evaluate(model, head, fresh_stored, fresh_records, 'free')
            log('free_probe', step, free_probe, {})
            log('free_fresh', step, free_fresh, {})
            if free_probe['task'] == 8 and free_fresh['task'] == 32:
                reason = 'free_passed'
                break
        validation = evaluate(model, head, val_stored, multi_val, 'free')
        validation_teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
        log('validation_free', step, validation, {})
        log('validation_teacher', step, validation_teacher, {})
    oracle_end = stable.official_metrics(head, phase_cache, phase_val)['task']
    single_end = score(head, single_stored, single_val)['task']
    unchanged = all(torch.equal(before[name], parameter) for name, parameter in head.named_parameters())
    history.close()
    val_improved = validation is not None and validation['task'] > 4
    status = 'PASS' if passed_x and val_improved and oracle_end >= 58 and single_end >= 56 else 'FAIL'
    stable.write(ROOT/'result.json', dict(
        status=status, stop=reason, teacher_x_passed=passed_x, validation_improved=val_improved,
        full_training_started=False, offset_supervision_same=False, gradient=grad_report,
        baseline_gradient=BASELINE_GRAD,
        teacher_probe=None if probe_now is None else dict(task=probe_now['task'], failures=probe_now['failures'],
                                                          position_mm=probe_now['position_mm'],
                                                          horizontal_gap_mm=probe_now['horizontal_gap_mm'],
                                                          vertical_gap_mm=probe_now['vertical_gap_mm']),
        teacher_fresh=None if fresh_now is None else dict(task=fresh_now['task'], failures=fresh_now['failures'],
                                                          position_mm=fresh_now['position_mm'],
                                                          horizontal_gap_mm=fresh_now['horizontal_gap_mm'],
                                                          vertical_gap_mm=fresh_now['vertical_gap_mm']),
        free_probe=None if free_probe is None else dict(task=free_probe['task'], by_count={
            key: value['success'] for key, value in free_probe['by_count'].items()},
            horizontal_gap_mm=free_probe['horizontal_gap_mm'], failures=free_probe['failures']),
        free_fresh=None if free_fresh is None else dict(task=free_fresh['task'], by_count={
            key: value['success'] for key, value in free_fresh['by_count'].items()},
            horizontal_gap_mm=free_fresh['horizontal_gap_mm'], failures=free_fresh['failures']),
        validation_free=None if validation is None else dict(task=validation['task'], by_count={
            key: value['success'] for key, value in validation['by_count'].items()},
            position_mm=validation['position_mm'], horizontal_gap_mm=validation['horizontal_gap_mm'],
            failures=validation['failures']),
        validation_teacher=None if validation_teacher is None else dict(
            task=validation_teacher['task'], by_count={
                key: value['success'] for key, value in validation_teacher['by_count'].items()},
            position_mm=validation_teacher['position_mm'], horizontal_gap_mm=validation_teacher['horizontal_gap_mm'],
            failures=validation_teacher['failures']),
        oracle_start=oracle, single_start=single, oracle_end=oracle_end, single_end=single_end,
        relation_parameters_unchanged=unchanged, modular_checkpoint_unchanged=stamp(MODULAR) == held,
        probe_checkpoint_unchanged=stamp(PROBE) == probe_held, full_9600=False, joint=False,
        full_1273=False, test_used=False))
    print('DONE', json.dumps(dict(status=status, stop=reason, teacher_probe=probe_now['task'],
                                 teacher_fresh=fresh_now['task'],
                                 gap_probe=probe_now['horizontal_gap_mm']['median'],
                                 gap_fresh=fresh_now['horizontal_gap_mm']['median'],
                                 val=None if validation is None else validation['task'],
                                 oracle=oracle_end, single=single_end, unchanged=unchanged,
                                 val_improved=val_improved)), flush=True)


if __name__ == '__main__':
    main()
