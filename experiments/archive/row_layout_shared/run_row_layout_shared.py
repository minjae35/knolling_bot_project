"""One limited run of shared-Y, width-conditioned row placement.

Teacher rows and order supervise the 8 memorized scenes plus 32 other
training scenes. Free rollout and the 152-scene validation are scored only
after that teacher placement succeeds. The relation head stays frozen.
"""
import json
import time
from pathlib import Path

import torch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_layout import NeuralRowPlacement, assignment_loss, order_loss, placement_loss
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from run_full_arrangement import cache_records, score, stamp, subset
from run_relation_joint import install_live_text
from run_row_layout import attach_rows, collate_rows, hidden_of
from run_row_transition import choose

ROOT = Path('artifacts/row_layout_shared')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
PROBE = Path('artifacts/row_layout/checkpoint-probe.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-3
COORDINATE_STEPS = 1500
SELECTOR_STEPS = 800
EVAL_EVERY = 25
LOSS = dict(xy=0.0169582087546587, angle=0.00024378746456932276, coordinate=0.017201995477080345,
            row_ce=2.3417627811431885, order_ce=0.28069183230400085, selector=2.6224546432495117,
            weight=0.05, weighted_selector=0.13112273411635655, total=0.1483247295934369)


def train_loss(model, head, packed):
    hidden = hidden_of(head, packed['features'])
    output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['row'], packed['slot'], False)
    losses = placement_loss(output, packed['target_xy'], packed['target_angle'], packed['valid'])
    return losses, output


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('shared row result already exists')
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
    train_index = probe_index+fresh_index
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
    record_40 = [records[index] for index in train_index]
    stored = attach_rows(record_40, cache_records(body, record_40))
    if len(stored) != 40:
        raise RuntimeError('the limited training cache has the wrong size')
    probe_records = record_40[:8]
    fresh_records = record_40[8:]
    probe_stored, fresh_stored = stored[:8], stored[8:]
    val_stored = attach_rows(multi_val, cache_records(body, multi_val))
    single_stored = cache_records(body, single_val)
    del body
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    print('START', json.dumps(dict(oracle=oracle, single=single, loss=LOSS)), flush=True)
    if (oracle, single) != (60, 62):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='initialization', oracle=oracle, single=single,
                                              loss_check=LOSS, modular_checkpoint_unchanged=stamp(MODULAR) == held))
        print('STOP initialization mismatch', flush=True)
        return
    model = NeuralRowPlacement().to(stable.DEVICE)
    saved = torch.load(PROBE, map_location=stable.DEVICE, weights_only=False)['model']
    own = model.state_dict()
    transferred = {key: value for key, value in saved.items() if key in own and own[key].shape == value.shape}
    own.update(transferred)
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
                                     mm=point['position_mm']['median'], failures=point['failures'])), flush=True)
        return payload

    probe_now = teacher_of(probe_stored, probe_records)
    fresh_now = teacher_of(fresh_stored, fresh_records)
    log('teacher_probe', 0, probe_now, {})
    log('teacher_fresh', 0, fresh_now, {})
    optimizer = torch.optim.AdamW(model.coordinate_parameters(), lr=LR, weight_decay=0., foreach=False)
    reason = 'coordinate_failed'
    best_mm = fresh_now['position_mm']['median']
    stale = 0
    for step in range(1, COORDINATE_STEPS+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, _ = train_loss(model, head, packed)
        losses['loss'].backward()
        if step == 1:
            module_grad = sum(parameter.grad.abs().sum().item() for parameter in model.coordinate_parameters()
                              if parameter.grad is not None)
            head_grad = any(parameter.grad is not None for parameter in head.parameters())
            print('PATH', json.dumps(dict(coordinate=module_grad, head=head_grad,
                                         xy=float(losses['xy'].detach()), angle=float(losses['angle'].detach()))),
                  flush=True)
            if module_grad == 0 or head_grad:
                reason = 'path_blocked'
                break
        torch.nn.utils.clip_grad_norm_(model.coordinate_parameters(), 10.)
        optimizer.step()
        if step % EVAL_EVERY and step != COORDINATE_STEPS:
            continue
        probe_now = teacher_of(probe_stored, probe_records)
        fresh_now = teacher_of(fresh_stored, fresh_records)
        log('teacher_probe', step, probe_now, dict(xy=float(losses['xy'].detach())))
        log('teacher_fresh', step, fresh_now, dict(xy=float(losses['xy'].detach())))
        if probe_now['task'] == 8 and fresh_now['task'] == 32:
            reason = 'teacher_passed'
            break
        median = fresh_now['position_mm']['median']
        if median is not None and (best_mm is None or median < best_mm):
            best_mm = median
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            reason = 'coordinate_plateau'
            break
    passed_teacher = reason == 'teacher_passed'
    free_probe = free_fresh = validation = validation_teacher = None
    if passed_teacher:
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
            output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['row'], packed['slot'], True)
            loss = assignment_loss(output, packed['row'], packed['valid'])+order_loss(output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.selector_parameters(), 1.)
            optimizer.step()
            if step % EVAL_EVERY and step != SELECTOR_STEPS:
                continue
            free_probe = evaluate(model, head, probe_stored, probe_records, 'free')
            free_fresh = evaluate(model, head, fresh_stored, fresh_records, 'free')
            held_probe = teacher_of(probe_stored, probe_records)
            held_fresh = teacher_of(fresh_stored, fresh_records)
            log('free_probe', step, free_probe, {})
            log('free_fresh', step, free_fresh, {})
            if held_probe['task'] != 8 or held_fresh['task'] != 32:
                reason = 'coordinates_moved'
                break
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
    val_improved = False
    if validation is not None and validation_teacher is not None:
        val_improved = validation['task'] > 4 or (
            validation_teacher['position_mm']['median'] is not None and validation_teacher['position_mm']['median'] <= 20
            and validation_teacher['task'] > 3)
    status = 'PASS' if reason == 'free_passed' and oracle_end >= 58 and single_end >= 56 else 'FAIL'
    stable.write(ROOT/'result.json', dict(
        status=status, stop=reason, teacher_passed=passed_teacher, validation_improved=val_improved,
        full_training_started=False, loss_check=LOSS,
        teacher_probe=None if probe_now is None else dict(task=probe_now['task'], failures=probe_now['failures'],
                                                          position_mm=probe_now['position_mm'],
                                                          horizontal_gap_mm=probe_now['horizontal_gap_mm'],
                                                          vertical_gap_mm=probe_now['vertical_gap_mm']),
        teacher_fresh=None if fresh_now is None else dict(task=fresh_now['task'], failures=fresh_now['failures'],
                                                          position_mm=fresh_now['position_mm'],
                                                          horizontal_gap_mm=fresh_now['horizontal_gap_mm'],
                                                          vertical_gap_mm=fresh_now['vertical_gap_mm']),
        free_probe=None if free_probe is None else dict(task=free_probe['task'], by_count={
            key: value['success'] for key, value in free_probe['by_count'].items()}),
        free_fresh=None if free_fresh is None else dict(task=free_fresh['task'], by_count={
            key: value['success'] for key, value in free_fresh['by_count'].items()}),
        validation_free=None if validation is None else dict(task=validation['task'], by_count={
            key: value['success'] for key, value in validation['by_count'].items()},
            position_mm=validation['position_mm'], failures=validation['failures']),
        validation_teacher=None if validation_teacher is None else dict(
            task=validation_teacher['task'], by_count={
                key: value['success'] for key, value in validation_teacher['by_count'].items()},
            position_mm=validation_teacher['position_mm'], failures=validation_teacher['failures']),
        oracle_start=oracle, single_start=single, oracle_end=oracle_end, single_end=single_end,
        relation_parameters_unchanged=unchanged, modular_checkpoint_unchanged=stamp(MODULAR) == held,
        probe_checkpoint_unchanged=stamp(PROBE) == probe_held, full_9600=False, joint=False, test_used=False))
    print('DONE', json.dumps(dict(status=status, stop=reason, teacher_probe=probe_now['task'],
                                 teacher_fresh=fresh_now['task'],
                                 val=None if validation is None else validation['task'],
                                 oracle=oracle_end, single=single_end, unchanged=unchanged,
                                 val_improved=val_improved)), flush=True)


if __name__ == '__main__':
    main()
