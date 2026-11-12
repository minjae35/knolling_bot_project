"""Update only the row-origin Y readout on hard 8-object training rows.

The last linear layer emits X and Y. Earlier origin layers feed both, so they
stay frozen with Δx, row, order, and the relation head. GT row, order, and Y
are labels. Free rollout does not receive them. The family holdout is excluded
from this fit and is the only selection set.
"""
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_checkpoint import load_checkpoint, save_checkpoint
from knolling_model.experiments.row_layout import NeuralRowPlacement
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from run_full_arrangement import cache_records, score, stamp, subset
from run_relation_joint import install_live_text
from run_row_layout import attach_rows, collate_rows, hidden_of

ROOT = Path('artifacts/experiments/row_layout_origin_y_hard')
BEST = Path('artifacts/row_layout_origin_y/checkpoint-best.pt')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
HOLDOUT = Path('artifacts/dev_holdout/multi_row_families.json')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-4
BATCH = 16
EVAL_EVERY = 50
MAX_UPDATES = 200
MIN_UPDATES = 50
PATIENCE = 2
Y_LIMIT_M = .002
CONFIG = dict(supervision='hard_8_origin_y', lr=LR, batch=BATCH, eval_every=EVAL_EVERY,
              max_updates=MAX_UPDATES, min_updates=MIN_UPDATES, patience_evals=PATIENCE,
              y_limit_m=Y_LIMIT_M, seed=1203, init='row_layout_origin_y', coordinate_clip=1.,
              selection='family_holdout_v1', validation_used_for_selection=False)


def prepare(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.row_layout[-1].weight.requires_grad_(True)
    model.row_layout[-1].bias.requires_grad_(True)


def zero_x_grad(layer):
    layer.weight.grad[0].zero_()
    layer.bias.grad[0].zero_()


def hard_y_loss(output, target_xy, valid, slot):
    error = (output['xy'][..., 1]-target_xy[..., 1]).detach().abs()
    chosen = valid & (slot == 0) & (error > Y_LIMIT_M)
    if not bool(chosen.any()):
        return None, 0
    loss = F.smooth_l1_loss(output['xy'][..., 1][chosen], target_xy[..., 1][chosen], beta=.010)
    return loss, int(chosen.sum())


def counts_of(point):
    return {count: dict(teacher=point['teacher']['by_count'][count]['success'],
                        teacher_total=point['teacher']['by_count'][count]['total'],
                        free=point['free']['by_count'][count]['success'],
                        free_total=point['free']['by_count'][count]['total'],
                        origin_y_mm=point['teacher']['by_count'][count]['origin_y_mm'])
            for count in ('2', '4', '6', '8')}


def brief(point):
    counts = counts_of(point)
    return dict(update=point['update'],
                teacher={count: counts[count]['teacher'] for count in counts},
                free={count: counts[count]['free'] for count in counts},
                origin_y_8=counts['8']['origin_y_mm'],
                horizontal_teacher=point['teacher']['horizontal_gap_mm']['median'],
                horizontal_free=point['free']['horizontal_gap_mm']['median'],
                loss=point.get('loss'))


def no_regression(point, baseline):
    if point['teacher']['horizontal_gap_mm']['median'] > baseline['teacher']['horizontal_gap_mm']['median']+0.02:
        return False
    if point['free']['horizontal_gap_mm']['median'] > baseline['free']['horizontal_gap_mm']['median']+0.02:
        return False
    for count in ('2', '4', '6', '8'):
        if point['teacher']['by_count'][count]['success'] < baseline['teacher']['by_count'][count]['success']:
            return False
        if point['free']['by_count'][count]['success'] < baseline['free']['by_count'][count]['success']:
            return False
    return True


def improved(point, baseline):
    current = point['teacher']['by_count']['8']
    base = baseline['teacher']['by_count']['8']
    return current['success'] > base['success'] and current['origin_y_mm']['median'] < base['origin_y_mm']['median']


def better_than(point, best):
    current = point['teacher']['by_count']['8']
    held = best['teacher']['by_count']['8']
    return (current['success'], point['free']['by_count']['8']['success'], -current['origin_y_mm']['median']) > (
        held['success'], best['free']['by_count']['8']['success'], -held['origin_y_mm']['median'])


def forward(model, head, packed):
    hidden = hidden_of(head, packed['features'])
    return model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                 packed['row'], packed['slot'], False)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('hard origin Y result already exists')
    if not BEST.is_file() or not HOLDOUT.is_file():
        raise RuntimeError('best checkpoint or family holdout is missing')
    ROOT.mkdir(parents=True, exist_ok=True)
    best_stamp = stamp(BEST)
    modular_stamp = stamp(MODULAR)
    protocol = json.loads(HOLDOUT.read_text())
    runner.setup()
    train_data = ExpandedDataset(DATA, 'train')
    val_data = ExpandedDataset(DATA, 'validation')
    train = [record for record in train_data.records if multi_row(record)]
    by_id = {record['sample_id']: record for record in train}
    dev_ids = set(protocol['dev_sample_ids'])
    fit_ids = set(protocol['fit_sample_ids'])
    if dev_ids & fit_ids or set(by_id) != dev_ids | fit_ids:
        raise RuntimeError('family holdout does not partition the train multi-row scenes')
    dev = [by_id[sample_id] for sample_id in protocol['dev_sample_ids']]
    fit = [by_id[sample_id] for sample_id in protocol['fit_sample_ids']]
    dev_counts = Counter(len(record['source']['objects']) for record in dev)
    if dict(dev_counts) != {2: 38, 4: 75, 6: 70, 8: 77} or len(fit) != 1013:
        raise RuntimeError(f'holdout pools changed: {dict(dev_counts)} fit={len(fit)}')
    fit8 = [record for record in fit if len(record['source']['objects']) == 8]
    multi_val = [record for record in val_data.records if multi_row(record)]
    single_val = [record for record in val_data.records if is_full(record) and len(record['supervision_v3']['rows']) == 1]
    phase_val = subset(val_data, 'validation', 94)
    val8 = [record for record in multi_val if len(record['source']['objects']) == 8]
    if len(fit8) != 279 or len(multi_val) != 152 or len(val8) != 42 or len(single_val) != 104:
        raise RuntimeError('scene counts changed')
    body = load_integrated(device=stable.DEVICE)
    install_live_text(body)
    body.eval()
    head = cap.load_head(128, torch.load(MODULAR, map_location='cpu', weights_only=False)['model'])
    head.requires_grad_(False)
    head.eval()
    head_before = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    phase_cache = stable.cache_oracle(body, phase_val)
    fit_stored = attach_rows(fit8, cache_records(body, fit8))
    dev_stored = attach_rows(dev, cache_records(body, dev))
    val_stored = attach_rows(multi_val, cache_records(body, multi_val))
    single_stored = cache_records(body, single_val)
    del body
    model = NeuralRowPlacement().to(stable.DEVICE)
    saved = torch.load(BEST, map_location=stable.DEVICE, weights_only=False)
    model.load_state_dict(saved['model'])
    model.eval()
    oracle = stable.official_metrics(head, phase_cache, phase_val)['task']
    single = score(head, single_stored, single_val)['task']
    val_teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
    val_free = evaluate(model, head, val_stored, multi_val, 'free')
    print('RELOAD', json.dumps(dict(oracle=oracle, single=single, teacher=val_teacher['task'], free=val_free['task'],
                                   horizontal=val_teacher['horizontal_gap_mm']['median'])), flush=True)
    if (oracle, single, val_teacher['task'], val_free['task']) != (60, 62, 95, 66):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reload_baseline', trained=False))
        print('STOP best checkpoint did not reproduce', flush=True)
        return
    hold_teacher = evaluate(model, head, dev_stored, dev, 'teacher')
    hold_free = evaluate(model, head, dev_stored, dev, 'free')
    baseline = dict(update=0, oracle=oracle, single=single, teacher=hold_teacher, free=hold_free)
    print('HOLDOUT', json.dumps(brief(baseline)), flush=True)
    hard = []
    hard_origins = 0
    with torch.no_grad():
        for start in range(0, len(fit8), 16):
            indices = range(start, min(start+16, len(fit8)))
            packed = collate_rows([fit_stored[index] for index in indices], stable.DEVICE)
            output = forward(model, head, packed)
            error = (output['xy'][..., 1]-packed['target_xy'][..., 1]).abs()
            slot0 = packed['valid'] & (packed['slot'] == 0)
            for local, index in enumerate(indices):
                chosen = slot0[local] & (error[local] > Y_LIMIT_M)
                if bool(chosen.any()):
                    hard.append(index)
                    hard_origins += int(chosen.sum())
    print('HARD', json.dumps(dict(scenes=len(hard), origins=hard_origins, fit8=len(fit8))), flush=True)
    if not hard:
        stable.write(ROOT/'result.json', dict(status='FAIL', stage='no_hard_cases', trained=False,
                                              holdout_previously_in_training=True))
        print('STOP no fit origin exceeds 2 mm', flush=True)
        return
    prepare(model)
    x_weight = model.row_layout[-1].weight.detach()[0].clone()
    x_bias = model.row_layout[-1].bias.detach()[0].clone()
    y_weight = model.row_layout[-1].weight.detach()[1].clone()
    y_bias = model.row_layout[-1].bias.detach()[1].clone()
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if parameter is not model.row_layout[-1].weight and parameter is not model.row_layout[-1].bias}
    optimizer = torch.optim.AdamW((model.row_layout[-1].weight, model.row_layout[-1].bias),
                                  lr=LR, weight_decay=0., foreach=False)
    generator = torch.Generator().manual_seed(cap.SEED)
    order = []
    cursor = 0

    def next_batch():
        nonlocal order, cursor
        picked = []
        while len(picked) < min(BATCH, len(hard)):
            if cursor >= len(order):
                order = torch.randperm(len(hard), generator=generator).tolist()
                cursor = 0
            take = min(min(BATCH, len(hard))-len(picked), len(order)-cursor)
            picked.extend(hard[order[index]] for index in range(cursor, cursor+take))
            cursor += take
        return picked

    probe = collate_rows([fit_stored[index] for index in next_batch()], stable.DEVICE)
    reference = NeuralRowPlacement().to(stable.DEVICE)
    reference.load_state_dict(saved['model'])
    reference.eval()
    with torch.no_grad():
        before_probe = forward(reference, head, probe)
    save_checkpoint(ROOT/'checkpoint-0000.pt', model, optimizer, 0, CONFIG)
    copy = NeuralRowPlacement().to(stable.DEVICE)
    prepare(copy)
    copy_optimizer = torch.optim.AdamW((copy.row_layout[-1].weight, copy.row_layout[-1].bias),
                                       lr=LR, weight_decay=0., foreach=False)
    loaded = load_checkpoint(ROOT/'checkpoint-0000.pt', copy, copy_optimizer, map_location=stable.DEVICE)
    with torch.no_grad():
        restored = forward(copy, head, probe)
    if not torch.equal(before_probe['xy'], restored['xy']) or loaded['update'] != 0:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reload', trained=False))
        print('STOP checkpoint reload mismatch', flush=True)
        return
    del copy, copy_optimizer, restored
    history = (ROOT/'history.jsonl').open('w')
    history.write(json.dumps(brief(baseline))+'\n')
    history.flush()
    started = time.perf_counter()
    best = baseline
    save_checkpoint(ROOT/'checkpoint-best.pt', model, optimizer, 0, CONFIG)
    reason = 'max_updates'
    stale = 0
    skips = 0
    running = 0.
    seen = 0
    path_ok = False
    for update in range(1, MAX_UPDATES+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        packed = collate_rows([fit_stored[index] for index in next_batch()], stable.DEVICE)
        output = forward(model, head, packed)
        loss, used = hard_y_loss(output, packed['target_xy'], packed['valid'], packed['slot'])
        if loss is None:
            skips += 1
            if skips >= 20:
                reason = 'under_2mm'
                break
            continue
        skips = 0
        loss.backward()
        zero_x_grad(model.row_layout[-1])
        if not path_ok:
            y_grad = float(model.row_layout[-1].weight.grad[1].norm())
            x_grad = float(model.row_layout[-1].weight.grad[0].norm())
            leaked = [name for name, parameter in model.named_parameters()
                      if parameter.grad is not None and parameter is not model.row_layout[-1].weight
                      and parameter is not model.row_layout[-1].bias]
            x_before = output['xy'][..., 0].detach().clone()
            print('GRAD', json.dumps(dict(y_grad=y_grad, x_grad=x_grad, leaked=leaked, origins=used,
                                         loss=float(loss.detach()))), flush=True)
            if y_grad == 0 or x_grad != 0 or leaked:
                reason = 'path_blocked'
                break
        torch.nn.utils.clip_grad_norm_((model.row_layout[-1].weight, model.row_layout[-1].bias), 1.)
        optimizer.step()
        x_same = torch.equal(model.row_layout[-1].weight[0], x_weight) and torch.equal(model.row_layout[-1].bias[0], x_bias)
        frozen_same = all(torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters() if name in frozen)
        y_moved = not torch.equal(model.row_layout[-1].weight[1], y_weight) or not torch.equal(model.row_layout[-1].bias[1], y_bias)
        if not x_same or not frozen_same:
            reason = 'x_moved' if not x_same else 'frozen_moved'
            break
        if not path_ok:
            model.eval()
            with torch.no_grad():
                after = forward(model, head, packed)
            x_prediction_same = torch.equal(after['xy'][..., 0], x_before) and torch.equal(after['delta_x'], output['delta_x'])
            print('UPDATE', json.dumps(dict(x_parameters_same=x_same, x_prediction_same=x_prediction_same,
                                           y_parameters_moved=y_moved, frozen_same=frozen_same)), flush=True)
            if not x_prediction_same or not y_moved:
                reason = 'x_prediction_moved' if not x_prediction_same else 'y_not_updated'
                break
            path_ok = True
        running += float(loss.detach())
        seen += 1
        if update % EVAL_EVERY and update != MAX_UPDATES:
            continue
        point = dict(update=update, loss=running/seen, teacher=evaluate(model, head, dev_stored, dev, 'teacher'),
                     free=evaluate(model, head, dev_stored, dev, 'free'), seconds=time.perf_counter()-started)
        running = 0.
        seen = 0
        history.write(json.dumps(brief(point))+'\n')
        history.flush()
        print('EVAL', json.dumps(brief(point)), flush=True)
        if not no_regression(point, baseline):
            reason = 'holdout_regression'
            break
        if improved(point, baseline) and better_than(point, best):
            best = point
            stale = 0
            save_checkpoint(ROOT/'checkpoint-best.pt', model, optimizer, update, CONFIG)
        else:
            stale += 1
        if update >= MIN_UPDATES and stale >= PATIENCE:
            reason = 'plateau'
            break
    history.close()
    exposure = dict(protocol=protocol['protocol'], seed=protocol['seed'], fraction=protocol['fraction'],
                    previously_in_best_training=True, previously_used_for_selection=False,
                    excluded_from_this_fit=True, dev_scenes=len(dev), fit_scenes=len(fit),
                    note='Best was fit on all 1273 train scenes, including these 260. This run does not fit them.')
    if reason in ('path_blocked', 'x_moved', 'frozen_moved', 'x_prediction_moved', 'y_not_updated'):
        stable.write(ROOT/'result.json', dict(status='STOP', stage=reason, trained=path_ok, exposure=exposure,
                                              hard_scenes=len(hard), hard_origins=hard_origins,
                                              best_checkpoint_unchanged=stamp(BEST) == best_stamp))
        print('STOP', reason, flush=True)
        return
    reloaded = NeuralRowPlacement().to(stable.DEVICE)
    prepare(reloaded)
    best_state = load_checkpoint(ROOT/'checkpoint-best.pt', reloaded, map_location=stable.DEVICE)
    model.load_state_dict(reloaded.state_dict())
    final_teacher = evaluate(model, head, dev_stored, dev, 'teacher')
    final_free = evaluate(model, head, dev_stored, dev, 'free')
    final = dict(update=best_state['update'], teacher=final_teacher, free=final_free)
    reload_ok = brief(final)['teacher'] == brief(best)['teacher'] and brief(final)['free'] == brief(best)['free']
    with torch.no_grad():
        end_probe = forward(model, head, probe)
    x_prediction_same = torch.equal(end_probe['xy'][..., 0], before_probe['xy'][..., 0])
    x_parameters_same = torch.equal(model.row_layout[-1].weight[0], x_weight) and torch.equal(model.row_layout[-1].bias[0], x_bias)
    adopted = improved(final, baseline) and no_regression(final, baseline) and reload_ok and x_prediction_same
    val_end_teacher = val_end_free = None
    oracle_end = single_end = None
    if adopted:
        val_end_teacher = evaluate(model, head, val_stored, multi_val, 'teacher')
        val_end_free = evaluate(model, head, val_stored, multi_val, 'free')
        oracle_end = stable.official_metrics(head, phase_cache, phase_val)['task']
        single_end = score(head, single_stored, single_val)['task']
        val_point = dict(teacher=val_end_teacher, free=val_end_free)
        val_base = dict(teacher=val_teacher, free=val_free)
        adopted = (oracle_end >= 60 and single_end >= 62 and val_end_free['task'] >= 66
                   and no_regression(val_point, val_base))
    status = 'PASS' if adopted else 'FAIL'
    stable.write(ROOT/'result.json', dict(
        status=status, stop=reason, best_update=best_state['update'], exposure=exposure,
        hard_scenes=len(hard), hard_origins=hard_origins, config=CONFIG,
        holdout_baseline=brief(baseline), holdout_best=brief(final),
        reload_matches_logged_success=reload_ok, x_parameters_unchanged=x_parameters_same,
        x_prediction_unchanged=x_prediction_same,
        relation_parameters_unchanged=all(torch.equal(head_before[name], parameter) for name, parameter in head.named_parameters()),
        validation_baseline=dict(oracle=oracle, single=single, teacher=val_teacher['task'], free=val_free['task'],
                                 by_count=counts_of(dict(teacher=val_teacher, free=val_free)),
                                 horizontal_teacher=val_teacher['horizontal_gap_mm']['median'],
                                 horizontal_free=val_free['horizontal_gap_mm']['median']),
        validation_end=None if val_end_teacher is None else dict(
            oracle=oracle_end, single=single_end, teacher=val_end_teacher['task'], free=val_end_free['task'],
            by_count=counts_of(dict(teacher=val_end_teacher, free=val_end_free)),
            horizontal_teacher=val_end_teacher['horizontal_gap_mm']['median'],
            horizontal_free=val_end_free['horizontal_gap_mm']['median']),
        validation_used_for_selection=False, test_used=False, gt_at_inference=False,
        best_checkpoint_unchanged=stamp(BEST) == best_stamp,
        modular_checkpoint_unchanged=stamp(MODULAR) == modular_stamp))
    print('DONE', json.dumps(dict(status=status, stop=reason, update=best_state['update'],
                                 holdout=brief(final), x_prediction_same=x_prediction_same,
                                 oracle=oracle_end, single=single_end,
                                 val_teacher=None if val_end_teacher is None else val_end_teacher['task'],
                                 val_free=None if val_end_free is None else val_end_free['task'])), flush=True)


if __name__ == '__main__':
    main()
