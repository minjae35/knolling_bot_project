"""One from-scratch Hidden-256 relation run. The 128-d checkpoint is not modified.

Learning rate warms up to 0.0001, stays there through 5000 updates, then falls
to 0.00003 by update 8000. Best checkpoint is validation RGB success, then
4-object success, then smaller gap error.
"""
import json
import time
from pathlib import Path

import torch
from knolling_model.experiments.integration import load_integrated
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from knolling_model.experiments.relation_pose import relation_loss

ROOT = Path('artifacts/phase1_relation_h256')
BASELINE_CKPT = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
HIDDEN = 256
MAX_UPDATES = 8000
EVAL_EVERY = 250
WARMUP = 400
HOLD_UNTIL = 5000
BASE_LR = 1e-4
FLOOR_LR = 3e-5
PATIENCE_EVALS = 8
MIN_UPDATES = 4000
TRAIN_RGB_BASE = 479
VAL_RGB_BASE = 56


def learning_rate(step):
    if step <= WARMUP:
        return BASE_LR * step / WARMUP
    if step <= HOLD_UNTIL:
        return BASE_LR
    progress = min(1., (step-HOLD_UNTIL)/(MAX_UPDATES-HOLD_UNTIL))
    return BASE_LR+(FLOOR_LR-BASE_LR)*progress


def rank_of(measured):
    rgb = measured['val']['rgb']
    median = rgb['gap']['median'] if rgb['gap']['median'] is not None else 1e9
    p95 = rgb['gap']['p95'] if rgb['gap']['p95'] is not None else 1e9
    return (rgb['task'], rgb['count']['4']['success'], -median, -p95)


def verdict(measured):
    train = measured['train']['rgb']['task']
    val = measured['val']['rgb']['task']
    if train > TRAIN_RGB_BASE and val > VAL_RGB_BASE:
        return 'PASS'
    if train > TRAIN_RGB_BASE and val < VAL_RGB_BASE:
        return 'FAIL_OVERFIT'
    return 'FAIL'


def main():
    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError('256-d output already exists')
    stable.smoke()
    runner.setup()
    saved = torch.load(BASELINE_CKPT, map_location='cpu', weights_only=False)
    train_data, val_data = runner.datasets()
    data = dict(train=train_data, val=val_data)
    model = load_integrated(device=stable.DEVICE)
    print('CACHE start', flush=True)
    caches = dict(train_oracle=stable.cache_oracle(model, train_data),
                  val_oracle=stable.cache_oracle(model, val_data),
                  val_rgb=stable.cache_rgb(model, val_data),
                  train_rgb=stable.cache_rgb(model, train_data))
    del model
    torch.cuda.empty_cache()
    reference = cap.pairwise_ids()
    baseline_head = cap.load_head(128, saved['model'])
    print('BASELINE EVAL', cap.parameters(baseline_head), flush=True)
    baseline = cap.evaluate(baseline_head, caches, data)
    del baseline_head
    baseline_ids = {split: {mode: cap.ids_of(baseline[split][mode]) for mode in ('oracle', 'rgb')}
                    for split in ('train', 'val')}
    head = cap.load_head(HIDDEN)
    width = head.hidden[0].out_features
    count = cap.parameters(head)
    print('PARAMS', dict(hidden=width, parameters=count), flush=True)
    if width != 256:
        raise RuntimeError('hidden width was not expanded')
    ROOT.mkdir(parents=True)
    generator = torch.Generator().manual_seed(cap.SEED)
    optimizer = torch.optim.AdamW(head.parameters(), lr=BASE_LR, weight_decay=0., foreach=False)
    head.train()
    best = None
    stalled = 0
    step = 0
    cursor = 0
    order = []
    losses = []
    train_seconds = 0.
    eval_seconds = 0.
    started = time.perf_counter()
    handle = (ROOT/'history.jsonl').open('w')
    stop = 'max_updates'
    while step < MAX_UPDATES:
        if cursor >= len(order):
            order = torch.randperm(len(data['train']), generator=generator).tolist()
            cursor = 0
        choice = order[cursor:cursor+cap.BATCH]
        cursor += len(choice)
        step += 1
        rate = learning_rate(step)
        for group in optimizer.param_groups:
            group['lr'] = rate
        batch = stable.take(caches['train_oracle'], choice, stable.DEVICE)
        tick = time.perf_counter()
        output = head(batch['features'], batch['valid'], batch['goal'], batch['order'])
        terms = relation_loss(output, batch['target_xy'], batch['target_angle'], batch['order'])
        optimizer.zero_grad(set_to_none=True)
        terms['loss'].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        train_seconds += time.perf_counter()-tick
        losses.append(float(terms['loss'].detach()))
        if step % EVAL_EVERY != 0 and step != MAX_UPDATES:
            continue
        tick = time.perf_counter()
        measured = cap.evaluate(head, caches, data)
        eval_seconds += time.perf_counter()-tick
        head.train()
        point = dict(update=step, lr=rate, loss=sum(losses)/len(losses),
                     train_seconds=train_seconds, eval_seconds=eval_seconds,
                     metrics=cap.snapshot(measured, baseline_ids, reference))
        losses = []
        handle.write(json.dumps(point)+'\n')
        handle.flush()
        val = point['metrics']['val']['rgb']
        train = point['metrics']['train']['rgb']
        print('EVAL', dict(update=step, lr=rate, loss=point['loss'], val_rgb=val['task'],
                           val_rgb4=val['count']['4']['success'], val_gap=val['gap']['median'],
                           train_rgb=train['task'], train_rgb4=train['count']['4']['success']), flush=True)
        score = rank_of(measured)
        if best is None or score > best['rank']:
            best = dict(measured=measured, rank=score, update=step, lr=rate)
            stalled = 0
            torch.save(dict(model={k: v.detach().cpu() for k, v in head.state_dict().items()},
                            update=step, lr=rate, hidden=width, parameters=count,
                            selection='validation_rgb_then_4object_then_gap'), ROOT/'checkpoint-best.pt')
        else:
            stalled += 1
            if step >= MIN_UPDATES and stalled >= PATIENCE_EVALS:
                stop = 'plateau'
                break
    handle.close()
    chosen = cap.snapshot(best['measured'], baseline_ids, reference)
    report = dict(status=verdict(best['measured']), hidden=width, parameters=count,
                  baseline_parameters=141062, baseline_checkpoint=str(BASELINE_CKPT),
                  baseline_reeval=cap.snapshot(baseline, baseline_ids, reference),
                  schedule=dict(warmup=WARMUP, hold_until=HOLD_UNTIL, base_lr=BASE_LR, floor_lr=FLOOR_LR),
                  stop=stop, updates=step, best_update=best['update'], best_lr=best['lr'],
                  seconds=time.perf_counter()-started, train_seconds=train_seconds, eval_seconds=eval_seconds,
                  best=chosen, joint_training=False, test_used=False)
    stable.write(ROOT/'result.json', report)
    print('DONE', dict(status=report['status'], update=best['update'], val_rgb=chosen['val']['rgb']['task'],
                       train_rgb=chosen['train']['rgb']['task'], seconds=round(report['seconds'], 1)), flush=True)


if __name__ == '__main__':
    main()
