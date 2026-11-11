"""Separate too little training from too small a relation head.

First reproduce the saved 128-d checkpoint. Then continue it for at most 3000
updates at a lower learning rate. A 256-d model is trained from scratch only
if that continuation does not move validation and train success stays low.
"""
import json
import time
from pathlib import Path

import torch
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.relation_pose import NeuralRelationPose, relation_loss
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable

SOURCE = Path('artifacts/phase1_relation_full/checkpoint-best.pt')
PAIRWISE = Path('artifacts/phase1_pairwise_affine/evaluation-002250.json')
ROOT = Path('artifacts/phase1_relation_continue')
WIDE_ROOT = Path('artifacts/phase1_relation_h256')
SEED = 1203
BATCH = 16
EXPECTED = dict(train_oracle=389, train_rgb=362, val_oracle=53, val_rgb=48, train_rgb_4=35, val_rgb_4=2)
CONTINUE_LR = 1e-4
CONTINUE_LR_FLOOR = 1e-4/3
MAX_CONTINUE = 3000
EVAL_EVERY = 250
# A one-scene flicker is the noise seen at lr 0.0003. Count an improvement only
# when validation RGB rises by 2 scenes, or train 4-object RGB rises by 8
# without giving back more than 2 validation RGB scenes.
VAL_RGB_GAIN = 2
TRAIN4_GAIN = 8
VAL_RGB_TOLERANCE = 2
TRAIN_RGB_LOW = 500
TRAIN4_LOW = 64


def ids_of(metrics):
    return sorted(scene['sample_id'] for scene in metrics['per_scene'] if scene['task'])


def counts(metrics):
    return {key: metrics[key] for key in ('task', 'total', 'count', 'order', 'selection_order', 'gap', 'failures')}


def retain(current, reference):
    ref = set(reference)
    kept = len(set(current) & ref)
    return dict(kept=kept, total=len(ref))


def snapshot(measured, baseline_ids, pairwise_ids):
    out = {}
    for split in ('train', 'val'):
        out[split] = {}
        for mode in ('oracle', 'rgb'):
            metrics = measured[split][mode]
            found = ids_of(metrics)
            item = counts(metrics)
            item['retention_vs_checkpoint'] = retain(found, baseline_ids[split][mode])
            if split == 'val' and mode == 'rgb':
                item['retention_vs_pairwise23'] = retain(found, pairwise_ids)
            out[split][mode] = item
    return out


def rank_of(measured):
    val, train = measured['val'], measured['train']
    median = val['rgb']['gap']['median']
    return (val['rgb']['task'], train['rgb']['count']['4']['success'], val['oracle']['task'],
            train['oracle']['count']['4']['success'], -(median if median is not None else 1e9))


def meaningful(current, best):
    val_rgb = current['val']['rgb']['task']
    train4 = current['train']['rgb']['count']['4']['success']
    if val_rgb >= best['val']['rgb']['task'] + VAL_RGB_GAIN:
        return True
    return (train4 >= best['train']['rgb']['count']['4']['success'] + TRAIN4_GAIN
            and val_rgb >= best['val']['rgb']['task'] - VAL_RGB_TOLERANCE)


def evaluate(head, caches, data):
    return dict(train=dict(oracle=stable.official_metrics(head, caches['train_oracle'], data['train']),
                           rgb=stable.official_metrics(head, caches['train_rgb'], data['train'])),
                val=dict(oracle=stable.official_metrics(head, caches['val_oracle'], data['val']),
                         rgb=stable.official_metrics(head, caches['val_rgb'], data['val'])))


def parameters(head):
    return sum(p.numel() for p in head.parameters())


def load_head(hidden, state=None):
    torch.manual_seed(SEED)
    head = NeuralRelationPose(12, hidden=hidden).to(stable.DEVICE)
    if state is not None:
        head.load_state_dict(state)
    return head


def pairwise_ids():
    rgb = json.loads(PAIRWISE.read_text())['rgb']['per_scene']
    return sorted(row['sample_id'] for row in rgb if row['task'])


def log_point(handle, point):
    handle.write(json.dumps(point)+'\n')
    handle.flush()
    val, train = point['metrics']['val'], point['metrics']['train']
    print('EVAL', dict(update=point['update'], lr=point['lr'], loss=point.get('loss'),
                       val_rgb=val['rgb']['task'], val_oracle=val['oracle']['task'],
                       val_rgb4=val['rgb']['count']['4']['success'],
                       train_rgb=train['rgb']['task'], train_rgb4=train['rgb']['count']['4']['success'],
                       keep_val_rgb=val['rgb']['retention_vs_checkpoint']['kept']), flush=True)


def run_updates(head, caches, data, output, *, max_updates, lr, floor, warmup, patience, initial, baseline_ids, pairwise):
    output.mkdir(parents=True, exist_ok=True)
    head.train()
    generator = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0., foreach=False)
    best = None if initial is None else dict(measured=initial, rank=rank_of(initial), update=0)
    stalled = 0
    dropped = False
    meaningful_updates = []
    history = []
    step = 0
    cursor = 0
    order = []
    losses = []
    started = time.perf_counter()
    handle = (output/'history.jsonl').open('w')
    stop = 'max_updates'
    while step < max_updates:
        if cursor >= len(order):
            order = torch.randperm(len(data['train']), generator=generator).tolist()
            cursor = 0
        choice = order[cursor:cursor+BATCH]
        cursor += len(choice)
        step += 1
        rate = lr if step >= warmup else lr*step/max(1, warmup)
        if dropped and floor is not None:
            rate = floor
        for group in optimizer.param_groups:
            group['lr'] = rate
        batch = stable.take(caches['train_oracle'], choice, stable.DEVICE)
        output_pose = head(batch['features'], batch['valid'], batch['goal'], batch['order'])
        terms = relation_loss(output_pose, batch['target_xy'], batch['target_angle'], batch['order'])
        optimizer.zero_grad(set_to_none=True)
        terms['loss'].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        losses.append(float(terms['loss'].detach()))
        if step % EVAL_EVERY != 0 and step != max_updates:
            continue
        measured = evaluate(head, caches, data)
        head.train()
        point = dict(update=step, lr=rate, loss=sum(losses)/len(losses), seconds=time.perf_counter()-started,
                     metrics=snapshot(measured, baseline_ids, pairwise))
        losses = []
        history.append(point)
        log_point(handle, point)
        score = rank_of(measured)
        previous = best['measured'] if best is not None else None
        if best is None or score > best['rank']:
            best = dict(measured=measured, rank=score, update=step)
            torch.save(dict(model={k: v.detach().cpu() for k, v in head.state_dict().items()},
                            update=step, lr=rate, hidden=head.hidden[0].out_features,
                            parameters=parameters(head), optimizer_state_restored=False,
                            selection='validation_rgb_then_train4'), output/'checkpoint-best.pt')
        if previous is None:
            # First measurement of a from-scratch model. It is the reference, not a gain.
            pass
        elif meaningful(measured, previous):
            stalled = 0
            meaningful_updates.append(step)
        else:
            stalled += 1
            if floor is not None and not dropped and stalled >= 2:
                dropped = True
                stalled = 0
                print('LR FLOOR', floor, flush=True)
            elif stalled >= patience:
                stop = 'plateau'
                break
    handle.close()
    return dict(stop=stop, updates=step, seconds=time.perf_counter()-started, meaningful_updates=meaningful_updates,
                lr_floor_used=bool(dropped), best_update=best['update'],
                best=snapshot(best['measured'], baseline_ids, pairwise))


def reproduced(measured):
    got = dict(train_oracle=measured['train']['oracle']['task'], train_rgb=measured['train']['rgb']['task'],
               val_oracle=measured['val']['oracle']['task'], val_rgb=measured['val']['rgb']['task'],
               train_rgb_4=measured['train']['rgb']['count']['4']['success'],
               val_rgb_4=measured['val']['rgb']['count']['4']['success'])
    return got == EXPECTED, got


def train_still_low(measured):
    rgb = measured['train']['rgb']
    return rgb['task'] < TRAIN_RGB_LOW and rgb['count']['4']['success'] < TRAIN4_LOW


def main():
    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError('continuation output already exists')
    stable.smoke()
    runner.setup()
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False)
    if 'optimizer' in saved:
        raise RuntimeError('optimizer state is present; the resume path must use it')
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
    reference = pairwise_ids()
    head = load_head(128, saved['model'])
    print('REPRODUCE', flush=True)
    baseline = evaluate(head, caches, data)
    match, got = reproduced(baseline)
    baseline_ids = {split: {mode: ids_of(baseline[split][mode]) for mode in ('oracle', 'rgb')} for split in ('train', 'val')}
    ROOT.mkdir(parents=True)
    stable.write(ROOT/'reproduction.json', dict(match=match, expected=EXPECTED, got=got,
                                                source=str(SOURCE), source_update=saved['step'],
                                                optimizer_in_checkpoint=False, parameters=parameters(head),
                                                metrics=snapshot(baseline, baseline_ids, reference)))
    print('REPRODUCE', 'MATCH' if match else 'MISMATCH', got, flush=True)
    if not match:
        stable.write(ROOT/'result.json', dict(status='FAIL', stage='reproduction', continued=False))
        return
    print('CONTINUE', dict(lr=CONTINUE_LR, floor=CONTINUE_LR_FLOOR, max_updates=MAX_CONTINUE,
                           optimizer='fresh AdamW; checkpoint had no optimizer state'), flush=True)
    continued = run_updates(head, caches, data, ROOT, max_updates=MAX_CONTINUE, lr=CONTINUE_LR,
                            floor=CONTINUE_LR_FLOOR, warmup=0, patience=4, initial=baseline,
                            baseline_ids=baseline_ids, pairwise=reference)
    improved = bool(continued['meaningful_updates'])
    best_train_rgb = continued['best']['train']['rgb']
    low = best_train_rgb['task'] < TRAIN_RGB_LOW and best_train_rgb['count']['4']['success'] < TRAIN4_LOW
    report = dict(status='continued', reproduction=got, optimizer_state_restored=False,
                  optimizer_reason='checkpoint-best.pt stores weights only',
                  parameters=parameters(head), continuation=dict(
                      stop=continued['stop'], updates=continued['updates'], seconds=continued['seconds'],
                      meaningful_updates=continued['meaningful_updates'], lr_floor_used=continued['lr_floor_used'],
                      best_update=continued['best_update'], best=continued['best']),
                  train_still_low=low, meaningful_improvement=improved, capacity_experiment=False)
    stable.write(ROOT/'result.json', report)
    print('CONTINUE DONE', dict(stop=continued['stop'], updates=continued['updates'],
                                best=continued['best_update'], meaningful=continued['meaningful_updates'],
                                low_train=low), flush=True)
    if improved or not low:
        return
    if WIDE_ROOT.exists():
        raise RuntimeError('256-d output already exists')
    print('CAPACITY 256', flush=True)
    wide = load_head(256)
    wide_result = run_updates(wide, caches, data, WIDE_ROOT, max_updates=7500, lr=3e-4, floor=None,
                              warmup=200, patience=12, initial=None, baseline_ids=baseline_ids, pairwise=reference)
    report['capacity_experiment'] = True
    report['wide'] = dict(parameters=parameters(wide), stop=wide_result['stop'], updates=wide_result['updates'],
                          seconds=wide_result['seconds'], meaningful_updates=wide_result['meaningful_updates'],
                          best_update=wide_result['best_update'], best=wide_result['best'])
    stable.write(ROOT/'result.json', report)
    stable.write(WIDE_ROOT/'result.json', report['wide'])
    print('CAPACITY DONE', dict(updates=wide_result['updates'], best=wide_result['best_update'],
                                val_rgb=wide_result['best']['val']['rgb']['task']), flush=True)


if __name__ == '__main__':
    main()
