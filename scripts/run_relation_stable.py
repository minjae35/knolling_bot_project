"""Stable relation-head training. Architecture and data stay as they are.

The 24-scene check uses the new schedule. Full training starts only after that
check holds on three consecutive free-rollout evaluations. The saved training
checkpoint is the best validation placement, not the lowest loss.
"""
import json
import math
import os
from collections import Counter
from pathlib import Path

os.environ.update(dict(
    HF_HOME='/workspace/cache/huggingface',
    HF_HUB_CACHE='/workspace/cache/huggingface/hub',
    HUGGINGFACE_HUB_CACHE='/workspace/cache/huggingface/hub',
    TRANSFORMERS_CACHE='/workspace/cache/huggingface/hub',
    TORCH_HOME='/workspace/cache/torch',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    CUBLAS_WORKSPACE_CONFIG=':4096:8'))

import numpy as np
import torch
from knolling_dataset.validator import inferred_rows, polygon
from knolling_model.data import Batch
from knolling_model.diagnostics import positioned_objects
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.phase1_evaluation import semantic_errors_phase1
from knolling_model.experiments.relation_pose import NeuralRelationPose, relation_loss
from knolling_model.recovery.contracts import Objects, pack_sources
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.model import legacy_evaluation_view
from knolling_model.recovery.selection import select_queries
import run_phase1_pairwise_affine as runner
from run_relation_probe import choose, scene_success

SPEC = json.loads(Path('configs/phase1_relation_pose.json').read_text())
ROOT = Path(SPEC['output'])
WIDTH = 4
DEVICE = 'cuda'


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def learning_rate(step, total):
    warm = SPEC['warmup_steps']
    base, floor = SPEC['lr'], SPEC['lr_floor']
    if step <= warm:
        return floor+(base-floor)*step/warm
    progress = min(1., (step-warm)/max(1, total-warm))
    return floor+0.5*(base-floor)*(1+math.cos(math.pi*progress))


def smoke():
    head = NeuralRelationPose(12)
    features = torch.randn(2, WIDTH, 12)
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    goal = torch.zeros(2, 13)
    goal[:, 6] = 1
    order = torch.tensor([[0, 1, 2, 3], [1, 0, -1, -1]])
    output = head(features, valid, goal, order)
    loss = relation_loss(output, torch.rand(2, WIDTH, 2), torch.zeros(2, WIDTH), order)
    loss['loss'].backward()
    if head.relation[-1].weight.grad.abs().sum() == 0:
        raise RuntimeError('padded relation batch produced no relation gradient')


def blank(count):
    return dict(features=torch.zeros(count, WIDTH, 12), xy=torch.zeros(count, WIDTH, 2),
                size=torch.zeros(count, WIDTH, 2), angle=torch.zeros(count, WIDTH),
                color=torch.zeros(count, WIDTH, 3), shape=torch.zeros(count, WIDTH, 3),
                valid=torch.zeros(count, WIDTH, dtype=torch.bool),
                query_index=torch.full((count, WIDTH), -1, dtype=torch.long),
                goal=torch.zeros(count, 13), target_xy=torch.zeros(count, WIDTH, 2),
                target_angle=torch.zeros(count, WIDTH), order=torch.full((count, WIDTH), -1, dtype=torch.long))


def store_example(bucket, index, objects, goal, record, targets):
    k = objects.valid.shape[1]
    if k > WIDTH:
        raise RuntimeError('scene exceeds the phase-1 width')
    bucket['features'][index, :k] = objects.features()[0].detach().cpu()
    for name in ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index'):
        bucket[name][index, :k] = getattr(objects, name)[0].detach().cpu()
    bucket['goal'][index] = goal[0].detach().cpu()
    if not targets:
        return
    ids, truth = recovery_supervision(record)
    teacher = runner.teacher_order([record], [ids], k, objects.xy.device)[0]
    bucket['target_xy'][index, :len(ids)] = truth['target_xy']
    bucket['target_angle'][index, :len(ids)] = truth['target_angle']
    bucket['order'][index, :teacher.numel()] = teacher.cpu()


def cache_oracle(model, data):
    stored = blank(len(data))
    for index, record in enumerate(data.records):
        objects, text, mask, _ = runner.arguments(model, [record], DEVICE)
        with torch.no_grad():
            goal = model.layout.goal_encoder(objects, text, mask)['vector']
        store_example(stored, index, objects, goal, record, targets=True)
        if (index+1) % 200 == 0:
            print('CACHE oracle', index+1, flush=True)
    return stored


def cache_rgb(model, data):
    """Keep every detection. RGB can return more objects than the four-object pad."""
    stored = []
    for index, record in enumerate(data.records):
        sample = data[index]
        with torch.no_grad():
            visual = model.perception(sample['image'][None].to(DEVICE))
            objects = pack_sources(visual['source'], select_queries(visual['source'], .5))
            tokens = model.tokenizer([record['instruction']['text']], return_tensors='pt', padding=True, truncation=False)
            mask = tokens['attention_mask'].to(DEVICE).bool()
            text = model.arrangement.text_encoder(input_ids=tokens['input_ids'].to(DEVICE), attention_mask=mask).last_hidden_state
            goal = model.layout.goal_encoder(objects, text, mask)['vector']
        stored.append(dict(features=objects.features().detach().cpu(), valid=objects.valid.detach().cpu(),
                           query_index=objects.query_index.detach().cpu(), goal=goal.detach().cpu(),
                           source={key: value.detach().cpu() for key, value in visual['source'].items()}))
        if (index+1) % 100 == 0:
            print('CACHE rgb', index+1, flush=True)
    return stored


def take(stored, indices, device):
    chosen = torch.tensor(indices)
    data = {key: stored[key][chosen].to(device) for key in
            ('features', 'xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index', 'goal',
             'target_xy', 'target_angle', 'order')}
    return data


def moved_ids(record):
    return [oid for row in record['supervision_v3']['rows'] for oid in row
            if not record['supervision_v3']['copy_required'][oid]]


def gap_values(record, targets):
    if not targets:
        return []
    values = []
    for line in inferred_rows(positioned_objects(record, targets)):
        for left, right in zip(line, line[1:]):
            opening = polygon(right, right.target_pose).bounds[0]-polygon(left, left.target_pose).bounds[2]
            values.append(abs(opening-record['instruction']['gap_m'])*1000)
    return values


def official_metrics(head, stored, data):
    if isinstance(stored, list):
        return metrics_rgb(head, stored, data)
    head.eval()
    rows = []
    selection_hits = None if 'source' in stored else 0
    with torch.no_grad():
        block = 16
        outputs = []
        for start in range(0, len(data), block):
            outputs.append(head(stored['features'][start:start+block].to(DEVICE),
                                stored['valid'][start:start+block].to(DEVICE),
                                stored['goal'][start:start+block].to(DEVICE)))
        xy = torch.cat([item['xy'].cpu() for item in outputs])
        angle = torch.cat([item['angle'].cpu() for item in outputs])
        order = torch.cat([item['predicted_order'].cpu() for item in outputs])
    for index, record in enumerate(data.records):
        sample = data[index]
        item = take(stored, [index], DEVICE)
        layout = dict(xy=xy[index:index+1].to(DEVICE), angle=angle[index:index+1].to(DEVICE),
                      valid=item['valid'], query_index=item['query_index'], rollout='free_rollout')
        if 'source' in stored:
            source = {key: value.to(DEVICE) for key, value in stored['source'][index].items()}
            view = legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5))
        else:
            objects = Objects(item['xy'], item['size'], item['angle'], item['color'], item['shape'],
                              item['valid'], item['query_index'])
            view = _oracle_view(objects, layout)
        truth = {key: value.to(DEVICE) for key, value in recovery_supervision(record)[1].items()}
        row = evaluate_batch(view, Batch({}, [truth], [sample['metadata']]),
                             semantic_check=semantic_errors_phase1)[0]
        rows.append(row)
        if selection_hits is not None:
            selection_hits += int(order[index].tolist() == stored['order'][index].tolist())
    return pack_metrics(rows, data, selection_hits)


def metrics_rgb(head, stored, data):
    head.eval()
    rows = []
    with torch.no_grad():
        for index, (item, record) in enumerate(zip(stored, data.records)):
            output = head(item['features'].to(DEVICE), item['valid'].to(DEVICE), item['goal'].to(DEVICE))
            layout = dict(xy=output['xy'], angle=output['angle'], valid=item['valid'].to(DEVICE),
                          query_index=item['query_index'].to(DEVICE), rollout='free_rollout')
            source = {key: value.to(DEVICE) for key, value in item['source'].items()}
            view = legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5))
            truth = {key: value.to(DEVICE) for key, value in recovery_supervision(record)[1].items()}
            rows.append(evaluate_batch(view, Batch({}, [truth], [data[index]['metadata']]),
                                       semantic_check=semantic_errors_phase1)[0])
    return pack_metrics(rows, data, None)


def pack_metrics(rows, data, selection_hits):
    per_scene = []
    gaps = []
    order_hits = 0
    for record, row in zip(data.records, rows):
        values = gap_values(record, row.get('predicted_targets'))
        gaps.extend(values)
        targets = row.get('predicted_targets')
        ordered = targets is not None and sorted(targets, key=lambda oid: (targets[oid]['x_m'], oid)) == moved_ids(record)
        order_hits += int(ordered)
        per_scene.append(dict(sample_id=record['sample_id'], objects=len(record['source']['objects']),
                              task=bool(row['task_success']), gap_error_mm=values, errors=row['errors'], order=bool(ordered)))
    counts = {}
    for count in (2, 3, 4):
        chosen = [scene for scene in per_scene if scene['objects'] == count]
        counts[str(count)] = dict(success=sum(scene['task'] for scene in chosen), total=len(chosen),
                                  order=sum(scene['order'] for scene in chosen))
    return dict(task=sum(scene['task'] for scene in per_scene), total=len(per_scene),
                count=counts, order=order_hits, selection_order=selection_hits,
                failures=dict(Counter(error for row in rows for error in row['errors'])),
                gap=dict(pairs=len(gaps), median=None if not gaps else float(np.median(gaps)),
                         p95=None if not gaps else float(np.percentile(gaps, 95))),
                per_scene=per_scene)


def rank_of(metrics):
    return (metrics['oracle']['task'], metrics['rgb']['task'],
            metrics['oracle']['count']['4']['success'], metrics['rgb']['count']['4']['success'],
            metrics['oracle']['count']['3']['success'], metrics['rgb']['count']['3']['success'],
            -(metrics['oracle']['gap']['median'] if metrics['oracle']['gap']['median'] is not None else 1e9))


def overfit_check(train_cache, train_data):
    wanted = {record['sample_id'] for _, record in choose(train_data.records)}
    indices = [i for i, record in enumerate(train_data.records) if record['sample_id'] in wanted]
    if len(indices) != SPEC['overfit_scenes']:
        raise RuntimeError('the original 24 scenes were not found')
    rows = [train_data.records[i] for i in indices]
    packed = take(train_cache, indices, DEVICE)
    total = SPEC['overfit_max_steps']
    generator = torch.Generator().manual_seed(SPEC['seed'])
    head = NeuralRelationPose(12).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=SPEC['lr'], weight_decay=SPEC['weight_decay'], foreach=False)
    history, streak, best = [], 0, None
    for step in range(1, total+1):
        rate = learning_rate(step, total)
        for group in optimizer.param_groups:
            group['lr'] = rate
        choice = torch.randint(len(indices), (SPEC['overfit_batch'],), generator=generator)
        output = head(packed['features'][choice], packed['valid'][choice], packed['goal'][choice], packed['order'][choice])
        terms = relation_loss(output, packed['target_xy'][choice], packed['target_angle'][choice], packed['order'][choice])
        optimizer.zero_grad(set_to_none=True)
        terms['loss'].backward()
        grad = float(torch.nn.utils.clip_grad_norm_(head.parameters(), SPEC['grad_clip']))
        optimizer.step()
        if step % SPEC['overfit_eval_every'] == 0 or step == total:
            point = evaluate_overfit(head, rows, packed)
            point.update(step=step, lr=rate, loss=float(terms['loss'].detach()), grad_norm=grad)
            history.append(point)
            print('OVERFIT', {key: point[key] for key in ('step', 'lr', 'loss', 'grad_norm', 'success', 'within_2mm', 'pairs', 'gap_median')}, flush=True)
            passed = point['success'] == len(rows) and point['pairs'] == 3*len(rows) and point['within_2mm'] == point['pairs']
            streak = streak+1 if passed else 0
            if best is None or (point['success'], point['within_2mm'], -(point['gap_median'] or 1e9)) > best['key']:
                best = dict(key=(point['success'], point['within_2mm'], -(point['gap_median'] or 1e9)),
                            step=step, metrics=point, state={k: v.detach().cpu() for k, v in head.state_dict().items()})
            if streak >= SPEC['overfit_stable_evals']:
                break
    passed = streak >= SPEC['overfit_stable_evals']
    torch.save(dict(model=best['state'], step=best['step'], metrics=best['metrics'], scheduler=SPEC['scheduler'], lr=SPEC['lr']),
               ROOT/'overfit-24.pt')
    write(ROOT/'overfit-history.json', history)
    return dict(passed=passed, best_step=best['step'], best=best['metrics'], streak=streak, steps=history[-1]['step'], history=history)


def evaluate_overfit(head, rows, packed):
    head.eval()
    with torch.no_grad():
        output = head(packed['features'], packed['valid'], packed['goal'])
    xy, angle = output['xy'].cpu(), output['angle'].cpu()
    items = [scene_success(record, recovery_supervision(record)[0], xy[i, :len(recovery_supervision(record)[0])],
                           angle[i, :len(recovery_supervision(record)[0])]) for i, record in enumerate(rows)]
    gaps = [gap for item in items for gap in item['gaps']]
    head.train()
    return dict(success=sum(item['task'] for item in items), scenes=len(items), pairs=len(gaps),
                within_2mm=sum(abs(gap) <= 2 for gap in gaps),
                gap_median=None if not gaps else float(np.median(np.abs(gaps))))


def train_full(train_cache, val_oracle, val_rgb, train_data, val_data):
    epochs, batch = SPEC['max_epochs'], SPEC['batch_size']
    steps_per = math.ceil(len(train_data)/batch)
    total = epochs*steps_per
    generator = torch.Generator().manual_seed(SPEC['seed'])
    head = NeuralRelationPose(12).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=SPEC['lr'], weight_decay=SPEC['weight_decay'], foreach=False)
    best, waited, history = None, 0, []
    step = 0
    for epoch in range(1, epochs+1):
        order = torch.randperm(len(train_data), generator=generator).tolist()
        losses = []
        head.train()
        for start in range(0, len(order), batch):
            step += 1
            rate = learning_rate(step, total)
            for group in optimizer.param_groups:
                group['lr'] = rate
            choice = order[start:start+batch]
            data = take(train_cache, choice, DEVICE)
            output = head(data['features'], data['valid'], data['goal'], data['order'])
            terms = relation_loss(output, data['target_xy'], data['target_angle'], data['order'])
            optimizer.zero_grad(set_to_none=True)
            terms['loss'].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), SPEC['grad_clip'])
            optimizer.step()
            losses.append(float(terms['loss'].detach()))
        metrics = dict(oracle=official_metrics(head, val_oracle, val_data),
                       rgb=official_metrics(head, val_rgb, val_data))
        point = dict(epoch=epoch, step=step, lr=learning_rate(step, total), train_loss=float(np.mean(losses)),
                     oracle_task=metrics['oracle']['task'], rgb_task=metrics['rgb']['task'],
                     oracle_4=metrics['oracle']['count']['4']['success'], rgb_4=metrics['rgb']['count']['4']['success'],
                     oracle_gap_median=metrics['oracle']['gap']['median'], rgb_gap_median=metrics['rgb']['gap']['median'],
                     oracle_gap_p95=metrics['oracle']['gap']['p95'], rgb_gap_p95=metrics['rgb']['gap']['p95'],
                     oracle_order=metrics['oracle']['order'], rgb_order=metrics['rgb']['order'])
        history.append(point)
        with (ROOT/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(point)+'\n')
        print('EPOCH', point, flush=True)
        score = rank_of(metrics)
        if best is None or score > best['score']:
            best = dict(score=score, epoch=epoch, step=step, metrics=metrics,
                        state={key: value.detach().cpu() for key, value in head.state_dict().items()})
            waited = 0
            torch.save(dict(model=best['state'], epoch=epoch, step=step, metrics=slim(metrics),
                            selection='validation_free_rollout'), ROOT/'checkpoint-best.pt')
        else:
            waited += 1
            if waited >= SPEC['patience_epochs']:
                break
    return best, history


def slim(metrics):
    return {mode: {key: metrics[mode][key] for key in ('task', 'total', 'count', 'order', 'selection_order', 'failures', 'gap')}
            for mode in metrics}


def decision(metrics):
    oracle, rgb = metrics['oracle'], metrics['rgb']
    checks = dict(rgb_above_baseline=rgb['task'] > SPEC['require_rgb_task_greater_than'],
                  oracle_4=oracle['count']['4']['success'] >= SPEC['require_oracle_4_at_least'],
                  rgb_4=rgb['count']['4']['success'] >= SPEC['require_rgb_4_at_least'])
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                baseline_rgb=f"{SPEC['baseline_rgb_task']}/94")


def main():
    smoke()
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'history.jsonl').exists() or (ROOT/'checkpoint-best.pt').exists():
        raise RuntimeError('relation output already exists; refusing to overwrite a training run')
    train_data, val_data = runner.datasets()
    model = load_integrated(device=DEVICE)
    print('CACHE start', flush=True)
    train_cache = cache_oracle(model, train_data)
    val_oracle = cache_oracle(model, val_data)
    val_rgb = cache_rgb(model, val_data)
    del model
    torch.cuda.empty_cache()
    overfit = overfit_check(train_cache, train_data)
    print('OVERFIT STATUS', 'PASS' if overfit['passed'] else 'FAIL', 'streak', overfit['streak'], flush=True)
    if not overfit['passed']:
        write(ROOT/'result.json', dict(status='FAIL', stage='overfit', overfit=slim_overfit(overfit), continued=False))
        return
    best, history = train_full(train_cache, val_oracle, val_rgb, train_data, val_data)
    head = NeuralRelationPose(12).to(DEVICE)
    head.load_state_dict(best['state'])
    print('TRAIN EVAL', flush=True)
    model = load_integrated(device=DEVICE)
    train_rgb = cache_rgb(model, train_data)
    del model
    train_metrics = dict(oracle=official_metrics(head, train_cache, train_data),
                         rgb=official_metrics(head, train_rgb, train_data))
    verdict = decision(best['metrics'])
    write(ROOT/'result.json', dict(status=verdict['status'], decision=verdict, overfit=slim_overfit(overfit),
                                   selected=dict(epoch=best['epoch'], step=best['step']),
                                   validation=slim(best['metrics']), train=slim(train_metrics),
                                   epochs_ran=history[-1]['epoch'], updates=history[-1]['step'],
                                   continued_to_joint=False))
    print('STATUS', verdict['status'], 'selected epoch', best['epoch'], flush=True)


def slim_overfit(overfit):
    return {key: overfit[key] for key in ('passed', 'best_step', 'best', 'streak', 'steps')}


if __name__ == '__main__':
    main()
