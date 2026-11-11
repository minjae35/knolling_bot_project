"""Learn 2-to-8-object full arrangements with the existing relation head.

One run, initialized from the modular best. Language, goal, and vision stay
frozen. Partial and no-op scenes stay out. Inference does not read a target
pose. The modular checkpoint is only read.
"""
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import MAX_OBJECTS, arrangement_errors, collate, is_full
from knolling_model.experiments.relation_pose import relation_loss
from knolling_model.recovery.contracts import Objects, oracle_objects
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from run_relation_joint import changed, install_live_text, module_state

ROOT = Path('artifacts/full_arrangement_relation')
MODULAR = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
DATA = Path('artifacts/generalization_20000/dataset')
LR = 1e-4
SINGLE_UPDATES = 300
MULTI_UPDATES = 900
BATCH = 16


def stamp(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def subset(data, split, expected):
    manifest = json.loads(Path(runner.BASE[split+'_manifest']).read_text())
    by_id = {record['sample_id']: record for record in data.records}
    view = copy.copy(data)
    view.records = [by_id[sample_id] for sample_id in manifest['sample_ids']]
    view.pairs = manifest['counterfactual_pairs']
    if len(view.records) != expected:
        raise RuntimeError(f'{split} manifest count changed')
    return view


def census(records):
    counted = Counter()
    for record in records:
        counted[(len(record['source']['objects']), len(record['supervision_v3']['rows']))] += 1
    return {f'{count}obj_{rows}row': value for (count, rows), value in sorted(counted.items())}


def target_acceptance(records):
    failures = Counter()
    transitions = 0
    for record in records:
        report = arrangement_errors(record)
        transitions += int(report['row_transition'] and bool(report['vertical_mm']))
        if not report['task'] or not report['row_match']:
            failures.update(report['errors'] or ['row_match'])
    samples = {}
    for count in (2, 4, 6, 8):
        record = next(item for item in records if len(item['source']['objects']) == count
                      and len(item['supervision_v3']['rows']) == (1 if count < 6 else 2))
        samples[str(count)] = evaluator_accepts(record)
    return dict(scenes=len(records), failures=dict(failures), row_transitions=transitions,
                evaluator=samples, passed=not failures and all(samples.values()))


def evaluator_accepts(record):
    """Feed the dataset target through the official evaluator and require success."""
    ids, truth = recovery_supervision(record)
    objects = oracle_objects([{key: value for key, value in truth.items()}])
    layout = dict(xy=truth['target_xy'][None], angle=truth['target_angle'][None], valid=objects.valid,
                  query_index=objects.query_index, rollout='free_rollout')
    view = _oracle_view(objects, layout)
    meta = dict(sample_id=record['sample_id'], object_ids=ids, record=record)
    row = evaluate_batch(view, Batch({}, [truth], [meta]), semantic_check=semantic_errors_v3)[0]
    return bool(row['task_success'])


def cache_records(model, records):
    stored = []
    for start in range(0, len(records), BATCH):
        batch = records[start:start+BATCH]
        objects, text, mask, _ = runner.arguments(model, batch, stable.DEVICE)
        with torch.no_grad():
            goal = model.layout.goal_encoder(objects, text, mask)['vector']
        encoded = objects.features()
        for index, record in enumerate(batch):
            kept = int(objects.valid[index].sum())
            ids, truth = recovery_supervision(record)
            teacher = runner.teacher_order([record], [ids], kept, objects.xy.device)[0]
            stored.append(dict(features=encoded[index, :kept].detach().cpu(),
                               xy=objects.xy[index, :kept].detach().cpu(),
                               size=objects.size[index, :kept].detach().cpu(),
                               angle=objects.angle[index, :kept].detach().cpu(),
                               color=objects.color[index, :kept].detach().cpu(),
                               shape=objects.shape[index, :kept].detach().cpu(),
                               valid=objects.valid[index, :kept].detach().cpu(),
                               query_index=objects.query_index[index, :kept].detach().cpu(),
                               goal=goal[index].detach().cpu(),
                               target_xy=truth['target_xy'][:kept].detach().cpu(),
                               target_angle=truth['target_angle'][:kept].detach().cpu(),
                               order=teacher.detach().cpu()))
        if start % 1600 == 0:
            print('CACHE', start, len(records), flush=True)
    return stored


def score(head, stored, records):
    rows = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(records), BATCH):
            indices = range(start, min(start+BATCH, len(records)))
            packed = collate([stored[index] for index in indices], stable.DEVICE)
            output = head(packed['features'], packed['valid'], packed['goal'])
            for local, index in enumerate(indices):
                kept = stored[index]['valid'].shape[0]
                item = {key: packed[key][local:local+1, :kept].cpu() for key in
                        ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')}
                objects = Objects(item['xy'], item['size'], item['angle'], item['color'], item['shape'],
                                  item['valid'], item['query_index'])
                layout = dict(xy=output['xy'][local:local+1, :kept].cpu(),
                              angle=output['angle'][local:local+1, :kept].cpu(),
                              valid=item['valid'], query_index=item['query_index'], rollout='free_rollout')
                ids, truth = recovery_supervision(records[index])
                meta = dict(sample_id=records[index]['sample_id'], object_ids=ids, record=records[index])
                judged = evaluate_batch(_oracle_view(objects, layout), Batch({}, [truth], [meta]),
                                        semantic_check=semantic_errors_v3)[0]
                detail = arrangement_errors(records[index], judged['predicted_targets']) if judged.get('predicted_targets') else None
                sequence = output['predicted_order'][local, :kept].cpu().tolist() == stored[index]['order'].tolist()
                rows.append(dict(task=bool(judged['task_success']), errors=judged['errors'], sequence=bool(sequence),
                                 row_match=bool(detail and detail['row_match']),
                                 collision='collision' in judged['errors'], boundary='boundary' in judged['errors'],
                                 clearance='clearance' in judged['errors'],
                                 horizontal_mm=[] if detail is None else detail['horizontal_mm'],
                                 vertical_mm=[] if detail is None else detail['vertical_mm']))
    return pack(rows, records)


def pack(rows, records):
    def stats(values):
        if not values:
            return dict(median=None, p95=None, pairs=0)
        return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)), pairs=len(values))

    def brief(chosen):
        return dict(success=sum(row['task'] for row in chosen), total=len(chosen),
                    order=sum(row['sequence'] for row in chosen), rows=sum(row['row_match'] for row in chosen),
                    collision=sum(row['collision'] for row in chosen), boundary=sum(row['boundary'] for row in chosen))

    grouped = {count: [] for count in range(2, MAX_OBJECTS+1)}
    single, multi = [], []
    horizontal, vertical = [], []
    for row, record in zip(rows, records):
        grouped[len(record['source']['objects'])].append(row)
        (single if len(record['supervision_v3']['rows']) == 1 else multi).append(row)
        horizontal.extend(row['horizontal_mm'])
        vertical.extend(row['vertical_mm'])
    return dict(task=sum(row['task'] for row in rows), total=len(rows),
                by_count={str(count): brief(grouped[count]) for count in range(2, MAX_OBJECTS+1)},
                single_row=brief(single), multi_row=brief(multi),
                order=sum(row['sequence'] for row in rows), row_match=sum(row['row_match'] for row in rows),
                collision=sum(row['collision'] for row in rows), boundary=sum(row['boundary'] for row in rows),
                clearance=sum(row['clearance'] for row in rows),
                failures=dict(Counter(error for row in rows for error in row['errors'])),
                horizontal_gap_mm=stats(horizontal), vertical_gap_mm=stats(vertical))


def probe(head, stored):
    packed = collate(stored, stable.DEVICE)
    features = packed['features'].detach().requires_grad_(True)
    supervised = head(features, packed['valid'], packed['goal'], packed['order'])
    loss = relation_loss(supervised, packed['target_xy'], packed['target_angle'], packed['order'])['loss']
    features_changed = features.detach().clone()
    features_changed[~packed['valid']] = 3
    other = head(features_changed, packed['valid'], packed['goal'], packed['order'])
    other_loss = relation_loss(other, packed['target_xy'], packed['target_angle'], packed['order'])['loss']
    loss.backward()
    free = head(packed['features'], packed['valid'], packed['goal'])
    masks = []
    for local, item in enumerate(stored):
        kept = item['valid'].shape[0]
        chosen = free['predicted_order'][local].detach().cpu()
        masks.append(bool(chosen[kept:].eq(-1).all() and set(chosen[:kept].tolist()) == set(range(kept))))
    valid_grad = float(features.grad[packed['valid']].abs().sum())
    return dict(loss_finite=bool(torch.isfinite(loss)), padding_invariant=torch.allclose(loss, other_loss),
                valid_grad=valid_grad > 0, masks=all(masks),
                passed=bool(torch.isfinite(loss) and torch.allclose(loss, other_loss) and valid_grad > 0 and all(masks)))


def train(head, train_cache, val_cache, val_records, single_pool):
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0., foreach=False)
    generator = torch.Generator().manual_seed(cap.SEED)
    order, cursor = [], 0
    every = list(range(len(train_cache)))
    active = None
    best = None
    history = (ROOT/'history.jsonl').open('w')
    losses, started = [], time.perf_counter()

    def evaluate_at(step):
        point = score(head, val_cache, val_records)
        point.update(update=step, loss=None if not losses else sum(losses)/len(losses),
                     seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(dict(update=step, task=point['task'], total=point['total'],
                                     single=point['single_row']['success'], multi=point['multi_row']['success'],
                                     by_count={key: value['success'] for key, value in point['by_count'].items()})),
              flush=True)
        return point

    def rank_of(point):
        return (point['task'], point['multi_row']['success'], point['by_count']['8']['success'],
                point['by_count']['6']['success'], -(point['horizontal_gap_mm']['median'] or 1e9))

    baseline = evaluate_at(0)
    floor = rank_of(baseline)
    for step in range(1, SINGLE_UPDATES+MULTI_UPDATES+1):
        pool = single_pool if step <= SINGLE_UPDATES else every
        if pool is not active:
            order, cursor, active = [], 0, pool
        if cursor >= len(order):
            order = torch.randperm(len(pool), generator=generator).tolist()
            cursor = 0
        choice = [pool[index] for index in order[cursor:cursor+BATCH]]
        cursor += len(choice)
        packed = collate([train_cache[index] for index in choice], stable.DEVICE)
        head.train()
        output = head(packed['features'], packed['valid'], packed['goal'], packed['order'])
        terms = relation_loss(output, packed['target_xy'], packed['target_angle'], packed['order'])
        if not torch.isfinite(terms['loss']):
            history.close()
            return baseline, None, 'nonfinite_loss'
        optimizer.zero_grad(set_to_none=True)
        terms['loss'].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        rate = 1e-5+(LR-1e-5)*min(1., step/40)
        for group in optimizer.param_groups:
            group['lr'] = rate
        optimizer.step()
        losses.append(float(terms['loss'].detach()))
        if step in (SINGLE_UPDATES, SINGLE_UPDATES+300, SINGLE_UPDATES+MULTI_UPDATES):
            point = evaluate_at(step)
            losses = []
            rank = rank_of(point)
            if rank > floor and (best is None or rank > best['rank']):
                best = dict(rank=rank, update=step, metrics=point,
                            state={key: value.detach().cpu() for key, value in head.state_dict().items()})
                torch.save(dict(update=step, model=best['state'], source='modular_best', lr=LR,
                                frozen=('language_encoder', 'goal_encoder', 'visual_encoder')),
                           ROOT/'checkpoint-best.pt')
    history.close()
    return baseline, best, 'completed'


def main():
    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError('full-arrangement output already exists')
    held = stamp(MODULAR)
    runner.setup()
    train_data = ExpandedDataset(DATA, 'train')
    val_data = ExpandedDataset(DATA, 'validation')
    phase_val = subset(val_data, 'validation', 94)
    full_train = [record for record in train_data.records if is_full(record)]
    full_val = [record for record in val_data.records if is_full(record)]
    distribution = dict(train=census(full_train), validation=census(full_val),
                        train_scenes=len(full_train), validation_scenes=len(full_val),
                        train_single=sum(len(record['supervision_v3']['rows']) == 1 for record in full_train),
                        validation_single=sum(len(record['supervision_v3']['rows']) == 1 for record in full_val))
    print('CENSUS', json.dumps(dict(train=distribution['train_scenes'], val=distribution['validation_scenes'],
                                   train_single=distribution['train_single'])), flush=True)
    acceptance = target_acceptance(full_val)
    print('GT', json.dumps(acceptance['evaluator']), 'failures', acceptance['failures'], flush=True)
    ROOT.mkdir(parents=True)
    if not acceptance['passed']:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='target_evaluator', acceptance=acceptance,
                                              distribution=distribution))
        print('STOP target evaluator', flush=True)
        return
    from knolling_model.experiments.integration import load_integrated
    model = load_integrated(device=stable.DEVICE)
    install_live_text(model)
    model.eval()
    for module in (model.perception, model.arrangement.text_encoder, model.layout.goal_encoder):
        module.requires_grad_(False)
        module.eval()
    saved = torch.load(MODULAR, map_location='cpu', weights_only=False)
    head = cap.load_head(128, saved['model'])
    phase_oracle = stable.cache_oracle(model, phase_val)
    phase_rgb = stable.cache_rgb(model, phase_val)
    reproduced = stable.official_metrics(head, phase_oracle, phase_val)
    reproduced_rgb = stable.official_metrics(head, phase_rgb, phase_val)
    print('REPRODUCE', json.dumps(dict(oracle=reproduced['task'], rgb=reproduced_rgb['task'],
                                      rgb4=reproduced_rgb['count']['4']['success'])), flush=True)
    if (reproduced['task'], reproduced_rgb['task'], reproduced_rgb['count']['4']['success']) != (60, 56, 4):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='reproduction',
                                              oracle=reproduced['task'], rgb=reproduced_rgb['task']))
        print('STOP reproduction mismatch', flush=True)
        return
    examples = []
    for count in (2, 4, 6, 8):
        examples.append(next(record for record in full_train if len(record['source']['objects']) == count))
    probed = probe(head, cache_records(model, examples))
    head.zero_grad(set_to_none=True)
    print('PROBE', json.dumps(probed), flush=True)
    if not probed['passed']:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='probe', probe=probed, distribution=distribution))
        print('STOP probe', flush=True)
        return
    frozen = dict(perception=module_state(model.perception), language=module_state(model.arrangement.text_encoder),
                  goal=module_state(model.layout.goal_encoder))
    print('CACHE train', len(full_train), flush=True)
    train_cache = cache_records(model, full_train)
    print('CACHE val', len(full_val), flush=True)
    val_cache = cache_records(model, full_val)
    single_pool = [index for index, record in enumerate(full_train) if len(record['supervision_v3']['rows']) == 1]
    baseline, best, reason = train(head, train_cache, val_cache, full_val, single_pool)
    movement = dict(perception=changed(frozen['perception'], model.perception),
                    language=changed(frozen['language'], model.arrangement.text_encoder),
                    goal=changed(frozen['goal'], model.layout.goal_encoder))
    chosen = head
    if best is not None:
        chosen = cap.load_head(128, best['state'])
    train_metrics = score(chosen, train_cache, full_train)
    phase_after = stable.official_metrics(chosen, phase_oracle, phase_val)
    learned = best is not None and reason == 'completed' and (
        best['metrics']['by_count']['6']['success'] > baseline['by_count']['6']['success']
        or best['metrics']['by_count']['8']['success'] > baseline['by_count']['8']['success'])
    status = 'PASS' if learned else 'FAIL'
    report = dict(status=status, stop=reason, distribution=distribution, target_check=acceptance,
                  probe=probed, baseline=baseline, best=None if best is None else best['metrics'],
                  best_update=None if best is None else best['update'], train=train_metrics,
                  phase1_oracle_before=reproduced['task'], phase1_oracle_after=phase_after['task'],
                  phase1_rgb_before=reproduced_rgb['task'], frozen_movement=movement,
                  representation=dict(head='NeuralRelationPose', max_objects=MAX_OBJECTS,
                                      row_break='one neighbor displacement, no separate row latent',
                                      partial_noop='deferred'),
                  modular_checkpoint_unchanged=stamp(MODULAR) == held, test_used=False, adopted=status == 'PASS')
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(status=status, best=report['best_update'],
                                 val=None if best is None else best['metrics']['task'],
                                 phase1=phase_after['task'], movement=movement)), flush=True)


if __name__ == '__main__':
    main()
