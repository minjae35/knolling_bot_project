"""One pilot for the linked row Y. Selector, detector, language, size, angle, and Δx stay fixed."""
import json
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.inference import accepted_success
from knolling_model.recovery.contracts import Objects
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from run_full_arrangement import cache_records
from run_row_layout import attach_rows, collate_rows, hidden_of
from score_end_to_end import load_adopted, score_rgb
from train_set_context import fresh_split

ROOT = Path('artifacts/experiments/linked_y')
VAL_SEED = 90211
VAL_COUNT = 256
UPDATES = 200
BATCH = 8


def frozen_snapshot(model):
    placement = model.placement
    saved = {}
    for name, module in (
            ('layout', placement.row_layout), ('step', placement.step), ('angle', placement.angle),
            ('assign', placement.assign), ('order', placement.order_query)):
        for key, value in module.state_dict().items():
            saved[f'{name}.{key}'] = value.detach().cpu().clone()
    saved['order_begin'] = placement.order_begin.detach().cpu().clone()
    saved['context'] = {key: value.detach().cpu().clone() for key, value in model.context.state_dict().items()}
    return saved


def same_snapshot(model, saved):
    current = frozen_snapshot(model)
    for key, value in saved.items():
        if key == 'context':
            if any(not torch.equal(current['context'][name], tensor) for name, tensor in value.items()):
                return False
        elif not torch.equal(current[key], value):
            return False
    return True


def score_objects(model, relation, stored, records, forced):
    counts = Counter()
    tasks = []
    with torch.no_grad():
        for start in range(0, len(records), 16):
            chunk_s = stored[start:start+16]
            chunk_r = records[start:start+16]
            packed = collate_rows(chunk_s, stable.DEVICE)
            hidden = hidden_of(relation, packed['features'])
            widths, heights = packed['size'][..., 0], packed['size'][..., 1]
            row = packed['row'] if forced else None
            slot = packed['slot'] if forced else None
            output = model(hidden, packed['valid'], packed['goal'], widths, heights, row, slot, True,
                           colors=packed['color'])
            for local, record in enumerate(chunk_r):
                kept = chunk_s[local]['valid'].shape[0]
                item = {key: packed[key][local:local+1, :kept].cpu() for key in
                        ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')}
                objects = Objects(item['xy'], item['size'], item['angle'], item['color'], item['shape'],
                                  item['valid'], item['query_index'])
                layout = dict(xy=output['xy'][local:local+1, :kept].cpu(),
                              angle=output['angle'][local:local+1, :kept].cpu(),
                              valid=item['valid'], query_index=item['query_index'], rollout='free_rollout')
                ids, truth = recovery_supervision(record)
                judged = evaluate_batch(_oracle_view(objects, layout), Batch({}, [truth], [dict(
                    sample_id=record['sample_id'], object_ids=ids, record=record)]),
                    semantic_check=semantic_errors_v3)[0]
                task = bool(accepted_success(record, judged['task_success']))
                tasks.append(task)
                if not task:
                    counts.update(judged['errors'])
    return dict(task=sum(tasks), total=len(tasks), failures=dict(counts))


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('linked Y result already exists')
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    train = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    fit, dev, _blocked = fresh_split(train.records)
    if len(dev) != 512:
        raise RuntimeError(f'family-disjoint dev changed: {len(dev)}')
    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(VAL_SEED)).tolist()
    val_ids = [fit[index] for index in order[:VAL_COUNT]]
    train_ids = [fit[index] for index in order[VAL_COUNT:]]
    if set(val_ids) & set(dev):
        raise RuntimeError('validation overlaps the used dev split')
    system, relation, model, head, _saved = load_adopted()
    model.requires_grad_(False)
    for parameter in model.placement.y_parameters():
        parameter.requires_grad_(True)
    held = frozen_snapshot(model)
    val_records = [train.records[index] for index in val_ids]
    train_records = [train.records[index] for index in train_ids]
    val_stored = attach_rows(val_records, cache_records(system, val_records))
    train_stored = attach_rows(train_records, cache_records(system, train_records))
    before_rgb = score_rgb(system, relation, model, head, train, val_ids)
    before_free = score_objects(model, relation, val_stored, val_records, False)
    before_forced = score_objects(model, relation, val_stored, val_records, True)
    print('BEFORE', json.dumps(dict(rgb=before_rgb['task'], free=before_free['task'],
                                    forced=before_forced['task'], total=len(val_ids))), flush=True)
    optimizer = torch.optim.Adam(model.placement.y_parameters(), lr=1e-3)
    cursor = 0
    permutation = []
    losses = []
    for update in range(UPDATES):
        if cursor+BATCH > len(permutation):
            permutation = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(VAL_SEED+update)).tolist()
            cursor = 0
        choice = permutation[cursor:cursor+BATCH]
        cursor += BATCH
        packed = collate_rows([train_stored[index] for index in choice], stable.DEVICE)
        hidden = hidden_of(relation, packed['features'])
        output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                       packed['row'], packed['slot'], True, colors=packed['color'])
        loss = F.smooth_l1_loss(output['xy'][..., 1][packed['valid']], packed['target_xy'][..., 1][packed['valid']],
                                beta=.010)
        if not torch.isfinite(loss):
            raise RuntimeError('linked Y loss diverged')
        optimizer.zero_grad()
        loss.backward()
        if update == 0:
            placement = model.placement
            if placement.row_layout[-1].weight.grad is not None or placement.step[-1].weight.grad is not None:
                raise RuntimeError('the frozen origin or Δx received a gradient')
            if placement.row_gap[-1].weight.grad is None:
                raise RuntimeError('the gap head received no gradient')
        optimizer.step()
        if update in (0, 49, 199):
            losses.append(round(float(loss.detach()), 6))
            print('LOSS', update, losses[-1], flush=True)
    if not same_snapshot(model, held):
        raise RuntimeError('a frozen weight changed')
    after_rgb = score_rgb(system, relation, model, head, train, val_ids)
    after_free = score_objects(model, relation, val_stored, val_records, False)
    after_forced = score_objects(model, relation, val_stored, val_records, True)
    kept = len(before_rgb['successes'] & after_rgb['successes'])
    result = dict(
        status='GO' if after_rgb['task'] > before_rgb['task'] else 'NO-GO',
        adopted=False, test_used=False, dev_512_used=False,
        validation=len(val_ids), train=len(train_ids), updates=UPDATES,
        losses=losses,
        rgb=dict(before=before_rgb['task'], after=after_rgb['task'], total=len(val_ids),
                 preserved=kept, lost=before_rgb['task']-kept, gained=after_rgb['task']-kept,
                 before_failures=before_rgb['failure_counts'], after_failures=after_rgb['failure_counts'],
                 before_vertical=before_rgb['vertical'], after_vertical=after_rgb['vertical']),
        gt_free=dict(before=before_free, after=after_free),
        gt_row=dict(before=before_forced, after=after_forced))
    torch.save(dict(row_anchor=model.placement.row_anchor.state_dict(),
                    row_gap=model.placement.row_gap.state_dict(), update=UPDATES), ROOT/'checkpoint.pt')
    (ROOT/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
