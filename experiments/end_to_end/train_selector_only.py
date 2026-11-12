"""Train only the set encoder, row assignment, and ordering.

Origin, Δx, and angle stay at the adopted checkpoint. Their inputs stay the
original object features. GT object features are the training input. The
family-disjoint dev free rollout decides whether this selector is kept.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import time

import torch

from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.row_layout import assignment_loss, order_loss
from knolling_model.experiments.set_context import SetArrangement
from knolling_model.recovery.contracts import Objects
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from eval_row_layout import evaluate
from run_full_arrangement import cache_records
from run_row_layout import attach_rows, collate_rows, hidden_of
from score_end_to_end import JOINT_BEST, RELATION_BEST, ROW_BEST, SIZE_BEST, load_bundle
from train_set_context import fresh_split

ROOT = Path('artifacts/experiments/selector_only')
DATA = 'artifacts/generalization_20000/dataset'
BATCH = 8
MAX_UPDATES = 160
EVALS = (40, 80, 120, 160)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selector_parameters(model):
    placement = model.placement
    return (list(model.context.parameters())+list(placement.assign.parameters())
            +list(placement.order_query.parameters())+[placement.order_begin])


def pose_tensors(model):
    placement = model.placement
    found = {}
    for prefix, module in (('row_layout', placement.row_layout), ('step', placement.step), ('angle', placement.angle)):
        for name, tensor in module.state_dict().items():
            found[f'{prefix}.{name}'] = tensor.detach().cpu().clone()
    return found


def enable(model):
    model.requires_grad_(False)
    for parameter in selector_parameters(model):
        parameter.requires_grad_(True)


def selector_loss(relation, model, stored):
    packed = collate_rows(stored, stable.DEVICE)
    hidden = hidden_of(relation, packed['features'])
    output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                   packed['row'], packed['slot'], True)
    return assignment_loss(output, packed['row'], packed['valid'])+order_loss(output)


def rollout(model, relation, stored, records):
    model.eval()
    scenes = []
    with torch.no_grad():
        for start in range(0, len(records), 16):
            chunk = [records[index] for index in range(start, min(start+16, len(records)))]
            packed = collate_rows([stored[index] for index in range(start, start+len(chunk))], stable.DEVICE)
            hidden = hidden_of(relation, packed['features'])
            widths, heights = packed['size'][..., 0], packed['size'][..., 1]
            output = model(hidden, packed['valid'], packed['goal'], widths, heights, None, None, True)
            for local, record in enumerate(chunk):
                kept = stored[start+local]['valid'].shape[0]
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
                valid = packed['valid'][local, :kept]
                scenes.append(dict(sample_id=record['sample_id'], task=bool(judged['task_success']),
                                    errors=list(judged['errors']),
                                    row=[int(value) for value in output['row'][local, :kept][valid].cpu()],
                                    slot=[int(value) for value in output['slot'][local, :kept][valid].cpu()],
                                    xy=output['xy'][local, :kept][valid].cpu()))
    return scenes


def flips(before, after):
    by_id = {scene['sample_id']: scene for scene in after}
    lost, gained, preserved = [], [], 0
    leaked = 0
    shift_same, shift_changed = [], []
    for scene in before:
        other = by_id[scene['sample_id']]
        same = scene['row'] == other['row'] and scene['slot'] == other['slot']
        delta = torch.linalg.vector_norm(scene['xy']-other['xy'], dim=-1)
        (shift_same if same else shift_changed).extend(float(value) for value in delta)
        if same and float(delta.max()) > 1e-4:
            leaked += 1
        if scene['task'] and other['task']:
            preserved += 1
        elif scene['task'] and not other['task']:
            lost.append(dict(sample_id=scene['sample_id'], same=same,
                             row_changed=scene['row'] != other['row'],
                             order_changed=scene['slot'] != other['slot'], errors=other['errors']))
        elif other['task'] and not scene['task']:
            gained.append(dict(sample_id=scene['sample_id'], errors=scene['errors']))
    def median(values):
        if not values:
            return None
        return float(torch.tensor(values).median())
    return dict(baseline=sum(scene['task'] for scene in before), current=sum(scene['task'] for scene in after),
                preserved=preserved, lost=len(lost), gained=len(gained), pose_leaks=leaked,
                lost_row_changed=sum(item['row_changed'] for item in lost),
                lost_order_only=sum(item['order_changed'] and not item['row_changed'] for item in lost),
                lost_structure_same=sum(item['same'] for item in lost),
                lost_errors=dict(Counter(error for item in lost for error in item['errors'])),
                xy_shift_same_mm=None if median(shift_same) is None else median(shift_same)*1000,
                xy_shift_changed_mm=None if median(shift_changed) is None else median(shift_changed)*1000)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('selector-only result already exists')
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    stamps = dict(joint=sha256(JOINT_BEST), size=sha256(SIZE_BEST), row=sha256(ROW_BEST),
                  relation=sha256(RELATION_BEST))
    train = ExpandedDataset(DATA, 'train')
    fit, dev, _blocked = fresh_split(train.records)
    if len(dev) != 512:
        raise RuntimeError(f'family-disjoint dev changed: {len(dev)}')
    system, relation, placement, _head, checkpoint = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
    if int(checkpoint.get('update', -1)) != 30:
        raise RuntimeError('selector training must start from joint update 30')
    model = SetArrangement(placement).to(stable.DEVICE)
    enable(model)
    pose = pose_tensors(model)
    trainable = {id(parameter) for parameter in selector_parameters(model)}
    frozen_pose = [parameter for parameter in list(placement.row_layout.parameters())
                   +list(placement.step.parameters())+list(placement.angle.parameters())]
    if any(parameter.requires_grad or id(parameter) in trainable for parameter in frozen_pose):
        raise RuntimeError('pose parameters are reachable by the selector optimizer')
    fit_records = [train.records[index] for index in fit]
    dev_records = [train.records[index] for index in dev]
    fit_stored = attach_rows(fit_records, cache_records(system, fit_records))
    dev_stored = attach_rows(dev_records, cache_records(system, dev_records))
    packed = collate_rows(fit_stored[:4], stable.DEVICE)
    with torch.no_grad():
        hidden = hidden_of(relation, packed['features'])
        bare = placement(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                         None, None, True)
        wrapped = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                        None, None, True)
    identical = all(torch.equal(bare[key], wrapped[key]) for key in ('xy', 'origins', 'row_logits', 'angle'))
    print('IDENTITY', json.dumps(dict(identical=bool(identical))), flush=True)
    if not identical:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='identity', test_used=False, stage2=False))
        print('STOP the set encoder changed the pose before training', flush=True)
        return
    baseline_eval = evaluate(model, relation, dev_stored, dev_records, 'free')
    baseline_scenes = rollout(model, relation, dev_stored, dev_records)
    if sum(scene['task'] for scene in baseline_scenes) != baseline_eval['task']:
        raise RuntimeError('scene rollout does not match the dev evaluator')
    base_task = baseline_eval['task']
    print('BASE', json.dumps(dict(task=base_task, total=baseline_eval['total'], row=baseline_eval['row_accuracy'],
                                 slot=baseline_eval['slot_accuracy'], collision=baseline_eval['collision'],
                                 boundary=baseline_eval['boundary'], failures=baseline_eval['failures'])), flush=True)
    optimizer = torch.optim.AdamW((
        dict(params=list(model.context.parameters()), lr=1e-3),
        dict(params=list(placement.assign.parameters())+list(placement.order_query.parameters())
             +[placement.order_begin], lr=1e-4)), weight_decay=0.)
    if any(id(parameter) in {id(item) for item in frozen_pose} for group in optimizer.param_groups
           for parameter in group['params']):
        raise RuntimeError('optimizer includes a pose parameter')
    log = (ROOT/'history.jsonl').open('w')
    log.write(json.dumps(dict(update=0, task=base_task, row=baseline_eval['row_accuracy'],
                             slot=baseline_eval['slot_accuracy']))+'\n')
    best = dict(update=0, task=base_task)
    generator = torch.Generator().manual_seed(1703)
    order, cursor, update, reason = [], 0, 0, 'max_updates'
    started = time.perf_counter()
    while update < MAX_UPDATES:
        if cursor+BATCH > len(order):
            order = torch.randperm(len(fit_stored), generator=generator).tolist()
            cursor = 0
        choice = [fit_stored[index] for index in order[cursor:cursor+BATCH]]
        cursor += BATCH
        loss = selector_loss(relation, model, choice)
        if not torch.isfinite(loss):
            reason = 'diverged'
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if update == 0:
            opened = (model.context.attn.out_proj.weight.grad is not None
                      and float(model.context.attn.out_proj.weight.grad.detach().abs().sum()) > 0
                      and placement.assign[0].weight.grad is not None)
            held = all(parameter.grad is None for parameter in frozen_pose)
            print('GRAD', json.dumps(dict(selector=bool(opened), pose_frozen=bool(held))), flush=True)
            if not opened or not held:
                reason = 'gradient'
                break
        optimizer.step()
        update += 1
        if update not in EVALS and update != MAX_UPDATES:
            continue
        current = evaluate(model, relation, dev_stored, dev_records, 'free')
        point = dict(update=update, task=current['task'], row=current['row_accuracy'], slot=current['slot_accuracy'],
                     collision=current['collision'], boundary=current['boundary'], failures=current['failures'],
                     seconds=time.perf_counter()-started)
        log.write(json.dumps(point)+'\n')
        log.flush()
        print('EVAL', json.dumps(point), flush=True)
        if current['task'] > best['task']:
            best = dict(update=update, task=current['task'])
            torch.save(dict(placement=placement.state_dict(), context=model.context.state_dict(), update=update,
                            dev_task=current['task']), ROOT/'checkpoint-best.pt')
        if current['task'] < .75*base_task:
            reason = 'worse'
            break
    log.close()
    unchanged = all(torch.equal(pose[key], tensor.detach().cpu()) for key, tensor in pose_tensors(model).items())
    compared = flips(baseline_scenes, rollout(model, relation, dev_stored, dev_records))
    improved = best['task'] > base_task and unchanged and compared['pose_leaks'] == 0
    report = dict(status='STAGE1_PASS' if improved else 'STOP', stop=reason, stage2=False,
                  baseline=dict(task=base_task, total=512, row=baseline_eval['row_accuracy'],
                                slot=baseline_eval['slot_accuracy']),
                  best=best, flips=compared, pose_unchanged=bool(unchanged),
                  adopted=False, test_used=False, full_9600=False,
                  checkpoints_unchanged=all((
                      sha256(JOINT_BEST) == stamps['joint'], sha256(SIZE_BEST) == stamps['size'],
                      sha256(ROW_BEST) == stamps['row'], sha256(RELATION_BEST) == stamps['relation'])))
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(status=report['status'], best=best, flips=compared,
                                 pose_unchanged=bool(unchanged))), flush=True)


if __name__ == '__main__':
    main()
