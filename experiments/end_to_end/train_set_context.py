"""One object-attention pilot for row assignment, order, and origin.

Eight ground-truth scenes must be fit before any pilot step. Detector,
language, the relation MLP, the size head, angle, and Δx stay frozen.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.row_layout import assignment_loss, order_loss
from knolling_model.experiments.set_context import ContextualRows, ObjectSetContext
from knolling_model.experiments.size_refinement import encode_instruction, install_refined_size
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.selection import select_queries
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from joint_stage import matched_order
from run_full_arrangement import cache_records
from run_row_layout import attach_rows, collate_rows
from score_end_to_end import (
    JOINT_BEST, RELATION_BEST, ROW_BEST, SIZE_BEST, labels_of, load_bundle, public_rgb,
    score_rgb, validation_indices)
from train_row_assign import family_splits

ROOT = Path('artifacts/experiments/set_context_pilot')
DATA = 'artifacts/generalization_20000/dataset'
HOLDOUT_SEED = 1510
DEV_SEED = 1703
BATCH = 8
CONTEXT_LR = 1e-3
HEAD_LR = 1e-4
OVERFIT_STEPS = 300
MAX_UPDATES = 300
EVALS = (100, 200, 300)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh_split(records):
    _, held_dev, held_out = family_splits(records)
    del held_dev
    blocked = {records[index]['scene_family_id'] for index in held_out}
    found = {}
    for index, record in enumerate(records):
        if is_full(record) and record['scene_family_id'] not in found and record['scene_family_id'] not in blocked:
            found[record['scene_family_id']] = index
    ordered = [found[name] for name in sorted(found)]
    permutation = torch.randperm(len(ordered), generator=torch.Generator().manual_seed(DEV_SEED)).tolist()
    dev_count = len(ordered)//5
    dev = [ordered[index] for index in permutation[:dev_count]]
    fit = [ordered[index] for index in permutation[dev_count:]]
    return fit, dev, blocked


def overfit_records(records, fit):
    chosen = []
    for count in (8, 6, 4, 2):
        found = [records[index] for index in fit
                 if multi_row(records[index]) and len(records[index]['source']['objects']) == count]
        chosen.extend(found[:2])
    if len(chosen) != 8:
        raise RuntimeError('overfit pool does not have two scenes of each count')
    return chosen


def placement_parameters(placement):
    return (list(placement.assign.parameters())+list(placement.order_query.parameters())
            +[placement.order_begin]+list(placement.row_layout.parameters()))


def enable(wrapper):
    wrapper.placement.requires_grad_(False)
    wrapper.context.requires_grad_(True)
    for parameter in placement_parameters(wrapper.placement):
        parameter.requires_grad_(True)


def structure_loss(output, target_xy, teacher_row, teacher_slot, valid):
    assign = assignment_loss(output, teacher_row, valid)
    order = order_loss(output)
    origin = valid & (teacher_slot == 0)
    if not bool(origin.any()):
        raise ValueError('row origin loss needs the first object of a row')
    x = F.smooth_l1_loss(output['xy'][..., 0][origin], target_xy[..., 0][origin], beta=.010)
    y = F.smooth_l1_loss(output['xy'][..., 1][valid], target_xy[..., 1][valid], beta=.010)
    return dict(loss=assign+order+(x+y)/.010, assign=float(assign.detach()), order=float(order.detach()),
                origin=float((x+y).detach()))


def gt_loss(relation, wrapper, stored):
    packed = collate_rows(stored, stable.DEVICE)
    with torch.no_grad():
        hidden = relation._hidden(packed['features'])
    output = wrapper(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                     packed['row'], packed['slot'], True)
    losses = structure_loss(output, packed['target_xy'], packed['row'], packed['slot'], packed['valid'])
    return losses


def fit_eight(system, relation, placement, records):
    context = ObjectSetContext().to(stable.DEVICE)
    probe = copy.deepcopy(placement).to(stable.DEVICE)
    wrapper = ContextualRows(probe, context)
    enable(wrapper)
    stored = attach_rows(records, cache_records(system, records))
    before = evaluate(wrapper, relation, stored, records, 'free')
    optimizer = torch.optim.AdamW((
        dict(params=list(context.parameters()), lr=CONTEXT_LR),
        dict(params=placement_parameters(probe), lr=HEAD_LR)), weight_decay=0.)
    passed = False
    history = []
    for step in range(1, OVERFIT_STEPS+1):
        start = ((step-1)*BATCH) % len(stored)
        choice = [stored[(start+offset) % len(stored)] for offset in range(BATCH)]
        losses = gt_loss(relation, wrapper, choice)
        if not torch.isfinite(losses['loss']):
            break
        optimizer.zero_grad(set_to_none=True)
        losses['loss'].backward()
        if step == 1:
            opened = float(context.attn.out_proj.weight.grad.detach().abs().sum()) > 0
            frozen_step = probe.step[-1].weight.grad is None
            if not opened or not frozen_step:
                return dict(passed=False, reason='gradient', before=before['task'], opened=opened,
                            frozen_step=frozen_step)
        optimizer.step()
        if step not in (100, 200, 300):
            continue
        current = evaluate(wrapper, relation, stored, records, 'free')
        point = dict(step=step, task=current['task'], row_exact=current['row_exact_scenes'],
                     slot_exact=current['slot_exact_scenes'], row=current['row_accuracy'])
        history.append(point)
        print('OVERFIT', json.dumps(point), flush=True)
        if current['row_exact_scenes'] == len(records) and current['task'] >= before['task']:
            passed = True
            break
    return dict(passed=passed, before_task=before['task'], before_row_exact=before['row_exact_scenes'],
                history=history, scenes=len(records))


def rgb_loss(system, relation, wrapper, head, images, records):
    truths, identities = [], []
    for record in records:
        ids, truth = recovery_supervision(record)
        identities.append(ids)
        truths.append(truth)
    texts = [record['instruction']['text'] for record in records]
    with torch.no_grad():
        visual = install_refined_size(head, images, system.perception(images), system.threshold)
        text, mask = encode_instruction(system, texts, images.device)
    kept, orders = [], []
    for index, truth in enumerate(truths):
        source = {key: value[index:index+1] for key, value in visual['source'].items()}
        order = matched_order(source, {key: value.to(images.device) for key, value in truth.items()}, system.threshold)
        if order is None:
            continue
        kept.append(index)
        orders.append(order)
    if not kept:
        return None
    index = torch.tensor(kept, device=images.device)
    chosen_source = {key: value[index] for key, value in visual['source'].items()}
    chosen, width = [], max(len(order) for order in orders)
    teacher_row = torch.full((len(orders), width), -1, device=images.device)
    teacher_slot = torch.full((len(orders), width), -1, device=images.device)
    target = torch.zeros((len(orders), width, 2), device=images.device)
    for local, (record_index, order) in enumerate(zip(kept, orders)):
        chosen.append(torch.tensor([query for query, _ in order], device=images.device))
        row, slot = labels_of(records[record_index], identities[record_index])
        truth = truths[record_index]
        for slot_index, (_, gt) in enumerate(order):
            teacher_row[local, slot_index] = int(row[gt])
            teacher_slot[local, slot_index] = int(slot[gt])
            target[local, slot_index] = truth['target_xy'][gt].to(images.device)
    objects = pack_sources(chosen_source, chosen)
    with torch.no_grad():
        goal = system.layout.goal_encoder(objects, text[index], mask[index])['vector']
        hidden = relation._hidden(objects.features())
    output = wrapper(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                     teacher_row, teacher_slot, True)
    losses = structure_loss(output, target, teacher_row, teacher_slot, objects.valid)
    losses.update(kept=len(kept), skipped=len(records)-len(kept), output=output)
    return losses


def save(path, wrapper, optimizer, update, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(placement=wrapper.placement.state_dict(), context=wrapper.context.state_dict(),
                    optimizer=optimizer.state_dict(), update=update, **payload), path)


def load_weights(wrapper, path):
    saved = torch.load(path, map_location=stable.DEVICE, weights_only=False)
    wrapper.placement.load_state_dict(saved['placement'])
    wrapper.context.load_state_dict(saved['context'])
    enable(wrapper)
    return saved


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('set context pilot result already exists')
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    stamps = dict(joint=sha256(JOINT_BEST), size=sha256(SIZE_BEST), row=sha256(ROW_BEST),
                  relation=sha256(RELATION_BEST))
    train = ExpandedDataset(DATA, 'train')
    val = ExpandedDataset(DATA, 'validation')
    fit, dev, blocked = fresh_split(train.records)
    val_families = {record['scene_family_id'] for record in val.records}
    fit_families = {train.records[index]['scene_family_id'] for index in fit}
    dev_families = {train.records[index]['scene_family_id'] for index in dev}
    if fit_families & dev_families or fit_families & blocked or dev_families & blocked or val_families & (fit_families | dev_families | blocked):
        raise RuntimeError('source families overlap a held-out split')
    system, relation, placement, head, checkpoint = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
    if int(checkpoint.get('update', -1)) != 30:
        raise RuntimeError('set context pilot must start from joint update 30')
    for module in (system, relation, placement, head):
        module.eval()
        module.requires_grad_(False)
    print('OVERFIT start', flush=True)
    report = fit_eight(system, relation, placement, overfit_records(train.records, fit))
    print('OVERFIT done', json.dumps(report), flush=True)
    if not report['passed']:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='overfit', overfit=report,
                                              full_9600=False, test_used=False))
        print('STOP the eight ground-truth scenes were not fit', flush=True)
        return
    if not all(torch.equal(placement.state_dict()[key].cpu(), value.cpu())
               for key, value in torch.load(JOINT_BEST, map_location='cpu', weights_only=False)['row'].items()):
        raise RuntimeError('overfit changed the adopted row weights')
    context = ObjectSetContext().to(stable.DEVICE)
    wrapper = ContextualRows(placement, context)
    enable(wrapper)
    val_indices = validation_indices(val)
    baseline = score_rgb(system, relation, wrapper, head, val, val_indices)
    dev_base = score_rgb(system, relation, wrapper, head, train, dev)
    print('BASE', json.dumps(dict(rgb=baseline['task'], eight=baseline['by_count']['8'],
                                 dev=dev_base['task'], dev_n=len(dev), fit=len(fit))), flush=True)
    if baseline['task'] != 54 or baseline['by_count']['8']['success'] != 4:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='baseline', rgb=public_rgb(baseline),
                                              overfit=report))
        print('STOP zero context did not reproduce 54/152', flush=True)
        return
    optimizer = torch.optim.AdamW((
        dict(params=list(context.parameters()), lr=CONTEXT_LR),
        dict(params=placement_parameters(placement), lr=HEAD_LR)), weight_decay=0.)
    save(ROOT/'checkpoint-0000.pt', wrapper, optimizer, 0, {})
    save(ROOT/'checkpoint-best.pt', wrapper, optimizer, 0, dict(dev_rgb=dev_base['task']))
    history = (ROOT/'history.jsonl').open('w')
    history.write(json.dumps(dict(update=0, dev_rgb=dev_base['task'], dev_row=dev_base['row_accuracy']))+'\n')
    best = dict(update=0, rgb=dev_base['task'], row=dev_base['row_accuracy'])
    generator = torch.Generator().manual_seed(DEV_SEED)
    order, cursor, update, seen, skipped = [], 0, 0, 0, 0
    reason = 'max_updates'
    started = time.perf_counter()
    while update < MAX_UPDATES:
        if cursor+BATCH > len(order):
            order = torch.randperm(len(fit), generator=generator).tolist()
            cursor = 0
        choice = [fit[order[index]] for index in range(cursor, cursor+BATCH)]
        cursor += BATCH
        images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        records = [train.records[index] for index in choice]
        prepared = rgb_loss(system, relation, wrapper, head, images, records)
        seen += len(records)
        if prepared is None:
            skipped += len(records)
            continue
        skipped += prepared['skipped']
        if not torch.isfinite(prepared['loss']):
            reason = 'diverged'
            break
        optimizer.zero_grad(set_to_none=True)
        prepared['loss'].backward()
        if update == 0:
            leaked = placement.step[-1].weight.grad is not None or placement.angle[-1].weight.grad is not None
            opened = context.attn.out_proj.weight.grad is not None and float(
                context.attn.out_proj.weight.grad.detach().abs().sum()) > 0
            if leaked or not opened:
                reason = 'gradient'
                print('STOP pilot gradient', json.dumps(dict(leaked=leaked, opened=opened)), flush=True)
                break
        optimizer.step()
        update += 1
        if update not in EVALS and not (update == 100 and best['update'] == 0):
            continue
        current = score_rgb(system, relation, wrapper, head, train, dev)
        point = dict(update=update, dev_rgb=current['task'], dev_row=current['row_accuracy'],
                     dev_total=current['total'], seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(point), flush=True)
        if current['task'] < dev_base['task']*.75:
            reason = 'collapse'
            break
        if update >= 100 and current['task'] <= dev_base['task'] and current['row_accuracy'] <= dev_base['row_accuracy']:
            reason = 'no_gain'
            break
        if current['task'] > best['rgb'] and current['row_accuracy'] >= dev_base['row_accuracy']-.02:
            best = dict(update=update, rgb=current['task'], row=current['row_accuracy'])
            save(ROOT/'checkpoint-best.pt', wrapper, optimizer, update,
                 dict(dev_rgb=current['task'], dev_row=current['row_accuracy']))
    history.close()
    save(ROOT/'checkpoint-last.pt', wrapper, optimizer, update, dict(stop=reason))
    saved = load_weights(wrapper, ROOT/'checkpoint-best.pt')
    final = score_rgb(system, relation, wrapper, head, val, val_indices)
    dev_final = score_rgb(system, relation, wrapper, head, train, dev)
    kept = baseline['successes'] & final['successes']
    improved = (int(saved['update']) > 0 and final['task'] > 54 and final['by_count']['8']['success'] >= 4
                and len(kept) >= 48 and dev_final['task'] > dev_base['task'])
    test_report = None
    if improved:
        test = ExpandedDataset(DATA, 'test')
        test_families = {record['scene_family_id'] for record in test.records}
        if test_families & (fit_families | dev_families | blocked | val_families):
            raise RuntimeError('test families overlap a used split')
        indices = [index for index, record in enumerate(test.records) if multi_row(record)]
        bare, bare_relation, bare_row, bare_head, _ = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
        previous = score_rgb(bare, bare_relation, bare_row, bare_head, test, indices)
        current = score_rgb(system, relation, wrapper, head, test, indices)
        test_report = dict(total=len(indices), baseline_rgb=previous['task'], rgb=current['task'],
                           baseline_eight=previous['by_count']['8'], eight=current['by_count']['8'],
                           baseline_row=previous['row_accuracy'], row=current['row_accuracy'])
        improved = current['task'] > previous['task']
    adopted = bool(improved and test_report and test_report['rgb'] > test_report['baseline_rgb'])
    result = dict(
        status='PASS' if adopted else 'FAIL', stop=reason, best_update=int(saved['update']),
        overfit=report, unfrozen=('object_context', 'row_assign', 'row_order', 'row_origin'),
        frozen=('rt_detr', 'language', 'goal', 'relation', 'size_head', 'row_step', 'row_angle'),
        excluded_scenes=skipped, attempted_scenes=seen,
        validation=dict(baseline_rgb=baseline['task'], rgb=final['task'], total=final['total'],
                        baseline_eight=baseline['by_count']['8'], eight=final['by_count']['8'],
                        preserved=len(kept), lost=len(baseline['successes']-final['successes']),
                        gained=len(final['successes']-baseline['successes']),
                        baseline_row=baseline['row_accuracy'], row=final['row_accuracy'],
                        baseline_slot=baseline['slot_accuracy'], slot=final['slot_accuracy'],
                        baseline_failures=baseline['failure_counts'], failures=final['failure_counts']),
        dev=dict(baseline_rgb=dev_base['task'], rgb=dev_final['task'], total=dev_final['total'],
                 baseline_row=dev_base['row_accuracy'], row=dev_final['row_accuracy']),
        test=test_report, adopted=adopted,
        checkpoints_unchanged=all((
            sha256(JOINT_BEST) == stamps['joint'], sha256(SIZE_BEST) == stamps['size'],
            sha256(ROW_BEST) == stamps['row'], sha256(RELATION_BEST) == stamps['relation'])),
        test_used=test_report is not None, full_9600=False)
    stable.write(ROOT/'result.json', result)
    print('DONE', json.dumps(dict(status=result['status'], update=int(saved['update']),
                                 rgb=final['task'], dev=dev_final['task'], test=test_report)), flush=True)


if __name__ == '__main__':
    main()
