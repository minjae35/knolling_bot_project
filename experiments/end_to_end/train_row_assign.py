"""Train only row assignment from the adopted joint checkpoint.

Row ids are the top-to-bottom order used by the frozen origin. A permutation
is a different origin slot, so the loss keeps those ids. Detector, language,
goal, relation, order, angle, origin, step, and the size head stay frozen.
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
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.row_layout import assignment_loss
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import encode_instruction, poses_from_detection
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
from knolling_model.recovery.selection import select_queries
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from eval_row_layout import evaluate
from joint_stage import matched_order
from run_full_arrangement import cache_records
from run_row_layout import attach_rows
from score_end_to_end import (
    JOINT_BEST, RELATION_BEST, ROW_BEST, SIZE_BEST, labels_of, load_bundle, public_rgb,
    score_rgb, validation_indices)

ROOT = Path('artifacts/experiments/row_assign_pilot')
DATA = 'artifacts/generalization_20000/dataset'
SEED = 1510
BATCH = 4
LR = 1e-4
MAX_UPDATES = 40
EVALS = (10, 20, 40)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family_splits(records):
    found = {}
    for index, record in enumerate(records):
        if is_full(record) and record['scene_family_id'] not in found:
            found[record['scene_family_id']] = index
    ordered = [found[name] for name in sorted(found)]
    permutation = torch.randperm(len(ordered), generator=torch.Generator().manual_seed(SEED)).tolist()
    hold_count = len(ordered)//5
    dev_count = len(ordered)//5
    hold = [ordered[index] for index in permutation[:hold_count]]
    dev = [ordered[index] for index in permutation[hold_count:hold_count+dev_count]]
    fit = [ordered[index] for index in permutation[hold_count+dev_count:]]
    return fit, dev, hold


def assign_parameters(row_model):
    return list(row_model.assign.parameters())


def enable(row_model, head):
    head.requires_grad_(False)
    row_model.requires_grad_(False)
    for parameter in assign_parameters(row_model):
        parameter.requires_grad_(True)


def copy_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def same_state(module, saved):
    current = module.state_dict()
    return all(torch.equal(current[key], value.to(current[key].device)) for key, value in saved.items())


def grad_norm(parameters):
    total = 0.
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm())
    return total


def canonical_rows(records):
    checked = inverted = 0
    for record in records:
        rows = record['supervision_v3']['rows']
        if not is_full(record) or len(rows) < 2:
            continue
        identities, truth = recovery_supervision(record)
        location = {object_id: index for index, object_id in enumerate(identities)}
        means = []
        for members in rows:
            chosen = torch.stack([truth['target_xy'][location[object_id], 1] for object_id in members])
            means.append(float(chosen.mean()))
        checked += 1
        inverted += sum(upper <= lower for upper, lower in zip(means, means[1:]))
    return dict(checked=checked, inverted=inverted)


def prepare(system, relation, row_model, images, records):
    truths, identities = [], []
    for record in records:
        ids, truth = recovery_supervision(record)
        identities.append(ids)
        truths.append({key: value.to(images.device) for key, value in truth.items()})
    texts = [record['instruction']['text'] for record in records]
    with torch.no_grad():
        visual = system.perception(images)
        text, mask = encode_instruction(system, texts, images.device)
    kept, orders = [], []
    for index, truth in enumerate(truths):
        source = {key: value[index:index+1] for key, value in visual['source'].items()}
        order = matched_order(source, truth, system.threshold)
        if order is None:
            continue
        kept.append(index)
        orders.append(order)
    if not kept:
        return None
    index = torch.tensor(kept, device=images.device)
    chosen_source = {key: value[index] for key, value in visual['source'].items()}
    chosen = []
    width = max(len(order) for order in orders)
    teacher = torch.full((len(orders), width), -1, device=images.device, dtype=torch.long)
    for local, (record_index, order) in enumerate(zip(kept, orders)):
        queries = torch.tensor([query for query, _ in order], device=images.device)
        chosen.append(queries)
        row, _ = labels_of(records[record_index], identities[record_index])
        for slot, (_, gt) in enumerate(order):
            teacher[local, slot] = int(row[gt])
    objects = pack_sources(chosen_source, chosen)
    with torch.no_grad():
        goal = system.layout.goal_encoder(objects, text[index], mask[index])['vector']
        hidden = relation._hidden(objects.features())
    output = row_model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1], None, None, True)
    loss = assignment_loss(output, teacher, objects.valid)
    return dict(loss=loss, output=output, teacher=teacher, valid=objects.valid,
                kept=len(kept), skipped=len(records)-len(kept))


def cause(errors, row_exact, slot_exact):
    if 'count' in errors or 'source_identity' in errors:
        return 'detection'
    if not row_exact:
        return 'row_assignment'
    if not slot_exact:
        return 'ordering'
    if 'axis_alignment' in errors:
        return 'angle'
    if 'vertical_gap' in errors or 'row_alignment' in errors or 'horizontal_gap' in errors:
        return 'xy_placement'
    if 'size_order' in errors:
        return 'size_order'
    return 'other'


def inspect(system, relation, row_model, head, data, index):
    record = data.records[index]
    identities, truth = recovery_supervision(record)
    truth = {key: value.cpu() for key, value in truth.items()}
    image = data[index]['image'][None].to(stable.DEVICE)
    try:
        output, objects, visual = poses_from_detection(
            system, relation, row_model, image, [record['instruction']['text']], head)
    except ValueError:
        return dict(errors=['count'], row_exact=False, slot_exact=False, objects=len(identities))
    source = {key: value.detach().cpu() for key, value in visual['source'].items()}
    layout = dict(xy=output['xy'].detach().cpu(), angle=output['angle'].detach().cpu(),
                  valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                  rollout='free_rollout')
    view = legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5))
    meta = dict(sample_id=record['sample_id'], object_ids=identities, record=record)
    judged = evaluate_batch(view, Batch({}, [truth], [meta]), semantic_check=semantic_errors_v3)[0]
    row, slot = labels_of(record, identities)
    chosen = select_queries(visual['source'], system.threshold)[0]
    distance = torch.cdist(objects.xy[0, :len(chosen)], truth['source_xy'].to(objects.xy.device))
    used_query, used_gt, row_hits, slot_hits, labeled = set(), set(), 0, 0, 0
    for item in distance.flatten().argsort().tolist():
        query, gt = divmod(item, distance.shape[1])
        if query in used_query or gt in used_gt or float(distance[query, gt]) > .010:
            continue
        used_query.add(query)
        used_gt.add(gt)
        labeled += 1
        row_hits += int(int(output['row'][0, query]) == int(row[gt]))
        slot_hits += int(int(output['slot'][0, query]) == int(slot[gt]))
    exact = labeled == len(chosen) == len(identities)
    return dict(errors=list(judged['errors']), objects=len(identities),
                row_exact=exact and row_hits == len(chosen), slot_exact=exact and slot_hits == len(chosen),
                task=accepted_success(record, judged['task_success']))


def tally(system, relation, row_model, head, data, sample_ids):
    by_id = {record['sample_id']: index for index, record in enumerate(data.records)}
    scenes = [inspect(system, relation, row_model, head, data, by_id[sample_id]) for sample_id in sample_ids]
    counts = Counter(cause(scene['errors'], scene['row_exact'], scene['slot_exact']) for scene in scenes)
    return dict(counts)


def save(path, head, row_model, optimizer, update, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(head=head.state_dict(), row=row_model.state_dict(),
                    optimizer=optimizer.state_dict(), update=update, **payload), path)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('row assignment pilot result already exists')
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    stamps = dict(joint=sha256(JOINT_BEST), size=sha256(SIZE_BEST), row=sha256(ROW_BEST),
                  relation=sha256(RELATION_BEST))
    train = ExpandedDataset(DATA, 'train')
    val = ExpandedDataset(DATA, 'validation')
    fit, dev, hold = family_splits(train.records)
    groups = [{train.records[index]['scene_family_id'] for index in indices} for indices in (fit, dev, hold)]
    val_families = {record['scene_family_id'] for record in val.records}
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2] or val_families & set().union(*groups):
        raise RuntimeError('source families overlap a held-out split')
    covered = set().union(*groups)
    missing = {record['scene_family_id'] for record in train.records if is_full(record)}-covered
    if missing:
        raise RuntimeError('a full-arrangement family has no split')
    order_audit = canonical_rows([train.records[index] for index in fit+dev+hold])
    if order_audit['inverted']:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='row_order', **order_audit))
        print('STOP teacher row ids are not top-to-bottom', flush=True)
        return
    system, relation, row_model, head, checkpoint = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
    if int(checkpoint.get('update', -1)) != 30:
        raise RuntimeError('row assignment pilot must start from joint update 30')
    for module in (system, relation, row_model, head):
        module.eval()
    enable(row_model, head)
    val_indices = validation_indices(val)
    multi_val = [val.records[index] for index in val_indices]
    stored = attach_rows(multi_val, cache_records(system, multi_val))
    baseline = score_rgb(system, relation, row_model, head, val, val_indices)
    gt_base = evaluate(row_model, relation, stored, multi_val, 'free')
    dev_base = score_rgb(system, relation, row_model, head, train, dev)
    hold_base = score_rgb(system, relation, row_model, head, train, hold)
    print('BASE', json.dumps(dict(
        rgb=baseline['task'], eight=baseline['by_count']['8'], row=baseline['row_accuracy'],
        gt_row=gt_base['row_accuracy'], dev=dev_base['task'], hold=hold_base['task'],
        fit=len(fit), dev_n=len(dev), hold_n=len(hold))), flush=True)
    if baseline['task'] != 54 or baseline['by_count']['8']['success'] != 4:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='baseline', rgb=public_rgb(baseline)))
        print('STOP joint checkpoint did not reproduce 54/152', flush=True)
        return
    frozen = dict(perception=copy_state(system.perception), language=copy_state(system.arrangement.text_encoder),
                  goal=copy_state(system.layout.goal_encoder), relation=copy_state(relation),
                  row=copy_state(row_model), head=copy_state(head))
    probe = None
    for start in range(0, min(len(fit), 160), BATCH):
        choice = fit[start:start+BATCH]
        images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        records = [train.records[index] for index in choice]
        prepared = prepare(system, relation, row_model, images, records)
        if prepared is not None:
            probe = (images, records, prepared)
            break
    if probe is None:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='no_probe_batch'))
        print('STOP no matched training batch', flush=True)
        return
    images, records, prepared = probe
    swapped = prepared['teacher'].clone()
    swapped = torch.where(swapped == 0, torch.full_like(swapped, 1),
                           torch.where(swapped == 1, torch.zeros_like(swapped), swapped))
    swapped_loss = assignment_loss(
        dict(row_logits=prepared['output']['row_logits'].detach()), swapped, prepared['valid'])
    row_model.zero_grad(set_to_none=True)
    prepared['loss'].backward()
    assign_grad = grad_norm(assign_parameters(row_model))
    leaked = grad_norm([parameter for name, parameter in row_model.named_parameters()
                        if not name.startswith('assign.')])
    outside = [system.perception, system.arrangement.text_encoder, system.layout.goal_encoder, relation, head]
    outside_grad = any(parameter.grad is not None and float(parameter.grad.detach().norm()) > 0
                       for module in outside for parameter in module.parameters())
    connected = all(parameter.requires_grad and parameter.grad is not None
                    for parameter in assign_parameters(row_model))
    print('GRAD', json.dumps(dict(
        loss=float(prepared['loss']), swapped_loss=float(swapped_loss), assign=assign_grad,
        other_row=leaked, outside=outside_grad, connected=connected,
        index_sensitive=not torch.allclose(prepared['loss'].detach(), swapped_loss))), flush=True)
    label_changed = not torch.equal(swapped, prepared['teacher'])
    loss_changed = not torch.allclose(prepared['loss'].detach(), swapped_loss)
    class_diff = not torch.allclose(prepared['output']['row_logits'][..., 0],
                                    prepared['output']['row_logits'][..., 1])
    if not connected or assign_grad <= 0 or leaked > 0 or outside_grad or (
            label_changed and class_diff and not loss_changed):
        stable.write(ROOT/'result.json', dict(
            status='STOP', stage='gradient', assign=assign_grad, other_row=leaked,
            outside=outside_grad, connected=connected, label_changed=label_changed,
            loss_changed=loss_changed, row_order=order_audit))
        print('STOP assignment loss is not connected only to row.assign', flush=True)
        return
    row_model.load_state_dict({key: value.to(stable.DEVICE) for key, value in frozen['row'].items()})
    enable(row_model, head)
    optimizer = torch.optim.AdamW(assign_parameters(row_model), lr=LR, weight_decay=0.)
    fresh = prepare(system, relation, row_model, images, records)
    before = fresh['output']['row_logits'].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    fresh['loss'].backward()
    optimizer.step()
    after = prepare(system, relation, row_model, images, records)
    moved = float((after['output']['row_logits']-before).abs().max())
    frozen_same = all(same_state(module, frozen[name]) for name, module in (
        ('perception', system.perception), ('language', system.arrangement.text_encoder),
        ('goal', system.layout.goal_encoder), ('relation', relation), ('head', head)))
    assign_changed = not same_state(row_model.assign, {key.split('assign.', 1)[1]: value
                                                       for key, value in frozen['row'].items()
                                                       if key.startswith('assign.')})
    others = {key: value for key, value in frozen['row'].items() if not key.startswith('assign.')}
    others_same = all(torch.equal(row_model.state_dict()[key], value.to(stable.DEVICE))
                      for key, value in others.items())
    print('STEP', json.dumps(dict(logit_move=moved, frozen_same=frozen_same,
                                 assign_changed=assign_changed, others_same=others_same)), flush=True)
    row_model.load_state_dict({key: value.to(stable.DEVICE) for key, value in frozen['row'].items()})
    enable(row_model, head)
    if not frozen_same or not assign_changed or not others_same:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='step', logit_move=moved,
                                              frozen_same=frozen_same, assign_changed=assign_changed,
                                              others_same=others_same, row_order=order_audit))
        print('STOP the probe step left the assignment head', flush=True)
        return
    optimizer = torch.optim.AdamW(assign_parameters(row_model), lr=LR, weight_decay=0.)
    save(ROOT/'checkpoint-0000.pt', head, row_model, optimizer, 0, {})
    save(ROOT/'checkpoint-best.pt', head, row_model, optimizer, 0, dict(dev_rgb=dev_base['task']))
    history = (ROOT/'history.jsonl').open('w')
    history.write(json.dumps(dict(update=0, dev_rgb=dev_base['task'], dev_row=dev_base['row_accuracy']))+'\n')
    best = dict(update=0, rgb=dev_base['task'], row=dev_base['row_accuracy'])
    generator = torch.Generator().manual_seed(SEED)
    order, cursor, update, seen, skipped = [], 0, 0, 0, 0
    reason = 'max_updates'
    started = time.perf_counter()
    while update < MAX_UPDATES:
        if cursor+BATCH > len(order):
            order = torch.randperm(len(fit), generator=generator).tolist()
            cursor = 0
        choice = [fit[order[index]] for index in range(cursor, cursor+BATCH)]
        cursor += BATCH
        batch_images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        batch_records = [train.records[index] for index in choice]
        prepared = prepare(system, relation, row_model, batch_images, batch_records)
        seen += len(batch_records)
        if prepared is None:
            skipped += len(batch_records)
            continue
        skipped += prepared['skipped']
        if not torch.isfinite(prepared['loss']):
            reason = 'diverged'
            break
        optimizer.zero_grad(set_to_none=True)
        prepared['loss'].backward()
        optimizer.step()
        update += 1
        if update not in EVALS:
            continue
        current = score_rgb(system, relation, row_model, head, train, dev)
        point = dict(update=update, dev_rgb=current['task'], dev_row=current['row_accuracy'],
                     dev_total=current['total'], seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(point), flush=True)
        if current['task'] < dev_base['task']*.75 or current['row_accuracy'] < dev_base['row_accuracy']-.05:
            reason = 'collapse'
            break
        if current['row_accuracy'] > best['row'] and current['task'] >= dev_base['task']:
            best = dict(update=update, rgb=current['task'], row=current['row_accuracy'])
            save(ROOT/'checkpoint-best.pt', head, row_model, optimizer, update,
                 dict(dev_rgb=current['task'], dev_row=current['row_accuracy']))
    history.close()
    save(ROOT/'checkpoint-last.pt', head, row_model, optimizer, update, dict(stop=reason))
    saved = torch.load(ROOT/'checkpoint-best.pt', map_location=stable.DEVICE, weights_only=False)
    head.load_state_dict(saved['head'])
    row_model.load_state_dict(saved['row'])
    enable(row_model, head)
    final = score_rgb(system, relation, row_model, head, val, val_indices)
    hold_final = score_rgb(system, relation, row_model, head, train, hold)
    gt_final = evaluate(row_model, relation, stored, multi_val, 'free')
    kept = baseline['successes'] & final['successes']
    lost = sorted(baseline['successes']-final['successes'])
    gained = sorted(final['successes']-baseline['successes'])
    lost_causes = tally(system, relation, row_model, head, val, lost)
    gained_causes = tally(system, relation, row_model, head, val, gained)
    adopted = (int(saved['update']) > 0 and final['row_accuracy'] > baseline['row_accuracy']
               and final['task'] > baseline['task'] and len(kept) >= 48
               and hold_final['task'] >= hold_base['task']*.90)
    report = dict(
        status='PASS' if adopted else 'FAIL', stop=reason, best_update=int(saved['update']), lr=LR,
        unfrozen=('row_assign',), row_order=order_audit,
        frozen=('rt_detr', 'physical_heads', 'language', 'goal', 'relation', 'size_head',
                'row_step', 'row_origin', 'row_order', 'row_angle'),
        excluded_scenes=skipped, attempted_scenes=seen,
        excluded_rate=None if not seen else skipped/seen,
        validation=dict(baseline_rgb=baseline['task'], rgb=final['task'], total=final['total'],
                        baseline_eight=baseline['by_count']['8'], eight=final['by_count']['8'],
                        baseline_row_predicted=baseline['row_accuracy'], row_predicted=final['row_accuracy'],
                        baseline_row_gt=gt_base['row_accuracy'], row_gt=gt_final['row_accuracy'],
                        baseline_gt_task=gt_base['task'], gt_task=gt_final['task'],
                        preserved=len(kept), lost=len(lost), gained=len(gained),
                        lost_causes=lost_causes, gained_causes=gained_causes),
        holdout=dict(baseline_rgb=hold_base['task'], rgb=hold_final['task'], total=hold_final['total'],
                     baseline_row_predicted=hold_base['row_accuracy'], row_predicted=hold_final['row_accuracy'],
                     baseline_eight=hold_base['by_count']['8'], eight=hold_final['by_count']['8']),
        checkpoints_unchanged=all((
            sha256(JOINT_BEST) == stamps['joint'], sha256(SIZE_BEST) == stamps['size'],
            sha256(ROW_BEST) == stamps['row'], sha256(RELATION_BEST) == stamps['relation'])),
        adopted=adopted, test_used=False, full_9600=False)
    stable.write(ROOT/'result.json', report)
    stable.write(ROOT/'delta.json', dict(lost=lost, gained=gained))
    print('DONE', json.dumps(dict(status=report['status'], update=int(saved['update']),
                                 rgb=final['task'], row=final['row_accuracy'],
                                 preserved=len(kept), hold=hold_final['task'],
                                 lost=lost_causes, gained=gained_causes)), flush=True)


if __name__ == '__main__':
    main()
