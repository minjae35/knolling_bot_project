"""Train object attention, row assignment, and ordering. Origin stays frozen.

The attention still feeds the frozen origin, so a row change can move the
placement. This run measures that effect and does not open the test split.
"""
import hashlib
import json
from pathlib import Path
import time

import torch

from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.row_layout import assignment_loss, order_loss
from knolling_model.experiments.set_context import ContextualRows, ObjectSetContext
from knolling_model.experiments.size_refinement import encode_instruction, install_refined_size, poses_from_detection
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.data import recovery_supervision
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from joint_stage import matched_order
from score_end_to_end import (
    JOINT_BEST, RELATION_BEST, ROW_BEST, SIZE_BEST, labels_of, load_bundle, public_rgb,
    score_rgb, validation_indices)
from train_set_context import fresh_split

ROOT = Path('artifacts/experiments/assign_order_pilot')
DATA = 'artifacts/generalization_20000/dataset'
BATCH = 8
CONTEXT_LR = 1e-3
HEAD_LR = 1e-4
MAX_UPDATES = 40
EVALS = (20, 40)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_of(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def same_state(module, saved):
    current = module.state_dict()
    return all(torch.equal(current[key].detach().cpu(), value) for key, value in saved.items())


def changed_keys(module, saved):
    current = module.state_dict()
    return sorted(key for key, value in saved.items() if not torch.equal(current[key].detach().cpu(), value))


def selector_parameters(placement):
    return list(placement.assign.parameters())+list(placement.order_query.parameters())+[placement.order_begin]


def enable(wrapper):
    wrapper.placement.requires_grad_(False)
    wrapper.context.requires_grad_(True)
    for parameter in selector_parameters(wrapper.placement):
        parameter.requires_grad_(True)


def prepare(system, relation, wrapper, head, images, records):
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
    chosen = []
    width = max(len(order) for order in orders)
    teacher_row = torch.full((len(orders), width), -1, device=images.device)
    teacher_slot = torch.full((len(orders), width), -1, device=images.device)
    for local, (record_index, order) in enumerate(zip(kept, orders)):
        chosen.append(torch.tensor([query for query, _ in order], device=images.device))
        row, slot = labels_of(records[record_index], identities[record_index])
        for slot_index, (_, gt) in enumerate(order):
            teacher_row[local, slot_index] = int(row[gt])
            teacher_slot[local, slot_index] = int(slot[gt])
    objects = pack_sources(chosen_source, chosen)
    with torch.no_grad():
        goal = system.layout.goal_encoder(objects, text[index], mask[index])['vector']
        hidden = relation._hidden(objects.features())
    output = wrapper(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                     teacher_row, teacher_slot, True)
    loss = assignment_loss(output, teacher_row, objects.valid)+order_loss(output)
    return dict(loss=loss, output=output, kept=len(kept), skipped=len(records)-len(kept))


def forward_pair(placement, wrapper, hidden, valid, goal, widths, heights):
    with torch.no_grad():
        bare = placement(hidden, valid, goal, widths, heights, None, None, True)
        mixed = wrapper(hidden, valid, goal, widths, heights, None, None, True)
    return bare, mixed


def median(values):
    if not values:
        return None
    return float(torch.tensor(values).median())


def origin_link(system, relation, baseline, trained, head, data, indices):
    """How a new row or order moves the frozen origin, in millimetres."""
    same_origin, changed_origin, xy_shift, base_error, trained_error = [], [], [], [], []
    row_same = row_changed = 0
    with torch.no_grad():
        for index in indices:
            record = data.records[index]
            identities, truth = recovery_supervision(record)
            image = data[index]['image'][None].to(stable.DEVICE)
            text = [record['instruction']['text']]
            try:
                before, objects, _ = poses_from_detection(system, relation, baseline, image, text, head)
                after, _, _ = poses_from_detection(system, relation, trained, image, text, head)
            except ValueError:
                continue
            valid = objects.valid[0]
            count = int(valid.sum())
            row_same += int(torch.equal(before['row'][0, :count], after['row'][0, :count]))
            row_changed += int(not torch.equal(before['row'][0, :count], after['row'][0, :count]))
            xy_shift.extend(torch.linalg.vector_norm(before['xy']-after['xy'], dim=-1)[0, :count].mul(1000).tolist())
            location = {object_id: slot for slot, object_id in enumerate(identities)}
            del location
            target = truth['target_xy'].to(before['xy'].device)
            # Greedy source match, same 10 mm gate as the official score.
            distance = torch.cdist(objects.xy[0, :count], truth['source_xy'].to(objects.xy.device))
            used_query, used_gt = set(), set()
            for item in distance.flatten().argsort().tolist():
                query, gt = divmod(item, distance.shape[1])
                if query in used_query or gt in used_gt or float(distance[query, gt]) > .010:
                    continue
                used_query.add(query)
                used_gt.add(gt)
                base_error.append(float(torch.linalg.vector_norm(before['xy'][0, query]-target[gt]))*1000)
                trained_error.append(float(torch.linalg.vector_norm(after['xy'][0, query]-target[gt]))*1000)
            for row_id in range(before['origins'].shape[1]):
                members = tuple(int(query) for query in range(count) if int(before['row'][0, query]) == row_id)
                other = tuple(int(query) for query in range(count) if int(after['row'][0, query]) == row_id)
                if not members and not other:
                    continue
                shift = float(torch.linalg.vector_norm(before['origins'][0, row_id]-after['origins'][0, row_id]))*1000
                (same_origin if members == other else changed_origin).append(shift)
    return dict(scenes_same_row=row_same, scenes_changed_row=row_changed,
                xy_shift_mm=median(xy_shift), xy_error_mm=dict(baseline=median(base_error), trained=median(trained_error)),
                origin_shift_mm=dict(same_members=median(same_origin), changed_members=median(changed_origin),
                                     same_n=len(same_origin), changed_n=len(changed_origin)))


def save(path, wrapper, optimizer, update, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(placement=wrapper.placement.state_dict(), context=wrapper.context.state_dict(),
                    optimizer=optimizer.state_dict(), update=update, **payload), path)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('assign-order pilot result already exists')
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
    if fit_families & dev_families or (fit_families | dev_families) & blocked or val_families & (fit_families | dev_families | blocked):
        raise RuntimeError('source families overlap a held-out split')
    system, relation, placement, head, checkpoint = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
    if int(checkpoint.get('update', -1)) != 30:
        raise RuntimeError('pilot must start from joint update 30')
    for module in (system, relation, placement, head):
        module.eval()
        module.requires_grad_(False)
    context = ObjectSetContext().to(stable.DEVICE)
    wrapper = ContextualRows(placement, context)
    enable(wrapper)
    frozen = dict(
        perception=state_of(system.perception), language=state_of(system.arrangement.text_encoder),
        goal=state_of(system.layout.goal_encoder), relation=state_of(relation), head=state_of(head),
        origin=state_of(placement.row_layout), step=state_of(placement.step), angle=state_of(placement.angle))
    found = None
    for start in range(0, min(len(fit), 160), BATCH):
        choice = fit[start:start+BATCH]
        images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        records = [train.records[index] for index in choice]
        prepared = prepare(system, relation, wrapper, head, images, records)
        if prepared is not None:
            found = (images, records, prepared)
            break
    if found is None:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='no_batch', test_used=False))
        print('STOP no matched batch', flush=True)
        return
    images, records, prepared = found
    reference = prepare(system, relation, placement, head, images, records)
    wrapped = prepare(system, relation, wrapper, head, images, records)
    identical = torch.allclose(reference['output']['xy'], wrapped['output']['xy']) and torch.allclose(
        reference['output']['row_logits'], wrapped['output']['row_logits']) and torch.allclose(
            reference['output']['origins'], wrapped['output']['origins'])
    print('IDENTITY', json.dumps(dict(identical=bool(identical))), flush=True)
    if not identical:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='identity', test_used=False))
        print('STOP zero attention changed the adopted inference', flush=True)
        return
    trainable = list(context.parameters())+selector_parameters(placement)
    optimizer = torch.optim.AdamW((
        dict(params=list(context.parameters()), lr=CONTEXT_LR),
        dict(params=selector_parameters(placement), lr=HEAD_LR)), weight_decay=0.)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    if optimized != {id(parameter) for parameter in trainable}:
        raise RuntimeError('optimizer does not match the attention, assignment, and order parameters')
    wrapper.zero_grad(set_to_none=True)
    wrapped['loss'].backward()
    graded = {name for name, parameter in wrapper.named_parameters()
              if parameter.grad is not None and float(parameter.grad.detach().abs().sum()) > 0}
    wanted = {'context.attn.out_proj.weight', 'placement.assign.0.weight', 'placement.order_query.0.weight',
              'placement.order_begin'}
    frozen_names = [name for name, parameter in wrapper.named_parameters()
                    if parameter.requires_grad and (
                        name.startswith('placement.row_layout') or name.startswith('placement.step')
                        or name.startswith('placement.angle'))]
    leaked = any(parameter.grad is not None for parameter in list(placement.row_layout.parameters())
                 +list(placement.step.parameters())+list(placement.angle.parameters())
                 +list(relation.parameters())+list(head.parameters())+list(system.perception.parameters())
                 +list(system.arrangement.text_encoder.parameters())+list(system.layout.goal_encoder.parameters()))
    print('GRAD', json.dumps(dict(opened=wanted <= graded, leaked=leaked, frozen_trainable=frozen_names,
                                 graded=sorted(graded))), flush=True)
    if not wanted <= graded or leaked or frozen_names:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='gradient', graded=sorted(graded),
                                              leaked=leaked, test_used=False))
        print('STOP gradients left the selector', flush=True)
        return
    optimizer.step()
    original_row = torch.load(JOINT_BEST, map_location='cpu', weights_only=False)['row']
    placement_changed = sorted(key for key, value in original_row.items()
                               if not torch.equal(placement.state_dict()[key].detach().cpu(), value))
    context_changed = any(not torch.equal(parameter.detach().cpu(), torch.zeros_like(parameter.detach().cpu()))
                          for name, parameter in context.named_parameters()
                          if name.endswith('out_proj.weight') or name.endswith('out_proj.bias'))
    frozen_same = all((
        same_state(system.perception, frozen['perception']),
        same_state(system.arrangement.text_encoder, frozen['language']),
        same_state(system.layout.goal_encoder, frozen['goal']),
        same_state(relation, frozen['relation']), same_state(head, frozen['head']),
        same_state(placement.row_layout, frozen['origin']), same_state(placement.step, frozen['step']),
        same_state(placement.angle, frozen['angle'])))
    print('STEP', json.dumps(dict(placement_changed=placement_changed, context_changed=bool(context_changed),
                                 frozen_same=bool(frozen_same))), flush=True)
    allowed = {'assign.0.weight', 'assign.0.bias', 'assign.2.weight', 'assign.2.bias',
               'order_query.0.weight', 'order_query.0.bias', 'order_query.2.weight', 'order_query.2.bias',
               'order_begin'}
    if not frozen_same or not context_changed or not set(placement_changed) <= allowed or not placement_changed:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='step', placement_changed=placement_changed,
                                              context_changed=bool(context_changed), frozen_same=bool(frozen_same),
                                              test_used=False))
        print('STOP the probe step changed a frozen parameter or buffer', flush=True)
        return
    placement.load_state_dict({key: value.to(stable.DEVICE) for key, value in original_row.items()})
    context.load_state_dict({key: value.to(stable.DEVICE) for key, value in ObjectSetContext().state_dict().items()})
    enable(wrapper)
    if not (same_state(placement.row_layout, frozen['origin']) and same_state(placement.assign, {
            key.split('assign.', 1)[1]: value for key, value in original_row.items() if key.startswith('assign.')})):
        raise RuntimeError('probe weights were not restored')
    val_indices = validation_indices(val)
    baseline = score_rgb(system, relation, wrapper, head, val, val_indices)
    dev_base = score_rgb(system, relation, wrapper, head, train, dev)
    print('BASE', json.dumps(dict(rgb=baseline['task'], eight=baseline['by_count']['8'], row=baseline['row_accuracy'],
                                 slot=baseline['slot_accuracy'], dev=dev_base['task'], dev_n=len(dev))), flush=True)
    if baseline['task'] != 54 or baseline['by_count']['8']['success'] != 4 or dev_base['task'] != 74 or len(dev) != 512:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='baseline', rgb=public_rgb(baseline),
                                              dev=dev_base['task'], test_used=False))
        print('STOP the zero attention did not reproduce the adopted scores', flush=True)
        return
    optimizer = torch.optim.AdamW((
        dict(params=list(context.parameters()), lr=CONTEXT_LR),
        dict(params=selector_parameters(placement), lr=HEAD_LR)), weight_decay=0.)
    save(ROOT/'checkpoint-0000.pt', wrapper, optimizer, 0, {})
    save(ROOT/'checkpoint-best.pt', wrapper, optimizer, 0, dict(dev_rgb=dev_base['task']))
    history = (ROOT/'history.jsonl').open('w')
    history.write(json.dumps(dict(update=0, dev_rgb=74, dev_row=dev_base['row_accuracy'],
                                 dev_slot=dev_base['slot_accuracy']))+'\n')
    best = dict(update=0, task=dev_base['task'])
    generator = torch.Generator().manual_seed(1703)
    order, cursor, update, seen, skipped = [], 0, 0, 0, 0
    reason = 'max_updates'
    started = time.perf_counter()
    while update < MAX_UPDATES:
        if cursor+BATCH > len(order):
            order = torch.randperm(len(fit), generator=generator).tolist()
            cursor = 0
        choice = [fit[order[item]] for item in range(cursor, cursor+BATCH)]
        cursor += BATCH
        batch_images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        batch_records = [train.records[index] for index in choice]
        batch = prepare(system, relation, wrapper, head, batch_images, batch_records)
        seen += len(batch_records)
        if batch is None:
            skipped += len(batch_records)
            continue
        skipped += batch['skipped']
        if not torch.isfinite(batch['loss']):
            reason = 'diverged'
            break
        optimizer.zero_grad(set_to_none=True)
        batch['loss'].backward()
        optimizer.step()
        update += 1
        if update not in EVALS:
            continue
        current = score_rgb(system, relation, wrapper, head, train, dev)
        point = dict(update=update, dev_rgb=current['task'], dev_row=current['row_accuracy'],
                     dev_slot=current['slot_accuracy'], seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(point), flush=True)
        if current['task'] > best['task']:
            best = dict(update=update, task=current['task'])
            save(ROOT/'checkpoint-best.pt', wrapper, optimizer, update, dict(dev_rgb=current['task']))
        if current['task'] < dev_base['task']:
            reason = 'worse'
            break
    history.close()
    save(ROOT/'checkpoint-last.pt', wrapper, optimizer, update, dict(stop=reason))
    if best['update'] > 0:
        saved = torch.load(ROOT/'checkpoint-best.pt', map_location=stable.DEVICE, weights_only=False)
        wrapper.placement.load_state_dict(saved['placement'])
        wrapper.context.load_state_dict(saved['context'])
        enable(wrapper)
    final = score_rgb(system, relation, wrapper, head, val, val_indices)
    link = origin_link(system, relation, ContextualRows(
        load_bundle(JOINT_BEST, row_path=JOINT_BEST)[2], ObjectSetContext().to(stable.DEVICE)).eval(),
        wrapper, head, val, val_indices)
    kept = baseline['successes'] & final['successes']
    adopted = (best['update'] > 0 and final['task'] > 54 and final['by_count']['8']['success'] >= 4
               and len(kept) >= 48 and best['task'] >= 74
               and same_state(placement.row_layout, frozen['origin']) and same_state(placement.step, frozen['step'])
               and same_state(placement.angle, frozen['angle']))
    result = dict(
        status='PASS' if adopted else 'FAIL', stop=reason, best_update=best['update'], evaluated_update=update,
        unfrozen=('object_context', 'row_assign', 'row_order'),
        frozen=('rt_detr', 'language', 'goal', 'relation', 'size_head', 'row_origin', 'row_step', 'row_angle'),
        validation=dict(baseline_rgb=54, rgb=final['task'], total=final['total'],
                        baseline_eight=dict(success=4, total=42), eight=final['by_count']['8'],
                        preserved=len(kept), lost=len(baseline['successes']-final['successes']),
                        gained=len(final['successes']-baseline['successes']),
                        baseline_row=baseline['row_accuracy'], row=final['row_accuracy'],
                        baseline_slot=baseline['slot_accuracy'], slot=final['slot_accuracy'],
                        baseline_failures=baseline['failure_counts'], failures=final['failure_counts']),
        dev=dict(baseline_rgb=74, best_rgb=best['task'], total=512),
        origin_link=link, adopted=adopted,
        checkpoints_unchanged=all((
            sha256(JOINT_BEST) == stamps['joint'], sha256(SIZE_BEST) == stamps['size'],
            sha256(ROW_BEST) == stamps['row'], sha256(RELATION_BEST) == stamps['relation'])),
        test_used=False)
    stable.write(ROOT/'result.json', result)
    print('DONE', json.dumps(dict(status=result['status'], update=best['update'], rgb=final['task'],
                                 row=final['row_accuracy'], slot=final['slot_accuracy'],
                                 preserved=len(kept), link=link)), flush=True)


if __name__ == '__main__':
    main()
