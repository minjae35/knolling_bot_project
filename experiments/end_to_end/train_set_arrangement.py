"""Replace pointwise row features with a set encoder that cannot move a fixed pose.

Pose is initialized from the adopted row checkpoint. The set encoder starts at
zero, so an unchanged row and order reproduce that checkpoint. Detector,
language, goal, relation, size, and angle stay frozen. The 152 scenes are not
used to choose a checkpoint, and the test split is not opened.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.row_layout import assignment_loss, order_loss
from knolling_model.experiments.set_context import SetArrangement
from knolling_model.experiments.size_refinement import encode_instruction, install_refined_size
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.data import recovery_supervision
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from eval_row_layout import evaluate
from joint_stage import matched_order
from run_full_arrangement import cache_records
from run_row_layout import attach_rows, collate_rows
from score_end_to_end import (
    JOINT_BEST, RELATION_BEST, ROW_BEST, SIZE_BEST, labels_of, load_bundle, score_rgb)
from train_set_context import fresh_split, overfit_records

ROOT = Path('artifacts/experiments/set_arrangement')
DATA = 'artifacts/generalization_20000/dataset'
BATCH = 8
MAX_UPDATES = 80
EVALS = (40, 80)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selector_parameters(model):
    placement = model.placement
    return (list(model.context.parameters())+list(placement.assign.parameters())
            +list(placement.order_query.parameters())+[placement.order_begin])


def pose_parameters(model):
    return list(model.placement.row_layout.parameters())+list(model.placement.step.parameters())


def enable(model):
    model.requires_grad_(False)
    for parameter in selector_parameters(model)+pose_parameters(model):
        parameter.requires_grad_(True)


def pose_loss(output, target_xy, valid):
    return F.smooth_l1_loss(output['xy'][valid], target_xy[valid], beta=.010)


def gt_step(relation, model, stored):
    packed = collate_rows(stored, stable.DEVICE)
    with torch.no_grad():
        hidden = relation._hidden(packed['features'])
    output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                   packed['row'], packed['slot'], True)
    valid = packed['valid']
    loss = (assignment_loss(output, packed['row'], valid)+order_loss(output)
            +pose_loss(output, packed['target_xy'], valid)/.010)
    return loss


def rgb_step(system, relation, model, head, images, records):
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
    output = model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                   teacher_row, teacher_slot, True)
    valid = objects.valid
    loss = assignment_loss(output, teacher_row, valid)+order_loss(output)+pose_loss(output, target, valid)/.010
    return dict(loss=loss, skipped=len(records)-len(kept))


def save(path, model, optimizer, update, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(placement=model.placement.state_dict(), context=model.context.state_dict(),
                    optimizer=optimizer.state_dict(), update=update, **payload), path)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('set arrangement result already exists')
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    stamps = dict(joint=sha256(JOINT_BEST), size=sha256(SIZE_BEST), row=sha256(ROW_BEST),
                  relation=sha256(RELATION_BEST))
    train = ExpandedDataset(DATA, 'train')
    fit, dev, _blocked = fresh_split(train.records)
    system, relation, placement, head, checkpoint = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
    if int(checkpoint.get('update', -1)) != 30:
        raise RuntimeError('arrangement replacement must start from joint update 30')
    for module in (system, relation, placement, head):
        module.eval()
        module.requires_grad_(False)
    model = SetArrangement(placement).to(stable.DEVICE)
    enable(model)
    scenes = overfit_records(train.records, fit)
    stored = attach_rows(scenes, cache_records(system, scenes))
    packed = collate_rows(stored, stable.DEVICE)
    with torch.no_grad():
        hidden = relation._hidden(packed['features'])
        bare = placement(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                         None, None, True)
        wrapped = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                        None, None, True)
    identical = all(torch.allclose(bare[key], wrapped[key]) for key in ('xy', 'origins', 'row_logits', 'angle'))
    print('IDENTITY', json.dumps(dict(identical=bool(identical))), flush=True)
    if not identical:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='identity', test_used=False))
        print('STOP the set encoder changed a pose before training', flush=True)
        return
    probe = copy.deepcopy(model)
    enable(probe)
    before = evaluate(probe, relation, stored, scenes, 'free')
    optimizer = torch.optim.AdamW((
        dict(params=list(probe.context.parameters()), lr=1e-3),
        dict(params=list(probe.placement.assign.parameters())+list(probe.placement.order_query.parameters())
             +[probe.placement.order_begin], lr=1e-4),
        dict(params=pose_parameters(probe), lr=1e-5)), weight_decay=0.)
    history = []
    passed = False
    for step in range(1, 201):
        start = ((step-1)*4) % len(stored)
        choice = [stored[(start+offset) % len(stored)] for offset in range(4)]
        loss = gt_step(relation, probe, choice)
        if not torch.isfinite(loss):
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            selector_open = float(probe.context.attn.out_proj.weight.grad.detach().abs().sum()) > 0
            pose_open = float(probe.placement.row_layout[-1].weight.grad.detach().abs().sum()) > 0
            separated = probe.placement.angle[-1].weight.grad is None
            print('OVERFIT_GRAD', json.dumps(dict(selector=selector_open, pose=pose_open, angle_frozen=separated)),
                  flush=True)
            if not selector_open or not pose_open or not separated:
                stable.write(ROOT/'result.json', dict(status='STOP', stage='overfit_gradient', test_used=False))
                print('STOP row and pose losses are not separated', flush=True)
                return
        optimizer.step()
        if step not in (50, 100, 200):
            continue
        current = evaluate(probe, relation, stored, scenes, 'free')
        point = dict(step=step, task=current['task'], row_exact=current['row_exact_scenes'],
                     slot_exact=current['slot_exact_scenes'], collision=current['collision'],
                     boundary=current['boundary'], failures=current['failures'])
        history.append(point)
        print('OVERFIT', json.dumps(point), flush=True)
        if current['row_exact_scenes'] == 8 and current['task'] >= 6 and current['collision'] <= 2:
            passed = True
            break
    overfit = dict(passed=passed, before_task=before['task'], before_row_exact=before['row_exact_scenes'],
                   before_collision=before['collision'], history=history)
    print('OVERFIT done', json.dumps(overfit), flush=True)
    if not passed:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='overfit', overfit=overfit, test_used=False,
                                              full_9600=False))
        print('STOP the eight scenes did not learn a consistent pose', flush=True)
        return
    if not torch.equal(placement.row_layout[-1].weight.detach().cpu(),
                       torch.load(JOINT_BEST, map_location='cpu', weights_only=False)['row']['row_layout.2.weight']):
        raise RuntimeError('the overfit copy changed the adopted pose')
    dev_base = score_rgb(system, relation, model, head, train, dev)
    print('BASE', json.dumps(dict(dev=dev_base['task'], dev_n=len(dev), row=dev_base['row_accuracy'])), flush=True)
    if dev_base['task'] != 74 or len(dev) != 512:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='baseline', dev=dev_base['task'],
                                              overfit=overfit, test_used=False))
        print('STOP the initial arrangement did not reproduce dev 74/512', flush=True)
        return
    optimizer = torch.optim.AdamW((
        dict(params=list(model.context.parameters()), lr=1e-3),
        dict(params=list(model.placement.assign.parameters())+list(model.placement.order_query.parameters())
             +[model.placement.order_begin], lr=1e-4),
        dict(params=pose_parameters(model), lr=1e-5)), weight_decay=0.)
    save(ROOT/'checkpoint-0000.pt', model, optimizer, 0, {})
    save(ROOT/'checkpoint-best.pt', model, optimizer, 0, dict(dev_rgb=74))
    log = (ROOT/'history.jsonl').open('w')
    log.write(json.dumps(dict(update=0, dev_rgb=74))+'\n')
    best = dict(update=0, task=74)
    generator = torch.Generator().manual_seed(1703)
    order, cursor, update, reason = [], 0, 0, 'max_updates'
    started = time.perf_counter()
    while update < MAX_UPDATES:
        if cursor+BATCH > len(order):
            order = torch.randperm(len(fit), generator=generator).tolist()
            cursor = 0
        choice = [fit[order[item]] for item in range(cursor, cursor+BATCH)]
        cursor += BATCH
        images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        records = [train.records[index] for index in choice]
        batch = rgb_step(system, relation, model, head, images, records)
        if batch is None or not torch.isfinite(batch['loss']):
            reason = 'diverged' if batch is not None else reason
            if batch is not None:
                break
            continue
        optimizer.zero_grad(set_to_none=True)
        batch['loss'].backward()
        if update == 0:
            leaked = model.placement.angle[-1].weight.grad is not None
            opened = (model.context.attn.out_proj.weight.grad is not None
                      and model.placement.row_layout[-1].weight.grad is not None)
            if leaked or not opened:
                reason = 'gradient'
                print('STOP pilot gradient', flush=True)
                break
        optimizer.step()
        update += 1
        if update not in EVALS:
            continue
        current = score_rgb(system, relation, model, head, train, dev)
        point = dict(update=update, dev_rgb=current['task'], dev_row=current['row_accuracy'],
                     dev_slot=current['slot_accuracy'], seconds=time.perf_counter()-started)
        log.write(json.dumps(point)+'\n')
        log.flush()
        print('EVAL', json.dumps(point), flush=True)
        if current['task'] > best['task']:
            best = dict(update=update, task=current['task'])
            save(ROOT/'checkpoint-best.pt', model, optimizer, update, dict(dev_rgb=current['task']))
        if current['task'] < 55:
            reason = 'worse'
            break
    log.close()
    save(ROOT/'checkpoint-last.pt', model, optimizer, update, dict(stop=reason))
    report = dict(status='FAIL', stop=reason, best_update=best['update'], overfit=overfit,
                  dev=dict(baseline_rgb=74, best_rgb=best['task'], total=512),
                  adopted=False, test_used=False, full_9600=False,
                  checkpoints_unchanged=all((
                      sha256(JOINT_BEST) == stamps['joint'], sha256(SIZE_BEST) == stamps['size'],
                      sha256(ROW_BEST) == stamps['row'], sha256(RELATION_BEST) == stamps['relation'])))
    if best['update'] > 0:
        saved = torch.load(ROOT/'checkpoint-best.pt', map_location=stable.DEVICE, weights_only=False)
        model.placement.load_state_dict(saved['placement'])
        model.context.load_state_dict(saved['context'])
        enable(model)
        from score_end_to_end import validation_indices
        val = ExpandedDataset(DATA, 'validation')
        indices = validation_indices(val)
        final = score_rgb(system, relation, model, head, val, indices)
        # The initial successes are the adopted 54. Reload is unnecessary: baseline score was not
        # stored, so compare against the known adopted successes by scoring the untouched checkpoint.
        bare_system, bare_relation, bare_row, bare_head, _ = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
        base = score_rgb(bare_system, bare_relation, bare_row, bare_head, val, indices)
        kept = base['successes'] & final['successes']
        report['validation'] = dict(baseline_rgb=base['task'], rgb=final['task'], total=final['total'],
                                    baseline_eight=base['by_count']['8'], eight=final['by_count']['8'],
                                    preserved=len(kept), lost=len(base['successes']-final['successes']),
                                    gained=len(final['successes']-base['successes']),
                                    baseline_row=base['row_accuracy'], row=final['row_accuracy'],
                                    baseline_slot=base['slot_accuracy'], slot=final['slot_accuracy'],
                                    failures=final['failure_counts'])
        report['adopted'] = (final['task'] > 54 and final['by_count']['8']['success'] >= 4 and len(kept) >= 48
                             and best['task'] > 74)
        report['status'] = 'PASS' if report['adopted'] else 'FAIL'
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(status=report['status'], dev=best['task'],
                                 validation=report.get('validation'))), flush=True)


if __name__ == '__main__':
    main()
