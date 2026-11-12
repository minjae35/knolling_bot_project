"""Fine-tune only the adopted size residual with size loss and free-rollout pose loss.

Arrangement, goal, language, and the detector stay frozen. This is not a joint
update of those modules.
"""
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.row_layout import placement_loss
from knolling_model.experiments.size_refinement import encode_instruction, live_poses
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.selection import select_queries
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from run_full_arrangement import cache_records, score, subset
from run_row_layout import attach_rows
from score_end_to_end import (
    SIZE_BEST, load_bundle, public_rgb, score_rgb, size_report, validation_indices)

ROOT = Path('artifacts/experiments/size_pose')
DATA = 'artifacts/generalization_20000/dataset'
ROW_BEST = 'artifacts/row_layout_origin_y/checkpoint-best.pt'
RELATION_BEST = 'artifacts/phase1_relation_continue/checkpoint-best.pt'
SEED = 1203
BATCH = 4
LR = 1e-4
MAX_UPDATES = 150
EVALS = (10, 25, 50, 100, 150)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family_indices(records):
    found = {}
    for index, record in enumerate(records):
        if is_full(record) and record['scene_family_id'] not in found:
            found[record['scene_family_id']] = index
    ordered = [found[name] for name in sorted(found)]
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(len(ordered), generator=generator).tolist()
    dev_count = len(ordered)//5
    dev = [ordered[index] for index in permutation[:dev_count]]
    fit = [ordered[index] for index in permutation[dev_count:]]
    return fit, dev


def matched_order(source, truth, threshold):
    chosen = select_queries(source, threshold)[0]
    count = len(truth['source_xy'])
    if len(chosen) != count or not 1 <= count <= 8:
        return None
    distance = torch.cdist(source['uv'][0, chosen]*source['uv'].new_tensor([.6, .4]), truth['source_xy'])
    used_query, used_gt, pairs = set(), set(), {}
    for item in distance.flatten().argsort().tolist():
        query, gt = divmod(item, distance.shape[1])
        if query in used_query or gt in used_gt:
            continue
        if float(distance[query, gt]) > .010:
            break
        used_query.add(query)
        used_gt.add(gt)
        pairs[int(chosen[query])] = gt
    if len(pairs) != count:
        return None
    return [(int(query), pairs[int(query)]) for query in chosen.tolist()]


def batch_losses(system, relation, row_model, head, images, records):
    truths = []
    for record in records:
        _, truth = recovery_supervision(record)
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
    chosen_visual = dict(source={key: value[index] for key, value in visual['source'].items()},
                         stages=[{key: value[index] for key, value in stage.items()} for stage in visual['stages']])
    output, objects, refined, _ = live_poses(
        system, relation, row_model, head, images[index], [texts[item] for item in kept],
        chosen_visual, text[index], mask[index])
    target_xy = objects.xy.new_zeros(objects.xy.shape)
    target_angle = objects.angle.new_zeros(objects.angle.shape)
    gt_size = []
    for local, (truth, order) in enumerate(zip([truths[item] for item in kept], orders)):
        for slot, (_, gt) in enumerate(order):
            target_xy[local, slot] = truth['target_xy'][gt]
            target_angle[local, slot] = truth['target_angle'][gt]
            gt_size.append(truth['size'][gt])
    if refined.shape[0] != len(gt_size):
        raise RuntimeError('refined sizes and matched targets diverged')
    size_loss = F.smooth_l1_loss(refined, torch.stack(gt_size), beta=.001)
    pose = placement_loss(output, target_xy, target_angle, objects.valid)
    return dict(size=size_loss, pose=pose['loss'], kept=len(kept), skipped=len(records)-len(kept))


def grad_norm(module):
    total = 0.
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm())
    return total


def copy_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def same_state(module, saved):
    current = module.state_dict()
    return all(torch.equal(current[key], value.to(current[key].device)) for key, value in saved.items())


def find_batch(system, relation, row_model, head, train, fit):
    for start in range(0, min(len(fit), 80), BATCH):
        choice = fit[start:start+BATCH]
        images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        records = [train.records[index] for index in choice]
        losses = batch_losses(system, relation, row_model, head, images, records)
        if losses is not None:
            return images, records, losses
    return None


def save(path, head, optimizer, update, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=head.state_dict(), optimizer=optimizer.state_dict(), update=update, **payload), path)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('size pose result already exists')
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    stamps = dict(size=sha256(SIZE_BEST), row=sha256(ROW_BEST), relation=sha256(RELATION_BEST))
    train = ExpandedDataset(DATA, 'train')
    val = ExpandedDataset(DATA, 'validation')
    fit, dev = family_indices(train.records)
    fit_families = {train.records[index]['scene_family_id'] for index in fit}
    dev_families = {train.records[index]['scene_family_id'] for index in dev}
    val_families = {record['scene_family_id'] for record in val.records}
    if fit_families & dev_families or fit_families & val_families or dev_families & val_families:
        raise RuntimeError('source families overlap a held-out split')
    system, relation, row_model, head, checkpoint = load_bundle(SIZE_BEST)
    if int(checkpoint.get('update', -1)) != 200:
        raise RuntimeError('size pose pilot must start from the adopted size update 200')
    val_indices = validation_indices(val)
    if len(val_indices) != 152:
        raise RuntimeError('validation comparison set changed')
    phase_val = subset(val, 'validation', 94)
    multi_val = [record for record in val.records if multi_row(record)]
    single_val = [record for record in val.records if is_full(record) and len(record['supervision_v3']['rows']) == 1]
    phase_cache = stable.cache_oracle(system, phase_val)
    multi_stored = attach_rows(multi_val, cache_records(system, multi_val))
    single_stored = cache_records(system, single_val)
    oracle = stable.official_metrics(relation, phase_cache, phase_val)['task']
    single = score(relation, single_stored, single_val)['task']
    modular = evaluate(row_model, relation, multi_stored, multi_val, 'free')
    baseline_rgb = score_rgb(system, relation, row_model, head, val, val_indices)
    baseline_size = size_report(system, head, val, val_indices)
    dev_rgb = score_rgb(system, relation, row_model, head, train, dev)
    dev_size = size_report(system, head, train, dev)
    print('BASE', json.dumps(dict(oracle=oracle, single=single, modular=modular['task'],
                                 rgb=baseline_rgb['task'], dev_rgb=dev_rgb['task'],
                                 width=baseline_size['width'])), flush=True)
    if (oracle, single, modular['task'], baseline_rgb['task']) != (60, 62, 66, 42):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='baseline', oracle=oracle, single=single,
                                              modular=modular['task'], rgb=baseline_rgb['task']))
        print('STOP adopted checkpoints did not reproduce', flush=True)
        return
    frozen = dict(perception=copy_state(system.perception), language=copy_state(system.arrangement.text_encoder),
                  goal=copy_state(system.layout.goal_encoder), relation=copy_state(relation),
                  row=copy_state(row_model), head=copy_state(head))
    found = find_batch(system, relation, row_model, head, train, fit)
    if found is None:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='no_matched_batch'))
        print('STOP no fully matched batch', flush=True)
        return
    images, records, losses = found
    with torch.no_grad():
        detected = {key: system.perception(images)['source'][key].clone()
                    for key in ('uv', 'angle', 'objectness_logits', 'size')}
    head.zero_grad(set_to_none=True)
    losses['size'].backward(retain_graph=True)
    size_norm = grad_norm(head)
    head.zero_grad(set_to_none=True)
    losses['pose'].backward()
    pose_norm = grad_norm(head)
    leaked = any(parameter.grad is not None for parameter in
                 list(system.perception.parameters())+list(relation.parameters())+list(row_model.parameters())
                 +list(system.layout.goal_encoder.parameters())+list(system.arrangement.text_encoder.parameters()))
    weight = 1. if pose_norm <= size_norm else size_norm/max(pose_norm, 1e-12)
    print('GRAD', json.dumps(dict(size=size_norm, pose=pose_norm, weight=weight, leaked=bool(leaked),
                                 kept=losses['kept'], skipped=losses['skipped'])), flush=True)
    if leaked or size_norm <= 0 or pose_norm <= 0:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='gradient', size=size_norm, pose=pose_norm,
                                              leaked=bool(leaked)))
        print('STOP pose loss did not stay on the size residual', flush=True)
        return
    head.load_state_dict({key: value.to(stable.DEVICE) for key, value in frozen['head'].items()})
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.)
    fresh = batch_losses(system, relation, row_model, head, images, records)
    optimizer.zero_grad(set_to_none=True)
    (fresh['size']+weight*fresh['pose']).backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
    optimizer.step()
    with torch.no_grad():
        again = system.perception(images)['source']
    detection_same = all(torch.equal(again[key], detected[key]) for key in detected)
    head_changed = not same_state(head, frozen['head'])
    weights_same = all(same_state(module, frozen[name]) for name, module in (
        ('perception', system.perception), ('language', system.arrangement.text_encoder),
        ('goal', system.layout.goal_encoder), ('relation', relation), ('row', row_model)))
    print('STEP', json.dumps(dict(passed=bool(head_changed and weights_same and detection_same),
                                 head_changed=bool(head_changed), weights_same=bool(weights_same),
                                 detection_same=bool(detection_same))), flush=True)
    head.load_state_dict({key: value.to(stable.DEVICE) for key, value in frozen['head'].items()})
    if not (head_changed and weights_same and detection_same) or not same_state(head, frozen['head']):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='step', head_changed=bool(head_changed),
                                              weights_same=bool(weights_same), detection_same=bool(detection_same)))
        print('STOP the optimizer step did not preserve frozen modules', flush=True)
        return
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.)
    save(ROOT/'checkpoint-0000.pt', head, optimizer, 0, dict(weight=weight))
    history = (ROOT/'history.jsonl').open('w')
    history.write(json.dumps(dict(update=0, dev_rgb=dev_rgb['task'], dev_width=dev_size['width']))+'\n')
    best = dict(update=0, rgb=dev_rgb['task'])
    save(ROOT/'checkpoint-best.pt', head, optimizer, 0, dict(dev_rgb=dev_rgb['task'], weight=weight))
    generator = torch.Generator().manual_seed(SEED)
    order, cursor, update, stall = [], 0, 0, 0
    seen = skipped = 0
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
        head.train()
        prepared = batch_losses(system, relation, row_model, head, batch_images, batch_records)
        seen += len(batch_records)
        if prepared is None:
            skipped += len(batch_records)
            continue
        skipped += prepared['skipped']
        optimizer.zero_grad(set_to_none=True)
        loss = prepared['size']+weight*prepared['pose']
        if not torch.isfinite(loss):
            reason = 'diverged'
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        update += 1
        if update not in EVALS:
            continue
        current_size = size_report(system, head, train, dev)
        current_rgb = score_rgb(system, relation, row_model, head, train, dev)
        point = dict(update=update, loss=float(loss.detach()), dev_rgb=current_rgb['task'],
                     dev_by={count: current_rgb['by_count'][count]['success'] for count in current_rgb['by_count']},
                     dev_width=current_size['width'], horizontal=current_rgb['horizontal'],
                     vertical=current_rgb['vertical'], row_accuracy=current_rgb['row_accuracy'],
                     slot_accuracy=current_rgb['slot_accuracy'], count_errors=current_rgb['count_errors'],
                     seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(point), flush=True)
        if (current_rgb['task'] < dev_rgb['task']*.75
                or current_size['width']['median'] > dev_size['width']['median']+.5):
            reason = 'collapse'
            break
        if (current_size['width']['median'] <= dev_size['width']['median']+.3
                and current_rgb['task'] > best['rgb']):
            best = dict(update=update, rgb=current_rgb['task'])
            stall = 0
            save(ROOT/'checkpoint-best.pt', head, optimizer, update,
                 dict(dev_rgb=current_rgb['task'], weight=weight))
        else:
            stall += 1
        if update >= 50 and stall >= 2:
            reason = 'plateau'
            break
    history.close()
    save(ROOT/'checkpoint-last.pt', head, optimizer, update, dict(stop=reason, weight=weight))
    saved = torch.load(ROOT/'checkpoint-best.pt', map_location=stable.DEVICE, weights_only=False)
    head.load_state_dict(saved['model'])
    final_rgb = score_rgb(system, relation, row_model, head, val, val_indices)
    final_size = size_report(system, head, val, val_indices)
    final_dev = score_rgb(system, relation, row_model, head, train, dev)
    final_oracle = stable.official_metrics(relation, phase_cache, phase_val)['task']
    final_single = score(relation, single_stored, single_val)['task']
    final_modular = evaluate(row_model, relation, multi_stored, multi_val, 'free')
    from knolling_model.experiments.size_refinement import ObjectSizeRefinement
    reloaded = ObjectSizeRefinement().to(stable.DEVICE)
    reloaded.load_state_dict(saved['model'])
    reload_rgb = score_rgb(system, relation, row_model, reloaded, val, val_indices)
    adopted = (int(saved['update']) > 0 and final_rgb['task'] > 42 and reload_rgb['task'] == final_rgb['task']
               and final_size['width']['median'] <= baseline_size['width']['median']+.3
               and final_size['width']['p95'] <= baseline_size['width']['p95']+.5
               and (final_oracle, final_single, final_modular['task']) == (60, 62, 66))
    report = dict(
        status='PASS' if adopted else 'FAIL', stop=reason, best_update=int(saved['update']),
        pose_weight=weight, lr=LR, size_grad=size_norm, pose_grad=pose_norm,
        excluded_scenes=skipped, attempted_scenes=seen,
        excluded_rate=None if not seen else skipped/seen,
        baseline=dict(oracle=oracle, single=single, modular=modular['task'], rgb=public_rgb(baseline_rgb),
                      width=baseline_size['width'], dev_rgb=dev_rgb['task'], dev_width=dev_size['width']),
        validation=dict(rgb=public_rgb(final_rgb), width=final_size['width'], height=final_size['height'],
                        footprint=final_size['footprint'],
                        preserved=len(baseline_rgb['successes'] & final_rgb['successes'])),
        dev=dict(rgb=final_dev['task'], total=final_dev['total'],
                 by_count={count: final_dev['by_count'][count]['success'] for count in final_dev['by_count']}),
        preserved=dict(oracle=final_oracle, single=final_single, modular=final_modular['task']),
        reload_matches=reload_rgb['task'] == final_rgb['task'],
        checkpoints_unchanged=(sha256(SIZE_BEST) == stamps['size'] and sha256(ROW_BEST) == stamps['row']
                               and sha256(RELATION_BEST) == stamps['relation']),
        test_used=False, gt_at_inference=False, full_joint=False)
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(status=report['status'], stop=reason, update=int(saved['update']),
                                 rgb=final_rgb['task'], width=final_size['width'])), flush=True)


if __name__ == '__main__':
    main()
