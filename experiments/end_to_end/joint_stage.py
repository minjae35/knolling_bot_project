"""One joint stage from the adopted checkpoints.

Detection, size, and pose are not allowed to update the same weights.
Perception, language, goal, the relation head, and row assignment stay frozen.
The size head receives only the size loss. Pose and the anchor to the frozen
arrangement update only the row step and the origin's last layer.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from knolling_model.coordinates import angle_delta
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.row_layout import placement_loss
from knolling_model.experiments.size_refinement import (
    encode_instruction, matched_queries, scatter_refined)
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.selection import select_queries
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from run_full_arrangement import cache_records, score, subset
from run_row_layout import attach_rows
from score_end_to_end import (
    RELATION_BEST, ROW_BEST, SIZE_BEST, load_bundle, public_rgb, score_rgb, size_report,
    validation_indices)

ROOT = Path('artifacts/experiments/joint_stage')
DATA = 'artifacts/generalization_20000/dataset'
SEED = 1203
BATCH = 4
LR = 1e-5
MAX_UPDATES = 30
EVALS = (10, 30)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family_indices(records):
    found = {}
    for index, record in enumerate(records):
        if is_full(record) and record['scene_family_id'] not in found:
            found[record['scene_family_id']] = index
    ordered = [found[name] for name in sorted(found)]
    permutation = torch.randperm(len(ordered), generator=torch.Generator().manual_seed(SEED)).tolist()
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
        if query in used_query or gt in used_gt or float(distance[query, gt]) > .010:
            if float(distance[query, gt]) > .010:
                break
            continue
        used_query.add(query)
        used_gt.add(gt)
        pairs[int(chosen[query])] = gt
    if len(pairs) != count:
        return None
    return [(int(query), pairs[int(query)]) for query in chosen.tolist()]


def trainable(row_model):
    return list(row_model.step.parameters())+[row_model.row_layout[-1].weight, row_model.row_layout[-1].bias]


def enable(head, row_model):
    head.requires_grad_(True)
    row_model.requires_grad_(False)
    for parameter in trainable(row_model):
        parameter.requires_grad_(True)


def grad_norm(parameters):
    total = 0.
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm())
    return total


def zero(*modules):
    for module in modules:
        module.zero_grad(set_to_none=True)


def copy_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def same_state(module, saved):
    current = module.state_dict()
    return all(torch.equal(current[key], value.to(current[key].device)) for key, value in saved.items())


def forward_split(system, relation, row_model, frozen_row, head, images, records):
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
    chosen_visual = dict(
        source={key: value[index] for key, value in visual['source'].items()},
        stages=[{key: value[index] for key, value in stage.items()} for stage in visual['stages']])
    source = chosen_visual['source']
    chosen = select_queries(source, system.threshold)
    pairs = [(image, int(query)) for image, queries in enumerate(chosen) for query in queries.tolist()]
    rgb, tokens, boxes, base = matched_queries(
        chosen_visual['stages'][-1]['tokens'], images[index], chosen_visual['stages'][-1]['boxes'],
        source['size'], pairs)
    refined = head(tokens, rgb, boxes, base)[:, 0]
    size = scatter_refined(source['size'], pairs, refined)
    width = max(len(order) for order in orders)
    target_xy = refined.new_zeros(len(orders), width, 2)
    target_angle = refined.new_zeros(len(orders), width)
    gt_size = []
    for local, (truth, order) in enumerate(zip([truths[item] for item in kept], orders)):
        for slot, (_, gt) in enumerate(order):
            target_xy[local, slot] = truth['target_xy'][gt]
            target_angle[local, slot] = truth['target_angle'][gt]
            gt_size.append(truth['size'][gt])
    if refined.shape[0] != len(gt_size):
        raise RuntimeError('refined sizes and matched targets diverged')
    size_loss = F.smooth_l1_loss(refined, torch.stack(gt_size), beta=.001)
    replaced = dict(source)
    replaced['size'] = size.detach()
    objects = pack_sources(replaced, chosen)
    with torch.no_grad():
        goal = system.layout.goal_encoder(objects, text[index], mask[index])['vector']
        hidden = relation._hidden(objects.features())
    valid = objects.valid
    widths, heights = objects.size[..., 0].detach(), objects.size[..., 1].detach()
    output = row_model(hidden, valid, goal, widths, heights, None, None, True)
    with torch.no_grad():
        reference = frozen_row(hidden, valid, goal, widths, heights, None, None, True)
    position = torch.linalg.vector_norm(reference['xy']-target_xy, dim=-1)
    angular = angle_delta(reference['angle'], target_angle).abs()
    good = valid & (position < .002) & (angular < math.radians(3))
    fail = valid & ~good
    pose = placement_loss(output, target_xy, target_angle, fail)['loss'] if bool(fail.any()) else output['xy'].sum()*0
    anchor = (F.smooth_l1_loss(output['xy'][good], reference['xy'][good], beta=.001)
              if bool(good.any()) else output['xy'].sum()*0)
    return dict(size=size_loss, pose=pose, anchor=anchor, kept=len(kept), skipped=len(records)-len(kept),
                good=int(good.sum()), fail=int(fail.sum()), output=output, reference=reference)


def find_batch(system, relation, row_model, frozen_row, head, train, fit):
    for start in range(0, min(len(fit), 80), BATCH):
        choice = fit[start:start+BATCH]
        images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        records = [train.records[index] for index in choice]
        losses = forward_split(system, relation, row_model, frozen_row, head, images, records)
        if losses is not None and losses['fail'] > 0 and losses['good'] > 0:
            return images, records, losses
    return None


def save(path, head, row_model, optimizer, update, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(head=head.state_dict(), row=row_model.state_dict(),
                    optimizer=optimizer.state_dict(), update=update, **payload), path)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('joint stage result already exists')
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    stamps = dict(size=sha256(SIZE_BEST), row=sha256(ROW_BEST), relation=sha256(RELATION_BEST))
    train = ExpandedDataset(DATA, 'train')
    val = ExpandedDataset(DATA, 'validation')
    fit, dev = family_indices(train.records)
    groups = (
        {train.records[index]['scene_family_id'] for index in fit},
        {train.records[index]['scene_family_id'] for index in dev},
        {record['scene_family_id'] for record in val.records})
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RuntimeError('source families overlap a held-out split')
    system, relation, row_model, head, checkpoint = load_bundle(SIZE_BEST)
    if int(checkpoint.get('update', -1)) != 200:
        raise RuntimeError('joint stage must start from the adopted size update 200')
    for module in (system, relation, row_model, head):
        module.eval()
        module.requires_grad_(False)
    enable(head, row_model)
    frozen_row = copy.deepcopy(row_model).eval()
    frozen_row.requires_grad_(False)
    val_indices = validation_indices(val)
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
                                 rgb=baseline_rgb['task'], dev_rgb=dev_rgb['task'])), flush=True)
    if (oracle, single, modular['task'], baseline_rgb['task']) != (60, 62, 66, 42):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='baseline', oracle=oracle, single=single,
                                              modular=modular['task'], rgb=baseline_rgb['task']))
        print('STOP adopted checkpoints did not reproduce', flush=True)
        return
    frozen = dict(perception=copy_state(system.perception), language=copy_state(system.arrangement.text_encoder),
                  goal=copy_state(system.layout.goal_encoder), relation=copy_state(relation),
                  row=copy_state(row_model), head=copy_state(head))
    found = find_batch(system, relation, row_model, frozen_row, head, train, fit)
    if found is None:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='no_probe_batch'))
        print('STOP no batch contained both a preserved pose and a pose failure', flush=True)
        return
    images, records, losses = found
    zero(head, row_model, relation, system.perception)
    losses['size'].backward(retain_graph=True)
    size_head, size_row = grad_norm(head.parameters()), grad_norm(trainable(row_model))
    zero(head, row_model, relation)
    losses['pose'].backward(retain_graph=True)
    pose_head = grad_norm(head.parameters())
    pose_step = grad_norm(row_model.step.parameters())
    pose_origin = grad_norm([row_model.row_layout[-1].weight, row_model.row_layout[-1].bias])
    pose_relation = grad_norm(relation.parameters())
    pose_assign = grad_norm(row_model.assign.parameters())
    zero(head, row_model)
    losses['anchor'].backward(retain_graph=True)
    anchor_step = grad_norm(row_model.step.parameters())+grad_norm(
        [row_model.row_layout[-1].weight, row_model.row_layout[-1].bias])
    zero(head, row_model, system.perception)
    detected = system.perception(images)
    detection, _ = system.perception.detection_loss(detected, [
        {key: value.to(images.device) for key, value in recovery_supervision(record)[1].items()}
        for record in records])
    detection_total = sum(value for value in detection.values() if torch.is_tensor(value) and value.numel() == 1)
    detection_connected = bool(detection_total.requires_grad)
    if detection_connected:
        detection_total.backward()
    detection_leak = grad_norm(list(head.parameters())+trainable(row_model)+list(relation.parameters()))
    print('GRAD', json.dumps(dict(
        size_head=size_head, size_row=size_row, pose_head=pose_head, pose_step=pose_step,
        pose_origin=pose_origin, pose_relation=pose_relation, pose_assign=pose_assign,
        anchor_row=anchor_step, detection_connected=detection_connected, detection_leak=detection_leak,
        good=losses['good'], fail=losses['fail'])), flush=True)
    leaked = (size_row > 0 or pose_head > 0 or pose_relation > 0 or pose_assign > 0 or detection_leak > 0
              or size_head <= 0 or (pose_step <= 0 and pose_origin <= 0))
    if leaked:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='gradient', size_head=size_head,
                                              size_row=size_row, pose_head=pose_head, pose_step=pose_step,
                                              pose_origin=pose_origin, pose_relation=pose_relation,
                                              detection_leak=detection_leak))
        print('STOP loss gradients crossed modules', flush=True)
        return
    head.load_state_dict({key: value.to(stable.DEVICE) for key, value in frozen['head'].items()})
    row_model.load_state_dict({key: value.to(stable.DEVICE) for key, value in frozen['row'].items()})
    enable(head, row_model)
    optimizer = torch.optim.AdamW([parameter for parameter in list(head.parameters())+trainable(row_model)
                                   if parameter.requires_grad], lr=LR, weight_decay=0.)
    fresh = forward_split(system, relation, row_model, frozen_row, head, images, records)
    before_xy = fresh['reference']['xy'].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    (fresh['size']+fresh['pose']+fresh['anchor']).backward()
    optimizer.step()
    after = forward_split(system, relation, row_model, frozen_row, head, images, records)
    moved = float((after['output']['xy']-before_xy).norm(dim=-1).max())*1000
    frozen_same = all(same_state(module, frozen[name]) for name, module in (
        ('perception', system.perception), ('language', system.arrangement.text_encoder),
        ('goal', system.layout.goal_encoder), ('relation', relation)))
    head_changed = not same_state(head, frozen['head'])
    print('STEP', json.dumps(dict(moved_mm=moved, frozen_same=bool(frozen_same),
                                 head_changed=bool(head_changed))), flush=True)
    head.load_state_dict({key: value.to(stable.DEVICE) for key, value in frozen['head'].items()})
    row_model.load_state_dict({key: value.to(stable.DEVICE) for key, value in frozen['row'].items()})
    enable(head, row_model)
    if not frozen_same or not head_changed or moved > 2:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='step', moved_mm=moved,
                                              frozen_same=bool(frozen_same), head_changed=bool(head_changed)))
        print('STOP the probe step did not stay on the unfrozen slice', flush=True)
        return
    optimizer = torch.optim.AdamW([parameter for parameter in list(head.parameters())+trainable(row_model)
                                   if parameter.requires_grad], lr=LR, weight_decay=0.)
    save(ROOT/'checkpoint-0000.pt', head, row_model, optimizer, 0, {})
    save(ROOT/'checkpoint-best.pt', head, row_model, optimizer, 0, dict(dev_rgb=dev_rgb['task']))
    history = (ROOT/'history.jsonl').open('w')
    history.write(json.dumps(dict(update=0, dev_rgb=dev_rgb['task'], dev_width=dev_size['width']))+'\n')
    best = dict(update=0, rgb=dev_rgb['task'])
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
        prepared = forward_split(system, relation, row_model, frozen_row, head, batch_images, batch_records)
        seen += len(batch_records)
        if prepared is None:
            skipped += len(batch_records)
            continue
        skipped += prepared['skipped']
        loss = prepared['size']+prepared['pose']+prepared['anchor']
        if not torch.isfinite(loss):
            reason = 'diverged'
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        update += 1
        if update not in EVALS:
            continue
        current_size = size_report(system, head, train, dev)
        current_rgb = score_rgb(system, relation, row_model, head, train, dev)
        current_oracle = stable.official_metrics(relation, phase_cache, phase_val)['task']
        current_modular = evaluate(row_model, relation, multi_stored, multi_val, 'free')['task']
        point = dict(update=update, dev_rgb=current_rgb['task'], dev_width=current_size['width'],
                     oracle=current_oracle, modular=current_modular, seconds=time.perf_counter()-started)
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(point), flush=True)
        if (current_rgb['task'] < dev_rgb['task']*.75
                or current_size['width']['median'] > dev_size['width']['median']+.5
                or current_oracle < 60 or current_modular < 60):
            reason = 'collapse'
            break
        if (current_size['width']['median'] <= dev_size['width']['median']+.3
                and current_rgb['task'] > best['rgb'] and current_modular >= 64):
            best = dict(update=update, rgb=current_rgb['task'])
            save(ROOT/'checkpoint-best.pt', head, row_model, optimizer, update, dict(dev_rgb=current_rgb['task']))
    history.close()
    save(ROOT/'checkpoint-last.pt', head, row_model, optimizer, update, dict(stop=reason))
    saved = torch.load(ROOT/'checkpoint-best.pt', map_location=stable.DEVICE, weights_only=False)
    head.load_state_dict(saved['head'])
    row_model.load_state_dict(saved['row'])
    enable(head, row_model)
    final_rgb = score_rgb(system, relation, row_model, head, val, val_indices)
    final_size = size_report(system, head, val, val_indices)
    final_oracle = stable.official_metrics(relation, phase_cache, phase_val)['task']
    final_single = score(relation, single_stored, single_val)['task']
    final_modular = evaluate(row_model, relation, multi_stored, multi_val, 'free')
    adopted = (int(saved['update']) > 0 and final_rgb['task'] > 42
               and final_size['width']['median'] <= baseline_size['width']['median']+.3
               and final_size['width']['p95'] <= baseline_size['width']['p95']+.5
               and final_oracle == 60 and final_single == 62 and final_modular['task'] >= 64)
    report = dict(
        status='PASS' if adopted else 'FAIL', stop=reason, best_update=int(saved['update']), lr=LR,
        unfrozen=('size_refinement', 'row_step', 'row_origin_last_layer'),
        frozen=('rt_detr', 'physical_heads', 'language', 'goal', 'relation', 'row_assign', 'row_order', 'row_angle'),
        excluded_scenes=skipped, attempted_scenes=seen,
        excluded_rate=None if not seen else skipped/seen,
        baseline=dict(oracle=oracle, single=single, modular=modular['task'], rgb=public_rgb(baseline_rgb),
                      width=baseline_size['width'], dev_rgb=dev_rgb['task'], dev_width=dev_size['width']),
        validation=dict(rgb=public_rgb(final_rgb), width=final_size['width'],
                        preserved=len(baseline_rgb['successes'] & final_rgb['successes'])),
        preserved=dict(oracle=final_oracle, single=final_single, modular=final_modular['task']),
        checkpoints_unchanged=(sha256(SIZE_BEST) == stamps['size'] and sha256(ROW_BEST) == stamps['row']
                               and sha256(RELATION_BEST) == stamps['relation']),
        test_used=False, full_9600=False)
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(status=report['status'], stop=reason, update=int(saved['update']),
                                 rgb=final_rgb['task'], modular=final_modular['task'],
                                 oracle=final_oracle, width=final_size['width'])), flush=True)


if __name__ == '__main__':
    main()
