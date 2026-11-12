"""Train origin and Δx on the selector's own rows, never on a mismatched teacher row.

A predicted grouping gets a target only when the existing teacher can place that
grouping legally. Successes keep the pose they already produce. Illegal
groupings are skipped. The selector, angle, detector, and language stay frozen.
"""
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from knolling_dataset.schema import Pose, REGIONS, instruction_from_dict, scene_from_dict, with_targets
from knolling_dataset.teacher_balanced import templates
from knolling_dataset.teacher_legacy import _translations
from knolling_dataset.validator import geometry, validate_layout
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.set_context import SetArrangement
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from eval_row_layout import evaluate
from run_full_arrangement import cache_records
from run_row_layout import attach_rows, collate_rows, hidden_of
from score_end_to_end import JOINT_BEST, RELATION_BEST, ROW_BEST, SIZE_BEST, load_bundle, score_rgb
from train_selector_only import flips, pose_tensors, rollout, sha256
from train_set_context import fresh_split

ROOT = Path('artifacts/experiments/predicted_pose')
SELECTOR = Path('artifacts/experiments/selector_only/checkpoint-best.pt')
DATA = 'artifacts/generalization_20000/dataset'
BATCH = 8
MAX_UPDATES = 80
EVALS = (40, 80)


def rows_of(ids, row, slot):
    groups = {}
    for object_id, row_id, slot_id in zip(ids, row, slot):
        groups.setdefault(int(row_id), []).append((int(slot_id), object_id))
    if not groups or sorted(groups) != list(range(len(groups))):
        return None
    return [[object_id for _, object_id in sorted(items)] for _, items in sorted(groups.items())]


def teacher_pose(record, predicted):
    scene = scene_from_dict(record['source'])
    instruction = instruction_from_dict(record['instruction'])
    matched = None
    for template in templates(scene, instruction):
        rows = [[obj.object_id for obj in row] for row in template['rows']]
        if rows == predicted:
            matched = template
            break
    if matched is None:
        return None
    region = REGIONS[instruction.region]
    translations = _translations(matched['width'], matched['height'], region, bool(instruction.locked_ids))
    for x_m, y_m in translations:
        poses = {object_id: Pose(x_m+pose.x_m, y_m+pose.y_m, pose.theta_rad)
                 for object_id, pose in matched['relative'].items()}
        target = with_targets(scene, poses, instruction.locked_ids)
        if not validate_layout(scene, target, instruction, 'full').valid:
            continue
        if semantic_errors_v3(target.objects, record) or not geometry(target.objects, 'target_pose', .010, .010).valid:
            continue
        return poses
    return None


def labeled(model, relation, stored, records):
    scenes = rollout(model, relation, stored, records)
    examples, counts, gaps = [], {'success': 0, 'exact': 0, 'other': 0, 'illegal': 0}, []
    for scene, item, record in zip(scenes, stored, records):
        ids, _truth = recovery_supervision(record)
        predicted = rows_of(ids, scene['row'], scene['slot'])
        teacher = record['supervision_v3']['rows']
        example = dict(item)
        example['train'] = False
        example['row'] = torch.tensor(scene['row'])
        example['slot'] = torch.tensor(scene['slot'])
        if scene['task']:
            counts['success'] += 1
            example['target_xy'] = scene['xy'].clone()
            example['train'] = True
        elif predicted is not None and predicted == teacher:
            counts['exact'] += 1
            example['train'] = True
        else:
            poses = None if predicted is None else teacher_pose(record, predicted)
            if poses is None:
                counts['illegal'] += 1
            else:
                counts['other'] += 1
                example['target_xy'] = torch.tensor([[poses[object_id].x_m, poses[object_id].y_m] for object_id in ids])
                example['train'] = True
                gaps.append(float(torch.linalg.vector_norm(example['target_xy']-item['target_xy'], dim=-1).median()))
        examples.append(example)
    return examples, counts, gaps


def pose_loss(relation, model, stored):
    chosen = [item for item in stored if item['train']]
    if not chosen:
        return None
    packed = collate_rows(chosen, stable.DEVICE)
    hidden = hidden_of(relation, packed['features'])
    output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                   packed['row'], packed['slot'], False)
    valid = packed['valid']
    return F.smooth_l1_loss(output['xy'][valid], packed['target_xy'][valid], beta=.010)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('predicted pose result already exists')
    if not SELECTOR.is_file():
        raise RuntimeError('selector checkpoint is missing')
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    stamps = dict(joint=sha256(JOINT_BEST), size=sha256(SIZE_BEST), row=sha256(ROW_BEST),
                  relation=sha256(RELATION_BEST), selector=sha256(SELECTOR))
    train = ExpandedDataset(DATA, 'train')
    fit, dev, _blocked = fresh_split(train.records)
    system, relation, placement, head, checkpoint = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
    if int(checkpoint.get('update', -1)) != 30:
        raise RuntimeError('pose training must start from the adopted row checkpoint')
    saved = torch.load(SELECTOR, map_location=stable.DEVICE, weights_only=False)
    if int(saved['update']) != 160:
        raise RuntimeError('selector checkpoint is not the dev-selected update')
    placement.load_state_dict(saved['placement'])
    model = SetArrangement(placement).to(stable.DEVICE)
    model.context.load_state_dict(saved['context'])
    model.requires_grad_(False)
    def snapshot(module, prefix):
        return {f'{prefix}.{name}': tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}

    selector = snapshot(model.context, 'context')
    selector.update(snapshot(placement.assign, 'assign'))
    selector.update(snapshot(placement.order_query, 'order'))
    selector['order_begin'] = placement.order_begin.detach().cpu().clone()
    pose = pose_tensors(model)
    fit_records = [train.records[index] for index in fit]
    dev_records = [train.records[index] for index in dev]
    recoverable = sum(teacher_pose(record, record['supervision_v3']['rows']) is None for record in fit_records[:32])
    if recoverable:
        raise RuntimeError('the teacher cannot place its own dataset rows')
    fit_stored = attach_rows(fit_records, cache_records(system, fit_records))
    dev_stored = attach_rows(dev_records, cache_records(system, dev_records))
    before = evaluate(model, relation, dev_stored, dev_records, 'free')
    if before['task'] != 110:
        raise RuntimeError(f'selector dev task changed: {before["task"]}')
    baseline_scenes = rollout(model, relation, dev_stored, dev_records)
    fit_examples, fit_counts, fit_gaps = labeled(model, relation, fit_stored, fit_records)
    dev_examples, dev_counts, _dev_gaps = labeled(model, relation, dev_stored, dev_records)
    covered = fit_counts['exact']+fit_counts['other']
    print('LABELS', json.dumps(dict(fit=fit_counts, dev=dev_counts, covered_failures=covered,
                                   other_shift_mm=None if not fit_gaps else float(torch.tensor(fit_gaps).median())*1000)),
          flush=True)
    if covered == 0:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='labels', fit=fit_counts, dev=dev_counts,
                                              stage2=False, test_used=False, adopted=False))
        print('STOP no predicted row has a legal target pose', flush=True)
        return
    for parameter in list(placement.row_layout.parameters())+list(placement.step.parameters()):
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(list(placement.row_layout.parameters())+list(placement.step.parameters()),
                                  lr=1e-5, weight_decay=0.)
    trainable = [item for item in fit_examples if item['train']]
    log = (ROOT/'history.jsonl').open('w')
    log.write(json.dumps(dict(update=0, task=before['task'], row=before['row_accuracy']))+'\n')
    best = dict(update=0, task=before['task'])
    torch.save(dict(placement=placement.state_dict(), context=model.context.state_dict(), update=0,
                    dev_task=before['task']), ROOT/'checkpoint-best.pt')
    generator = torch.Generator().manual_seed(1703)
    order, cursor, update, reason = [], 0, 0, 'max_updates'
    started = time.perf_counter()
    while update < MAX_UPDATES:
        if cursor+BATCH > len(order):
            order = torch.randperm(len(trainable), generator=generator).tolist()
            cursor = 0
        choice = [trainable[index] for index in order[cursor:cursor+BATCH]]
        cursor += BATCH
        loss = pose_loss(relation, model, choice)
        if loss is None or not torch.isfinite(loss):
            reason = 'diverged'
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if update == 0:
            opened = (placement.row_layout[-1].weight.grad is not None
                      and float(placement.row_layout[-1].weight.grad.detach().abs().sum()) > 0)
            held = model.context.attn.out_proj.weight.grad is None and placement.assign[0].weight.grad is None
            held = held and placement.angle[-1].weight.grad is None
            print('GRAD', json.dumps(dict(pose=bool(opened), selector_frozen=bool(held))), flush=True)
            if not opened or not held:
                reason = 'gradient'
                break
        for parameter in optimizer.param_groups[0]['params']:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        update += 1
        if update not in EVALS:
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
        if current['task'] < 98:
            reason = 'worse'
            break
    log.close()
    chosen = torch.load(ROOT/'checkpoint-best.pt', map_location=stable.DEVICE, weights_only=False)
    placement.load_state_dict(chosen['placement'])
    model.context.load_state_dict(chosen['context'])
    current_selector = snapshot(model.context, 'context')
    current_selector.update(snapshot(placement.assign, 'assign'))
    current_selector.update(snapshot(placement.order_query, 'order'))
    current_selector['order_begin'] = placement.order_begin.detach().cpu().clone()
    selector_held = all(torch.equal(selector[key], value) for key, value in current_selector.items())
    compared = flips(baseline_scenes, rollout(model, relation, dev_stored, dev_records))
    report = dict(status='STOP', stop=reason, stage2=True, labels=dict(fit=fit_counts, dev=dev_counts),
                  gt_dev=dict(baseline=before['task'], best=best['task'], total=512), best_update=best['update'],
                  flips=compared, selector_frozen=bool(selector_held),
                  pose_changed=not all(torch.equal(pose[key], tensor.detach().cpu())
                                       for key, tensor in pose_tensors(model).items()) if best['update'] else False,
                  adopted=False, test_used=False, full_9600=False,
                  checkpoints_unchanged=all((
                      sha256(JOINT_BEST) == stamps['joint'], sha256(SIZE_BEST) == stamps['size'],
                      sha256(ROW_BEST) == stamps['row'], sha256(RELATION_BEST) == stamps['relation'],
                      sha256(SELECTOR) == stamps['selector'])))
    if best['task'] > before['task'] and compared['pose_leaks'] == 0 and selector_held:
        bare_system, bare_relation, bare_row, bare_head, _bare = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
        base = score_rgb(bare_system, bare_relation, bare_row, bare_head, train, dev)
        final = score_rgb(system, relation, model, head, train, dev)
        kept = base['successes'] & final['successes']
        report['rgb_dev'] = dict(baseline=base['task'], task=final['task'], total=len(dev),
                                 preserved=len(kept), lost=len(base['successes']-final['successes']),
                                 gained=len(final['successes']-base['successes']),
                                 collision=final['failure_counts'].get('collision', 0),
                                 boundary=final['failure_counts'].get('boundary', 0),
                                 clearance=final['failure_counts'].get('clearance', 0),
                                 failures=final['failure_counts'])
        report['status'] = 'PASS' if final['task'] > 74 else 'STOP'
        report['adopted'] = False
        print('RGB', json.dumps(report['rgb_dev']), flush=True)
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(status=report['status'], best=best, flips=compared)), flush=True)


if __name__ == '__main__':
    main()
