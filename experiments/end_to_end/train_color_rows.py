"""Fine-tune the adopted selector so different colors stop sharing a row.

Origin, Δx, angle, and the perception stack stay frozen. The extra loss is
supervision only. Family-disjoint dev RGB success decides whether to keep it.
"""
import json
from pathlib import Path
import time

import torch

from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.row_layout import assignment_loss, color_separation_loss, order_loss
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from eval_row_layout import evaluate
from run_full_arrangement import cache_records
from run_row_layout import attach_rows, collate_rows, hidden_of
from score_end_to_end import (
    JOINT_BEST, RELATION_BEST, ROW_BEST, SELECTOR_BEST, SIZE_BEST, load_adopted, score_rgb)
from train_selector_only import pose_tensors, selector_parameters, sha256
from train_set_context import fresh_split

ROOT = Path('artifacts/experiments/color_rows')
DATA = 'artifacts/generalization_20000/dataset'
BATCH = 8
MAX_UPDATES = 80
EVALS = (40, 80)


def enable(model):
    model.requires_grad_(False)
    for parameter in selector_parameters(model):
        parameter.requires_grad_(True)


def batch_loss(relation, model, stored, records):
    packed = collate_rows(stored, stable.DEVICE)
    hidden = hidden_of(relation, packed['features'])
    output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                   packed['row'], packed['slot'], True)
    group = torch.tensor([record['instruction']['group_by'] == 'color' for record in records],
                         device=packed['valid'].device)
    loss = (assignment_loss(output, packed['row'], packed['valid'])+order_loss(output)
            +color_separation_loss(output['row_logits'], packed['color'], packed['valid'], group))
    return loss


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('color row result already exists')
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    stamps = dict(joint=sha256(JOINT_BEST), size=sha256(SIZE_BEST), row=sha256(ROW_BEST),
                  relation=sha256(RELATION_BEST), selector=sha256(SELECTOR_BEST))
    train = ExpandedDataset(DATA, 'train')
    fit, dev, _blocked = fresh_split(train.records)
    system, relation, model, head, saved = load_adopted()
    if int(saved['update']) != 160:
        raise RuntimeError('color fine-tune must start from selector update 160')
    enable(model)
    pose = pose_tensors(model)
    frozen = list(model.placement.row_layout.parameters())+list(model.placement.step.parameters())+list(
        model.placement.angle.parameters())
    if any(parameter.requires_grad for parameter in frozen):
        raise RuntimeError('pose parameters are trainable')
    fit_records = [train.records[index] for index in fit]
    dev_records = [train.records[index] for index in dev]
    fit_stored = attach_rows(fit_records, cache_records(system, fit_records))
    dev_stored = attach_rows(dev_records, cache_records(system, dev_records))
    before_gt = evaluate(model, relation, dev_stored, dev_records, 'free')
    if before_gt['task'] != 110:
        raise RuntimeError(f'selector GT task changed: {before_gt["task"]}')
    before_rgb = score_rgb(system, relation, model, head, train, dev)
    if before_rgb['task'] != 93:
        raise RuntimeError(f'selector RGB task changed: {before_rgb["task"]}')
    probe = [index for index, record in enumerate(fit_records) if record['instruction']['group_by'] == 'color'][:64]
    with torch.no_grad():
        packed = collate_rows([fit_stored[index] for index in probe], stable.DEVICE)
        hidden = hidden_of(relation, packed['features'])
        output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                       None, None, True)
        group = torch.ones(len(probe), dtype=torch.bool, device=packed['valid'].device)
        opened = float(color_separation_loss(output['row_logits'], packed['color'], packed['valid'], group))
    print('COLOR', json.dumps(dict(loss=opened, gt=before_gt['task'], rgb=before_rgb['task'])), flush=True)
    if opened <= 0:
        stable.write(ROOT/'result.json', dict(status='STOP', stage='loss', test_used=False, adopted=False))
        print('STOP color rows are already separated', flush=True)
        return
    optimizer = torch.optim.AdamW((
        dict(params=list(model.context.parameters()), lr=1e-4),
        dict(params=list(model.placement.assign.parameters())+list(model.placement.order_query.parameters())
             +[model.placement.order_begin], lr=1e-5)), weight_decay=0.)
    best = dict(update=0, rgb=93)
    torch.save(dict(placement=model.placement.state_dict(), context=model.context.state_dict(), update=0,
                    dev_rgb=93), ROOT/'checkpoint-best.pt')
    generator = torch.Generator().manual_seed(1703)
    order, cursor, update, reason = [], 0, 0, 'max_updates'
    started = time.perf_counter()
    history = []
    while update < MAX_UPDATES:
        if cursor+BATCH > len(order):
            order = torch.randperm(len(fit_stored), generator=generator).tolist()
            cursor = 0
        choice = order[cursor:cursor+BATCH]
        cursor += BATCH
        loss = batch_loss(relation, model, [fit_stored[index] for index in choice],
                          [fit_records[index] for index in choice])
        if not torch.isfinite(loss):
            reason = 'diverged'
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if update == 0:
            pose_held = all(parameter.grad is None for parameter in frozen)
            moved = model.placement.assign[0].weight.grad is not None
            print('GRAD', json.dumps(dict(selector=bool(moved), pose_frozen=bool(pose_held))), flush=True)
            if not moved or not pose_held:
                reason = 'gradient'
                break
        optimizer.step()
        update += 1
        if update not in EVALS:
            continue
        current = score_rgb(system, relation, model, head, train, dev)
        gt = evaluate(model, relation, dev_stored, dev_records, 'free')
        kept = before_rgb['successes'] & current['successes']
        point = dict(update=update, rgb=current['task'], gt=gt['task'], row=current['row_accuracy'],
                     slot=current['slot_accuracy'], preserved=len(kept),
                     lost=len(before_rgb['successes']-current['successes']),
                     gained=len(current['successes']-before_rgb['successes']),
                     failures=current['failure_counts'], gt_failures=gt['failures'],
                     seconds=time.perf_counter()-started)
        history.append(point)
        print('EVAL', json.dumps(point), flush=True)
        if current['task'] > best['rgb']:
            best = dict(update=update, rgb=current['task'], gt=gt['task'], preserved=len(kept),
                        lost=point['lost'], gained=point['gained'], failures=current['failure_counts'])
            torch.save(dict(placement=model.placement.state_dict(), context=model.context.state_dict(),
                            update=update, dev_rgb=current['task']), ROOT/'checkpoint-best.pt')
        if current['task'] < 80:
            reason = 'worse'
            break
    chosen = torch.load(ROOT/'checkpoint-best.pt', map_location=stable.DEVICE, weights_only=False)
    model.placement.load_state_dict(chosen['placement'])
    model.context.load_state_dict(chosen['context'])
    pose_held = all(torch.equal(pose[key], tensor.detach().cpu()) for key, tensor in pose_tensors(model).items())
    adopted = best['rgb'] > 93 and best.get('lost', 1) <= best.get('gained', 0) and pose_held
    report = dict(status='PASS' if adopted else 'STOP', stop=reason, best=best, history=history,
                  baseline=dict(rgb=93, gt=110), pose_unchanged=bool(pose_held), adopted=bool(adopted),
                  test_used=False, full_16000=False, pose_retrained=False,
                  checkpoints_unchanged=all((
                      sha256(JOINT_BEST) == stamps['joint'], sha256(SIZE_BEST) == stamps['size'],
                      sha256(ROW_BEST) == stamps['row'], sha256(RELATION_BEST) == stamps['relation'],
                      sha256(SELECTOR_BEST) == stamps['selector'])))
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(status=report['status'], best=best, pose_unchanged=bool(pose_held))), flush=True)


if __name__ == '__main__':
    main()
