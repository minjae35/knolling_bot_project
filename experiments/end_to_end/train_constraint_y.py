"""Evaluator-aligned Row-Y loss. One overfit gate, then at most one pilot.

Selector, detector, language, goal, size, angle, and Δx stay fixed. Illegal rows
are not given a pose target. Inference is unchanged: no teacher and no corrector.
"""
import json
from collections import Counter
from pathlib import Path

import torch

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from analyze_linked_y_loss import collect, openings, solve_y
from knolling_dataset.schema import HEIGHT_M, REGIONS, scene_from_dict
from knolling_dataset.validator import inferred_rows, polygon
from knolling_model.data import Batch
from knolling_model.diagnostics import positioned_objects
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import encode_instruction, install_refined_size
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
from knolling_model.recovery.selection import select_queries
from score_end_to_end import load_adopted, score_rgb
from train_linked_y import VAL_COUNT, VAL_SEED, frozen_snapshot, same_snapshot
from train_set_context import fresh_split

ROOT = Path('artifacts/experiments/constraint_y')
OVERFIT_FAILS = 8
OVERFIT_KEEP = 4
OVERFIT_STEPS = 400
PILOT_STEPS = 200
BATCH = 8
LR = 1e-3
EXACT_M = 0.002
CLEARANCE_M = 0.010 - 0.0001
BOUNDARY_LOW = 0.010 - 0.0001
BOUNDARY_HIGH = HEIGHT_M - 0.010 + 0.0001
REGION_SLACK = 0.0001


def box_y(center, theta, major, minor):
    """Vertical bounds of the evaluator's rotated envelope. Center stays live."""
    cosine, sine = torch.cos(theta), torch.sin(theta)
    offsets = []
    for sign_x, sign_y in ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)):
        offsets.append(sine*(sign_x*major/2)+cosine*(sign_y*minor/2))
    offset = torch.stack(offsets)
    return center+offset.min(), center+offset.max()


def scene_loss(xy, theta, row, assets, record, baseline_y, preserve):
    """Exact 2mm opening band, minimum clearance, boundary, and region.

    A coordinate already inside a band contributes nothing, so it has no gradient.
    """
    theta = theta.detach()
    x = xy[:, 0].detach()
    y = xy[:, 1]
    valid = list(range(len(assets)))
    bounds = [box_y(y[index], theta[index], assets[index]['a'], assets[index]['b']) for index in valid]
    total = y.new_zeros(())
    counted = 0
    region = REGIONS[record['instruction']['region']]
    for index in valid:
        ymin, ymax = bounds[index]
        pieces = (
            torch.relu(BOUNDARY_LOW-ymin)+torch.relu(ymax-BOUNDARY_HIGH),
            torch.relu((region[1]-REGION_SLACK)-ymin)+torch.relu(ymax-(region[3]+REGION_SLACK)))
        total = total+pieces[0]+pieces[1]
        counted += 2
    groups = {}
    for index in valid:
        groups.setdefault(int(row[index]), []).append(index)
    ordered = sorted(groups, key=lambda key: -float(y.detach()[groups[key]].mean()))
    gap = record['instruction']['gap_m']
    for upper_key, lower_key in zip(ordered, ordered[1:]):
        upper, lower = groups[upper_key], groups[lower_key]
        opening = torch.stack([bounds[index][0] for index in upper]).min()
        opening = opening-torch.stack([bounds[index][1] for index in lower]).max()
        upper_color = assets[min(upper, key=lambda index: float(x[index]))]['color']
        lower_color = assets[min(lower, key=lambda index: float(x[index]))]['color']
        factor = 2 if record['instruction']['group_by'] == 'color' and upper_color != lower_color else 1
        required = factor*gap
        total = total+torch.relu((opening-required).abs()-EXACT_M)+torch.relu(CLEARANCE_M-opening)
        counted += 2
    if preserve:
        total = total+(y-baseline_y).abs().mean()
        counted += 1
    if counted == 0:
        raise RuntimeError('constraint loss had no term')
    return total/counted


def collate(items, device):
    width = max(item['count'] for item in items)

    def pad(value, fill):
        missing = width-value.shape[0]
        if missing <= 0:
            return value[:width]
        tail = value.new_full((missing, *value.shape[1:]), fill)
        return torch.cat((value, tail), 0)

    return dict(
        hidden=torch.stack([pad(item['hidden'], 0.) for item in items]).to(device),
        valid=torch.stack([pad(item['valid'], False) for item in items]).to(device),
        goal=torch.stack([item['goal'] for item in items]).to(device),
        widths=torch.stack([pad(item['widths'], 0.) for item in items]).to(device),
        heights=torch.stack([pad(item['heights'], 0.) for item in items]).to(device),
        colors=torch.stack([pad(item['colors'], 0.) for item in items]).to(device))


def forward_loss(model, items):
    packed = collate(items, stable.DEVICE)
    output = model(packed['hidden'], packed['valid'], packed['goal'], packed['widths'], packed['heights'],
                   None, None, True, colors=packed['colors'])
    loss = output['xy'].new_zeros(())
    for index, item in enumerate(items):
        count = item['count']
        loss = loss+scene_loss(
            output['xy'][index, :count], output['angle'][index, :count], output['row'][index, :count],
            item['assets'], item['record'], item['baseline_y'].to(output['xy'].device), item['kind'] == 'success')
    return loss/len(items), output


def poses_of(item, xy, angle):
    poses = {}
    for index, asset in enumerate(item['assets']):
        poses[asset['id']] = dict(x_m=float(xy[index, 0]), y_m=float(xy[index, 1]), theta_rad=float(angle[index]))
    return poses


def task_count(model, items):
    packed = collate(items, stable.DEVICE)
    with torch.no_grad():
        output = model(packed['hidden'], packed['valid'], packed['goal'], packed['widths'], packed['heights'],
                       None, None, True, colors=packed['colors'])
    wins = []
    errors = Counter()
    for index, item in enumerate(items):
        count = item['count']
        detail = arrangement_errors(item['record'], poses_of(
            item, output['xy'][index, :count].cpu(), output['angle'][index, :count].cpu()))
        wins.append(bool(detail['task']))
        if not detail['task']:
            errors.update(detail['errors'])
    return wins, errors


def capture(system, relation, model, head, data, index):
    record = data.records[index]
    image = data[index]['image'][None].to(stable.DEVICE)
    identities, truth = recovery_supervision(record)
    truth = {key: value.cpu() for key, value in truth.items()}
    with torch.no_grad():
        visual = system.perception(image)
        visual = install_refined_size(head, image, visual, system.threshold)
        queries = select_queries(visual['source'], system.threshold)
        if any(len(chosen) < 1 or len(chosen) > 8 for chosen in queries):
            return None, 'count'
        objects = pack_sources(visual['source'], queries)
        text, mask = encode_instruction(system, [record['instruction']['text']], image.device)
        goal = system.layout.goal_encoder(objects, text, mask)['vector']
        hidden = relation._hidden(objects.features())
        output = model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                       None, None, True, colors=objects.color)
    count = int(objects.valid[0].sum())
    if not bool(objects.valid[0, :count].all()):
        return None, 'count'
    sources = scene_from_dict(record['source']).objects
    used = set()
    assets = []
    for slot in range(count):
        point = objects.xy[0, slot].detach().cpu()
        best, distance = None, 1e9
        for obj in sources:
            if obj.object_id in used:
                continue
            gap = ((obj.source_pose.x_m-float(point[0]))**2+(obj.source_pose.y_m-float(point[1]))**2)**.5
            if gap < distance:
                best, distance = obj, gap
        if best is None or distance > .010:
            return None, 'match'
        used.add(best.object_id)
        assets.append(dict(id=best.object_id, a=best.asset.a_m, b=best.asset.b_m, color=best.color))
    source = {key: value.detach().cpu() for key, value in visual['source'].items()}
    layout = dict(xy=output['xy'].detach().cpu(), angle=output['angle'].detach().cpu(),
                  valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                  rollout='free_rollout')
    judged = evaluate_batch(legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5)),
                            Batch({}, [truth], [dict(sample_id=record['sample_id'], object_ids=identities,
                                                     record=record)]),
                            semantic_check=semantic_errors_v3)[0]
    task = bool(accepted_success(record, judged['task_success']))
    official = judged.get('predicted_targets')
    if official is None:
        return None, 'match'
    for asset, slot in zip(assets, range(count)):
        if asset['id'] not in official:
            return None, 'match'
        if abs(official[asset['id']]['y_m']-float(output['xy'][0, slot, 1])) > 1e-4:
            return None, 'match'
    item = dict(
        index=index, record=record, task=task, assets=assets, count=count,
        hidden=hidden[0, :count].detach().cpu(), valid=torch.ones(count, dtype=torch.bool),
        goal=goal[0].detach().cpu(), widths=objects.size[0, :count, 0].detach().cpu(),
        heights=objects.size[0, :count, 1].detach().cpu(), colors=objects.color[0, :count].detach().cpu(),
        baseline_y=output['xy'][0, :count, 1].detach().cpu(),
        baseline_xy=output['xy'][0, :count].detach().cpu(),
        baseline_angle=output['angle'][0, :count].detach().cpu())
    if task:
        item['kind'] = 'success'
        return item, 'success'
    poses = poses_of(item, item['baseline_xy'], item['baseline_angle'])
    _updated, reason = solve_y(record, poses)
    if reason == 'pass':
        item['kind'] = 'fix'
    return item, reason


def opening_errors(record, poses):
    values = openings(inferred_rows(positioned_objects(record, poses)), record)
    return [abs(value-target)*1000 for value, target in values]


def prove_loss(failure, success):
    """Direct coordinate step, before any weight changes."""
    posed = positioned_objects(failure['record'], poses_of(failure, failure['baseline_xy'], failure['baseline_angle']))
    obj = next(item for item in posed if item.object_id == failure['assets'][0]['id'])
    ymin, ymax = box_y(torch.tensor(obj.target_pose.y_m), torch.tensor(obj.target_pose.theta_rad),
                       obj.asset.a_m, obj.asset.b_m)
    bounds = polygon(obj, obj.target_pose).bounds
    if abs(float(ymin)-bounds[1]) > 1e-6 or abs(float(ymax)-bounds[3]) > 1e-6:
        raise RuntimeError('the constraint box does not match the evaluator polygon')
    y = failure['baseline_xy'].clone()
    y.requires_grad_(True)
    loss = scene_loss(y, failure['baseline_angle'], failure_rows(failure), failure['assets'], failure['record'],
                      failure['baseline_y'], False)
    loss.backward()
    before = float(loss.detach())
    direction = y.grad.detach()
    stepped = y.detach()-1e-3*direction.sign()
    after = float(scene_loss(stepped, failure['baseline_angle'], failure_rows(failure), failure['assets'],
                             failure['record'], failure['baseline_y'], False))
    before_mm = opening_errors(failure['record'], poses_of(failure, failure['baseline_xy'], failure['baseline_angle']))
    after_mm = opening_errors(failure['record'], poses_of(failure, stepped, failure['baseline_angle']))
    kept = success['baseline_xy'].clone().requires_grad_(True)
    kept_loss = scene_loss(kept, success['baseline_angle'], failure_rows(success), success['assets'],
                           success['record'], success['baseline_y'], False)
    kept_loss.backward()
    outside = kept.detach().clone()
    top = int(failure_rows(success)[int(kept[:, 1].argmax())])
    for index, row_id in enumerate(failure_rows(success).tolist()):
        if row_id == top:
            outside[index, 1] = outside[index, 1]+0.005
    outside_value = scene_loss(outside, success['baseline_angle'], failure_rows(success), success['assets'],
                               success['record'], success['baseline_y'], False)
    report = dict(
        failure_loss_before=before, failure_loss_after_1mm=after,
        failure_opening_mm_before=before_mm, failure_opening_mm_after=after_mm,
        failure_grad_abs_max=float(direction.abs().max()),
        success_loss=float(kept_loss.detach()), success_grad_abs_max=float(kept.grad.detach().abs().max()),
        success_outside_5mm=float(outside_value.detach()))
    if not after < before-1e-6:
        raise RuntimeError(f'a 1mm step did not reduce the constraint: {report}')
    if sum(after_mm) >= sum(before_mm)-0.5:
        raise RuntimeError(f'the evaluator opening did not improve: {report}')
    if float(kept.grad.detach().abs().max()) > 1e-8:
        raise RuntimeError(f'an in-band success still has a gradient: {report}')
    if float(outside_value.detach()) <= 0:
        raise RuntimeError(f'a 5mm push did not leave the allowed band: {report}')
    return report


def failure_rows(item):
    """Row ids of the adopted forward. Recomputed once and stored on the item."""
    if 'rows' not in item:
        raise RuntimeError('row ids were not cached')
    return item['rows']


def cache_rows(model, item):
    packed = collate([item], stable.DEVICE)
    with torch.no_grad():
        output = model(packed['hidden'], packed['valid'], packed['goal'], packed['widths'], packed['heights'],
                       None, None, True, colors=packed['colors'])
    item['rows'] = output['row'][0, :item['count']].detach().cpu()
    return item


def take(pool, kind, count):
    return [item for item in pool if item['kind'] == kind][:count]


def row_count(item):
    return len(set(int(value) for value in item['rows'].tolist()))


def ready(pool):
    fixes = [item for item in pool if item['kind'] == 'fix']
    keeps = [item for item in pool if item['kind'] == 'success']
    violating = [item for item in fixes if row_count(item) > 1 and max(opening_errors(
        item['record'], poses_of(item, item['baseline_xy'], item['baseline_angle'])), default=0) > 2]
    return (len(fixes) >= OVERFIT_FAILS and len(keeps) >= OVERFIT_KEEP
            and any(row_count(item) > 1 for item in keeps) and bool(violating))


def overfit(model, items):
    optimizer = torch.optim.Adam([parameter for parameter in model.placement.y_parameters() if parameter.requires_grad], lr=LR)
    history = []
    for step in range(OVERFIT_STEPS):
        optimizer.zero_grad()
        loss, _output = forward_loss(model, items)
        if not torch.isfinite(loss):
            raise RuntimeError('constraint loss diverged during overfit')
        loss.backward()
        if step == 0:
            placement = model.placement
            if placement.row_layout[-1].weight.grad is not None or placement.step[-1].weight.grad is not None:
                raise RuntimeError('frozen origin or Δx received a gradient')
            if placement.row_gap[-1].weight.grad is None and placement.row_anchor.weight.grad is None:
                raise RuntimeError('neither Y head received a gradient')
        optimizer.step()
        if step % 20 == 0 or step == OVERFIT_STEPS-1:
            wins, errors = task_count(model, items)
            fixes = [item['kind'] == 'fix' for item in items]
            history.append(dict(step=step, loss=float(loss.detach()),
                                fix=sum(win for win, fix in zip(wins, fixes) if fix),
                                fix_total=sum(fixes),
                                kept=sum(win for win, fix in zip(wins, fixes) if not fix),
                                kept_total=sum(not fix for fix in fixes),
                                errors=dict(errors)))
            print('OVERFIT', json.dumps(history[-1]), flush=True)
            if history[-1]['fix'] == history[-1]['fix_total'] and history[-1]['kept'] == history[-1]['kept_total']:
                break
    return history


def pilot(model, items):
    optimizer = torch.optim.Adam([parameter for parameter in model.placement.y_parameters() if parameter.requires_grad], lr=LR)
    cursor = 0
    order = []
    losses = []
    for update in range(PILOT_STEPS):
        if cursor+BATCH > len(order):
            order = torch.randperm(len(items), generator=torch.Generator().manual_seed(VAL_SEED+update)).tolist()
            cursor = 0
        choice = [items[index] for index in order[cursor:cursor+BATCH]]
        cursor += BATCH
        optimizer.zero_grad()
        loss, _output = forward_loss(model, choice)
        if not torch.isfinite(loss):
            raise RuntimeError('constraint loss diverged during the pilot')
        loss.backward()
        optimizer.step()
        if update in (0, 49, PILOT_STEPS-1):
            losses.append(dict(step=update, loss=round(float(loss.detach()), 6)))
            print('PILOT', json.dumps(losses[-1]), flush=True)
    return losses


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('constraint Y result already exists')
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    data = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    fit, dev, _blocked = fresh_split(data.records)
    if len(dev) != 512:
        raise RuntimeError(f'family-disjoint dev changed: {len(dev)}')
    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(VAL_SEED)).tolist()
    val_ids = [fit[index] for index in order[:VAL_COUNT]]
    train_ids = [fit[index] for index in order[VAL_COUNT:]]
    if set(val_ids) & set(dev) or set(train_ids) & set(dev) or set(val_ids) & set(train_ids):
        raise RuntimeError('the pilot split overlaps a sealed or used split')
    system, relation, model, head, _saved = load_adopted()
    model.requires_grad_(False)
    for parameter in model.placement.y_parameters():
        parameter.requires_grad_(True)
    held = frozen_snapshot(model)
    initial = dict(row_anchor=model.placement.row_anchor.state_dict(),
                   row_gap=model.placement.row_gap.state_dict())
    before = score_rgb(system, relation, model, head, data, val_ids)
    print('BEFORE', before['task'], flush=True)
    if before['task'] != 47:
        raise RuntimeError(f'adopted baseline on the 256 is {before["task"]}, not 47')
    pool = []
    seen = Counter()
    scanned = set()
    for index in train_ids:
        scanned.add(index)
        item, reason = capture(system, relation, model, head, data, index)
        seen[reason] += 1
        if item is not None and reason in ('success', 'pass'):
            pool.append(cache_rows(model, item))
        if ready(pool):
            break
    print('SCAN', json.dumps(dict(seen)), flush=True)
    failures = take(pool, 'fix', OVERFIT_FAILS)
    keeps = take(pool, 'success', OVERFIT_KEEP)
    if not ready(pool):
        raise RuntimeError(f'not enough Y-fixable train scenes: {len(failures)} fixes, {len(keeps)} successes')
    probe_fail = next(item for item in pool if item['kind'] == 'fix' and row_count(item) > 1 and max(opening_errors(
        item['record'], poses_of(item, item['baseline_xy'], item['baseline_angle'])), default=0) > 2)
    probe_keep = next(item for item in pool if item['kind'] == 'success' and row_count(item) > 1)
    proof = prove_loss(probe_fail, probe_keep)
    print('PROOF', json.dumps(proof), flush=True)
    history = overfit(model, failures+keeps)
    official = collect(system, relation, model, head, data, [item['index'] for item in failures+keeps])
    fix_ids = {item['record']['sample_id'] for item in failures}
    official_fix = sum(item['task'] for item in official if item['sample_id'] in fix_ids)
    official_kept = sum(item['task'] for item in official if item['sample_id'] not in fix_ids)
    print('OVERFIT_RGB', official_fix, official_kept, flush=True)
    model.placement.row_anchor.load_state_dict(initial['row_anchor'])
    model.placement.row_gap.load_state_dict(initial['row_gap'])
    passed = official_fix == OVERFIT_FAILS and official_kept == OVERFIT_KEEP
    result = dict(status='NO-GO', adopted=False, test_used=False, dev_512_used=False, pilot=False,
                  proof=proof, overfit=history, overfit_rgb=dict(fix=official_fix, kept=official_kept),
                  scan=dict(seen), baseline=before['task'])
    if not passed:
        (ROOT/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return
    for index in train_ids:
        if index in scanned:
            continue
        scanned.add(index)
        item, reason = capture(system, relation, model, head, data, index)
        seen[reason] += 1
        if item is not None and reason in ('success', 'pass'):
            pool.append(cache_rows(model, item))
        if len(scanned) % 200 == 0:
            print('SCAN', json.dumps(dict(seen)), flush=True)
    print('ELIGIBLE', sum(item['kind'] == 'fix' for item in pool), sum(item['kind'] == 'success' for item in pool), flush=True)
    losses = pilot(model, pool)
    if not same_snapshot(model, held):
        raise RuntimeError('a frozen weight changed')
    after = score_rgb(system, relation, model, head, data, val_ids)
    kept = len(before['successes'] & after['successes'])
    result.update(
        pilot=True, losses=losses, scan=dict(seen),
        train_fix=sum(item['kind'] == 'fix' for item in pool),
        train_success=sum(item['kind'] == 'success' for item in pool),
        rgb=dict(before=before['task'], after=after['task'], total=len(val_ids),
                 preserved=kept, lost=before['task']-kept, gained=after['task']-kept,
                 before_failures=before['failure_counts'], after_failures=after['failure_counts'],
                 before_vertical=before['vertical'], after_vertical=after['vertical']))
    result['status'] = 'GO' if after['task'] > before['task'] else 'NO-GO'
    torch.save(dict(row_anchor=model.placement.row_anchor.state_dict(),
                    row_gap=model.placement.row_gap.state_dict(), adopted=False), ROOT/'checkpoint.pt')
    (ROOT/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
