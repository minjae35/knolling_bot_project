"""Train only the object-wise size residual. Detector and arrangement stay frozen."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors, is_full
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_layout import NeuralRowPlacement
from knolling_model.experiments.size_refinement import (
    ObjectSizeRefinement, horizontal_extent, install_refined_size, matched_queries,
    poses_from_detection)
from knolling_model.recovery.data import recovery_supervision, unique_source_indices
from knolling_model.recovery.model import legacy_evaluation_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.selection import select_queries
import run_phase1_pairwise_affine as runner
import run_relation_capacity as cap
import run_relation_stable as stable
from diagnose_row_transition import multi_row
from eval_row_layout import evaluate
from run_full_arrangement import cache_records, score, subset
from run_row_layout import attach_rows

ROOT = Path('artifacts/experiments/size_refinement')
ROW_BEST = Path('artifacts/row_layout_origin_y/checkpoint-best.pt')
RELATION_BEST = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
DATA = 'artifacts/generalization_20000/dataset'
SEED = 1203
BATCH = 8
LR = 1e-3
EVAL_EVERY = 100
RGB_EVERY = 200
MAX_UPDATES = 1000
MIN_UPDATES = 200
PATIENCE = 3
WIDTH_GAIN_MM = 0.02


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    if not values:
        return dict(median=None, p95=None, n=0)
    tensor = torch.tensor(values, dtype=torch.float)
    return dict(median=float(tensor.median()), p95=float(tensor.quantile(.95)), n=len(values))


def family_split(records):
    indices = unique_source_indices(records)
    ordered = sorted(indices, key=lambda index: records[index]['scene_family_id'])
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(len(ordered), generator=generator).tolist()
    dev_count = len(ordered)//5
    dev = [ordered[index] for index in permutation[:dev_count]]
    fit = [ordered[index] for index in permutation[dev_count:]]
    return fit, dev


def full_records(records):
    found = {}
    for index, record in enumerate(records):
        if is_full(record) and record['scene_family_id'] not in found:
            found[record['scene_family_id']] = index
    return found


def correspond(source, truths, threshold):
    pairs = []
    summary = Counter()
    scale = source['uv'].new_tensor([.6, .4])
    for batch, truth in enumerate(truths):
        chosen = select_queries({'objectness_logits': source['objectness_logits'][batch:batch+1]}, threshold)[0]
        truth_xy = truth['source_xy']
        summary['gt'] += len(truth_xy)
        summary['scenes'] += 1
        if len(chosen) != len(truth_xy):
            summary['count_mismatch'] += 1
        if not len(chosen):
            summary['unmatched_gt'] += len(truth_xy)
            continue
        distance = torch.cdist(source['uv'][batch, chosen]*scale, truth_xy)
        used_query, used_gt = set(), set()
        for item in distance.flatten().argsort().tolist():
            query, gt = divmod(item, distance.shape[1])
            if query in used_query or gt in used_gt:
                continue
            if float(distance[query, gt]) > .010:
                break
            used_query.add(query)
            used_gt.add(gt)
            pairs.append((batch, int(chosen[query]), gt))
        summary['matched'] += len(used_gt)
        summary['unmatched_gt'] += len(truth_xy)-len(used_gt)
    return pairs, summary


def labels_of(record, identities):
    row = torch.full((len(identities),), -1)
    slot = torch.full((len(identities),), -1)
    location = {object_id: index for index, object_id in enumerate(identities)}
    for row_id, members in enumerate(record['supervision_v3']['rows']):
        for slot_id, object_id in enumerate(members):
            row[location[object_id]] = row_id
            slot[location[object_id]] = slot_id
    return row, slot


def size_report(system, head, data, indices):
    width, height, footprint = [], [], []
    summary = Counter()
    system.perception.eval()
    if head is not None:
        head.eval()
    with torch.no_grad():
        for start in range(0, len(indices), BATCH):
            chunk = indices[start:start+BATCH]
            images = torch.stack([data[index]['image'] for index in chunk]).to(stable.DEVICE)
            truths = []
            for index in chunk:
                _, truth = recovery_supervision(data.records[index])
                truths.append({key: value.to(stable.DEVICE) for key, value in truth.items()})
            visual = system.perception(images)
            if head is not None:
                visual = install_refined_size(head, images, visual, system.threshold)
            pairs, counts = correspond(visual['source'], truths, system.threshold)
            summary.update(counts)
            for batch, query, gt in pairs:
                predicted = visual['source']['size'][batch, query]
                truth = truths[batch]['size'][gt]
                width.append(float((predicted[0]-truth[0]).abs())*1000)
                height.append(float((predicted[1]-truth[1]).abs())*1000)
                predicted_extent = horizontal_extent(predicted, visual['source']['angle'][batch, query])
                truth_extent = horizontal_extent(truth, truths[batch]['source_angle'][gt])
                footprint.append(float((predicted_extent-truth_extent).abs())*1000)
    return dict(width=describe(width), height=describe(height), footprint=describe(footprint),
                matched=int(summary['matched']), gt=int(summary['gt']),
                count_mismatch=int(summary['count_mismatch']), scenes=int(summary['scenes']))


def score_rgb(system, relation, row_model, head, data, indices):
    system.perception.eval()
    relation.eval()
    row_model.eval()
    if head is not None:
        head.eval()
    scenes = []
    with torch.no_grad():
        for index in indices:
            record = data.records[index]
            image = data[index]['image'][None].to(stable.DEVICE)
            identities, truth = recovery_supervision(record)
            truth = {key: value.cpu() for key, value in truth.items()}
            try:
                output, objects, visual = poses_from_detection(
                    system, relation, row_model, image, [record['instruction']['text']], head)
            except ValueError:
                scenes.append(dict(sample_id=record['sample_id'], objects=len(identities), task=False,
                                   errors=['count'], horizontal=[], vertical=[], row_match=False,
                                   row_hits=0, slot_hits=0, labeled=0))
                continue
            source = {key: value.detach().cpu() for key, value in visual['source'].items()}
            layout = dict(xy=output['xy'].detach().cpu(), angle=output['angle'].detach().cpu(),
                          valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                          rollout='free_rollout')
            view = legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5))
            meta = dict(sample_id=record['sample_id'], object_ids=identities, record=record)
            judged = evaluate_batch(view, Batch({}, [truth], [meta]), semantic_check=semantic_errors_v3)[0]
            detail = arrangement_errors(record, judged['predicted_targets']) if judged.get('predicted_targets') else None
            row_hits = slot_hits = labeled = 0
            if judged['count_correct']:
                row, slot = labels_of(record, identities)
                chosen = select_queries(visual['source'], system.threshold)[0]
                predicted_xy = objects.xy[0, :len(chosen)]
                distance = torch.cdist(predicted_xy, truth['source_xy'].to(predicted_xy.device))
                used_query, used_gt = set(), set()
                for item in distance.flatten().argsort().tolist():
                    query, gt = divmod(item, distance.shape[1])
                    if query in used_query or gt in used_gt or float(distance[query, gt]) > .010:
                        continue
                    used_query.add(query)
                    used_gt.add(gt)
                    labeled += 1
                    row_hits += int(int(output['row'][0, query]) == int(row[gt]))
                    slot_hits += int(int(output['slot'][0, query]) == int(slot[gt]))
            scenes.append(dict(sample_id=record['sample_id'], objects=len(identities),
                               task=bool(judged['task_success']), errors=list(judged['errors']),
                               horizontal=[] if detail is None else detail['horizontal_mm'],
                               vertical=[] if detail is None else detail['vertical_mm'],
                               row_match=bool(detail and detail['row_match']),
                               row_hits=row_hits, slot_hits=slot_hits, labeled=labeled))
    return pack_rgb(scenes)


def pack_rgb(scenes):
    horizontal = [value for scene in scenes for value in scene['horizontal']]
    vertical = [value for scene in scenes for value in scene['vertical']]
    labeled = sum(scene['labeled'] for scene in scenes)

    def bucket(count):
        chosen = [scene for scene in scenes if scene['objects'] == count]
        return dict(success=sum(scene['task'] for scene in chosen), total=len(chosen))

    return dict(task=sum(scene['task'] for scene in scenes), total=len(scenes),
                by_count={str(count): bucket(count) for count in (2, 4, 6, 8)},
                horizontal=describe(horizontal), vertical=describe(vertical),
                count_errors=sum('count' in scene['errors'] for scene in scenes),
                row_match=sum(scene['row_match'] for scene in scenes),
                row_accuracy=None if not labeled else sum(scene['row_hits'] for scene in scenes)/labeled,
                slot_accuracy=None if not labeled else sum(scene['slot_hits'] for scene in scenes)/labeled,
                successes={scene['sample_id'] for scene in scenes if scene['task']})


def same_state(module, saved):
    current = module.state_dict()
    return all(torch.equal(current[key], value.to(current[key].device)) for key, value in saved.items())


def step_check(system, relation, row_model, head, train, fit):
    frozen = {
        'perception': copy_state(system.perception),
        'language': copy_state(system.arrangement.text_encoder),
        'goal': copy_state(system.layout.goal_encoder),
        'relation': copy_state(relation),
        'row': copy_state(row_model),
    }
    for module in (system.perception, system.arrangement, relation, row_model):
        module.requires_grad_(False)
        module.eval()
    head.train()
    trainable = [name for name, parameter in head.named_parameters() if parameter.requires_grad]
    others = [parameter for module in (system.perception, system.arrangement, relation, row_model)
              for parameter in module.parameters() if parameter.requires_grad]
    choice = fit[:BATCH]
    images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
    truths = []
    for index in choice:
        _, truth = recovery_supervision(train.records[index])
        truths.append({key: value.to(stable.DEVICE) for key, value in truth.items()})
    with torch.no_grad():
        visual = system.perception(images)
    reference = {key: visual['source'][key].detach().clone() for key in ('uv', 'angle', 'objectness_logits', 'size')}
    counts = [len(queries) for queries in select_queries(visual['source'], system.threshold)]
    pairs, _ = correspond(visual['source'], truths, system.threshold)
    if not pairs or others or any(not name.startswith(('crop.', 'readout.')) for name in trainable):
        return dict(passed=False, reason='trainable_path', pairs=len(pairs), trainable=trainable)
    rgb, tokens, boxes, base = matched_queries(
        visual['stages'][-1]['tokens'], images, visual['stages'][-1]['boxes'], visual['source']['size'],
        [(batch, query) for batch, query, _ in pairs])
    before = head(tokens, rgb, boxes, base).detach()
    target = torch.stack([truths[batch]['size'][gt] for batch, _, gt in pairs])
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.)
    optimizer.zero_grad(set_to_none=True)
    refined = head(tokens, rgb, boxes, base)[:, 0]
    loss = F.smooth_l1_loss(refined, target, beta=.001)
    loss.backward()
    leaked = any(parameter.grad is not None for parameter in
                 list(system.perception.parameters())+list(relation.parameters())+list(row_model.parameters()))
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
    optimizer.step()
    after = head(tokens, rgb, boxes, base).detach()
    with torch.no_grad():
        again = system.perception(images)
    counts_again = [len(queries) for queries in select_queries(again['source'], system.threshold)]
    detection_same = (counts == counts_again and torch.equal(again['source']['uv'], reference['uv'])
                      and torch.equal(again['source']['angle'], reference['angle'])
                      and torch.equal(again['source']['objectness_logits'], reference['objectness_logits'])
                      and torch.equal(again['source']['size'], reference['size']))
    weights_same = all(same_state(module, frozen[name]) for name, module in (
        ('perception', system.perception), ('language', system.arrangement.text_encoder),
        ('goal', system.layout.goal_encoder), ('relation', relation), ('row', row_model)))
    passed = (not leaked and detection_same and weights_same and not torch.allclose(before, after)
              and torch.isfinite(loss))
    return dict(passed=bool(passed), leaked=bool(leaked), detection_same=bool(detection_same),
                weights_same=bool(weights_same), size_changed=bool(not torch.allclose(before, after)),
                loss=float(loss.detach()), pairs=len(pairs), trainable=len(trainable))


def copy_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_checkpoint(path, head, optimizer, update, payload):
    torch.save(dict(model=head.state_dict(), optimizer=optimizer.state_dict(), update=update, **payload), path)


def main():
    if (ROOT/'result.json').exists():
        raise RuntimeError('size refinement result already exists')
    stamps = dict(row=sha256(ROW_BEST), relation=sha256(RELATION_BEST))
    runner.setup()
    ROOT.mkdir(parents=True, exist_ok=True)
    train = ExpandedDataset(DATA, 'train')
    val = ExpandedDataset(DATA, 'validation')
    fit, dev = family_split(train.records)
    fit_families = {train.records[index]['scene_family_id'] for index in fit}
    dev_families = {train.records[index]['scene_family_id'] for index in dev}
    val_families = {record['scene_family_id'] for record in val.records}
    if fit_families & dev_families or fit_families & val_families or dev_families & val_families:
        raise RuntimeError('source families overlap a held-out split')
    full = full_records(train.records)
    dev_rgb_indices = [full[train.records[index]['scene_family_id']] for index in dev
                       if train.records[index]['scene_family_id'] in full]
    print('SPLIT', json.dumps(dict(train_families=len(fit)+len(dev), fit=len(fit), dev=len(dev),
                                  dev_full=len(dev_rgb_indices))), flush=True)
    system = load_integrated(device=stable.DEVICE)
    relation = cap.load_head(128, torch.load(RELATION_BEST, map_location='cpu', weights_only=False)['model']).to(stable.DEVICE)
    row_model = NeuralRowPlacement().to(stable.DEVICE)
    row_model.load_state_dict(torch.load(ROW_BEST, map_location=stable.DEVICE, weights_only=False)['model'])
    for module in (system, relation, row_model):
        module.eval()
        module.requires_grad_(False)
    head = ObjectSizeRefinement().to(stable.DEVICE)
    phase_val = subset(val, 'validation', 94)
    multi_val = [record for record in val.records if multi_row(record)]
    single_val = [record for record in val.records if is_full(record) and len(record['supervision_v3']['rows']) == 1]
    val_indices = [index for index, record in enumerate(val.records) if multi_row(record)]
    if len(multi_val) != 152 or len(val_indices) != 152:
        raise RuntimeError('validation comparison set changed')
    phase_cache = stable.cache_oracle(system, phase_val)
    multi_stored = attach_rows(multi_val, cache_records(system, multi_val))
    single_stored = cache_records(system, single_val)
    oracle = stable.official_metrics(relation, phase_cache, phase_val)['task']
    single = score(relation, single_stored, single_val)['task']
    modular = evaluate(row_model, relation, multi_stored, multi_val, 'free')
    baseline_rgb = score_rgb(system, relation, row_model, head, val, val_indices)
    baseline_size = size_report(system, head, val, val_indices)
    dev_size = size_report(system, head, train, dev)
    print('BASE', json.dumps(dict(oracle=oracle, single=single, modular=modular['task'],
                                 rgb=baseline_rgb['task'], width=baseline_size['width'])), flush=True)
    if (oracle, single, modular['task'], baseline_rgb['task']) != (60, 62, 66, 24):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='baseline', oracle=oracle, single=single,
                                              modular=modular['task'], rgb=baseline_rgb['task'],
                                              checkpoints_unchanged=True))
        print('STOP baseline did not reproduce', flush=True)
        return
    initial = copy_state(head)
    check = step_check(system, relation, row_model, head, train, fit)
    print('STEP', json.dumps(check), flush=True)
    head.load_state_dict({key: value.to(stable.DEVICE) for key, value in initial.items()})
    if not check['passed'] or not same_state(head, initial):
        stable.write(ROOT/'result.json', dict(status='STOP', stage='step_check', step=check))
        print('STOP size step did not preserve detection and arrangement', flush=True)
        return
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.)
    save_checkpoint(ROOT/'checkpoint-0000.pt', head, optimizer, 0, dict(dev_width=dev_size['width']))
    history = (ROOT/'history.jsonl').open('w')
    history.write(json.dumps(dict(update=0, dev_width=dev_size['width'], dev_height=dev_size['height'],
                                 dev_footprint=dev_size['footprint']))+'\n')
    best = dict(update=0, width=dev_size['width']['median'], p95=dev_size['width']['p95'])
    save_checkpoint(ROOT/'checkpoint-best.pt', head, optimizer, 0, dict(dev_width=dev_size['width']))
    generator = torch.Generator().manual_seed(SEED)
    order, cursor = [], 0
    running, seen = 0., 0
    stale = 0
    reason = 'max_updates'
    update = 0
    started = time.perf_counter()
    while update < MAX_UPDATES:
        if cursor+BATCH > len(order):
            order = torch.randperm(len(fit), generator=generator).tolist()
            cursor = 0
        choice = [fit[order[index]] for index in range(cursor, cursor+BATCH)]
        cursor += BATCH
        images = torch.stack([train[index]['image'] for index in choice]).to(stable.DEVICE)
        truths = []
        for index in choice:
            _, truth = recovery_supervision(train.records[index])
            truths.append({key: value.to(stable.DEVICE) for key, value in truth.items()})
        head.train()
        with torch.no_grad():
            visual = system.perception(images)
        pairs, _ = correspond(visual['source'], truths, system.threshold)
        if not pairs:
            continue
        rgb, tokens, boxes, base = matched_queries(
            visual['stages'][-1]['tokens'], images, visual['stages'][-1]['boxes'], visual['source']['size'],
            [(batch, query) for batch, query, _ in pairs])
        target = torch.stack([truths[batch]['size'][gt] for batch, _, gt in pairs])
        optimizer.zero_grad(set_to_none=True)
        refined = head(tokens, rgb, boxes, base)[:, 0]
        loss = F.smooth_l1_loss(refined, target, beta=.001)
        if not torch.isfinite(loss):
            reason = 'diverged'
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        update += 1
        running += float(loss.detach())
        seen += 1
        if update % 25 == 0:
            print('LOSS', json.dumps(dict(update=update, loss=running/seen)), flush=True)
        if update % EVAL_EVERY:
            continue
        current = size_report(system, head, train, dev)
        point = dict(update=update, loss=running/seen, dev_width=current['width'],
                     dev_height=current['height'], dev_footprint=current['footprint'],
                     matched=current['matched'], seconds=time.perf_counter()-started)
        if update % RGB_EVERY == 0:
            dev_rgb = score_rgb(system, relation, row_model, head, train, dev_rgb_indices)
            point['dev_rgb'] = dev_rgb['task']
            point['dev_rgb_by'] = {count: dev_rgb['by_count'][count]['success'] for count in dev_rgb['by_count']}
        running, seen = 0., 0
        history.write(json.dumps(point)+'\n')
        history.flush()
        print('EVAL', json.dumps(point), flush=True)
        improved = (current['width']['median'] < best['width']-WIDTH_GAIN_MM
                    or (current['width']['median'] <= best['width'] and current['width']['p95'] < best['p95']-WIDTH_GAIN_MM))
        if improved:
            best = dict(update=update, width=current['width']['median'], p95=current['width']['p95'])
            stale = 0
            save_checkpoint(ROOT/'checkpoint-best.pt', head, optimizer, update, dict(dev_width=current['width']))
        else:
            stale += 1
        if update >= MIN_UPDATES and stale >= PATIENCE:
            reason = 'plateau'
            break
    history.close()
    save_checkpoint(ROOT/'checkpoint-last.pt', head, optimizer, update, dict(stop=reason))
    saved = torch.load(ROOT/'checkpoint-best.pt', map_location=stable.DEVICE, weights_only=False)
    head.load_state_dict(saved['model'])
    final_dev = size_report(system, head, train, dev)
    final_dev_rgb = score_rgb(system, relation, row_model, head, train, dev_rgb_indices)
    final_val_size = size_report(system, head, val, val_indices)
    final_rgb = score_rgb(system, relation, row_model, head, val, val_indices)
    final_oracle = stable.official_metrics(relation, phase_cache, phase_val)['task']
    final_single = score(relation, single_stored, single_val)['task']
    final_modular = evaluate(row_model, relation, multi_stored, multi_val, 'free')
    kept = baseline_rgb['successes'] & final_rgb['successes']
    size_better = (final_val_size['width']['median'] < baseline_size['width']['median']-.2
                   and final_val_size['width']['p95'] < baseline_size['width']['p95'])
    adopted = (size_better and final_rgb['task'] > baseline_rgb['task'] and saved['update'] > 0
               and (final_oracle, final_single, final_modular['task']) == (60, 62, 66))
    report = dict(status='PASS' if adopted else 'FAIL', stop=reason, best_update=saved['update'],
                  config=dict(lr=LR, batch=BATCH, eval_every=EVAL_EVERY, rgb_every=RGB_EVERY,
                              max_updates=MAX_UPDATES, min_updates=MIN_UPDATES, patience=PATIENCE,
                              seed=SEED, families=len(fit)+len(dev), fit=len(fit), dev=len(dev)),
                  baseline=dict(oracle=oracle, single=single, modular=modular['task'],
                                rgb=baseline_rgb['task'], rgb_by={count: baseline_rgb['by_count'][count]['success']
                                                                  for count in baseline_rgb['by_count']},
                                width=baseline_size['width'], height=baseline_size['height'],
                                footprint=baseline_size['footprint'],
                                horizontal=baseline_rgb['horizontal'], vertical=baseline_rgb['vertical'],
                                count_errors=baseline_rgb['count_errors'], row_match=baseline_rgb['row_match'],
                                row_accuracy=baseline_rgb['row_accuracy'], slot_accuracy=baseline_rgb['slot_accuracy']),
                  dev=dict(width=final_dev['width'], height=final_dev['height'], footprint=final_dev['footprint'],
                           rgb=final_dev_rgb['task'], total=final_dev_rgb['total'],
                           by_count=final_dev_rgb['by_count']),
                  validation=dict(width=final_val_size['width'], height=final_val_size['height'],
                                  footprint=final_val_size['footprint'], rgb=final_rgb['task'],
                                  rgb_by={count: final_rgb['by_count'][count]['success'] for count in final_rgb['by_count']},
                                  horizontal=final_rgb['horizontal'], vertical=final_rgb['vertical'],
                                  count_errors=final_rgb['count_errors'], row_match=final_rgb['row_match'],
                                  row_accuracy=final_rgb['row_accuracy'], slot_accuracy=final_rgb['slot_accuracy'],
                                  preserved_successes=len(kept), baseline_successes=len(baseline_rgb['successes'])),
                  preserved=dict(oracle=final_oracle, single=final_single, modular=final_modular['task']),
                  checkpoints_unchanged=sha256(ROW_BEST) == stamps['row'] and sha256(RELATION_BEST) == stamps['relation'],
                  test_used=False, gt_at_inference=False)
    # Sample ids are recorded for the preservation count, not written out.
    stable.write(ROOT/'result.json', report)
    print('DONE', json.dumps(dict(status=report['status'], stop=reason, update=saved['update'],
                                 rgb=final_rgb['task'], width=final_val_size['width'])), flush=True)


if __name__ == '__main__':
    main()
