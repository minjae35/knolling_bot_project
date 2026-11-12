"""Train the adopted selector on full arrangements.

The detector, language encoder, goal encoder, relation hidden, size head,
angle head, and pose weights stay frozen. Loss is teacher row and order
cross-entropy. Inference scoring still uses constrained decoding and the
geometry decoder. The adopted checkpoint is never overwritten.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'experiments' / 'end_to_end')]
if os.environ.get('CUDA_VISIBLE_DEVICES', None) == '':
    del os.environ['CUDA_VISIBLE_DEVICES']
os.environ.update(dict(
    HF_HOME='/workspace/cache/huggingface',
    HF_HUB_CACHE='/workspace/cache/huggingface/hub',
    HUGGINGFACE_HUB_CACHE='/workspace/cache/huggingface/hub',
    TRANSFORMERS_CACHE='/workspace/cache/huggingface/hub',
    TORCH_HOME='/workspace/cache/torch',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    CUBLAS_WORKSPACE_CONFIG=':4096:8'))

import torch
from knolling_dataset.schema import digest
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import SCHEMA, SEED, ExpandedDataset, partition
from knolling_model.experiments.full_arrangement import arrangement_errors, is_full
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.row_layout import assignment_loss, order_loss
from knolling_model.experiments.size_refinement import (
    encode_instruction, install_refined_size, poses_from_detection)
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
from knolling_model.recovery.selection import select_queries
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from dev_splits import fresh_split, heldout_validation
from score_end_to_end import DATA, SELECTOR_BEST, labels_of, load_adopted

OUT = Path('artifacts/experiments/selector_full')
ADOPTED = Path('artifacts/experiments/selector_only/checkpoint-best.pt')
BATCH = 32
LR = 1e-4
WEIGHT_DECAY = 0.01
MIN_EPOCHS = 8
MAX_EPOCHS = 20
PATIENCE = 5
HEADROOM = 0.02
SOURCE_KEYS = ('objectness_logits', 'uv', 'size', 'angle', 'color_logits', 'shape_logits')
ERROR_KEYS = (
    'mixed_color_row', 'size_order', 'horizontal_gap', 'vertical_gap', 'requested_region',
    'collision', 'clearance', 'boundary', 'count', 'row_alignment', 'axis_alignment',
    'source_identity', 'noncontiguous_color_group', 'alignment')


def official_family(index):
    return 'expanded_' + digest([SCHEMA, SEED, index])[:24]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def audit():
    """Count full training scenes and prove the family splits do not overlap."""
    print('AUDIT load train', flush=True)
    train = ExpandedDataset(DATA, 'train')
    print('AUDIT load validation', flush=True)
    validation = ExpandedDataset(DATA, 'validation')
    computed = {name: set() for name in ('train', 'validation', 'test')}
    for index in range(4000):
        computed[partition(index)].add(official_family(index))
    train_ids = {record['scene_family_id'] for record in train.records}
    val_ids = {record['scene_family_id'] for record in validation.records}
    if train_ids != computed['train'] or val_ids != computed['validation']:
        raise RuntimeError('official family ids do not match the partition')
    if train_ids & val_ids or train_ids & computed['test'] or val_ids & computed['test']:
        raise RuntimeError('official train, validation, and test families overlap')
    fit, dev, blocked = fresh_split(train.records)
    dev256 = heldout_validation(fit)
    families = {
        'dev512': {train.records[index]['scene_family_id'] for index in dev},
        'dev256': {train.records[index]['scene_family_id'] for index in dev256},
        'holdout640': set(blocked),
    }
    overlaps = {
        'dev512_dev256': sorted(families['dev512'] & families['dev256']),
        'dev512_holdout': sorted(families['dev512'] & families['holdout640']),
        'dev256_holdout': sorted(families['dev256'] & families['holdout640']),
    }
    excluded = families['dev512'] | families['dev256'] | families['holdout640']
    trainable = train_ids - excluded
    if trainable & val_ids or trainable & computed['test'] or trainable & excluded:
        raise RuntimeError('training families leak into a held-out split')
    full = [index for index, record in enumerate(train.records)
            if is_full(record) and record['scene_family_id'] in trainable]
    val_full = [index for index, record in enumerate(validation.records) if is_full(record)]
    if {validation.records[index]['scene_family_id'] for index in val_full} - val_ids:
        raise RuntimeError('validation full scenes left their split')
    counts = Counter(len(train.records[index]['source']['objects']) for index in full)
    report = dict(
        train_families=len(train_ids), validation_families=len(val_ids), test_families=len(computed['test']),
        dev512_families=len(families['dev512']), dev256_families=len(families['dev256']),
        holdout_families=len(families['holdout640']),
        excluded_overlaps={key: len(value) for key, value in overlaps.items()},
        training_families=len(trainable), full_training_samples=len(full),
        full_by_objects={str(count): counts[count] for count in range(2, 9)},
        official_validation_full=len(val_full),
        official_overlap=dict(train_validation=0, train_test=0, validation_test=0),
        test_scored=False)
    print('AUDIT', json.dumps(report), flush=True)
    return train, validation, report, full, val_full, dev, dev256


def configure(model, system, relation, head):
    system.requires_grad_(False)
    relation.requires_grad_(False)
    head.requires_grad_(False)
    model.requires_grad_(False)
    params = list(model.context.parameters()) + model.placement.selector_parameters()
    for param in params:
        param.requires_grad_(True)
    return params


def perceive(system, head, images):
    visual = system.perception(images)
    return install_refined_size(head, images, visual, system.threshold)


def decode_layout(system, relation, model, visual, texts, teacher_row, teacher_slot, search):
    queries = select_queries(visual['source'], system.threshold)
    if any(len(chosen) < 1 or len(chosen) > 8 for chosen in queries):
        raise ValueError('detection count is outside 1..8')
    objects = pack_sources(visual['source'], queries)
    text, mask = encode_instruction(system, texts, objects.xy.device)
    goal = system.layout.goal_encoder(objects, text, mask)['vector']
    hidden = relation._hidden(objects.features())
    output = model(
        hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
        teacher_row, teacher_slot, True, colors=objects.color)
    if search:
        output = select_layout_of(model, hidden, objects, goal, output)
    return apply_geometry_of(output, objects, goal), objects


def select_layout_of(model, hidden, objects, goal, output):
    from knolling_model.experiments.constraint_decode import select_layout
    return select_layout(model, hidden, objects, goal, output)


def apply_geometry_of(output, objects, goal):
    from knolling_model.experiments.geometry_decoder import apply_geometry
    return apply_geometry(output, objects, goal)


def align_teacher(xy, truth_xy, record, identities):
    """Place teacher row and slot in packed detection order. None if any object is unmatched."""
    if xy.shape[0] != truth_xy.shape[0] or xy.shape[0] == 0:
        return None
    row, slot = labels_of(record, identities)
    distance = torch.cdist(xy, truth_xy.to(device=xy.device, dtype=xy.dtype))
    used_query, used_gt = set(), set()
    teacher_row = torch.full((xy.shape[0],), -1, dtype=torch.long)
    teacher_slot = torch.full((xy.shape[0],), -1, dtype=torch.long)
    for item in distance.flatten().argsort().tolist():
        query, gt = divmod(item, distance.shape[1])
        if query in used_query or gt in used_gt or float(distance[query, gt]) > .010:
            continue
        used_query.add(query)
        used_gt.add(gt)
        teacher_row[query] = int(row[gt])
        teacher_slot[query] = int(slot[gt])
    if len(used_query) != xy.shape[0] or bool((teacher_row < 0).any()) or bool((teacher_slot < 0).any()):
        return None
    for row_id in teacher_row.unique().tolist():
        members = teacher_slot[teacher_row == row_id]
        if sorted(members.tolist()) != list(range(len(members))):
            return None
    return teacher_row, teacher_slot


def blank(record, errors, objects, seconds, geometry):
    return dict(sample_id=record['sample_id'], objects=objects, task=False, errors=list(errors),
                row_match=False, row_hits=0, slot_hits=0, labeled=0, seconds=seconds, geometry=geometry)


def judge_scene(output, objects, visual, record, identities, truth, seconds, threshold, detail=True):
    source = {key: value[0:1].detach().cpu() for key, value in visual['source'].items()}
    layout = dict(xy=output['xy'][0:1].detach().cpu(), angle=output['angle'][0:1].detach().cpu(),
                  valid=objects.valid[0:1].detach().cpu(), query_index=objects.query_index[0:1].detach().cpu(),
                  rollout='free_rollout')
    judged = evaluate_batch(
        legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5)),
        Batch({}, [truth], [dict(sample_id=record['sample_id'], object_ids=identities, record=record)]),
        semantic_check=semantic_errors_v3)[0]
    detail_row = arrangement_errors(record, judged['predicted_targets']) if detail and judged.get('predicted_targets') else None
    row_hits = slot_hits = labeled = 0
    if judged['count_correct']:
        row, slot = labels_of(record, identities)
        chosen = select_queries(visual['source'], threshold)[0]
        distance = torch.cdist(objects.xy[0, :len(chosen)], truth['source_xy'].to(objects.xy.device))
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
    geometry = '' if not output.get('geometry_status') else str(output['geometry_status'][0])
    return dict(sample_id=record['sample_id'], objects=len(identities),
                task=bool(accepted_success(record, judged['task_success'])), errors=list(judged['errors']),
                row_match=bool(detail_row and detail_row['row_match']), row_hits=row_hits, slot_hits=slot_hits,
                labeled=labeled, seconds=seconds, geometry=geometry)


def score_one(system, relation, model, head, data, index, teacher, detail=True):
    record = data.records[index]
    identities, truth = recovery_supervision(record)
    truth = {key: value.cpu() for key, value in truth.items()}
    image = data[index]['image'][None].to(stable.DEVICE)
    started = time.perf_counter()
    with torch.no_grad():
        try:
            visual = perceive(system, head, image)
            queries = select_queries(visual['source'], system.threshold)
            if len(queries[0]) < 1 or len(queries[0]) > 8:
                raise ValueError('detection count is outside 1..8')
            objects = pack_sources(visual['source'], queries)
            teacher_row = teacher_slot = None
            if teacher:
                aligned = align_teacher(objects.xy[0, :int(objects.valid[0].sum())], truth['source_xy'], record, identities)
                if aligned is None:
                    return blank(record, ['alignment'], len(identities), time.perf_counter() - started, '')
                teacher_row = aligned[0][None].to(stable.DEVICE)
                teacher_slot = aligned[1][None].to(stable.DEVICE)
            output, objects = decode_layout(
                system, relation, model, visual, [record['instruction']['text']], teacher_row, teacher_slot, not teacher)
        except ValueError:
            return blank(record, ['count'], len(identities), time.perf_counter() - started, '')
    return judge_scene(
        output, objects, visual, record, identities, truth, time.perf_counter() - started, system.threshold, detail)


def score_indices(system, relation, model, head, data, indices, teacher=False, detail=True):
    model.eval()
    scenes = []
    started = time.perf_counter()
    for done, index in enumerate(indices, start=1):
        scenes.append(score_one(system, relation, model, head, data, index, teacher, detail))
        if done % 100 == 0 or done == len(indices):
            print(f'SCORE teacher={teacher} {done}/{len(indices)} task={sum(item["task"] for item in scenes)} '
                  f'elapsed={time.perf_counter()-started:.0f}s', flush=True)
    return summarize(scenes), scenes


def summarize(scenes):
    labeled = sum(scene['labeled'] for scene in scenes)
    failures = Counter(name for scene in scenes if not scene['task'] for name in scene['errors'])
    by_count = {}
    for count in range(2, 9):
        chosen = [scene for scene in scenes if scene['objects'] == count]
        by_count[str(count)] = dict(success=sum(scene['task'] for scene in chosen), total=len(chosen))
    return dict(
        task=sum(scene['task'] for scene in scenes), total=len(scenes), by_count=by_count,
        failures={key: int(failures.get(key, 0)) for key in ERROR_KEYS},
        row_match=sum(scene['row_match'] for scene in scenes),
        row_accuracy=None if not labeled else sum(scene['row_hits'] for scene in scenes) / labeled,
        slot_accuracy=None if not labeled else sum(scene['slot_hits'] for scene in scenes) / labeled,
        labeled=labeled,
        horizontal_room=sum('horizontal_room' in scene['geometry'] for scene in scenes),
        vertical_room=sum('vertical_room' in scene['geometry'] for scene in scenes),
        successes=sorted(scene['sample_id'] for scene in scenes if scene['task']))


def public_summary(summary):
    return {key: value for key, value in summary.items() if key != 'successes'}


def stack_sources(cache, renders, device):
    source = {}
    for key in SOURCE_KEYS:
        source[key] = torch.cat([cache[render][key] for render in renders], 0).to(device)
    return source


def remember_source(cache, render, visual):
    if render not in cache:
        cache[render] = {key: visual['source'][key].detach().cpu() for key in SOURCE_KEYS}
    return cache


def prepare_samples(system, head, data, indices):
    cache, samples, skipped = {}, [], Counter()
    started = time.perf_counter()
    for done, index in enumerate(indices, start=1):
        record = data.records[index]
        render = record['source_render_dir']
        if render not in cache:
            image = data[index]['image'][None].to(stable.DEVICE)
            with torch.no_grad():
                visual = perceive(system, head, image)
            remember_source(cache, render, visual)
        source = stack_sources(cache, [render], stable.DEVICE)
        queries = select_queries(source, system.threshold)
        identities, truth = recovery_supervision(record)
        if len(queries[0]) < 2 or len(queries[0]) > 8:
            skipped['count'] += 1
        else:
            objects = pack_sources(source, queries)
            aligned = align_teacher(
                objects.xy[0, :int(objects.valid[0].sum())], truth['source_xy'], record, identities)
            if aligned is None:
                skipped['alignment'] += 1
            else:
                samples.append(dict(
                    index=index, render=render, text=record['instruction']['text'],
                    family=record['scene_family_id'], objects=len(identities),
                    teacher_row=aligned[0], teacher_slot=aligned[1]))
        if done % 200 == 0 or done == len(indices):
            print(f'CACHE {done}/{len(indices)} usable={len(samples)} skipped={dict(skipped)} '
                  f'elapsed={time.perf_counter()-started:.0f}s', flush=True)
    return cache, samples, dict(skipped)


def selector_forward(system, relation, model, cache, batch):
    source = stack_sources(cache, [item['render'] for item in batch], stable.DEVICE)
    queries = select_queries(source, system.threshold)
    if any(len(chosen) < 2 or len(chosen) > 8 for chosen in queries):
        raise RuntimeError('a cached training detection changed')
    objects = pack_sources(source, queries)
    text, mask = encode_instruction(system, [item['text'] for item in batch], stable.DEVICE)
    with torch.no_grad():
        goal = system.layout.goal_encoder(objects, text, mask)['vector'].detach()
        hidden = relation._hidden(objects.features()).detach()
    width = objects.valid.shape[1]
    teacher_row = torch.full((len(batch), width), -1, dtype=torch.long, device=stable.DEVICE)
    teacher_slot = torch.full((len(batch), width), -1, dtype=torch.long, device=stable.DEVICE)
    for index, item in enumerate(batch):
        count = int(objects.valid[index].sum())
        if item['teacher_row'].shape[0] != count:
            raise RuntimeError('cached teacher labels do not match the detection')
        teacher_row[index, :count] = item['teacher_row'].to(stable.DEVICE)
        teacher_slot[index, :count] = item['teacher_slot'].to(stable.DEVICE)
    output = model(
        hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
        teacher_row, teacher_slot, True, colors=objects.color.detach())
    row = assignment_loss(output, teacher_row, objects.valid)
    order = order_loss(output)
    active = objects.valid & (teacher_row >= 0)
    row_hits = (output['row_logits'].argmax(-1)[active] == teacher_row[active]).sum()
    targets = output['order_targets']
    order_active = targets >= 0
    order_hits = (output['order_logits'].argmax(-1)[order_active] == targets[order_active]).sum()
    return row, order, int(row_hits), int(active.sum()), int(order_hits), int(order_active.sum()), output['order_logits']


def smoke(system, relation, model, head, data, indices):
    """One cached batch: finite loss, selector gradients, and no frozen gradients."""
    print('SMOKE', flush=True)
    groups = {}
    for index in indices:
        groups.setdefault(data.records[index]['source_render_dir'], []).append(index)
    render = next(key for key, values in groups.items() if len(values) >= 2)
    seen = {'calls': 0}
    original = system.perception.forward

    def counted(*args, **kwargs):
        seen['calls'] += 1
        return original(*args, **kwargs)

    system.perception.forward = counted
    try:
        cache, samples, skipped = prepare_samples(system, head, data, groups[render][:2])
    finally:
        system.perception.forward = original
    if seen['calls'] != 1:
        raise RuntimeError(f'detector cache reused the same image {seen["calls"]} times')
    if len(samples) < 1:
        raise RuntimeError(f'smoke could not align a training scene: {skipped}')
    extra = [index for index in indices if data.records[index]['source_render_dir'] != render]
    more_cache, more_samples, _skipped = prepare_samples(system, head, data, extra)
    cache.update(more_cache)
    samples.extend(more_samples)
    multi = [item for item in samples if int(item['teacher_slot'].max()) >= 1]
    if not multi:
        raise RuntimeError('smoke found no row with two or more objects')
    chosen = multi[:2] + [item for item in samples if item not in multi][:2]
    params = configure(model, system, relation, head)
    model.train()
    system.eval()
    relation.eval()
    head.eval()
    row, order, row_hits, row_count, order_hits, order_count, order_logits = selector_forward(
        system, relation, model, cache, chosen)
    loss = row + order
    if not torch.isfinite(loss) or float(row.detach()) <= 0:
        raise RuntimeError(f'selector loss is not usable: row={float(row.detach())} order={float(order.detach())}')
    wiring = torch.autograd.grad(
        order_logits.sum(), model.placement.order_query[-1].weight, retain_graph=True, allow_unused=True)[0]
    if wiring is None or float(wiring.detach().abs().sum()) == 0:
        raise RuntimeError('order loss is not connected to the order head')
    loss.backward()
    watched = {
        'context': list(model.context.parameters()),
        'assign': list(model.placement.assign.parameters()),
        'order_query': list(model.placement.order_query.parameters()),
        'order_begin': [model.placement.order_begin],
    }
    gradients = {}
    for name, group in watched.items():
        total = 0.
        for param in group:
            if param.grad is None:
                raise RuntimeError(f'{name} did not receive a gradient')
            total += float(param.grad.detach().abs().sum())
        if name in ('context', 'assign') and total < 1e-8:
            raise RuntimeError(f'{name} gradient is zero')
        gradients[name] = total
    frozen = [model.placement.row_layout, model.placement.step, model.placement.angle,
              model.placement.row_anchor, model.placement.row_gap, relation, head,
              system.layout.goal_encoder, system.perception]
    for module in frozen:
        for param in module.parameters():
            if param.requires_grad or param.grad is not None:
                raise RuntimeError(f'a frozen module received a gradient: {type(module).__name__}')
    trainable = set(id(param) for param in params)
    unexpected = [name for name, param in model.named_parameters() if param.requires_grad and id(param) not in trainable]
    if unexpected:
        raise RuntimeError(f'unexpected trainable parameter: {unexpected}')
    model.zero_grad(set_to_none=True)
    model.eval()
    report = dict(loss=float(loss.detach()), row=float(row.detach()), order=float(order.detach()),
                  gradients=gradients, trainable=sum(param.numel() for param in params),
                  detector_calls=seen['calls'], samples=len(chosen),
                  row_accuracy=row_hits / row_count, order_accuracy=order_hits / order_count)
    print('SMOKE', json.dumps(report), flush=True)
    return report


def agreement(system, relation, model, head, data, indices):
    """Batched layout must match the adopted single-scene path before any score is trusted."""
    print('AGREEMENT', len(indices), flush=True)
    with torch.no_grad():
        _agreement_body(system, relation, model, head, data, indices)
    print('AGREEMENT ok', flush=True)


def _agreement_body(system, relation, model, head, data, indices):
    compared = 0
    for index in indices:
        record = data.records[index]
        image = data[index]['image'][None].to(stable.DEVICE)
        try:
            adopted, _objects, _visual = poses_from_detection(
                system, relation, model, image, [record['instruction']['text']], head)
            visual = perceive(system, head, image)
            output, _objects = decode_layout(
                system, relation, model, visual, [record['instruction']['text']], None, None, True)
        except ValueError:
            continue
        if not torch.allclose(adopted['xy'], output['xy'], atol=1e-5, rtol=0):
            raise RuntimeError('batched layout does not match poses_from_detection')
        if adopted.get('geometry_status') != output.get('geometry_status'):
            raise RuntimeError('geometry status does not match the adopted path')
        compared += 1
        if compared >= 4:
            break
    if compared < 4:
        raise RuntimeError('agreement did not find four detectable scenes')


def state_of(model):
    return dict(
        context={key: value.detach().cpu().clone() for key, value in model.context.state_dict().items()},
        placement={key: value.detach().cpu().clone() for key, value in model.placement.state_dict().items()})


def restore(model, state):
    model.context.load_state_dict(state['context'])
    model.placement.load_state_dict(state['placement'])
    model.eval()


def same_state(left, right):
    for part in ('context', 'placement'):
        if left[part].keys() != right[part].keys():
            return False
        for key, value in left[part].items():
            if not torch.equal(value, right[part][key]):
                return False
    return True


def save_checkpoint(path, model, epoch, updates, val_task):
    if path.resolve() == ADOPTED.resolve():
        raise RuntimeError('refusing to overwrite the adopted checkpoint')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(context=model.context.state_dict(), placement=model.placement.state_dict(),
                    epoch=epoch, update=updates, val_task=val_task, adopted_overwritten=False), path)


def set_lr(optimizer, step, warmup, total):
    if step < warmup:
        scale = (step + 1) / warmup
    else:
        progress = (step - warmup) / max(1, total - warmup)
        scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.)))
    for group in optimizer.param_groups:
        group['lr'] = LR * scale
    return LR * scale


def train_epoch(system, relation, model, optimizer, cache, samples, generator, step, warmup, total):
    model.train()
    system.eval()
    relation.eval()
    order = torch.randperm(len(samples), generator=generator).tolist()
    totals = Counter()
    for start in range(0, len(order), BATCH):
        batch = [samples[index] for index in order[start:start + BATCH]]
        lr = set_lr(optimizer, step, warmup, total)
        optimizer.zero_grad(set_to_none=True)
        row, order_loss_value, row_hits, row_count, order_hits, order_count, _order_logits = selector_forward(
            system, relation, model, cache, batch)
        loss = row + order_loss_value
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite loss at update {step}')
        loss.backward()
        optimizer.step()
        step += 1
        totals['loss'] += float(loss) * len(batch)
        totals['row'] += float(row) * len(batch)
        totals['order'] += float(order_loss_value) * len(batch)
        totals['rows'] += row_hits
        totals['row_objects'] += row_count
        totals['orders'] += order_hits
        totals['order_objects'] += order_count
        totals['scenes'] += len(batch)
    model.eval()
    scenes = totals['scenes']
    return step, lr, dict(
        loss=totals['loss'] / scenes, row=totals['row'] / scenes, order=totals['order'] / scenes,
        row_accuracy=totals['rows'] / totals['row_objects'],
        order_accuracy=totals['orders'] / totals['order_objects'])


def score_gt(system, relation, model, records):
    model.eval()
    scenes = []
    started = time.perf_counter()
    for start in range(0, len(records), 8):
        chunk = records[start:start + 8]
        objects, text, mask, truths = runner.arguments(system, chunk, stable.DEVICE)
        with torch.no_grad():
            goal = system.layout.goal_encoder(objects, text, mask)['vector']
            hidden = relation._hidden(objects.features())
            output = model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                           None, None, True, colors=objects.color)
            output = select_layout_of(model, hidden, objects, goal, output)
            output = apply_geometry_of(output, objects, goal)
        for local, record in enumerate(chunk):
            identities, _truth = recovery_supervision(record)
            count = len(identities)
            if int(objects.valid[local].sum()) != count:
                scenes.append(blank(record, ['count'], count, 0, ''))
                continue
            one_objects = pack_one(objects, local, count)
            layout = dict(xy=output['xy'][local:local + 1, :count].detach().cpu(),
                          angle=output['angle'][local:local + 1, :count].detach().cpu(),
                          valid=one_objects.valid, query_index=one_objects.query_index, rollout='free_rollout')
            truth = {key: value.cpu() for key, value in truths[local].items()}
            judged = evaluate_batch(
                _oracle_view(one_objects, layout),
                Batch({}, [truth], [dict(sample_id=record['sample_id'], object_ids=identities, record=record)]),
                semantic_check=semantic_errors_v3)[0]
            row, slot = labels_of(record, identities)
            predicted_row = output['row'][local, :count].detach().cpu()
            predicted_slot = output['slot'][local, :count].detach().cpu()
            geometry = '' if not output.get('geometry_status') else str(output['geometry_status'][local])
            detail = arrangement_errors(record, judged['predicted_targets']) if judged.get('predicted_targets') else None
            scenes.append(dict(
                sample_id=record['sample_id'], objects=count,
                task=bool(accepted_success(record, judged['task_success'])), errors=list(judged['errors']),
                row_match=bool(detail and detail['row_match']), row_hits=int((predicted_row == row).sum()),
                slot_hits=int((predicted_slot == slot).sum()), labeled=count, seconds=0, geometry=geometry))
        if len(scenes) % 200 == 0 or len(scenes) == len(records):
            print(f'GT {len(scenes)}/{len(records)} task={sum(item["task"] for item in scenes)} '
                  f'elapsed={time.perf_counter()-started:.0f}s', flush=True)
    return summarize(scenes)


def pack_one(objects, local, count):
    from knolling_model.recovery.contracts import Objects
    return Objects(
        objects.xy[local:local + 1, :count].detach().cpu(), objects.size[local:local + 1, :count].detach().cpu(),
        objects.angle[local:local + 1, :count].detach().cpu(), objects.color[local:local + 1, :count].detach().cpu(),
        objects.shape[local:local + 1, :count].detach().cpu(), objects.valid[local:local + 1, :count].detach().cpu(),
        objects.query_index[local:local + 1, :count].detach().cpu())


def compare_sets(baseline, candidate):
    before = set(baseline)
    after = set(candidate)
    return dict(preserved=len(before & after), lost=len(before - after), gained=len(after - before),
                baseline_count=len(before), candidate_count=len(after))


def scaling_verdict(curve, baseline_task, best_task):
    if len(curve) < MIN_EPOCHS:
        return 'incomplete', 'training stopped before the minimum epoch'
    train = [row['train_row_accuracy'] for row in curve]
    val = [row['val_task'] for row in curve]
    last = curve[-1]
    best_epoch = max(range(len(curve)), key=lambda index: (val[index], -index))
    train_up = train[-1] > train[0] + .01
    val_up = best_task > baseline_task
    flat_train = max(train[-5:]) - min(train[-5:]) < .01
    flat_val = max(val[-5:]) - min(val[-5:]) <= 1
    if train[-1] < .70 and flat_train and flat_val and best_task - baseline_task < 0.02 * last['val_total']:
        return 'bottleneck', 'train and validation stayed low and flat'
    if train_up and val[best_epoch] > val[-1] and last['epoch'] - curve[best_epoch]['epoch'] >= 2:
        return 'overfit', 'training accuracy kept rising after validation peaked'
    if curve[best_epoch]['epoch'] >= last['epoch'] - 1 and train_up and val_up:
        return 'more_training', 'training and validation were still improving at the stop'
    if val_up:
        return 'plateau', 'validation improved and then stopped rising'
    return 'no_gain', 'validation did not improve'


def finish(system, relation, model, head, train, validation, families, val_indices, dev512, dev256):
    """Score the saved best checkpoint. Training is not repeated."""
    preflight = json.loads((OUT / 'preflight.json').read_text())
    current = preflight['current']
    teacher = preflight['teacher']
    curve = [json.loads(line) for line in (OUT / 'curve.jsonl').read_text().splitlines() if line.strip()]
    wall = None
    for line in (OUT / 'train.log').read_text().splitlines():
        if line.startswith('TRAIN WALL'):
            wall = float(line.split()[2][:-1])
    saved = torch.load(OUT / 'checkpoint-best.pt', map_location='cpu', weights_only=False)
    print('DEV baseline', flush=True)
    dev_base_512, _ = score_indices(system, relation, model, head, train, dev512, teacher=False)
    dev_base_256, _ = score_indices(system, relation, model, head, train, dev256, teacher=False)
    if dev_base_512['task'] != 146 or dev_base_256['task'] != 79:
        raise RuntimeError('baseline dev score changed')
    print('GT baseline', flush=True)
    gt_base = score_gt(system, relation, model, [validation.records[index] for index in val_indices])
    model.context.load_state_dict(saved['context'])
    model.placement.load_state_dict(saved['placement'])
    model.eval()
    print('DEV best', flush=True)
    dev_best_512, _ = score_indices(system, relation, model, head, train, dev512, teacher=False)
    dev_best_256, _ = score_indices(system, relation, model, head, train, dev256, teacher=False)
    print('GT best', flush=True)
    gt_best = score_gt(system, relation, model, [validation.records[index] for index in val_indices])
    final_summary, _scenes = score_indices(system, relation, model, head, validation, val_indices, teacher=False)
    if final_summary['task'] != saved['val_task']:
        raise RuntimeError('reloaded best checkpoint does not match its validation score')
    best_task = final_summary['task']
    adopted = best_task > current['task']
    verdict, reason = scaling_verdict(curve, current['task'], best_task)
    result = dict(
        training_families=families['training_families'], full_training_samples=families['full_training_samples'],
        usable_samples=5085, skipped=dict(alignment=249, count=42), epochs=curve[-1]['epoch'], updates=curve[-1]['update'],
        gpu_seconds=wall, curve=curve, baseline_val=current, teacher_val=teacher,
        best_val=public_summary(final_summary), best_epoch=saved['epoch'], adopted=adopted,
        checkpoint=None if not adopted else str(OUT / 'checkpoint-best.pt'),
        dev512=dict(baseline=dev_base_512['task'], best=dev_best_512['task'], total=len(dev512),
                    **compare_sets(dev_base_512['successes'], dev_best_512['successes'])),
        dev256=dict(baseline=dev_base_256['task'], best=dev_best_256['task'], total=len(dev256),
                    **compare_sets(dev_base_256['successes'], dev_best_256['successes'])),
        gt_object=dict(baseline=public_summary(gt_base), best=public_summary(gt_best)),
        scaling=dict(verdict=verdict, reason=reason), test_scored=False,
        adopted_checkpoint_overwritten=False)
    write_json(OUT / 'result.json', result)
    print('RESULT', json.dumps(dict(adopted=adopted, best_epoch=saved['epoch'], best=best_task,
                                    baseline=current['task'], verdict=verdict,
                                    dev512=result['dev512'], dev256=result['dev256'],
                                    gt_base=gt_base['task'], gt_best=gt_best['task'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('debug', 'all', 'finish'), default='all')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    runner.setup()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for selector training')
    train, validation, families, full_indices, val_indices, dev512, dev256 = audit()
    agreement_indices = val_indices[:40]
    if args.limit:
        val_indices = val_indices[:args.limit]
    print('LOAD adopted', flush=True)
    system, relation, model, head, _saved = load_adopted()
    configure(model, system, relation, head)
    if args.stage == 'finish':
        finish(system, relation, model, head, train, validation, families, val_indices, dev512, dev256)
        return
    agreement(system, relation, model, head, validation, agreement_indices)
    smoke_report = smoke(system, relation, model, head, train, full_indices[:40])
    print('CEILING current', flush=True)
    current, _current_scenes = score_indices(system, relation, model, head, validation, val_indices, teacher=False)
    print('CEILING teacher', flush=True)
    teacher, _teacher_scenes = score_indices(system, relation, model, head, validation, val_indices, teacher=True)
    if not teacher['labeled'] or (teacher['row_accuracy'] or 0) < .99:
        raise RuntimeError('teacher rows were not actually used')
    gain = teacher['task'] - current['task']
    passed = gain >= HEADROOM * max(current['total'], 1)
    preflight = dict(families=families, smoke=smoke_report, current=public_summary(current),
                     teacher=public_summary(teacher), gain=gain, headroom=HEADROOM, passed=passed,
                     test_scored=False)
    write_json(OUT / ('preflight-debug.json' if args.stage == 'debug' or args.limit else 'preflight.json'), preflight)
    print('GATE', json.dumps(dict(current=current['task'], teacher=teacher['task'], total=current['total'],
                                  gain=gain, passed=passed)), flush=True)
    if args.stage == 'debug' or not passed:
        print('STOP before training', flush=True)
        return
    print('CACHE training scenes', flush=True)
    cache, samples, skipped = prepare_samples(system, head, train, full_indices)
    if len(samples) < BATCH:
        raise RuntimeError('not enough aligned full scenes to train')
    probe = torch.randperm(len(samples), generator=torch.Generator().manual_seed(7)).tolist()[:256]
    probe_indices = [samples[index]['index'] for index in probe]
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY, foreach=False)
    steps_per_epoch = math.ceil(len(samples) / BATCH)
    warmup = steps_per_epoch
    total_steps = steps_per_epoch * MAX_EPOCHS
    generator = torch.Generator().manual_seed(20261111)
    baseline_state = state_of(model)
    best_state = baseline_state
    best_task = current['task']
    best_summary = current
    best_epoch = 0
    step = 0
    curve = []
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    curve_path = OUT / 'curve.jsonl'
    curve_path.write_text('')
    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            step, lr, train_metrics = train_epoch(
                system, relation, model, optimizer, cache, samples, generator, step, warmup, total_steps)
            val_summary, _scenes = score_indices(
                system, relation, model, head, validation, val_indices, teacher=False, detail=False)
            probe_summary, _probe_scenes = score_indices(
                system, relation, model, head, train, probe_indices, teacher=False, detail=False)
            row = dict(epoch=epoch, update=step, lr=lr, **{f'train_{key}': value for key, value in train_metrics.items()},
                       val_task=val_summary['task'], val_total=val_summary['total'],
                       val_row_accuracy=val_summary['row_accuracy'], val_slot_accuracy=val_summary['slot_accuracy'],
                       train_probe_task=probe_summary['task'], train_probe_total=probe_summary['total'])
            curve.append(row)
            with curve_path.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            print('EPOCH', json.dumps(row), flush=True)
            if val_summary['task'] > best_task:
                best_task = val_summary['task']
                best_epoch = epoch
                best_state = state_of(model)
                best_summary = val_summary
                save_checkpoint(OUT / 'checkpoint-best.pt', model, epoch, step, best_task)
            save_checkpoint(OUT / 'checkpoint-last.pt', model, epoch, step, val_summary['task'])
            if epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
                print(f'EARLY STOP epoch={epoch} best_epoch={best_epoch}', flush=True)
                break
    finally:
        elapsed = time.perf_counter() - started
        print(f'TRAIN WALL {elapsed:.0f}s updates={step}', flush=True)
    adopted = best_task > current['task']
    print('DEV baseline', flush=True)
    restore(model, baseline_state)
    dev_base_512, _ = score_indices(system, relation, model, head, train, dev512, teacher=False)
    dev_base_256, _ = score_indices(system, relation, model, head, train, dev256, teacher=False)
    print('GT baseline', flush=True)
    gt_base = score_gt(system, relation, model, [validation.records[index] for index in val_indices])
    probe_base, _ = score_indices(system, relation, model, head, train, probe_indices, teacher=False)
    if same_state(best_state, baseline_state):
        dev_best_512, dev_best_256, gt_best, probe_best = dev_base_512, dev_base_256, gt_base, probe_base
        final_summary = current
    else:
        restore(model, best_state)
        print('DEV best', flush=True)
        dev_best_512, _ = score_indices(system, relation, model, head, train, dev512, teacher=False)
        dev_best_256, _ = score_indices(system, relation, model, head, train, dev256, teacher=False)
        print('GT best', flush=True)
        gt_best = score_gt(system, relation, model, [validation.records[index] for index in val_indices])
        probe_best, _ = score_indices(system, relation, model, head, train, probe_indices, teacher=False)
        final_summary, _scenes = score_indices(system, relation, model, head, validation, val_indices, teacher=False)
        if final_summary['task'] != best_task:
            raise RuntimeError('reloaded best checkpoint does not match its validation score')
    verdict, reason = scaling_verdict(curve, current['task'], best_task)
    result = dict(
        training_families=families['training_families'], full_training_samples=families['full_training_samples'],
        usable_samples=len(samples), skipped=skipped, epochs=curve[-1]['epoch'] if curve else 0, updates=step,
        gpu_seconds=elapsed, curve=curve, baseline_val=public_summary(current), teacher_val=public_summary(teacher),
        best_val=public_summary(final_summary), best_epoch=best_epoch, adopted=adopted,
        checkpoint=None if not adopted else str(OUT / 'checkpoint-best.pt'),
        dev512=dict(baseline=dev_base_512['task'], best=dev_best_512['task'], total=len(dev512),
                    **compare_sets(dev_base_512['successes'], dev_best_512['successes'])),
        dev256=dict(baseline=dev_base_256['task'], best=dev_best_256['task'], total=len(dev256),
                    **compare_sets(dev_base_256['successes'], dev_best_256['successes'])),
        gt_object=dict(baseline=public_summary(gt_base), best=public_summary(gt_best)),
        train_probe=dict(baseline=probe_base['task'], best=probe_best['task'], total=probe_base['total']),
        scaling=dict(verdict=verdict, reason=reason), test_scored=False,
        adopted_checkpoint_overwritten=False)
    write_json(OUT / 'result.json', result)
    print('RESULT', json.dumps(dict(adopted=adopted, best_epoch=best_epoch, best=best_task,
                                    baseline=current['task'], verdict=verdict)), flush=True)


if __name__ == '__main__':
    main()
