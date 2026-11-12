"""Train only the source-angle direction head, then only the selection linear layer.

The detector, selector, and geometry decoder stay frozen. Test is not opened.
A reserved family slice is excluded from both updates.
"""
import math
import os
import sys
from collections import defaultdict
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
from torch.nn import functional as F
import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_model.coordinates import angle_delta
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import is_full
from knolling_model.experiments.size_refinement import pack_sources, poses_from_detection
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.selection import select_queries
from dev_splits import family_splits, fresh_split, heldout_validation
from score_end_to_end import DATA
from train_selector_full import judge_scene, load_adopted

ANGLE_PATH = Path('artifacts/experiments/selector_full/direction-head.pt')
SELECTION_PATH = Path('artifacts/experiments/selector_full/selection-head.pt')
RESERVED_SEED = 44021
RESERVED_COUNT = 128
MATCH_M = 0.010


def families(records):
    _fit, _dev, hold = family_splits(records)
    _fit2, dev512, _blocked = fresh_split(records)
    dev256 = heldout_validation(_fit2)
    blocked = {records[index]['scene_family_id'] for index in hold + dev512 + dev256}
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record['scene_family_id']].append(index)
    allowed = sorted(set(grouped) - blocked)
    order = torch.randperm(len(allowed), generator=torch.Generator().manual_seed(RESERVED_SEED)).tolist()
    reserved = {allowed[index] for index in order[:RESERVED_COUNT]}
    trainable = [name for name in allowed if name not in reserved]
    return grouped, trainable, reserved, blocked


def _pairs(xy, truth):
    distance = torch.cdist(xy.detach(), truth['source_xy'].to(xy.device))
    used_query, used_gt, pairs = set(), set(), []
    for item in distance.flatten().argsort().tolist():
        query, gt = divmod(item, distance.shape[1])
        if query in used_query or gt in used_gt or float(distance[query, gt]) > MATCH_M:
            continue
        used_query.add(query)
        used_gt.add(gt)
        pairs.append((query, gt))
    return pairs


def direction_angle(module, tokens):
    raw = module(tokens)
    near_zero = raw.square().sum(-1) < 1e-12
    horizontal = torch.where(near_zero, raw[..., 0] + 1e-6, raw[..., 0])
    return .5 * torch.atan2(raw[..., 1], horizontal)


def angle_errors(system, data, indices):
    errors = []
    for index in indices:
        record = data.records[index]
        identities, truth = recovery_supervision(record)
        image = data[index]['image'][None].to(stable.DEVICE)
        with torch.no_grad():
            visual = system.perception(image)
        chosen = select_queries(visual['source'], system.threshold)
        if not chosen or not (1 <= len(chosen[0]) <= 8):
            continue
        objects = pack_sources(visual['source'], chosen)
        count = int(objects.valid[0].sum())
        for query, gt in _pairs(objects.xy[0, :count], truth):
            errors.append(float(angle_delta(
                objects.angle[0, query], truth['source_angle'][gt].to(stable.DEVICE)).abs()) * 180 / math.pi)
    return errors


def full_probe(system, relation, model, head, data, indices):
    passed = 0
    for index in indices:
        record = data.records[index]
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
        image = data[index]['image'][None].to(stable.DEVICE)
        try:
            output, objects, visual = poses_from_detection(
                system, relation, model, image, [record['instruction']['text']], head)
        except ValueError:
            continue
        passed += int(judge_scene(
            output, objects, visual, record, identities, truth, 0, system.threshold, detail=False)['task'])
    return passed


def train_direction(system, data, grouped, trainable):
    module = system.perception.physical.direction
    snapshot = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=0)
    system.eval()
    losses, seen = [], 0
    for name in trainable:
        index = grouped[name][0]
        record = data.records[index]
        _identities, truth = recovery_supervision(record)
        image = data[index]['image'][None].to(stable.DEVICE)
        with torch.no_grad():
            visual = system.perception(image)
        chosen = select_queries(visual['source'], system.threshold)
        if not chosen or not (1 <= len(chosen[0]) <= 8):
            continue
        tokens = visual['stages'][-1]['tokens'][0, chosen[0]]
        angle = direction_angle(module, tokens)
        xy = visual['source']['uv'][0, chosen[0]] * angle.new_tensor([.6, .4])
        pairs = _pairs(xy, truth)
        if not pairs:
            continue
        predicted = angle[[query for query, _gt in pairs]]
        target = truth['source_angle'].to(angle.device)[[gt for _query, gt in pairs]]
        delta = angle_delta(predicted, target) / (math.pi / 180)
        loss = F.smooth_l1_loss(delta, torch.zeros_like(delta))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        seen += 1
        if seen % 400 == 0:
            print('direction', seen, 'loss', round(sum(losses[-400:]) / 400, 4), flush=True)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return snapshot, seen, (sum(losses) / len(losses) if losses else None)


def selection_labels(record, identities, pairs, count, device):
    instruction = record['instruction']
    moving = set() if instruction['keep_all'] else set(instruction['selected_ids'])
    labels = torch.zeros(count, device=device)
    for query, gt in pairs:
        labels[query] = 0. if instruction['keep_all'] or identities[gt] not in moving else 1.
    return labels


def selection_exact(system, data, indices):
    correct = scored = 0
    for index in indices:
        record = data.records[index]
        identities, truth = recovery_supervision(record)
        image = data[index]['image'][None].to(stable.DEVICE)
        with torch.no_grad():
            visual = system.perception(image)
            from knolling_model.experiments.size_refinement import encode_instruction
            text, mask = encode_instruction(system, [record['instruction']['text']], image.device)
            chosen = select_queries(visual['source'], system.threshold)
            if not chosen or not (1 <= len(chosen[0]) <= 8):
                continue
            objects = pack_sources(visual['source'], chosen)
            encoded = system.layout.goal_encoder(objects, text, mask)
        count = int(objects.valid[0].sum())
        pairs = _pairs(objects.xy[0, :count], truth)
        if len(pairs) != count or len(pairs) != len(identities):
            continue
        labels = selection_labels(record, identities, pairs, count, encoded['selected'].device)
        predicted = encoded['selected'][0, :count] >= 0.5
        correct += int(bool(torch.equal(predicted, labels.bool())))
        scored += 1
    return correct, scored


def train_selection(system, data, grouped, trainable):
    layer = system.layout.goal_encoder.selection
    snapshot = {key: value.detach().cpu().clone() for key, value in layer.state_dict().items()}
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    for parameter in layer.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3, weight_decay=0)
    system.eval()
    from knolling_model.experiments.size_refinement import encode_instruction
    losses, seen = [], 0
    picks = []
    for name in trainable:
        rows = grouped[name]
        picks.append(rows[2])
        if len(picks) % 3 == 0:
            picks.extend((rows[0], rows[3]))
    for index in picks:
        record = data.records[index]
        identities, truth = recovery_supervision(record)
        image = data[index]['image'][None].to(stable.DEVICE)
        with torch.no_grad():
            visual = system.perception(image)
            text, mask = encode_instruction(system, [record['instruction']['text']], image.device)
        chosen = select_queries(visual['source'], system.threshold)
        if not chosen or not (1 <= len(chosen[0]) <= 8):
            continue
        objects = pack_sources(visual['source'], chosen)
        encoded = system.layout.goal_encoder(objects, text, mask)
        count = int(objects.valid[0].sum())
        pairs = _pairs(objects.xy[0, :count].detach(), truth)
        if len(pairs) != count:
            continue
        labels = selection_labels(record, identities, pairs, count, encoded['selected_logit'].device)
        logit = encoded['selected_logit'][0, :count]
        loss = F.binary_cross_entropy_with_logits(logit, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        seen += 1
        if seen % 400 == 0:
            print('selection', seen, 'loss', round(sum(losses[-400:]) / 400, 4), flush=True)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    return snapshot, seen, (sum(losses) / len(losses) if losses else None)


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main():
    runner.setup()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    train = ExpandedDataset(DATA, 'train')
    validation = ExpandedDataset(DATA, 'validation')
    grouped, trainable, reserved, blocked = families(train.records)
    val_families = {record['scene_family_id'] for record in validation.records}
    if blocked & val_families or reserved & blocked or set(trainable) & blocked:
        raise RuntimeError('training families overlap a held-out split')
    noop = [index for index, record in enumerate(validation.records) if record['mode'] == 'no_op'][:80]
    partial = [index for index, record in enumerate(validation.records) if record['mode'] == 'partial'][:80]
    full = [index for index, record in enumerate(validation.records) if is_full(record)][:64]
    system, relation, model, head, _saved = load_adopted()
    before_angle = angle_errors(system, validation, noop)
    before_full = full_probe(system, relation, model, head, validation, full)
    before_selection = selection_exact(system, validation, partial)
    print('before', dict(
        angle_median=round(_median(before_angle), 3), angle_n=len(before_angle),
        within_1deg=sum(value <= 1 for value in before_angle),
        full=before_full, full_n=len(full), selection=before_selection), flush=True)
    angle_snapshot, angle_seen, angle_loss = train_direction(system, train, grouped, trainable)
    after_angle = angle_errors(system, validation, noop)
    after_full = full_probe(system, relation, model, head, validation, full)
    angle_gain = _median(before_angle) - _median(after_angle)
    print('direction_epoch', dict(
        seen=angle_seen, loss=None if angle_loss is None else round(angle_loss, 4),
        angle_median=round(_median(after_angle), 3),
        within_1deg=sum(value <= 1 for value in after_angle),
        full=after_full, gain=round(angle_gain, 3)), flush=True)
    keep_angle = angle_gain >= 0.15 and after_full >= before_full - 1
    if keep_angle:
        ANGLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'direction': system.perception.physical.direction.state_dict()}, ANGLE_PATH)
        print('kept direction', ANGLE_PATH, flush=True)
    else:
        system.perception.physical.direction.load_state_dict(angle_snapshot)
        print('reverted direction', flush=True)
    selection_snapshot, selection_seen, selection_loss = train_selection(system, train, grouped, trainable)
    after_selection = selection_exact(system, validation, partial)
    after_full_selection = full_probe(system, relation, model, head, validation, full)
    print('selection_epoch', dict(
        seen=selection_seen, loss=None if selection_loss is None else round(selection_loss, 4),
        selection=after_selection, full=after_full_selection), flush=True)
    if after_selection[0] >= before_selection[0] + 8 and after_full_selection >= before_full - 1:
        SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'selection': system.layout.goal_encoder.selection.state_dict()}, SELECTION_PATH)
        print('kept selection', SELECTION_PATH, flush=True)
    else:
        system.layout.goal_encoder.selection.load_state_dict(selection_snapshot)
        print('reverted selection', flush=True)


if __name__ == '__main__':
    main()
