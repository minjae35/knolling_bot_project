"""One conservative pass on the direction head. Stop if validation angle gets worse."""
import math
import os
import sys
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
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.selection import select_queries
from score_end_to_end import DATA
from train_angle_selection import angle_errors, direction_angle, families, full_probe
from train_selector_full import load_adopted

ANGLE_PATH = Path('artifacts/experiments/selector_full/direction-head.pt')


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main():
    runner.setup()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    train = ExpandedDataset(DATA, 'train')
    validation = ExpandedDataset(DATA, 'validation')
    grouped, trainable, _reserved, _blocked = families(train.records)
    noop = [index for index, record in enumerate(validation.records) if record['mode'] == 'no_op'][:80]
    full = [index for index, record in enumerate(validation.records) if is_full(record)][:64]
    system, relation, model, head, _saved = load_adopted()
    module = system.perception.physical.direction
    snapshot = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    start = angle_errors(system, validation, noop)
    start_full = full_probe(system, relation, model, head, validation, full)
    start_median = _median(start)
    print('start', dict(median=round(start_median, 3), within=sum(v <= 1 for v in start), full=start_full), flush=True)
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-5, weight_decay=0)
    system.eval()
    seen = 0
    best = start_median
    for name in trainable:
        index = grouped[name][0]
        record = train.records[index]
        _identities, truth = recovery_supervision(record)
        image = train[index]['image'][None].to(stable.DEVICE)
        with torch.no_grad():
            visual = system.perception(image)
        chosen = select_queries(visual['source'], system.threshold)
        if not chosen or not (1 <= len(chosen[0]) <= 8):
            continue
        tokens = visual['stages'][-1]['tokens'][0, chosen[0]]
        angle = direction_angle(module, tokens)
        xy = visual['source']['uv'][0, chosen[0]] * angle.new_tensor([.6, .4])
        distance = torch.cdist(xy.detach(), truth['source_xy'].to(xy.device))
        used_query, used_gt, pairs = set(), set(), []
        for item in distance.flatten().argsort().tolist():
            query, gt = divmod(item, distance.shape[1])
            if query in used_query or gt in used_gt or float(distance[query, gt]) > 0.010:
                continue
            used_query.add(query)
            used_gt.add(gt)
            pairs.append((query, gt))
        if not pairs:
            continue
        predicted = angle[[query for query, _gt in pairs]]
        target = truth['source_angle'].to(angle.device)[[gt for _query, gt in pairs]]
        delta = angle_delta(predicted, target) / (math.pi / 180)
        loss = F.smooth_l1_loss(delta, torch.zeros_like(delta))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        seen += 1
        if seen % 400 == 0:
            current = _median(angle_errors(system, validation, noop))
            print('step', seen, 'median', round(current, 3), 'loss', round(float(loss.detach()), 4), flush=True)
            if current > start_median + 0.05:
                module.load_state_dict(snapshot)
                print('reverted early', flush=True)
                return
            best = min(best, current)
    after = angle_errors(system, validation, noop)
    after_full = full_probe(system, relation, model, head, validation, full)
    gain = start_median - _median(after)
    print('done', dict(
        seen=seen, median=round(_median(after), 3), within=sum(v <= 1 for v in after),
        full=after_full, gain=round(gain, 3)), flush=True)
    if gain >= 0.10 and after_full >= start_full - 1:
        ANGLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'direction': {key: value.detach().cpu() for key, value in module.state_dict().items()}}, ANGLE_PATH)
        print('kept direction', flush=True)
    else:
        module.load_state_dict(snapshot)
        print('reverted direction', flush=True)


if __name__ == '__main__':
    main()
