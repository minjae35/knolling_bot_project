"""Score the adopted RGB path: size residual, relation hidden, and row layout.

Inference reads an image and an instruction. Target poses are not inputs.
Partial and no-op scenes are not part of this score.
"""
from collections import Counter
import json
from pathlib import Path
import time

import torch

from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors, is_full
from knolling_model.experiments.inference import accepted_success, instruction_supported
from knolling_model.experiments.integration import load_integrated
from knolling_model.experiments.row_layout import NeuralRowPlacement
from knolling_model.experiments.size_refinement import (
    ObjectSizeRefinement, horizontal_extent, poses_from_detection)
from knolling_model.arrangement.data import recovery_supervision
from knolling_model.arrangement.evaluation_v3 import semantic_errors_v3
from knolling_model.arrangement.model import legacy_evaluation_view
from knolling_model.arrangement.selection import select_queries
import run_relation_capacity as cap
import run_relation_stable as stable

ROW_BEST = Path('artifacts/row_layout_origin_y/checkpoint-best.pt')
RELATION_BEST = Path('artifacts/phase1_relation_continue/checkpoint-best.pt')
SIZE_BEST = Path('artifacts/experiments/size_refinement/checkpoint-best.pt')
JOINT_BEST = Path('artifacts/experiments/joint_stage/checkpoint-best.pt')
SELECTOR_BEST = Path('artifacts/experiments/selector_full/checkpoint-best.pt')
DATA = 'artifacts/generalization_20000/dataset'


def describe(values):
    if not values:
        return dict(median=None, p95=None, n=0)
    tensor = torch.tensor(values, dtype=torch.float)
    return dict(median=float(tensor.median()), p95=float(tensor.quantile(.95)), n=len(values))


def _weights(checkpoint):
    if 'model' in checkpoint:
        return checkpoint['model']
    if 'head' in checkpoint:
        return checkpoint['head']
    raise KeyError('checkpoint has no model or head weights')


def load_bundle(size_path=SIZE_BEST, device=None, row_path=None):
    device = stable.DEVICE if device is None else device
    system = load_integrated(device=device)
    relation = cap.load_head(128, torch.load(RELATION_BEST, map_location='cpu', weights_only=False)['model']).to(device)
    row_model = NeuralRowPlacement().to(device)
    row_checkpoint = torch.load(ROW_BEST if row_path is None else row_path, map_location=device, weights_only=False)
    row_model.load_state_dict(row_checkpoint['row'] if row_path is not None and 'row' in row_checkpoint else row_checkpoint['model'])
    head = ObjectSizeRefinement().to(device)
    checkpoint = torch.load(size_path, map_location=device, weights_only=False)
    head.load_state_dict(_weights(checkpoint))
    for module in (system, relation, row_model, head):
        module.eval()
    for module in (system, relation, row_model):
        module.requires_grad_(False)
    return system, relation, row_model, head, checkpoint


def load_adopted(device=None):
    """Adopted inference: set encoder for row and order, original pose weights."""
    from knolling_model.experiments.set_context import SetArrangement
    device = stable.DEVICE if device is None else device
    system, relation, placement, head, _checkpoint = load_bundle(JOINT_BEST, device=device, row_path=JOINT_BEST)
    saved = torch.load(SELECTOR_BEST, map_location=device, weights_only=False)
    placement.load_state_dict(saved['placement'])
    model = SetArrangement(placement).to(device)
    model.context.load_state_dict(saved['context'])
    model.eval()
    model.requires_grad_(False)
    return system, relation, model, head, saved


def labels_of(record, identities):
    row = torch.full((len(identities),), -1)
    slot = torch.full((len(identities),), -1)
    location = {object_id: index for index, object_id in enumerate(identities)}
    for row_id, members in enumerate(record['supervision_v3']['rows']):
        for slot_id, object_id in enumerate(members):
            row[location[object_id]] = row_id
            slot[location[object_id]] = slot_id
    return row, slot


def blank(record, errors, objects, seconds, geometry):
    return dict(sample_id=record['sample_id'], objects=objects, task=False, errors=list(errors),
                row_match=False, row_hits=0, slot_hits=0, labeled=0, seconds=seconds, geometry=geometry)


def judge_scene(output, objects, visual, record, identities, truth, seconds, threshold, detail=True):
    """Judge one RGB prediction. Ground truth is not an inference input."""
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


def score_rgb(system, relation, row_model, head, data, indices):
    system.perception.eval()
    relation.eval()
    row_model.eval()
    head.eval()
    scenes = []
    with torch.no_grad():
        for index in indices:
            record = data.records[index]
            image = data[index]['image'][None].to(stable.DEVICE)
            identities, truth = recovery_supervision(record)
            truth = {key: value.cpu() for key, value in truth.items()}
            if not instruction_supported(record):
                raise RuntimeError('official comparison included an unsupported instruction')
            started = time.perf_counter()
            try:
                output, objects, visual = poses_from_detection(
                    system, relation, row_model, image, [record['instruction']['text']], head)
                elapsed = time.perf_counter()-started
            except ValueError:
                elapsed = time.perf_counter()-started
                scenes.append(dict(sample_id=record['sample_id'], objects=len(identities), task=False,
                                   errors=['count'], horizontal=[], vertical=[], row_match=False,
                                   row_hits=0, slot_hits=0, labeled=0, seconds=elapsed))
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
            scenes.append(dict(sample_id=record['sample_id'], objects=len(identities),
                               task=accepted_success(record, judged['task_success']), errors=list(judged['errors']),
                               horizontal=[] if detail is None else detail['horizontal_mm'],
                               vertical=[] if detail is None else detail['vertical_mm'],
                               row_match=bool(detail and detail['row_match']),
                               row_hits=row_hits, slot_hits=slot_hits, labeled=labeled, seconds=elapsed))
    return pack_rgb(scenes)


def pack_rgb(scenes):
    horizontal = [value for scene in scenes for value in scene['horizontal']]
    vertical = [value for scene in scenes for value in scene['vertical']]
    labeled = sum(scene['labeled'] for scene in scenes)

    def bucket(count):
        chosen = [scene for scene in scenes if scene['objects'] == count]
        return dict(success=sum(scene['task'] for scene in chosen), total=len(chosen))

    failures = Counter(name for scene in scenes if not scene['task'] for name in scene['errors'])
    seconds = [scene['seconds'] for scene in scenes]
    return dict(task=sum(scene['task'] for scene in scenes), total=len(scenes),
                by_count={str(count): bucket(count) for count in (2, 4, 6, 8)},
                horizontal=describe(horizontal), vertical=describe(vertical),
                count_errors=sum('count' in scene['errors'] for scene in scenes),
                failure_counts=dict(failures),
                row_match=sum(scene['row_match'] for scene in scenes),
                row_accuracy=None if not labeled else sum(scene['row_hits'] for scene in scenes)/labeled,
                slot_accuracy=None if not labeled else sum(scene['slot_hits'] for scene in scenes)/labeled,
                seconds=describe(seconds) | dict(mean=None if not seconds else float(sum(seconds)/len(seconds))),
                successes={scene['sample_id'] for scene in scenes if scene['task']})


def size_report(system, head, data, indices):
    width, height, footprint = [], [], []
    summary = Counter()
    head.eval()
    with torch.no_grad():
        for index in indices:
            record = data.records[index]
            image = data[index]['image'][None].to(stable.DEVICE)
            _, truth = recovery_supervision(record)
            truth = {key: value.to(stable.DEVICE) for key, value in truth.items()}
            visual = system.perception(image)
            from knolling_model.experiments.size_refinement import install_refined_size
            visual = install_refined_size(head, image, visual, system.threshold)
            chosen = select_queries(visual['source'], system.threshold)[0]
            summary['scenes'] += 1
            summary['gt'] += len(truth['source_xy'])
            if len(chosen) != len(truth['source_xy']):
                summary['count_mismatch'] += 1
            if not len(chosen):
                continue
            distance = torch.cdist(visual['source']['uv'][0, chosen]*visual['source']['uv'].new_tensor([.6, .4]),
                                   truth['source_xy'])
            used_query, used_gt = set(), set()
            for item in distance.flatten().argsort().tolist():
                query, gt = divmod(item, distance.shape[1])
                if query in used_query or gt in used_gt or float(distance[query, gt]) > .010:
                    continue
                used_query.add(query)
                used_gt.add(gt)
                predicted = visual['source']['size'][0, chosen[query]]
                target = truth['size'][gt]
                width.append(float((predicted[0]-target[0]).abs())*1000)
                height.append(float((predicted[1]-target[1]).abs())*1000)
                predicted_extent = horizontal_extent(predicted, visual['source']['angle'][0, chosen[query]])
                truth_extent = horizontal_extent(target, truth['source_angle'][gt])
                footprint.append(float((predicted_extent-truth_extent).abs())*1000)
            summary['matched'] += len(used_gt)
    return dict(width=describe(width), height=describe(height), footprint=describe(footprint),
                matched=int(summary['matched']), gt=int(summary['gt']),
                count_mismatch=int(summary['count_mismatch']), scenes=int(summary['scenes']))


def validation_indices(data):
    from diagnose_row_transition import multi_row
    return [index for index, record in enumerate(data.records) if multi_row(record)]


def public_rgb(point):
    return {key: value for key, value in point.items() if key != 'successes'}


def main():
    import run_phase1_pairwise_affine as runner
    from diagnose_row_transition import multi_row
    from eval_row_layout import evaluate
    from knolling_model.experiments.inference import SCOPE
    from run_full_arrangement import cache_records, score, subset
    from run_row_layout import attach_rows
    runner.setup()
    val = ExpandedDataset(DATA, 'validation')
    system, relation, row_model, head, checkpoint = load_bundle(JOINT_BEST, row_path=JOINT_BEST)
    indices = validation_indices(val)
    phase_val = subset(val, 'validation', 94)
    multi_val = [record for record in val.records if multi_row(record)]
    single_val = [record for record in val.records if is_full(record) and len(record['supervision_v3']['rows']) == 1]
    oracle = stable.official_metrics(relation, stable.cache_oracle(system, phase_val), phase_val)['task']
    single = score(relation, cache_records(system, single_val), single_val)['task']
    modular = evaluate(row_model, relation, attach_rows(multi_val, cache_records(system, multi_val)), multi_val, 'free')
    rgb = score_rgb(system, relation, row_model, head, val, indices)
    report = dict(update=checkpoint.get('update'), scope=SCOPE, oracle=oracle, single=single,
                  modular=modular['task'], rgb=public_rgb(rgb),
                  partial_supported=False, no_op_supported=False, test_used=False)
    path = Path('artifacts/experiments/end_to_end/status.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    stable.write(path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
