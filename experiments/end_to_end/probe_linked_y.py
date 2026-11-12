"""Can a linked row Y express the teacher, without moving an unchanged row? No training."""
import json
from collections import Counter

import torch

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_dataset.schema import REGIONS, scene_from_dict
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.row_layout import chain_centers
from knolling_model.recovery.contracts import Objects
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from run_full_arrangement import cache_records
from run_row_layout import attach_rows, collate_rows, hidden_of
from score_end_to_end import load_adopted
from train_set_context import fresh_split

Y_MIN, Y_MAX = REGIONS['full'][1], REGIONS['full'][3]


def centers_of(record, stored):
    ids, _truth = recovery_supervision(record)
    groups = {}
    for index, (row_id, slot_id) in enumerate(zip(stored['row'].tolist(), stored['slot'].tolist())):
        groups.setdefault(int(row_id), []).append((int(slot_id), index))
    rows = [sorted(groups[key]) for key in sorted(groups)]
    extent = stored['size'][:, 0]*stored['target_angle'].sin().abs()+stored['size'][:, 1]*stored['target_angle'].cos().abs()
    centers, heights = [], []
    for row in rows:
        picked = [index for _, index in row]
        centers.append(float(stored['target_xy'][picked, 1].mean()))
        heights.append(float(extent[picked].max()))
    return torch.tensor(centers), torch.tensor(heights)


def main():
    runner.setup()
    train = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    _fit, dev, _blocked = fresh_split(train.records)
    system, relation, model, _head, _saved = load_adopted()
    records = [train.records[index] for index in dev]
    stored = attach_rows(records, cache_records(system, records))
    identity = []
    tasks = []
    y_error = []
    rule_error = []
    center_error = []
    gaps = []
    outside = 0
    too_tall = 0
    corrections = []
    anchors = []
    gap_values = Counter()
    with torch.no_grad():
        for start in range(0, len(records), 16):
            chunk_s = stored[start:start+16]
            chunk_r = records[start:start+16]
            packed = collate_rows(chunk_s, stable.DEVICE)
            hidden = hidden_of(relation, packed['features'])
            widths, heights = packed['size'][..., 0], packed['size'][..., 1]
            output = model(hidden, packed['valid'], packed['goal'], widths, heights, None, None, True,
                           colors=packed['color'])
            forced = model(hidden, packed['valid'], packed['goal'], widths, heights,
                           packed['row'], packed['slot'], True, colors=packed['color'])
            base = model.placement.row_layout(model.placement.origin_inputs(
                hidden, packed['valid'], packed['goal'], forced['row'], widths, heights, forced['slot']))
            identity.append(float((forced['origins']-base).abs().max()))
            angle = forced['angle']
            extent = widths*angle.sin().abs()+heights*angle.cos().abs()
            for local, record in enumerate(chunk_r):
                kept = chunk_s[local]['valid'].shape[0]
                item = {key: packed[key][local:local+1, :kept].cpu() for key in
                        ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')}
                objects = Objects(item['xy'], item['size'], item['angle'], item['color'], item['shape'],
                                  item['valid'], item['query_index'])
                layout = dict(xy=output['xy'][local:local+1, :kept].cpu(),
                              angle=output['angle'][local:local+1, :kept].cpu(),
                              valid=item['valid'], query_index=item['query_index'], rollout='free_rollout')
                ids, truth = recovery_supervision(record)
                judged = evaluate_batch(_oracle_view(objects, layout), Batch({}, [truth], [dict(
                    sample_id=record['sample_id'], object_ids=ids, record=record)]),
                    semantic_check=semantic_errors_v3)[0]
                tasks.append(bool(accepted_success(record, judged['task_success'])))
                centers, row_heights = centers_of(record, chunk_s[local])
                if centers.shape[0] < 2:
                    continue
                opening = torch.tensor([
                    (centers[index]-row_heights[index]/2)-(centers[index+1]+row_heights[index+1]/2)
                    for index in range(centers.shape[0]-1)])
                rebuilt = chain_centers(centers[0], row_heights, opening)
                y_error.append(float((rebuilt-centers).abs().max())*1000)
                palette = {obj.object_id: obj.color for obj in scene_from_dict(record['source']).objects}
                rows = record['supervision_v3']['rows']
                doubled = record['instruction']['group_by'] == 'color'
                gap_m = record['instruction']['gap_m']
                for index in range(len(rows)-1):
                    factor = 2 if doubled and palette[rows[index][0]] != palette[rows[index+1][0]] else 1
                    rule_error.append(abs(float(opening[index])-gap_m*factor)*1000)
                span = float((centers[0]+row_heights[0]/2)-(centers[-1]-row_heights[-1]/2))
                center_error.append(abs(float(centers[0])-(.20+span/2-float(row_heights[0])/2))*1000)
                gaps.extend(float(value)*1000 for value in opening)
                top = float(centers[0]+row_heights[0]/2)
                bottom = float(centers[-1]-row_heights[-1]/2)
                outside += int(top > Y_MAX+.002 or bottom < Y_MIN-.002)
                too_tall += int((top-bottom) > (Y_MAX-Y_MIN)+.002)
                members = packed['valid'][local] & (forced['row'][local] >= 0)
                order = []
                for row_id in range(int(forced['row'][local, members].max())+1):
                    chosen = (forced['row'][local] == row_id) & packed['valid'][local]
                    if bool(chosen.any()):
                        order.append(row_id)
                base_y = base[local, :, 1]
                row_extent = []
                for row_id in order:
                    chosen = (forced['row'][local] == row_id) & packed['valid'][local]
                    row_extent.append(float(extent[local, chosen].max()))
                instructed = float(model.placement._gap(packed['goal'][local:local+1])[0])
                for index in range(len(order)-1):
                    implied = ((base_y[order[index]]-row_extent[index]/2)
                               -(base_y[order[index+1]]+row_extent[index+1]/2))
                    gap_values[round(float(opening[index])*1000, 1)] += 1
                corrections.append((round(float(implied), 4), round(instructed, 4),
                                    round(float(opening[index])*1000, 1), float(opening[index]-implied)))
                if index == 0:
                    anchors.append((round(float(base_y[order[0]]), 3), round(float(sum(row_heights)), 3),
                                    round(float(centers[0]-base_y[order[0]]), 4)))
    groups = {}
    for key in corrections:
        groups.setdefault(key[:2], []).append(key[3])
    spreads = [max(values)-min(values) for values in groups.values() if len(values) > 1]
    anchor_groups = {}
    for base_y, height, delta in anchors:
        anchor_groups.setdefault((base_y, height), []).append(delta)
    anchor_spreads = [max(values)-min(values) for values in anchor_groups.values() if len(values) > 1]
    print(json.dumps(dict(
        free_task=sum(tasks), total=len(tasks),
        origin_delta_mm=max(identity)*1000,
        teacher_y_error_mm_max=max(y_error),
        gap_rule_mm_max=max(rule_error), center_rule_mm_max=max(center_error),
        gap_mm_min=min(gaps), gap_mm_median=float(torch.tensor(gaps).median()),
        negative_gaps=sum(value < -0.5 for value in gaps),
        outside_workspace=outside, taller_than_region=too_tall,
        gap_values=dict(gap_values),
        correction_pairs=len(corrections), colliding_feature_groups=len(spreads),
        correction_spread_mm_max=0 if not spreads else max(spreads)*1000,
        anchor_spread_mm_max=0 if not anchor_spreads else max(anchor_spreads)*1000), indent=2))


if __name__ == '__main__':
    main()
