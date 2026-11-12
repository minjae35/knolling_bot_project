"""GT size is a diagnostic input to the decoder only. It never enters inference."""
import json
from collections import Counter

import torch

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_dataset.schema import Pose
from knolling_dataset.validator import polygon
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.geometry_decoder import decode_scene, extent_offsets
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import encode_instruction, install_refined_size
from knolling_model.losses import match_source
from knolling_model.recovery.contracts import pack_sources
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
from knolling_model.recovery.selection import select_queries
from score_end_to_end import load_adopted
from train_linked_y import VAL_COUNT, VAL_SEED
from train_set_context import fresh_split

STRUCTURAL = {'mixed_color_row', 'size_order', 'noncontiguous_color_group', 'axis_alignment',
               'count', 'source_identity'}
GEOMETRY = {'horizontal_gap', 'vertical_gap', 'row_alignment', 'clearance', 'collision',
            'boundary', 'requested_region'}
GAP_NAMES = ('horizontal_gap', 'vertical_gap', 'requested_region', 'clearance', 'collision', 'boundary')


class _Asset:
    def __init__(self, major, minor, footprint):
        self.a_m, self.b_m, self.footprint = major, minor, footprint


class _Object:
    def __init__(self, major, minor, footprint):
        self.asset = _Asset(major, minor, footprint)


def judge(record, identities, truth, source, objects, xy, angle):
    layout = dict(xy=xy.detach().cpu(), angle=angle.detach().cpu(),
                  valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                  rollout='free_rollout')
    judged = evaluate_batch(
        legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5)),
        Batch({}, [truth], [dict(sample_id=record['sample_id'], object_ids=identities, record=record)]),
        semantic_check=semantic_errors_v3)[0]
    task = bool(accepted_success(record, judged['task_success']))
    return task, list(judged['errors'])


def matched_sizes(source, queries, truth):
    query, gt = match_source(source, [truth])[0]
    table = {int(left): int(right) for left, right in zip(query.tolist(), gt.tolist())}
    chosen = [int(index) for index in queries.tolist()]
    if any(index not in table for index in chosen):
        return None, None
    order = [table[index] for index in chosen]
    return truth['size'][order], order


def envelope_residual(angle, major, minor):
    xmin, xmax, ymin, ymax = extent_offsets(angle, major, minor)
    bounds = polygon(_Object(float(major), float(minor), None), Pose(0., 0., float(angle))).bounds
    return max(abs(bounds[0]-float(xmin)), abs(bounds[2]-float(xmax)),
               abs(bounds[1]-float(ymin)), abs(bounds[3]-float(ymax)))


def footprint_delta(angle, asset):
    pose = Pose(0., 0., float(angle))
    obj = _Object(asset['a_m'], asset['b_m'], asset['footprint'])
    envelope = polygon(obj, pose, envelope=True).bounds
    contour = polygon(obj, pose, envelope=False).bounds
    width = (contour[2]-contour[0])-(envelope[2]-envelope[0])
    height = (contour[3]-contour[1])-(envelope[3]-envelope[1])
    return abs(width), abs(height)


def bucket(task, errors, status):
    if task:
        return 'success'
    if str(status).startswith('infeasible'):
        return 'infeasible'
    names = set(errors)
    has_structure, has_geometry = bool(names & STRUCTURAL), bool(names & GEOMETRY)
    if has_geometry and not has_structure:
        return 'geometry'
    if has_structure and not has_geometry:
        return 'structural'
    if has_structure and has_geometry:
        return 'both'
    return 'other'


def stats(values):
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    return dict(n=len(values), median=float(tensor.median()), mean=float(tensor.mean()),
                p95=float(torch.quantile(tensor, .95)),
                abs_median=float(tensor.abs().median()))


def main():
    runner.setup()
    data = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    fit, dev, _blocked = fresh_split(data.records)
    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(VAL_SEED)).tolist()
    splits = {'dev512': dev, 'dev256': [fit[index] for index in order[:VAL_COUNT]]}
    system, relation, model, head, _saved = load_adopted()
    report = {}
    for name, indices in splits.items():
        rows = []
        size_signed, extent_signed, formula, contour = [], [], [], []
        for index in indices:
            record = data.records[index]
            image = data[index]['image'][None].to(stable.DEVICE)
            identities, truth = recovery_supervision(record)
            truth = {key: value.cpu() for key, value in truth.items()}
            by_id = {obj['object_id']: obj for obj in record['source']['objects']}
            try:
                with torch.no_grad():
                    visual = system.perception(image)
                    visual = install_refined_size(head, image, visual, system.threshold)
                    queries = select_queries(visual['source'], system.threshold)
                    if any(len(chosen) < 1 or len(chosen) > 8 for chosen in queries):
                        raise ValueError('count')
                    objects = pack_sources(visual['source'], queries)
                    text, mask = encode_instruction(system, [record['instruction']['text']], image.device)
                    goal = system.layout.goal_encoder(objects, text, mask)['vector']
                    hidden = relation._hidden(objects.features())
                    output = model(hidden, objects.valid, goal, objects.size[..., 0], objects.size[..., 1],
                                   None, None, True, colors=objects.color)
            except ValueError:
                rows.append(dict(id=record['sample_id'], before=False, after=False, kind='structural',
                                 before_errors=['count'], after_errors=['count']))
                continue
            count = int(objects.valid[0].sum())
            source = {key: value.detach().cpu() for key, value in visual['source'].items()}
            neural = output['xy'][0, :count].detach().cpu()
            angle = output['angle'][0, :count].detach().cpu()
            row = output['row'][0, :count].detach().cpu()
            slot = output['slot'][0, :count].detach().cpu()
            pred = objects.size[0, :count].detach().cpu()
            color = objects.color[0, :count].argmax(-1).detach().cpu()
            goal_row = goal[0].detach().cpu()
            decoded, status = decode_scene(neural, angle, row, slot, pred[:, 0], pred[:, 1], color, goal_row)
            pose = neural if str(status).startswith('infeasible') else decoded
            before_task, before_errors = judge(record, identities, truth, source, objects, pose[None], angle[None])
            kind = bucket(before_task, before_errors, status)
            gt_size, gt_order = matched_sizes(source, queries[0], truth)
            if gt_size is None or len(gt_size) != count:
                rows.append(dict(id=record['sample_id'], before=before_task, after=before_task, kind=kind,
                                 before_errors=before_errors, after_errors=before_errors, matched=False))
                continue
            for local, gt_index in enumerate(gt_order):
                signed = (pred[local]-gt_size[local])*1000
                size_signed.append(signed)
                pred_box = extent_offsets(angle[local], pred[local, 0], pred[local, 1])
                true_box = extent_offsets(angle[local], gt_size[local, 0], gt_size[local, 1])
                extent_signed.append(torch.tensor([
                    (pred_box[1]-pred_box[0])-(true_box[1]-true_box[0]),
                    (pred_box[3]-pred_box[2])-(true_box[3]-true_box[2])])*1000)
                formula.append(envelope_residual(angle[local], gt_size[local, 0], gt_size[local, 1])*1000)
                asset = by_id[identities[gt_index]]['asset']
                contour.append(torch.tensor(footprint_delta(angle[local], asset))*1000)
            oracle, oracle_status = decode_scene(
                neural, angle, row, slot, gt_size[:, 0], gt_size[:, 1], color, goal_row)
            oracle_pose = neural if str(oracle_status).startswith('infeasible') else oracle
            after_task, after_errors = judge(
                record, identities, truth, source, objects, oracle_pose[None], angle[None])
            rows.append(dict(id=record['sample_id'], before=before_task, after=after_task, kind=kind,
                             before_errors=before_errors, after_errors=after_errors,
                             before_status=status, after_status=oracle_status, matched=True))
        before_ids = {row['id'] for row in rows if row['before']}
        after_ids = {row['id'] for row in rows if row['after']}
        geometry_rows = [row for row in rows if row['kind'] == 'geometry']
        fixed = [row for row in geometry_rows if row['after']]
        size = torch.stack(size_signed) if size_signed else torch.zeros((0, 2))
        extent = torch.stack(extent_signed) if extent_signed else torch.zeros((0, 2))
        contour_mm = torch.stack(contour) if contour else torch.zeros((0, 2))
        before_fail = Counter(name for row in rows if not row['before'] for name in row['before_errors'])
        after_fail = Counter(name for row in rows if not row['after'] for name in row['after_errors'])
        cleared = Counter()
        for row in geometry_rows:
            if row['after']:
                cleared['fixed'] += 1
            else:
                cleared.update(name for name in row['after_errors'] if name in GAP_NAMES or name in GEOMETRY)
                cleared['still_'+bucket(False, row['after_errors'], row.get('after_status', ''))] += 1
        report[name] = dict(
            total=len(rows),
            before=len(before_ids), after=len(after_ids),
            preserved=len(before_ids & after_ids), lost=len(before_ids-after_ids),
            gained=len(after_ids-before_ids),
            geometry_only=len(geometry_rows), geometry_fixed_by_gt_size=len(fixed),
            geometry_still=len(geometry_rows)-len(fixed),
            kinds=dict(Counter(row['kind'] for row in rows)),
            size_error_mm=dict(major=stats(size[:, 0].tolist()), minor=stats(size[:, 1].tolist())),
            extent_error_mm=dict(width=stats(extent[:, 0].tolist()), height=stats(extent[:, 1].tolist())),
            formula_residual_mm=stats(formula),
            footprint_minus_envelope_mm=dict(width=stats(contour_mm[:, 0].tolist()),
                                             height=stats(contour_mm[:, 1].tolist())),
            geometry_cleared=dict(cleared),
            gap_errors=dict(before={name: before_fail.get(name, 0) for name in GAP_NAMES},
                            after={name: after_fail.get(name, 0) for name in GAP_NAMES}))
        print(name, report[name]['before'], report[name]['after'], 'geometry',
              report[name]['geometry_only'], 'fixed', report[name]['geometry_fixed_by_gt_size'], flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
