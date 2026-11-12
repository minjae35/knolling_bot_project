"""Score the frozen selector plus the geometry decoder. No training and no test split."""
import json
from collections import Counter

import torch

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors, collate
from knolling_model.experiments.geometry_decoder import decode_scene
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import encode_instruction, install_refined_size, poses_from_detection
from knolling_model.recovery.contracts import Objects, pack_sources
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation import _oracle_view
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
from knolling_model.recovery.selection import select_queries
from run_full_arrangement import cache_records
from run_row_layout import hidden_of
from score_end_to_end import load_adopted
from train_linked_y import VAL_COUNT, VAL_SEED
from train_set_context import fresh_split

STRUCTURAL = {'mixed_color_row', 'size_order', 'noncontiguous_color_group', 'axis_alignment',
               'count', 'source_identity'}
GEOMETRY = {'horizontal_gap', 'vertical_gap', 'row_alignment', 'clearance', 'collision',
            'boundary', 'requested_region'}


def judge_rgb(record, identities, truth, source, objects, xy, angle):
    layout = dict(xy=xy.detach().cpu(), angle=angle.detach().cpu(),
                  valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                  rollout='free_rollout')
    judged = evaluate_batch(
        legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5)),
        Batch({}, [truth], [dict(sample_id=record['sample_id'], object_ids=identities, record=record)]),
        semantic_check=semantic_errors_v3)[0]
    task = bool(accepted_success(record, judged['task_success']))
    detail = arrangement_errors(record, judged['predicted_targets']) if judged.get('predicted_targets') else None
    return task, list(judged['errors']), detail


def apply_decode(output, size, color, goal, count):
    return decode_scene(
        output['xy'][0, :count].detach().cpu(), output['angle'][0, :count].detach().cpu(),
        output['row'][0, :count].detach().cpu(), output['slot'][0, :count].detach().cpu(),
        size[0, :count, 0].detach().cpu(), size[0, :count, 1].detach().cpu(),
        color[0, :count].argmax(-1).detach().cpu(), goal[0].detach().cpu())


def pack_pair(before_task, before_errors, after_task, after_errors, status, move_mm):
    return dict(before=before_task, after=after_task, before_errors=before_errors,
                after_errors=after_errors, status=status, move_mm=move_mm)


def summarize(name, rows):
    before_ids = {row['id'] for row in rows if row['before']}
    after_ids = {row['id'] for row in rows if row['after']}
    kept = before_ids & after_ids
    failures = Counter()
    structural = geometry = both = infeasible = 0
    for row in rows:
        if row['after']:
            continue
        failures.update(row['after_errors'])
        names = set(row['after_errors'])
        if any(name.startswith('infeasible') or name.startswith('decoder_') for name in names):
            infeasible += 1
        has_structure = bool(names & STRUCTURAL)
        has_geometry = bool(names & GEOMETRY)
        structural += int(has_structure and not has_geometry)
        geometry += int(has_geometry and not has_structure)
        both += int(has_structure and has_geometry)
    moves = [row['move_mm'] for row in rows if row['status'] == 'placed']
    return dict(
        split=name, total=len(rows),
        before=len(before_ids), after=len(after_ids),
        preserved=len(kept), lost=len(before_ids-after_ids), gained=len(after_ids-before_ids),
        status=dict(Counter(row['status'] for row in rows)),
        placed_move_mm=None if not moves else dict(median=float(torch.tensor(moves).median()),
                                                    max=max(moves)),
        failure_counts=dict(failures),
        remaining_structural=structural, remaining_geometry=geometry,
        remaining_both=both, remaining_infeasible=infeasible)


def rgb_rows(system, relation, model, head, data, indices):
    found = []
    for index in indices:
        record = data.records[index]
        image = data[index]['image'][None].to(stable.DEVICE)
        identities, truth = recovery_supervision(record)
        truth = {key: value.cpu() for key, value in truth.items()}
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
            found.append(dict(id=record['sample_id'], before=False, after=False,
                              before_errors=['count'], after_errors=['count'], status='count', move_mm=0.))
            continue
        count = int(objects.valid[0].sum())
        source = {key: value.detach().cpu() for key, value in visual['source'].items()}
        before_task, before_errors, _ = judge_rgb(
            record, identities, truth, source, objects, output['xy'], output['angle'])
        decoded, status = apply_decode(output, objects.size, objects.color, goal, count)
        if status.startswith('infeasible') or status == 'empty':
            after_task, after_errors = False, ['decoder_'+status]
            move = 0.
        else:
            pose = output['xy'].detach().cpu().clone()
            pose[0, :count] = decoded
            after_task, after_errors, _ = judge_rgb(
                record, identities, truth, source, objects, pose, output['angle'])
            move = float((decoded-output['xy'][0, :count].cpu()).abs().max())*1000
        found.append(dict(id=record['sample_id'], before=before_task, after=after_task,
                          before_errors=before_errors, after_errors=after_errors,
                          status=status, move_mm=move))
    return found


def object_rows(model, relation, stored, records):
    found = []
    with torch.no_grad():
        for start in range(0, len(records), 16):
            chunk_s, chunk_r = stored[start:start+16], records[start:start+16]
            packed = collate(chunk_s, stable.DEVICE)
            hidden = hidden_of(relation, packed['features'])
            output = model(hidden, packed['valid'], packed['goal'], packed['size'][..., 0], packed['size'][..., 1],
                           None, None, True, colors=packed['color'])
            for local, record in enumerate(chunk_r):
                count = chunk_s[local]['valid'].shape[0]
                item = {key: packed[key][local:local+1, :count].cpu() for key in
                        ('xy', 'size', 'angle', 'color', 'shape', 'valid', 'query_index')}
                objects = Objects(item['xy'], item['size'], item['angle'], item['color'], item['shape'],
                                  item['valid'], item['query_index'])
                ids, truth = recovery_supervision(record)
                truth = {key: value.cpu() for key, value in truth.items()}

                def score(xy):
                    layout = dict(xy=xy, angle=output['angle'][local:local+1, :count].cpu(),
                                  valid=item['valid'], query_index=item['query_index'], rollout='free_rollout')
                    judged = evaluate_batch(
                        _oracle_view(objects, layout),
                        Batch({}, [truth], [dict(sample_id=record['sample_id'], object_ids=ids, record=record)]),
                        semantic_check=semantic_errors_v3)[0]
                    return bool(accepted_success(record, judged['task_success'])), list(judged['errors'])

                before_task, before_errors = score(output['xy'][local:local+1, :count].cpu())
                decoded, status = apply_decode(
                    dict(xy=output['xy'][local:local+1], angle=output['angle'][local:local+1],
                         row=output['row'][local:local+1], slot=output['slot'][local:local+1]),
                    packed['size'][local:local+1], packed['color'][local:local+1],
                    packed['goal'][local:local+1], count)
                if status.startswith('infeasible') or status == 'empty':
                    after_task, after_errors, move = False, ['decoder_'+status], 0.
                else:
                    pose = output['xy'][local:local+1, :count].cpu().clone()
                    pose[0] = decoded
                    after_task, after_errors = score(pose)
                    move = float((decoded-output['xy'][local, :count].cpu()).abs().max())*1000
                found.append(dict(id=record['sample_id'], before=before_task, after=after_task,
                                  before_errors=before_errors, after_errors=after_errors,
                                  status=status, move_mm=move))
    return found


def main():
    runner.setup()
    data = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    fit, dev, _blocked = fresh_split(data.records)
    if len(dev) != 512:
        raise RuntimeError(f'family-disjoint dev changed: {len(dev)}')
    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(VAL_SEED)).tolist()
    val_ids = [fit[index] for index in order[:VAL_COUNT]]
    if set(val_ids) & set(dev):
        raise RuntimeError('the 256 overlaps the 512')
    system, relation, model, head, _saved = load_adopted()
    # Touch the official RGB helper so this file cannot drift from an unused import.
    assert poses_from_detection is not None
    report = {}
    for name, indices in (('dev512', dev), ('dev256', val_ids)):
        rows = rgb_rows(system, relation, model, head, data, indices)
        print('RGB', name, sum(row['after'] for row in rows), len(rows), flush=True)
        records = [data.records[index] for index in indices]
        stored = cache_records(system, records)
        objects = object_rows(model, relation, stored, records)
        print('GT', name, sum(row['after'] for row in objects), len(objects), flush=True)
        report[name] = dict(rgb=summarize(name, rows), gt_object=summarize(name, objects))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
