"""Confirm the wired decoder on the official RGB path, and separate the remaining failures."""
import json
from collections import Counter

import torch

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.full_arrangement import arrangement_errors
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import poses_from_detection
from knolling_model.recovery.data import recovery_supervision
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3
from knolling_model.recovery.model import legacy_evaluation_view
from score_end_to_end import load_adopted
from train_linked_y import VAL_COUNT, VAL_SEED
from train_set_context import fresh_split

STRUCTURAL = {'mixed_color_row', 'size_order', 'noncontiguous_color_group', 'axis_alignment',
               'count', 'source_identity'}
GEOMETRY = {'horizontal_gap', 'vertical_gap', 'row_alignment', 'clearance', 'collision',
            'boundary', 'requested_region'}


def main():
    runner.setup()
    data = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    fit, dev, _blocked = fresh_split(data.records)
    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(VAL_SEED)).tolist()
    splits = {'dev512': dev, 'dev256': [fit[index] for index in order[:VAL_COUNT]]}
    system, relation, model, head, _saved = load_adopted()
    # Baseline successes are the adopted numbers already measured. Recompute with the wired path only.
    report = {}
    for name, indices in splits.items():
        tasks, statuses, buckets = [], Counter(), Counter()
        gap_mm, size_mm = [], []
        for index in indices:
            record = data.records[index]
            image = data[index]['image'][None].to(stable.DEVICE)
            identities, truth = recovery_supervision(record)
            truth = {key: value.cpu() for key, value in truth.items()}
            try:
                output, objects, visual = poses_from_detection(
                    system, relation, model, image, [record['instruction']['text']], head)
            except ValueError:
                tasks.append(False)
                statuses['count'] += 1
                buckets['structural'] += 1
                continue
            source = {key: value.detach().cpu() for key, value in visual['source'].items()}
            layout = dict(xy=output['xy'].detach().cpu(), angle=output['angle'].detach().cpu(),
                          valid=objects.valid.detach().cpu(), query_index=objects.query_index.detach().cpu(),
                          rollout='free_rollout')
            judged = evaluate_batch(
                legacy_evaluation_view(dict(source=source, layout=layout, threshold=.5)),
                Batch({}, [truth], [dict(sample_id=record['sample_id'], object_ids=identities, record=record)]),
                semantic_check=semantic_errors_v3)[0]
            task = bool(accepted_success(record, judged['task_success']))
            tasks.append(task)
            statuses.update(output.get('geometry_status') or ['missing'])
            if task:
                continue
            errors = set(judged['errors'])
            kind = 'both' if errors & STRUCTURAL and errors & GEOMETRY else (
                'structural' if errors & STRUCTURAL else 'geometry' if errors & GEOMETRY else 'other')
            buckets[kind] += 1
            if judged.get('predicted_targets') and kind == 'geometry':
                detail = arrangement_errors(record, judged['predicted_targets'])
                gap_mm.extend(detail['horizontal_mm']+detail['vertical_mm'])
                size_mm.extend(judged['source_size_mm'])
        gaps = torch.tensor(gap_mm) if gap_mm else torch.tensor([0.])
        sizes = torch.tensor(size_mm) if size_mm else torch.tensor([0.])
        report[name] = dict(task=sum(tasks), total=len(tasks), status=dict(statuses), buckets=dict(buckets),
                            geometry_only_gap_mm=dict(median=float(gaps.median()), p95=float(torch.quantile(gaps, .95))),
                            geometry_only_size_mm=dict(median=float(sizes.median()), p95=float(torch.quantile(sizes, .95))))
        print(name, report[name]['task'], flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
