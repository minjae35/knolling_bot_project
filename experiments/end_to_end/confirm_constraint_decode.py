"""Score the wired constraint search through the official RGB path."""
import json
import time
from collections import Counter

import run_phase1_pairwise_affine as runner
import run_relation_stable as stable
from knolling_model.data import Batch
from knolling_model.evaluation import evaluate_batch
from knolling_model.experiments.expanded_data import ExpandedDataset
from knolling_model.experiments.inference import accepted_success
from knolling_model.experiments.size_refinement import poses_from_detection
from knolling_model.arrangement.data import recovery_supervision
from knolling_model.arrangement.evaluation_v3 import semantic_errors_v3
from knolling_model.arrangement.model import legacy_evaluation_view
from dev_splits import fresh_split, heldout_validation
from score_end_to_end import load_adopted


def main():
    runner.setup()
    data = ExpandedDataset('artifacts/generalization_20000/dataset', 'train')
    fit, dev, _blocked = fresh_split(data.records)
    splits = {'dev512': dev, 'dev256': heldout_validation(fit)}
    system, relation, model, head, _saved = load_adopted()
    report = {}
    for name, indices in splits.items():
        tasks, failures, seconds, room = [], Counter(), [], 0
        for index in indices:
            record = data.records[index]
            image = data[index]['image'][None].to(stable.DEVICE)
            identities, truth = recovery_supervision(record)
            truth = {key: value.cpu() for key, value in truth.items()}
            started = time.perf_counter()
            try:
                output, objects, visual = poses_from_detection(
                    system, relation, model, image, [record['instruction']['text']], head)
            except ValueError:
                seconds.append(time.perf_counter()-started)
                tasks.append(False)
                failures['count'] += 1
                continue
            seconds.append(time.perf_counter()-started)
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
            if not task:
                failures.update(judged['errors'])
            room += any('horizontal_room' in item for item in output.get('geometry_status', []))
        report[name] = dict(task=sum(tasks), total=len(tasks), seconds=float(sum(seconds)/len(seconds)),
                            failures={key: failures.get(key, 0) for key in
                                      ('mixed_color_row', 'size_order', 'horizontal_gap', 'vertical_gap',
                                       'requested_region', 'collision', 'clearance', 'boundary', 'count')},
                            horizontal_room=room)
        print(name, report[name]['task'], flush=True)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
