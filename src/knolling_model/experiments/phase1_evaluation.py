"""Phase 1 scoring only; never called by inference or used to repair predictions.

Preserves v3 geometric/semantic tolerances. Canonical Teacher XY fidelity is
reported separately, not required for a valid translated single-row layout.
"""
from knolling_dataset.validator import inferred_rows
from knolling_model.diagnostics import positioned_objects
from knolling_model.recovery.evaluation_v3 import semantic_errors_v3


def semantic_errors_phase1(objects, record):
    errors = semantic_errors_v3(objects, record)
    if len(inferred_rows(objects)) != 1:
        errors = sorted(set(errors + ['single_row']))
    return errors


def phase1_gate(rows, records, pairs, expected_ids):
    ids = [r['sample_id'] for r in rows]
    complete = (len(ids) == len(set(ids)) == len(expected_ids) == len(set(expected_ids))
                and set(ids) == set(expected_ids)
                and {r['sample_id'] for r in records} == set(expected_ids))
    by_id = {r['sample_id']: r for r in rows}
    by_record = {r['sample_id']: r for r in records}
    checks = []
    for pair in pairs:
        a, b = pair['sample_ids']
        passed = False
        if a in by_id and b in by_id and a in by_record and b in by_record:
            ra, rb = by_id[a], by_id[b]
            passed = (ra['task_success'] and rb['task_success']
                and bool(semantic_errors_phase1(positioned_objects(by_record[a], ra['predicted_targets']), by_record[b]))
                and bool(semantic_errors_phase1(positioned_objects(by_record[b], rb['predicted_targets']), by_record[a])))
        checks.append(dict(pair_id=pair['pair_id'], factor=pair['factor'], passed=bool(passed)))
    pair_contract = (bool(pairs) and len({p['pair_id'] for p in pairs}) == len(pairs)
                     and {p['factor'] for p in pairs} == {'axis_alignment', 'gap_m', 'sort_direction'})
    return dict(complete=complete, samples=len(expected_ids),
                successful_scenes=sum(r['task_success'] for r in rows),
                counterfactual_passed=sum(c['passed'] for c in checks), counterfactual_total=len(pairs),
                checks=checks, passed=bool(complete and pair_contract and all(r['task_success'] for r in rows)
                                          and all(c['passed'] for c in checks)))
