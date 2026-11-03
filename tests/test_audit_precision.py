"""Roundoff allowance is restricted to derived movement scores, never GT."""
from copy import deepcopy
import math

import pytest

from knolling_dataset.audit_v13 import (
    _movement_score_matches, _movement_metrics_match, _partial_quality_matches,
    check_v13, read_jsonl,
)
from knolling_dataset.builder import prepare_annotations
from knolling_dataset.migration import initialize_balanced_sidecars, write_jsonl
from knolling_dataset.policies import BALANCED, policy_metadata


@pytest.mark.parametrize('stored,computed', [
    (4.440863602014575e-06, 4.440863602014576e-06),
    (.05376727341039903, .05376727341039904),
])
def test_observed_movement_roundoff_is_one_adjacent_float(stored, computed):
    assert computed-stored == math.ulp(stored)
    assert _movement_score_matches(stored, computed)
    assert _movement_score_matches(computed, stored)
    assert not _movement_score_matches(stored, math.nextafter(computed, math.inf))
    assert not _movement_score_matches(math.nextafter(stored, -math.inf), computed)


@pytest.mark.parametrize('stored,computed', [
    (0., math.nextafter(0., math.inf)), (-0., 0.), (-1., -1.),
    (float('nan'), 1.), (float('inf'), float('inf')), (1., float('nan')),
    (1, 1.), (True, 1.), (None, 1.), ('1.0', 1.),
])
def test_invalid_scores_and_noop_changes_still_fail(stored, computed):
    assert not _movement_score_matches(stored, computed)
    assert _movement_score_matches(0., 0.)


def test_other_metrics_keys_hashes_and_nested_coordinates_remain_exact():
    current = {'J_move': .05376727341039904, 'J_shape': .5,
               'teacher_config': {'comparison_decimals': 12}, 'target_hash': 'abc',
               'object_movements': [{'object_id': 'o1', 'translation_mm': 1.}]}
    saved = deepcopy(current)
    saved['J_move'] = math.nextafter(saved['J_move'], -math.inf)
    assert _movement_metrics_match(saved, current)
    for key, value in [('J_shape', math.nextafter(.5, math.inf)),
                       ('teacher_config', {'comparison_decimals': 13}),
                       ('target_hash', 'changed'),
                       ('object_movements', [{'object_id': 'o1', 'translation_mm': math.nextafter(1., math.inf)}])]:
        changed = deepcopy(saved)
        changed[key] = value
        assert not _movement_metrics_match(changed, current)
    assert not _movement_metrics_match({**saved, 'extra': 0}, current)
    assert not _movement_metrics_match({k:v for k,v in saved.items() if k != 'J_move'}, current)
    assert not _movement_metrics_match(None, current)


def test_partial_quality_changes_only_known_movement_fields_without_mutation():
    current = [{'sample_id': 'sample', 'selected_ids': ('o1',),
                'legacy': {'J_move': .02, 'J_shape': .5},
                'v1_3': {'J_move': 4.440863602014576e-06, 'J_shape': .4}}]
    saved = deepcopy(current)
    saved[0]['v1_3']['J_move'] = 4.440863602014575e-06
    before = deepcopy(saved)
    assert _partial_quality_matches(saved, current)
    assert saved == before
    # Reject an additional ULP, wrong identity, extra keys, missing samples,
    # or a last-bit change to an unrelated quality metric.
    for path, value in [(('v1_3', 'J_move'), math.nextafter(saved[0]['v1_3']['J_move'], -math.inf)),
                        (('legacy', 'J_shape'), math.nextafter(.5, math.inf)),
                        (('selected_ids',), ('other',)), (('sample_id',), 'different'),
                        (('extra',), True)]:
        changed = deepcopy(saved)
        target = changed[0]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert not _partial_quality_matches(changed, current)
    assert not _partial_quality_matches([], current)


@pytest.fixture(scope='module')
def balanced_records():
    return prepare_annotations(10, 1202, policy=BALANCED)[1]


def test_teacher_audit_accepts_roundoff_but_rejects_score_and_metadata_tampering(tmp_path, balanced_records):
    initialize_balanced_sidecars(tmp_path, balanced_records)
    path = tmp_path/'teacher_audit.jsonl'
    original = read_jsonl(path)
    index = next(i for i, row in enumerate(original) if row['J_move'] > 0)
    sid = original[index]['sample_id']
    changed = deepcopy(original)
    changed[index]['J_move'] = math.nextafter(original[index]['J_move'], math.inf)
    write_jsonl(path, changed)
    provenance = policy_metadata(BALANCED)
    assert check_v13(tmp_path, balanced_records, provenance, invariance=False)['valid']
    changed[index]['J_move'] = math.nextafter(changed[index]['J_move'], math.inf)
    write_jsonl(path, changed)
    assert sid+':teacher_scores_or_configuration' in check_v13(
        tmp_path, balanced_records, provenance, invariance=False)['errors']
    changed = deepcopy(original)
    changed[index]['geometry_checks'] += 1
    write_jsonl(path, changed)
    assert sid+':teacher_scores_or_configuration' in check_v13(
        tmp_path, balanced_records, provenance, invariance=False)['errors']
    changed = deepcopy(original)
    changed[index]['target_hash'] = 'corrupt'
    write_jsonl(path, changed)
    assert sid+':teacher_target_hash' in check_v13(
        tmp_path, balanced_records, provenance, invariance=False)['errors']
