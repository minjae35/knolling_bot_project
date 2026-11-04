"""Budget/admission tests ensure decreasing loss cannot launch a pilot."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'validation_gates', Path(__file__).parents[2] / 'scripts/s0_validation_gates.py')
gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gates)


def detection(**changes):
    return dict(count_correct=5, fn=0, fp=0, min_matched_confidence=.6,
                max_unmatched_confidence=.4, finite=True) | changes


def test_exact_counts_do_not_hide_fn_fp_or_low_margin():
    assert gates.detection_pass(detection())
    for changes in ({'fn': 1, 'fp': 1}, {'min_matched_confidence': .51},
                    {'max_unmatched_confidence': .49}, {'finite': False}):
        assert not gates.detection_pass(detection(**changes))


def test_budget_counts_actual_updates_and_rejects_larger_budget():
    assert gates.step_allowed(99, 100)
    assert not gates.step_allowed(100, 100)
    assert not gates.step_allowed(200, 200)
    for completed, limit in ((101, 100), (0, 2000), (-1, 100), (True, 100)):
        with pytest.raises(ValueError):
            gates.step_allowed(completed, limit)


def test_no_go_has_nonzero_exit_status_for_shell_admission():
    assert gates.decision_exit_code('GO') == 0
    assert gates.decision_exit_code('NO-GO') != 0
    with pytest.raises(ValueError):
        gates.decision_exit_code('loss_decreased')


def test_repair_stops_globally_after_second_failure():
    assert gates.repair_admitted('detection1', {})
    assert gates.repair_admitted('detection2', {'detection1': False})
    assert not gates.repair_admitted('detection2', {'detection1': True})
    assert not gates.repair_admitted('language1', {'detection1': False, 'detection2': False})
    assert gates.repair_admitted('language1', {'detection1': False, 'detection2': True})
    assert not gates.repair_admitted('joint', {'detection1': True})
    assert gates.repair_admitted('joint', {'detection1': True, 'language1': True})


def test_three_retained_metric_passes_required():
    assert not gates.stable([detection()] * 2, gates.detection_pass)
    assert gates.stable([detection()] * 3, gates.detection_pass)
    assert not gates.stable([detection(), detection(fn=1), detection()], gates.detection_pass)


def test_language_requires_both_ground_truths_and_sensitivity():
    r = dict(count_correct=2, fn=0, fp=0, finite=True,
             minimum_pair_angle_change_deg=89., instructions=[
                 dict(target_position_p95_mm=.8, target_angle_p95_deg=1.)] * 2)
    assert gates.language_pass(r)
    assert not gates.language_pass(r | {'minimum_pair_angle_change_deg': 1.})
    assert not gates.language_pass(r | {'instructions': []})
    assert not gates.language_pass(r | {'instructions': [r['instructions'][0],
        dict(target_position_p95_mm=.8, target_angle_p95_deg=70.)]})


def test_paired_family_weight_is_not_doubled_and_tail_normalizes():
    records = [dict(scene_family_id='a'), dict(scene_family_id='a'),
               dict(scene_family_id='b'), dict(scene_family_id='c')]
    groups = list(gates.family_groups(records, ['a', 'b', 'c'], width=2))
    assert groups == [[(0, .25), (1, .25), (2, .5)], [(3, 1.)]]
    assert all(sum(w for _, w in group) == 1 for group in groups)
