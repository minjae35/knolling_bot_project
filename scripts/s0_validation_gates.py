"""Predeclared admission rules for disposable validations, never S0 metrics."""
import math

MICRO_LIMIT = 100
PILOT_LIMIT = 200
STABLE_EVALUATIONS = 3
DETECTION_SAMPLES = (
    'scene_0000_primary', 'scene_0044_primary', 'scene_0070_primary',
    'scene_0065_primary', 'scene_0092_primary',
)
LANGUAGE_SAMPLES = ('scene_0051_primary', 'scene_0051_counterfactual')


def decision_exit_code(decision):
    if decision not in ('GO', 'NO-GO'):
        raise ValueError('unknown validation decision')
    return 0 if decision == 'GO' else 2


def repair_admitted(candidate, completed):
    """A failed second candidate terminates the entire repair sequence."""
    if any(completed.get(n) is False for n in ('detection2', 'language2')):
        return False
    if candidate == 'detection1':
        return not completed
    if candidate == 'detection2':
        return completed.get('detection1') is False
    detection_ok = any(completed.get(n) is True for n in ('detection1', 'detection2'))
    if candidate == 'language1':
        return detection_ok and 'language1' not in completed
    if candidate == 'joint':
        return detection_ok and completed.get('language1') is True
    return False


def step_allowed(completed, limit):
    if type(completed) is not int or type(limit) is not int:
        raise ValueError('update counts must be integers')
    if limit not in (MICRO_LIMIT, PILOT_LIMIT) or not 0 <= completed <= limit:
        raise ValueError('invalid bounded update budget')
    return completed < limit


def finite_le(value, limit):
    return isinstance(value, (int, float)) and math.isfinite(value) and value <= limit


def detection_pass(result):
    return (result['count_correct'] == len(DETECTION_SAMPLES)
            and result['fn'] == result['fp'] == 0
            and finite_le(result['max_unmatched_confidence'], .45)
            and finite_le(-result['min_matched_confidence'], -.55)
            and result['finite'])


def language_pass(result):
    return (result['count_correct'] == 2 and result['fn'] == result['fp'] == 0
            and len(result['instructions']) == 2
            and all(finite_le(v['target_position_p95_mm'], 1.)
                    and finite_le(v['target_angle_p95_deg'], 2.)
                    for v in result['instructions'])
            and finite_le(-result['minimum_pair_angle_change_deg'], -86.)
            and result['finite'])


def stable(history, predicate):
    return (len(history) >= STABLE_EVALUATIONS
            and all(predicate(r) for r in history[-STABLE_EVALUATIONS:]))


def pilot_pass(result, baseline):
    d, p = result['detection'], result['conditional_pose']
    return (d['count_correct'] == 32 and d['fn'] == d['fp'] == 0
            and finite_le(p['source_position_mm']['p95'], 1.)
            and finite_le(p['source_size_mm']['p95'], 1.)
            and finite_le(p['source_angle_deg']['p95'], 2.)
            and finite_le(p['target_position_mm']['p95'], 2.)
            and finite_le(p['target_angle_deg']['p95'], 4.)
            and language_pass(result['language'])
            and all(result['official']['by_mode'][mode]['successes'] >=
                    baseline['official']['by_mode'][mode]['successes']
                    for mode in ('full', 'partial', 'no_op'))
            and result['official']['task_success'] >= .75
            and result['finite'])


def family_groups(records, shuffled_families, width=16):
    """Each family has unit weight; its existing instructions split that weight."""
    families = {}
    for i, record in enumerate(records):
        families.setdefault(record['scene_family_id'], []).append(i)
    for start in range(0, len(shuffled_families), width):
        names = shuffled_families[start:start + width]
        yield [(i, 1 / len(families[name]) / len(names))
               for name in names for i in families[name]]


def protocol():
    return {
        'schema': 's0_bounded_validation_protocol_v1',
        'micro_limit_each': MICRO_LIMIT, 'joint_pilot_limit': PILOT_LIMIT,
        'evaluation_interval_micro': 10, 'evaluation_interval_pilot': 20,
        'consecutive_passing_evaluations_required': STABLE_EVALUATIONS,
        'detection_samples': DETECTION_SAMPLES, 'language_samples': LANGUAGE_SAMPLES,
        'detection_training': 'existing final 80-class affine classifier only; mean existing Detection loss; Target excluded; other weights/statistics frozen',
        'language_training': 'existing text projection and full Arrangement only; mean existing pair Target loss; actual predicted Source, frozen perception and MiniLM',
        'micro_batch': 2, 'micro_effective_batch': 'one mean over 5 A samples or 2 B samples; no stochastic repetition',
        'optimizer': 'fresh AdamW each phase; existing LR/decay/bias exemptions/clip; seed 1203; FP32 strict determinism',
        'detection_gate': '5/5 count; independent FN=FP=0; matched min>=.55, unmatched max<=.45; finite; all32 count must not decrease at end',
        'language_gate': 'both count-correct/FN=FP=0; each instruction Target P95<=1mm and<=2deg; every object H/V change>=86deg; finite',
        'pilot_admission': 'both independent micro-tests pass three successive evaluations; otherwise stop immediately',
        'pilot_initialization': 'original last.pt + A final classifier + B text projection/Arrangement; fresh joint AdamW',
        'pilot_training': 'all32 existing samples, all existing trainable modules, unchanged Detection+Target; paired family units, each family total weight1;16 families/effective update',
        'pilot_gate': '32/32 count;FN=FP=0;all93 Source center/size P95<=1mm,Source angle<=2deg;Target P95<=2mm/4deg;0051 language gate;Task>=75%;no mode Task regression;finite;3 successive evaluations',
        'early_stop': 'nonfinite tensor/gradient or CUDA error immediately; micro gate failure at100 forbids later phases; no candidate retries/sweeps',
        'official_s0_gate_unchanged': True, 'full_s0_automatic_start': False,
        'inference_threshold': .5, 'gt_assistance_in_inference': False,
    }
