"""Offline v1.3 reports. Structured labels never become model inputs or gates."""
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations

import numpy as np

from knolling_dataset.english import observable_subject
from knolling_dataset.schema import scene_from_dict, instruction_from_dict, Pose
from knolling_dataset.teacher_balanced import observed_rows
from knolling_dataset.validator import geometry


def distribution(values):
    return {'median': float(np.median(values)) if values else None,
            'p95': float(np.percentile(values, 95)) if values else None, 'n': len(values)}


def success_summary(rows):
    n = len(rows)
    successes = sum(r['task_success'] for r in rows)
    return {'n': n, 'successes': successes, 'task_success': successes/n if n else None,
            'raw_geometry_success': sum(r['raw_geometry_success'] for r in rows)/n if n else None,
            'failure_counts': dict(Counter(e for r in rows for e in r['errors'])),
            'evaluated': bool(n)}


def condition_labels(record):
    source = scene_from_dict(record['source'])
    ins = instruction_from_dict(record['instruction'])
    def composition(values):
        return '|'.join(f'{name}:{n}' for name, n in sorted(Counter(values).items()))
    labels = {'object_count': str(len(source.objects)),
              'shape_composition': composition(o.asset.shape for o in source.objects),
              # Exact dimensions avoid introducing undocumented size thresholds.
              'size_composition_mm': composition(f'{o.asset.a_m*1000:g}x{o.asset.b_m*1000:g}'
                                                 for o in source.objects),
              'color_composition': composition(o.color for o in source.objects),
              'mode': record['mode'],
              'english_instruction_type': 'keep_all' if ins.keep_all else observable_subject(source, ins)}
    for name in ('axis_alignment', 'gap_m', 'region', 'group_by', 'sort_direction'):
        labels[name] = 'not_applicable' if ins.keep_all else str(getattr(ins, name))
    if record['mode'] == 'partial':
        labels['partial_selection'] = 'single_object' if len(ins.selected_ids) == 1 else 'multi_object'
        # Same structural proxy as the dataset quality audit; not proof of a
        # prior interactive/organized episode or a human aesthetic judgment.
        ids = tuple(o.object_id for o in source.objects)
        organized = any(observed_rows(source, replace(ins, selected_ids=ids, locked_ids=(),
                                                     group_by=g, sort_direction=s))
                        for g in ('none', 'color') for s in ('asc', 'desc'))
        labels['partial_source_layout'] = 'organized' if organized else 'scattered'
    return labels


def positioned_objects(record, poses):
    source = scene_from_dict(record['source'])
    return [replace(o, target_pose=Pose(**poses[o.object_id])) for o in source.objects]


def language_sensitivity(rows, records, semantic_check):
    """Same RGB, different commands: own success AND rejection of both crosses.

    Exclude pairs whose GT targets do not fail both opposite commands, using
    the same prediction tolerances as the evaluated outputs. No diagnostic
    pair or missing prediction is silently counted as a success.
    """
    by_id = {r['sample_id']: r for r in rows}
    by_rgb = defaultdict(list)
    missing_rgb = 0
    for record in records:
        key = record['source_render_dir']+'/rgb.png'
        rgb_hash = record.get('file_hashes', {}).get(key)
        if rgb_hash is None:
            missing_rgb += 1
        else:
            by_rgb[(record['scene_family_id'], rgb_hash)].append(record)
    pairs = []
    for members in by_rgb.values():
        for a, b in combinations(members, 2):
            if a['instruction']['text'] == b['instruction']['text']:
                continue
            # Ground-truth targets test whether this command contrast is
            # actually diagnostic; they do not decide predicted success.
            gt_a = list(scene_from_dict(a['target']).objects)
            gt_b = list(scene_from_dict(b['target']).objects)
            diagnostic = bool(semantic_check(gt_a, b)) and bool(semantic_check(gt_b, a))
            ra, rb = by_id[a['sample_id']], by_id[b['sample_id']]
            own = ra['task_success'] and rb['task_success']
            crosses = []
            for row, other in ((ra, b), (rb, a)):
                poses = row.get('predicted_targets')
                if row['matched_objects'] != row['gt_count'] or poses is None:
                    crosses.append(None)
                else:
                    objects = positioned_objects(other, poses)
                    crosses.append(geometry(objects, 'target_pose', .010, .010).valid
                                   and not semantic_check(objects, other))
            pairs.append({'sample_ids': [a['sample_id'], b['sample_id']],
                          'eligible': diagnostic,
                          'exclusion_reason': None if diagnostic else 'no_bidirectional_gt_contrast',
                          'own_success': bool(own), 'a_under_b_success': crosses[0],
                          'b_under_a_success': crosses[1],
                          'distinguished': bool(diagnostic and own and crosses == [False, False])})
    eligible = [p for p in pairs if p['eligible']]
    n = len(eligible)
    own_n = sum(p['own_success'] for p in eligible)
    distinguished_n = sum(p['distinguished'] for p in eligible)
    return {'candidate_pairs': len(pairs), 'eligible_pairs': n,
            'excluded_pairs': len(pairs)-n, 'samples_missing_rgb_hash': missing_rgb,
            'own_success_pairs': own_n, 'own_success_rate': own_n/n if n else None,
            'distinguished_pairs': distinguished_n,
            'sensitivity': distinguished_n/n if n else None, 'evaluated': bool(n), 'pairs': pairs}


def summarize_diagnostics(rows, records, semantic_check, stage_count=3):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for name, value in row.get('conditions', {}).items():
            groups[name][value].append(row)
    for name, values in (('partial_selection', ('single_object', 'multi_object')),
                         ('partial_source_layout', ('organized', 'scattered'))):
        for value in values:
            groups[name][value]  # Missing coverage must be explicit, not omitted.
    conditions = {name: {value: success_summary(group) for value, group in sorted(buckets.items())}
                  for name, buckets in sorted(groups.items())}
    partial = [r for r in rows if r['mode'] == 'partial']
    checked = [r for r in partial if r.get('keep_violation') is not None]
    violations = sum(r['keep_violation'] for r in checked)
    stages = []
    for i in range(stage_count):
        observed = [r['stages'][i] for r in rows if len(r.get('stages', [])) > i]
        stages.append({'stage': i+1, 'n': len(rows), 'evaluated_n': len(observed),
                       'position_mm': distribution([v for s in observed for v in s['position_mm']]),
                       'angle_deg': distribution([v for s in observed for v in s['angle_deg']]),
                       'geometry_successes': sum(s['geometry_success'] for s in observed),
                       'geometry_success': sum(s['geometry_success'] for s in observed)/len(rows) if rows else None,
                       'geometry_failure_counts': dict(Counter(e for s in observed for e in s['geometry_errors']))})
    return {'by_condition': conditions,
            'keep_violation': {'n': len(partial), 'checked_n': len(checked),
                               'unchecked_n': len(partial)-len(checked), 'violations': violations,
                               'scene_rate': violations/len(partial) if partial and len(checked) == len(partial) else None,
                               'evaluated': bool(checked)},
            'refinement': stages,
            'language_sensitivity': language_sensitivity(rows, records, semantic_check)}
