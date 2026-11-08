"""Read-only supervision and deterministic exposure plans for recovery stages."""
from collections import defaultdict
from dataclasses import replace
import random

import torch

from knolling_dataset.english import english_instruction
from knolling_dataset.schema import check_instruction
from knolling_model.data import supervision

AXES = ('horizontal', 'vertical')
REGIONS = ('full', 'left', 'right')


def recovery_supervision(record):
    ids, truth = supervision(record)
    ins = record['instruction']
    selected = set(ins['selected_ids'])
    truth['selected'] = torch.tensor([i in selected for i in ids], dtype=torch.bool)
    # Exact copying is supervised separately from the old 3mm/3degree move metric.
    from knolling_model.coordinates import angle_delta
    truth['move'] = ((truth['target_xy']-truth['source_xy']).norm(dim=-1) > 1e-6) | (
        angle_delta(truth['target_angle'], truth['source_angle']).abs() > 1e-6)
    truth['axis'] = torch.tensor(-100 if ins['keep_all'] else AXES.index(ins['axis_alignment']))
    truth['region'] = torch.tensor(-100 if ins['keep_all'] else REGIONS.index(ins['region']))
    return ids, truth


def unique_source_indices(records):
    """Perception epochs visit each source once, independent of text frequency."""
    seen, result = {}, []
    for i, record in enumerate(records):
        key = record['scene_family_id']
        # A family must actually identify the same source, not just a split group.
        source = record['source']
        if key in seen and seen[key] != source:
            raise ValueError('family contains different source states')
        if key not in seen:
            seen[key] = source
            result.append(i)
    return result


def balanced_instruction_indices(records, seed, epoch):
    """Equal full/partial/no-op visits; every instruction occurs at least once.

    Oversampling is explicit, deterministic and local to layout training. It
    does not multiply detection loss by the number of language variants.
    """
    if not records:
        raise ValueError('empty record list')
    groups = defaultdict(list)
    for i, r in enumerate(records):
        ins = r['instruction']
        mode = 'no_op' if r.get('mode') == 'no_op' or ins['keep_all'] else (
            'partial' if ins['locked_ids'] else 'full')
        groups[mode].append(i)
    rng = random.Random(seed + 100003*epoch)
    count = max(map(len, groups.values()))
    output = []
    for mode in sorted(groups):
        pool = groups[mode][:]
        rng.shuffle(pool)
        output.extend(pool[i % len(pool)] for i in range(count))
    rng.shuffle(output)
    return output


def assert_family_disjoint(partitions):
    seen = {}
    for split, records in partitions.items():
        for r in records:
            family = r['scene_family_id']
            if family in seen and seen[family] != split:
                raise ValueError(f'source family leaked across splits: {family}')
            seen[family] = split


def counterfactual_pair(scene, instruction, factor, alternative):
    """Return two English instructions changing exactly one semantic factor.

    This creates no files or targets. Each variant still needs offline teacher
    solving, feasibility validation and its original family's split assignment.
    """
    choices = {'axis_alignment': AXES, 'region': REGIONS, 'gap_m': (.015, .020, .030),
               'sort_direction': ('asc', 'desc'), 'group_by': ('none', 'color')}
    if instruction.keep_all or factor not in choices or alternative not in choices[factor]:
        raise ValueError('unsupported counterfactual')
    if getattr(instruction, factor) == alternative:
        raise ValueError('counterfactual must change the factor')
    first = english_instruction(scene, instruction)
    second = english_instruction(scene, replace(instruction, **{factor: alternative}))
    check_instruction(scene, second)
    return first, second
