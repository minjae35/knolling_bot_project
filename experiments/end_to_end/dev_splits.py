"""Splits that reproduce the adopted Dev scores. They do not open the test set."""
from collections import defaultdict

import torch

from knolling_model.experiments.full_arrangement import is_full

HOLDOUT_SEED = 1510
DEV_SEED = 1703
VAL_SEED = 90211
VAL_COUNT = 256
RESERVED_SEED = 44021
RESERVED_COUNT = 128
V2_SEAL_SEED = 88021
V2_SEAL_COUNT = 400


def family_splits(records):
    """One full scene per family. The first slice is the sealed family holdout."""
    found = {}
    for index, record in enumerate(records):
        if is_full(record) and record['scene_family_id'] not in found:
            found[record['scene_family_id']] = index
    ordered = [found[name] for name in sorted(found)]
    permutation = torch.randperm(len(ordered), generator=torch.Generator().manual_seed(HOLDOUT_SEED)).tolist()
    hold_count = len(ordered)//5
    dev_count = len(ordered)//5
    hold = [ordered[index] for index in permutation[:hold_count]]
    dev = [ordered[index] for index in permutation[hold_count:hold_count+dev_count]]
    fit = [ordered[index] for index in permutation[hold_count+dev_count:]]
    return fit, dev, hold


def fresh_split(records):
    """Family-disjoint dev of 512. Fit excludes the sealed holdout families."""
    _fit, _dev, held_out = family_splits(records)
    blocked = {records[index]['scene_family_id'] for index in held_out}
    found = {}
    for index, record in enumerate(records):
        if is_full(record) and record['scene_family_id'] not in found and record['scene_family_id'] not in blocked:
            found[record['scene_family_id']] = index
    ordered = [found[name] for name in sorted(found)]
    permutation = torch.randperm(len(ordered), generator=torch.Generator().manual_seed(DEV_SEED)).tolist()
    dev_count = len(ordered)//5
    dev = [ordered[index] for index in permutation[:dev_count]]
    fit = [ordered[index] for index in permutation[dev_count:]]
    return fit, dev, blocked


def heldout_validation(fit):
    """256 scenes from the fit families. They do not overlap the 512-scene dev."""
    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(VAL_SEED)).tolist()
    return [fit[index] for index in order[:VAL_COUNT]]


def families(records):
    """Train families, with Dev, holdout, and one reserved slice held out."""
    _fit, _dev, hold = family_splits(records)
    _fit2, dev512, _blocked = fresh_split(records)
    dev256 = heldout_validation(_fit2)
    blocked = {records[index]['scene_family_id'] for index in hold + dev512 + dev256}
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record['scene_family_id']].append(index)
    allowed = sorted(set(grouped) - blocked)
    order = torch.randperm(len(allowed), generator=torch.Generator().manual_seed(RESERVED_SEED)).tolist()
    reserved = {allowed[index] for index in order[:RESERVED_COUNT]}
    trainable = [name for name in allowed if name not in reserved]
    return grouped, trainable, reserved, blocked


def v2_sealed_families(records):
    """400 train families for a later v2 test. Do not score them while tuning.

    They are disjoint from the v1 dev, holdout, validation probe, and the
    reserved slice that was already scored. Pass train records only, so the
    official validation and test splits stay out. Future training must skip
    this set too.
    """
    _grouped, trainable, reserved, blocked = families(records)
    if len(trainable) < V2_SEAL_COUNT:
        raise RuntimeError('not enough untouched train families for the v2 seal')
    order = torch.randperm(len(trainable), generator=torch.Generator().manual_seed(V2_SEAL_SEED)).tolist()
    sealed = [trainable[index] for index in order[:V2_SEAL_COUNT]]
    if len(set(sealed)) != V2_SEAL_COUNT or set(sealed) & set(reserved) or set(sealed) & set(blocked):
        raise RuntimeError('v2 seal overlaps a used slice')
    return sealed
