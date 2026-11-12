"""Splits that reproduce the adopted Dev scores. They do not open the test set."""
import torch

from knolling_model.experiments.full_arrangement import is_full

HOLDOUT_SEED = 1510
DEV_SEED = 1703
VAL_SEED = 90211
VAL_COUNT = 256


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
