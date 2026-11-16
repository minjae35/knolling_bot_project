"""Recompute paper statistics from archived score rows; never invokes inference."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'audit' / 'evidence'


def load(name):
    path = EVIDENCE / name
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len({row['sample_id'] for row in rows}) == 640
    assert all(row['sample_id'].startswith('v2hold_') for row in rows)
    return {row['sample_id']: row for row in rows}


v1, v2 = load('v1_holdout.jsonl'), load('v2_holdout.jsonl')
assert v1.keys() == v2.keys()
for sid in v1:
    assert all(v1[sid][k] == v2[sid][k] for k in ('scene_family_id', 'mode', 'objects'))


def summarize(rows):
    result = {}
    for mode in ('overall', 'full', 'partial', 'no_op'):
        group = [r for r in rows.values() if mode == 'overall' or r['mode'] == mode]
        success = sum(r['task'] for r in group)
        result[mode] = dict(success=success, total=len(group), percent=100 * success / len(group))
    result['error_tags'] = dict(Counter(e for r in rows.values() for e in r['errors']))
    result['unique_spacing_failure_scenes'] = sum(any(e in ('horizontal_gap', 'vertical_gap') for e in r['errors']) for r in rows.values())
    result['by_object_count'] = {}
    for n in range(2, 9):
        group = [r for r in rows.values() if r['objects'] == n]
        result['by_object_count'][n] = dict(success=sum(r['task'] for r in group), total=len(group))
    seconds = [r['seconds'] for r in rows.values()]
    result['timing_seconds'] = dict(median=statistics.median(seconds), mean=statistics.mean(seconds), note='Recorded wall times, not a hardware-controlled benchmark.')
    return result


families = defaultdict(list)
for sid, row in v1.items():
    families[row['scene_family_id']].append(sid)
assert len(families) == 128 and all(len(ids) == 5 for ids in families.values())
delta = {fid: sum(int(v2[s]['task']) - int(v1[s]['task']) for s in ids) for fid, ids in families.items()}
ids = sorted(families)
rng = random.Random(20261112)
draws = sorted(sum(delta[rng.choice(ids)] for _ in ids) / 640 * 100 for _ in range(10000))
pair = dict(preserved=sum(v1[s]['task'] and v2[s]['task'] for s in v1),
            gained=sum(not v1[s]['task'] and v2[s]['task'] for s in v1),
            lost=sum(v1[s]['task'] and not v2[s]['task'] for s in v1),
            both_failed=sum(not v1[s]['task'] and not v2[s]['task'] for s in v1),
            improved_families=sum(x > 0 for x in delta.values()),
            worsened_families=sum(x < 0 for x in delta.values()),
            tied_families=sum(x == 0 for x in delta.values()),
            difference_pp=sum(delta.values()) / 640 * 100,
            family_bootstrap=dict(seed=20261112, resamples=10000, families=128, percentile_indices=[250, 9749], ci95_pp=[draws[250], draws[9749]]))
assert pair['preserved'] == 489 and pair['gained'] == 51 and pair['lost'] == 0
result = dict(v1=summarize(v1), v2=summarize(v2), paired=pair,
              sources={name:hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest() for name in ('v1_holdout.jsonl','v2_holdout.jsonl')})
out = ROOT / 'audit' / 'verified_results.json'
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
