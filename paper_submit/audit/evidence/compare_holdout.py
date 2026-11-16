"""Compare the one v1 run and the one v2 run on the sealed holdout."""
import json
import random
from collections import Counter, defaultdict

v1 = {row['sample_id']: row for row in map(json.loads, open('/tmp/v1_holdout.jsonl'))}
v2 = {row['sample_id']: row for row in map(json.loads, open('/tmp/v2_holdout.jsonl'))}
assert set(v1) == set(v2) == set(v2)
assert len(v1) == 640

def rate(rows, mode=None):
    chosen = list(rows.values()) if mode is None else [row for row in rows.values() if row['mode'] == mode]
    return sum(row['task'] for row in chosen), len(chosen)

print('V1', {mode: rate(v1, mode) for mode in (None, 'full', 'partial', 'no_op')})
print('V2', {mode: rate(v2, mode) for mode in (None, 'full', 'partial', 'no_op')})

kept = lost = gained = 0
lost_mode = Counter()
gain_mode = Counter()
gain_from = Counter()
lost_to = Counter()
for sid, a in v1.items():
    b = v2[sid]
    if a['task'] and b['task']:
        kept += 1
    elif a['task'] and not b['task']:
        lost += 1
        lost_mode[a['mode']] += 1
        lost_to[tuple(b['errors'])] += 1
    elif not a['task'] and b['task']:
        gained += 1
        gain_mode[b['mode']] += 1
        gain_from[tuple(a['errors'])] += 1
print('kept', kept, 'lost', lost, dict(lost_mode), 'gained', gained, dict(gain_mode))
print('gained from', gain_from.most_common())
print('lost to', lost_to.most_common())

groups = {
    'detection': ('count', 'source_identity'),
    'ordering': ('size_order', 'mixed_color_row', 'noncontiguous_color_group', 'color_order'),
    'spacing': ('horizontal_gap', 'vertical_gap', 'row_alignment', 'axis_alignment'),
    'region': ('requested_region',),
    'keep': ('keep',),
    'collision': ('collision',),
    'boundary': ('boundary',),
    'clearance': ('clearance',),
}
for name, rows in (('v1', v1), ('v2', v2)):
    raw = Counter(error for row in rows.values() if not row['task'] for error in row['errors'])
    grouped = {}
    for label, names in groups.items():
        grouped[label] = sum(1 for row in rows.values() if not row['task'] and any(error in names for error in row['errors']))
    print(name, 'raw', dict(raw), 'grouped', grouped)

for name, rows in (('v1', v1), ('v2', v2)):
    print(name, 'by count')
    for count in range(2, 9):
        for mode in ('full', 'partial', 'no_op'):
            chosen = [row for row in rows.values() if row['objects'] == count and row['mode'] == mode]
            if not chosen:
                continue
            print(mode, count, sum(row['task'] for row in chosen), len(chosen))

def family_rates(rows):
    buckets = defaultdict(list)
    for row in rows.values():
        buckets[row['scene_family_id']].append(row['task'])
    assert len(buckets) == 128 and all(len(values) == 5 for values in buckets.values())
    return {family: sum(values) / 5 for family, values in buckets.items()}

f1 = family_rates(v1)
f2 = family_rates(v2)
families = sorted(f1)
rng = random.Random(20261112)
draws = []
for _ in range(10000):
    sample = [rng.choice(families) for _ in families]
    a = sum(f1[name] for name in sample) / 128
    b = sum(f2[name] for name in sample) / 128
    draws.append((a, b, b - a))
draws.sort(key=lambda item: item[2])
low, high = draws[250], draws[9749]
print('family mean v1', sum(f1.values()) / 128, 'v2', sum(f2.values()) / 128)
print('diff ci', low[2], high[2], 'point', sum(f2.values()) / 128 - sum(f1.values()) / 128)
print('families v2 better', sum(f2[name] > f1[name] for name in families),
      'worse', sum(f2[name] < f1[name] for name in families),
      'same', sum(f2[name] == f1[name] for name in families))
for name, rows in (('v1', v1), ('v2', v2)):
    seconds = sorted(row['seconds'] for row in rows.values())
    print(name, 'median', seconds[len(seconds) // 2], 'mean', sum(seconds) / len(seconds))
