"""Confirm the sealed holdout does not overlap the 20,000-scene dataset."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/workspace/knolling_bot_project')
official = json.loads((ROOT / 'artifacts/generalization_20000/dataset/annotations.json').read_text())
records = official['records']
official_families = {row['scene_family_id'] for row in records}
official_rgb = set()
official_assets = set()
official_shapes = Counter()
official_colors = Counter()
official_counts = Counter()
official_modes = Counter()
official_group = Counter()
official_gap = Counter()
official_region = Counter()
official_axis = Counter()
seen_family = set()
for row in records:
    for path, digest in row.get('file_hashes', {}).items():
        if path.endswith('rgb.png'):
            official_rgb.add(digest)
    if row['scene_family_id'] in seen_family:
        continue
    seen_family.add(row['scene_family_id'])
    objects = row['source']['objects']
    official_counts[len(objects)] += 1
    for obj in objects:
        official_shapes[obj['asset']['shape']] += 1
        official_colors[obj['color']] += 1
        official_assets.add(obj['asset']['family_id'])
for row in records:
    instruction = row['instruction']
    official_modes[row['mode']] += 1
    official_group[instruction['group_by']] += 1
    official_gap[instruction.get('gap_m')] += 1
    official_region[instruction.get('region')] += 1
    official_axis[instruction.get('axis_alignment')] += 1

hold_root = ROOT / 'artifacts/v2_independent_eval'
bundles = []
for path in sorted((hold_root / 'families').glob('*/family.json')):
    bundles.append(json.loads(path.read_text()))
hold_records = [row for bundle in bundles for row in bundle['records']]
hold_families = {row['scene_family_id'] for row in hold_records}
hold_rgb = set()
hold_assets = set()
hold_shapes = Counter()
hold_colors = Counter()
hold_counts = Counter()
hold_modes = Counter()
hold_group = Counter()
hold_gap = Counter()
hold_region = Counter()
hold_axis = Counter()
for bundle in bundles:
    row = bundle['records'][0]
    for path, digest in row['file_hashes'].items():
        if path.endswith('rgb.png'):
            hold_rgb.add(digest)
    objects = row['source']['objects']
    hold_counts[len(objects)] += 1
    for obj in objects:
        hold_shapes[obj['asset']['shape']] += 1
        hold_colors[obj['color']] += 1
        hold_assets.add(obj['asset']['family_id'])
for row in hold_records:
    instruction = row['instruction']
    hold_modes[row['mode']] += 1
    hold_group[instruction['group_by']] += 1
    hold_gap[round(instruction['gap_m'], 3)] += 1
    hold_region[instruction['region']] += 1
    hold_axis[instruction['axis_alignment']] += 1

official_gap = Counter({round(key, 3) if isinstance(key, float) else key: value for key, value in official_gap.items()})
print(json.dumps(dict(
    official_records=len(records),
    official_families=len(official_families),
    hold_records=len(hold_records),
    hold_families=len(hold_families),
    family_overlap=len(official_families & hold_families),
    rgb_overlap=len(official_rgb & hold_rgb),
    hold_rgb=len(hold_rgb),
    asset_overlap=len(official_assets & hold_assets),
    hold_assets=len(hold_assets),
    official_assets=len(official_assets),
    hold_counts=dict(sorted(hold_counts.items())),
    official_family_counts=dict(sorted(official_counts.items())),
    hold_shapes=dict(hold_shapes),
    official_shapes=dict(official_shapes),
    hold_colors=dict(hold_colors),
    official_colors=dict(official_colors),
    hold_modes=dict(hold_modes),
    official_modes=dict(official_modes),
    hold_group=dict(hold_group),
    official_group=dict(official_group),
    hold_gap=dict(hold_gap),
    official_gap=dict(official_gap),
    hold_region=dict(hold_region),
    official_region=dict(official_region),
    hold_axis=dict(hold_axis),
    official_axis=dict(official_axis),
), indent=2))
