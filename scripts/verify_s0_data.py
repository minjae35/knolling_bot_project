"""Read-only audit of the full English pilot and pinned 32-sample S0 selection."""
import argparse
import json
from pathlib import Path

from knolling_dataset.splits import audit_dataset
from knolling_model.data import S0Dataset


def verify(root, selection):
    result = audit_dataset(root, read_only=True)
    if not result['valid']:
        raise ValueError(f"dataset audit failed: {result['errors']}")
    if (result['samples'], result['scene_families']) != (130, 100):
        raise ValueError('expected the 130-sample, 100-family English pilot')
    dataset = S0Dataset(root, selection)
    for index in range(len(dataset)):
        dataset[index]  # Check every selected RGB shape/mode and source-target join.
    return {'valid': True, 'samples': result['samples'],
            'scene_families': result['scene_families'], 's0_samples': len(dataset),
            'manifest_sha256': dataset.selection['manifest_sha256'],
            'note': 'Data integrity only; not human approval or evidence of training success.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('artifacts/pilot_v1_3_en'))
    parser.add_argument('--selection', type=Path, default=Path('configs/s0_samples.json'))
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset, args.selection), indent=2))


if __name__ == '__main__':
    main()
