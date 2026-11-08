from copy import deepcopy
import hashlib
import json

import pytest

from knolling_model.recovery.counterfactuals import build_bundle, RecoverySuite, verify_pair, input_target_contract


@pytest.fixture
def recovery_suite(s0_dataset, tmp_path):
    built = build_bundle(s0_dataset, tmp_path/'controls', pair_limit=6, min_axis_pairs=2)
    return RecoverySuite(s0_dataset, built['path'], built['sha256']), built


def test_bundle_replay_source_identity_and_pure_contrasts(recovery_suite, s0_dataset):
    suite, built = recovery_suite
    assert len(suite) == 44 and built['samples'] == 12
    assert all(built['pairs'][k] for k in ('axis_alignment', 'region', 'selection'))
    assert len({r['scene_family_id'] for r in suite.records}) == 25
    for i in range(32, len(suite), 2):
        verify_pair(suite.records[i], suite.records[i+1])
        a, b = suite[i], suite[i+1]
        assert (a['image'] == b['image']).all()
        assert a['text'] != b['text']
    assert suite.records[:32] == s0_dataset.records
    with pytest.raises(ValueError, match='new directory'):
        build_bundle(s0_dataset, suite.root)


def test_loader_rejects_semantic_and_target_tampering(recovery_suite, s0_dataset, tmp_path):
    _, built = recovery_suite
    original = json.loads(open(built['path']).read())
    for field in ('instruction', 'target', 'split'):
        bundle = deepcopy(original)
        r = bundle['records'][0]
        if field == 'instruction':
            r['instruction']['text'] = 'Keep everything.'
        elif field == 'target':
            r['target']['objects'][0]['target_pose']['x_m'] += .001
        else:
            r['split'] = 'test'
        data = json.dumps(bundle).encode()
        path = tmp_path/f'{field}.json'
        path.write_bytes(data)
        with pytest.raises(ValueError):
            RecoverySuite(s0_dataset, path, hashlib.sha256(data).hexdigest())
    a=deepcopy(s0_dataset.records[0])
    b=deepcopy(a)
    b['target']['objects'][0]['target_pose']['x_m'] += .00001
    with pytest.raises(ValueError, match='conflicting Teacher'):
        input_target_contract([a,b])
