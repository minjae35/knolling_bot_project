"""Storage/atomicity tests on small tensors, not experiment training."""
from collections import OrderedDict
from copy import deepcopy
import shutil

import pytest
import torch

from knolling_model.arrangement import driver, storage
from test_recovery_integrated import equal
from test_recovery_readout import DEVICES


def state(device='cpu'):
    torch.manual_seed(43)
    x = OrderedDict((k, torch.randn(*shape, device=device)) for k, shape in [
        ('text_encoder.weight', (8, 4)), ('perception.weight', (4, 4)),
        ('perception.running_mean', (4,)), ('layout.goal_encoder.weight', (2, 4)),
        ('layout.pose.weight', (2, 4))])
    x['perception.num_batches_tracked'] = torch.tensor(7, device=device)
    x._metadata = {'': {'version': 1}}
    return x


def payload(x, stage='micro_pose'):
    return dict(schema=driver.CHECKPOINT_SCHEMA, stage=stage, status='paused', model=x,
        optimizer={'state': {0: {'step': torch.tensor(3.), 'exp_avg': torch.ones(2, 4),
                                'exp_avg_sq': torch.full((2, 4), .3)}}, 'param_groups': [{'params': [0]}]},
        rng=driver.rng_state(), cursor={'update': 3, 'epoch': 2, 'batch': 1, 'elapsed_seconds': 4.},
        integrated={'phase': 'head_xy', 'pending_head': {'phase_done': True},
                    'solver': {'history': [torch.ones(3)]}, 'fine_resets': 0,
                    'timing': {'phase': {'solver': .2}, 'total': {'solver': .2}}},
        baseline={'success': ['a']}, completed={'grounding': {'pass': True}})


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('stage', sorted(storage.STAGES))
def test_stage_partition_exact_model_buffers_and_execution_roundtrip(tmp_path, device, stage):
    store = storage.CheckpointStore(tmp_path/'checkpoint_store'); x = state(device)
    value = payload(x, stage); path = tmp_path/'state.pt'
    store.save_checkpoint(path, value)
    raw = torch.load(path, weights_only=True)
    prefixes = {r['prefix'] for r in raw['shared']}
    assert prefixes == {'text_encoder.'} | ({'perception.'} if stage in storage.FROZEN_PERCEPTION else set())
    if stage in ('perception', 'joint'):
        assert {'perception.weight', 'perception.running_mean', 'perception.num_batches_tracked'} <= raw['local'].keys()
    restored = storage.load_artifact(path)
    equal(restored, driver.cpu_copy(value))
    assert restored['model']._metadata == x._metadata
    store.save_anchor(tmp_path/'anchor.pt', x, stage)
    equal(storage.load_artifact(tmp_path/'anchor.pt'), driver.cpu_copy(x))
    # Mutable module state and buffers must survive a subsequent snapshot too.
    x['layout.pose.weight'].add_(2); x['perception.running_mean'].add_(.5)
    x['perception.num_batches_tracked'].add_(1)
    store.save_checkpoint(tmp_path/'next.pt', payload(x, stage))
    equal(storage.load_artifact(tmp_path/'next.pt')['model'], driver.cpu_copy(x))
    assert storage.load_artifact(path)['model']['perception.num_batches_tracked'].item() == 7


def test_cross_run_reuse_immutable_objects_and_no_overwrite(tmp_path):
    store = storage.CheckpointStore(tmp_path/'checkpoint_store'); x = state()
    store.save_checkpoint(tmp_path/'first.pt', payload(x))
    blobs = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns, storage._file_hash(p)) for p in store.root.glob('*.pt')}
    x['layout.pose.weight'].add_(3)
    storage.CheckpointStore(store.root).save_checkpoint(tmp_path/'second.pt', payload(x))
    assert blobs == {p.name: (p.stat().st_ino, p.stat().st_mtime_ns, storage._file_hash(p)) for p in store.root.glob('*.pt')}
    with pytest.raises(FileExistsError): store.save_checkpoint(tmp_path/'first.pt', payload(x))
    # A direct .data buffer mutation does not have to increment Tensor._version;
    # exact comparisons, not only version counters, detect and preserve it.
    x['perception.running_mean'].data.add_(1)
    store.save_checkpoint(tmp_path/'third.pt', payload(x))
    assert len(list(store.root.glob('*.pt'))) == 3
    equal(storage.load_artifact(tmp_path/'third.pt')['model'], driver.cpu_copy(x))


def test_shared_cache_preserves_signed_zero_and_nan_bits(tmp_path):
    store = storage.CheckpointStore(tmp_path/'checkpoint_store'); x = state()
    x['text_encoder.weight'][0, :2] = torch.tensor([0., float('nan')])
    store.save_checkpoint(tmp_path/'first.pt', payload(x))
    x['text_encoder.weight'][0, 0] = -0.
    store.save_checkpoint(tmp_path/'second.pt', payload(x))
    assert len(list(store.root.glob('*.pt'))) == 3
    restored = storage.load_artifact(tmp_path/'second.pt')['model']
    assert storage._equal_state(x, restored)
    store.save_checkpoint(tmp_path/'third.pt', payload(x))
    assert len(list(store.root.glob('*.pt'))) == 3


@pytest.mark.parametrize('corruption', ['missing', 'bytes', 'cached_mutation', 'missing_partition',
                                      'duplicate_partition', 'local', 'shape'])
def test_corrupt_objects_and_manifests_fail_closed(tmp_path, corruption):
    store = storage.CheckpointStore(tmp_path/'checkpoint_store'); x = state(); p = tmp_path/'state.pt'
    store.save_checkpoint(p, payload(x)); raw = torch.load(p, weights_only=True)
    blob = store.root/(raw['shared'][0]['content_sha256'] + '.pt')
    if corruption == 'missing': blob.unlink()
    elif corruption in ('bytes', 'cached_mutation'): blob.write_bytes(b'corrupt synthetic blob')
    elif corruption == 'missing_partition': raw['shared'].pop()
    elif corruption == 'duplicate_partition': raw['shared'].append(deepcopy(raw['shared'][0]))
    elif corruption == 'local': next(iter(raw['local'].values())).add_(1)
    elif corruption == 'shape': raw['descriptors']['layout.pose.weight']['shape'] = [100]
    if corruption not in ('missing', 'bytes', 'cached_mutation'):
        p = tmp_path/'bad-manifest.pt'; driver.save_exclusive(p, raw)
    with pytest.raises(ValueError): storage.load_artifact(p)
    if corruption == 'cached_mutation':
        with pytest.raises(ValueError, match='immutable shared object changed'):
            store.save_checkpoint(tmp_path/'new.pt', payload(x))
        assert not (tmp_path/'new.pt').exists()


@pytest.mark.parametrize('boundary', ['blob_write', 'after_blob', 'after_manifest'])
def test_atomic_publication_crash_boundaries(tmp_path, monkeypatch, boundary):
    store = storage.CheckpointStore(tmp_path/'checkpoint_store'); x = state(); p = tmp_path/'state.pt'
    original = driver.save_exclusive
    class Crash(BaseException): pass
    def save(path, value):
        is_blob = value.get('schema') == storage.BLOB
        if boundary == 'blob_write' and is_blob: raise Crash()
        if boundary == 'after_blob' and not is_blob: raise Crash()
        original(path, value)
        if boundary == 'after_manifest' and not is_blob: raise Crash()
    monkeypatch.setattr(driver, 'save_exclusive', save)
    with pytest.raises(Crash): store.save_checkpoint(p, payload(x))
    monkeypatch.setattr(driver, 'save_exclusive', original)
    if boundary == 'after_manifest': equal(storage.load_artifact(p), driver.cpu_copy(payload(x)))
    else:
        assert not p.exists()
        storage.CheckpointStore(store.root).save_checkpoint(p, payload(x))
        equal(storage.load_artifact(p), driver.cpu_copy(payload(x)))
    assert not list(tmp_path.rglob('.checkpoint-*'))


def test_mid_serialization_failure_leaves_prior_complete_state(tmp_path, monkeypatch):
    store = storage.CheckpointStore(tmp_path/'checkpoint_store'); x = state()
    store.save_checkpoint(tmp_path/'old.pt', payload(x)); old_hash = storage._file_hash(tmp_path/'old.pt')
    original = torch.save
    def fail(value, stream, *args, **kwargs):
        stream.write(b'partial'); raise OSError('simulated full disk')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError): store.save_checkpoint(tmp_path/'new.pt', payload(x))
    monkeypatch.setattr(torch, 'save', original)
    assert not (tmp_path/'new.pt').exists() and not list(tmp_path.rglob('.checkpoint-*'))
    assert storage._file_hash(tmp_path/'old.pt') == old_hash
    equal(storage.load_artifact(tmp_path/'old.pt'), driver.cpu_copy(payload(x)))


def test_legacy_read_export_and_bundle_relocation(tmp_path):
    root = tmp_path/'bundle'; root.mkdir(); x = state(); data = payload(x)
    driver.save_exclusive(root/'legacy.pt', data)
    old_hash = storage._file_hash(root/'legacy.pt')
    store = storage.CheckpointStore(root/'checkpoint_store')
    store.save_checkpoint(root/'split.pt', storage.load_artifact(root/'legacy.pt'))
    moved = tmp_path/'moved'; shutil.copytree(root, moved)
    equal(storage.load_artifact(moved/'split.pt'), driver.cpu_copy(data))
    # A standalone full export is an explicit new file, never an in-place rewrite.
    driver.save_exclusive(moved/'export.pt', storage.load_artifact(moved/'split.pt'))
    equal(torch.load(moved/'export.pt', weights_only=True), driver.cpu_copy(data))
    assert storage._file_hash(root/'legacy.pt') == old_hash
