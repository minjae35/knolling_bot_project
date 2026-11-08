"""Lossless, versioned model partitions; optimizer/execution payloads stay complete.

No training, migration, overwrite or garbage collection occurs on import/load.
Shared objects are immutable and must travel with their referencing artifacts.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import re

import torch

FORMAT = 'recovery_split_storage_v1'
BLOB = 'recovery_shared_state_v1'
STAGES = {'grounding', 'micro_pose', 'oracle_layout', 'predicted_layout', 'perception', 'joint'}
FROZEN_PERCEPTION = {'grounding', 'micro_pose', 'oracle_layout', 'predicted_layout'}
HEX = re.compile(r'^[0-9a-f]{64}$')


def common_store(output):
    output = Path(output).resolve()
    for parent in output.parents:
        if parent.name == 'artifacts': return parent/'checkpoint_store'
    return output.parent/'checkpoint_store'


def _stamp(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError(f'missing or linked shared object: {path}')
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def descriptors(state):
    if not isinstance(state, dict) or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                                         or v.layout != torch.strided for k, v in state.items()):
        raise ValueError('model state must contain named dense tensors, including all persistent buffers')
    return {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in sorted(state.items())}


def tensor_digest(state):
    """Content identity independent of torch ZIP serialization and key order."""
    from knolling_dataset.schema import canonical_json
    desc = descriptors(state)
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        header = canonical_json([k, desc[k]]).encode()
        h.update(len(header).to_bytes(8, 'big')); h.update(header)
        data = v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(len(data).to_bytes(8, 'big')); h.update(data)
    return h.hexdigest()


def _cpu(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _equal_state(a, b):
    # Value equality merges +0/-0 and rejects equal NaN bit patterns. Check the
    # actual stored bits so even frozen buffers reconstruct exactly.
    return descriptors(a) == descriptors(b) and all(torch.equal(
        v.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        b[k].detach().cpu().contiguous().reshape(-1).view(torch.uint8)) for k, v in a.items())


def _load_blob(path, expected_file_hash, content):
    before = _stamp(path)
    if _file_hash(path) != expected_file_hash: raise ValueError('shared object file hash mismatch')
    raw = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(raw, dict) or set(raw) != {'schema', 'content_sha256', 'state'}
            or raw['schema'] != BLOB or raw['content_sha256'] != content
            or tensor_digest(raw['state']) != content):
        raise ValueError('shared object content mismatch')
    if _stamp(path) != before: raise ValueError('shared object changed while loading')
    return raw['state']


def load_artifact(path):
    """Resolve split model/anchor state, or read an unchanged legacy artifact.

    Returns the same full dictionary expected by existing contract checks. Plain
    torch.load reads a split envelope, not an inference-ready model state.
    """
    path = Path(path)
    raw = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(raw, dict) or 'storage_schema' not in raw: return raw
    if (set(raw) != {'storage_schema', 'kind', 'stage', 'payload', 'local', 'local_sha256',
                     'shared', 'descriptors', 'key_order', 'state_metadata'}
            or raw['storage_schema'] != FORMAT or raw['kind'] not in ('checkpoint', 'anchor')
            or raw['stage'] not in STAGES or not isinstance(raw['payload'], dict)):
        raise ValueError('invalid split storage envelope')
    if raw['kind'] == 'checkpoint' and ('model' in raw['payload'] or raw['payload'].get('stage') != raw['stage']):
        raise ValueError('split checkpoint payload mismatch')
    if raw['kind'] == 'anchor' and raw['payload']: raise ValueError('anchor payload must be empty')
    if tensor_digest(raw['local']) != raw['local_sha256']: raise ValueError('mutable model content mismatch')
    merged = dict(raw['local']); prefixes = set()
    for ref in raw['shared']:
        if (set(ref) != {'prefix', 'store', 'content_sha256', 'sha256', 'keys'}
                or ref['prefix'] not in ('text_encoder.', 'perception.') or ref['prefix'] in prefixes
                or (ref['prefix'] == 'perception.' and raw['stage'] not in FROZEN_PERCEPTION)
                or not HEX.fullmatch(ref['content_sha256']) or not HEX.fullmatch(ref['sha256'])
                or not isinstance(ref['store'], str) or Path(ref['store']).is_absolute()):
            raise ValueError('invalid shared object reference')
        prefixes.add(ref['prefix'])
        blob = path.parent/ref['store']/(ref['content_sha256'] + '.pt')
        state = _load_blob(blob, ref['sha256'], ref['content_sha256'])
        if (sorted(state) != ref['keys'] or any(not k.startswith(ref['prefix']) for k in state)
                or set(state) & set(merged)):
            raise ValueError('shared model partition overlap or key mismatch')
        merged.update(state)
    if descriptors(merged) != raw['descriptors']:
        raise ValueError('missing model keys or shape/dtype mismatch')
    order = raw['key_order']
    if len(order) != len(set(order)) or set(order) != set(merged): raise ValueError('model key order mismatch')
    state = OrderedDict((k, merged[k]) for k in order)
    if raw['state_metadata'] is not None: state._metadata = raw['state_metadata']
    if raw['kind'] == 'anchor': return state
    return dict(raw['payload'], model=state)


class CheckpointStore:
    """Small complete manifests, with exact frozen-state checks before reusing blobs.

    Frozen decisions use stage policy, not requires_grad alone. All module state
    (including buffers) is compared exactly. An unexpected change creates a NEW
    content version; it is never silently omitted. Perception/Joint always keep
    the entire Perception component local, including nominally frozen buffers.
    """
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._cached = {}

    def _shared(self, prefix, state, destination):
        from .driver import save_exclusive
        old = self._cached.get(prefix)
        if old is not None:
            snapshot, ref, stamp = old
            if _stamp(self.root/(ref['content_sha256'] + '.pt')) != stamp:
                raise ValueError('immutable shared object changed')
            if _equal_state(state, snapshot):
                return dict(ref, store=os.path.relpath(self.root, destination.parent))
        snapshot = _cpu(state); content = tensor_digest(snapshot)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root/(content + '.pt')
        blob = dict(schema=BLOB, content_sha256=content, state=snapshot)
        if not path.exists():
            try:
                save_exclusive(path, blob)
            except FileExistsError:
                # Another writer may have published this content concurrently.
                pass
        file_digest = _file_hash(path)
        checked = _load_blob(path, file_digest, content)
        if not _equal_state(checked, snapshot):
            raise ValueError('existing shared object differs from source')
        ref = dict(prefix=prefix, store=os.path.relpath(self.root, destination.parent),
                   content_sha256=content, sha256=file_digest, keys=sorted(snapshot))
        self._cached[prefix] = (snapshot, ref, _stamp(path))
        return ref

    def _save(self, path, state, payload, stage, kind):
        from .driver import save_exclusive, cpu_copy
        if stage not in STAGES: raise ValueError('unsupported checkpoint stage')
        path = Path(path).resolve()
        if path.exists() or path.is_symlink(): raise FileExistsError(path)
        desc = descriptors(state)
        prefixes = ['text_encoder.'] + (['perception.'] if stage in FROZEN_PERCEPTION else [])
        shared = []; local = dict(state)
        for prefix in prefixes:
            part = {k: v for k, v in state.items() if k.startswith(prefix)}
            if not part: continue  # Small synthetic test fixtures need not contain a text encoder.
            shared.append(self._shared(prefix, part, path))
            for k in part: local.pop(k)
        local = _cpu(local)
        save_exclusive(path, dict(storage_schema=FORMAT, kind=kind, stage=stage,
            payload=cpu_copy(payload), local=local, local_sha256=tensor_digest(local), shared=shared,
            descriptors=desc, key_order=list(state), state_metadata=deepcopy(getattr(state, '_metadata', None))))

    def save_checkpoint(self, path, payload):
        data = dict(payload); state = data.pop('model')
        self._save(path, state, data, data['stage'], 'checkpoint')

    def save_anchor(self, path, state, stage):
        self._save(path, state, {}, stage, 'anchor')
