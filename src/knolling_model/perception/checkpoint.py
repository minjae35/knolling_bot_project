"""Independent atomic perception checkpoints with data/code/config provenance."""
import hashlib
import inspect
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from knolling_dataset.builder import file_hash
from .model import ARCHITECTURE

SCHEMA = 'standalone_perception_checkpoint_v1'


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


# Checkpoint receipts still name the package recovery. The bytes live in arrangement.
PROVENANCE_ALIASES = {
    'recovery/perception.py': 'arrangement/perception.py',
    'recovery/contracts.py': 'arrangement/contracts.py',
    'recovery/data.py': 'arrangement/data.py',
}


def code_provenance():
    root = Path(__file__).resolve().parents[1]
    names = [p.relative_to(root).as_posix() for p in sorted((root/'perception').glob('*.py'))]
    names += ['rtdetr.py', 'model.py', 'deterministic_sampling.py', 'coordinates.py',
              'recovery/perception.py', 'recovery/contracts.py', 'recovery/data.py']
    return {name: file_hash(root/PROVENANCE_ALIASES.get(name, name)) for name in names}


def model_provenance():
    """Only inference graph compatibility blocks reuse; adapter/layout edits do not.

    Hash physical head definitions instead of all of model.py, which also holds
    legacy Arrangement classes unrelated to this standalone graph.
    """
    from knolling_model.model import SourceHeads, PhysicalSourceHeads, mlp
    from knolling_model.coordinates import angle_delta, wrap_pi
    root = Path(__file__).resolve().parents[1]
    files = ['perception/model.py', 'rtdetr.py', 'deterministic_sampling.py', 'recovery/perception.py']
    result = {p: file_hash(root/PROVENANCE_ALIASES.get(p, p)) for p in files}
    for value in (SourceHeads, PhysicalSourceHeads, mlp, angle_delta, wrap_pi):
        result[value.__name__] = hashlib.sha256(inspect.getsource(value).encode()).hexdigest()
    return result


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def save(path, model, optimizer, *, config, pins, fingerprint, update, elapsed, metrics, purpose):
    """Only paths owned by this run; callers must never pass another agent's files."""
    path = Path(path)
    payload = dict(schema=SCHEMA, architecture=ARCHITECTURE, config=config, pins=pins,
        dataset_fingerprint=fingerprint, code=code_provenance(), model_code=model_provenance(),
        update=update, elapsed_seconds=elapsed,
        metrics=metrics, purpose=purpose, model=cpu_copy(model.state_dict()),
        optimizer=cpu_copy(optimizer.state_dict()) if optimizer is not None else None, rng=rng_state())
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checksum = file_hash(temporary)
        os.replace(temporary, path)
        sidecar = path.with_suffix(path.suffix+'.sha256')
        temporary_digest = Path(str(temporary)+'.sha256')
        temporary_digest.write_text(checksum+'\n')
        os.replace(temporary_digest, sidecar)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return checksum


def read(path, *, fingerprint=None, config=None, require_current_training_code=False):
    path = Path(path)
    if file_hash(path) != path.with_suffix(path.suffix+'.sha256').read_text().strip():
        raise ValueError('perception checkpoint SHA-256 mismatch')
    # Trusted, locally produced checkpoint; optimizer/RNG includes Python/NumPy state.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema') != SCHEMA or payload.get('architecture') != ARCHITECTURE:
        raise ValueError('not an independent perception checkpoint')
    if fingerprint is not None and payload['dataset_fingerprint'] != fingerprint:
        raise ValueError('perception checkpoint dataset mismatch')
    if config is not None and payload['config'] != config:
        raise ValueError('perception checkpoint config mismatch')
    if payload.get('model_code') != model_provenance():
        raise ValueError('perception model code changed; explicit reviewed migration required')
    if require_current_training_code and payload['code'] != code_provenance():
        raise ValueError('perception training code changed; repeat CPU verification')
    return payload


def restore(payload, model, optimizer=None, *, restore_random=False):
    model.load_state_dict(payload['model'], strict=True)
    if optimizer is not None:
        if payload['optimizer'] is None:
            raise ValueError('checkpoint has no optimizer state')
        optimizer.load_state_dict(payload['optimizer'])
    if restore_random:
        restore_rng(payload['rng'])
