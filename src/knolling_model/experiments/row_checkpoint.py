"""Checkpoint for the row-placement run. Weights, optimizer, step, and settings."""
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, update, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    update=int(update), config=dict(config)), path)


def load_checkpoint(path, model, optimizer=None, map_location=None):
    state = torch.load(path, map_location=map_location, weights_only=False)
    if 'model' not in state or 'optimizer' not in state or 'update' not in state or 'config' not in state:
        raise ValueError('row checkpoint is missing model, optimizer, update, or config')
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    return state
