"""Low-frequency terminal and JSON progress, independent of training math."""
import json
from pathlib import Path
import time

import torch
from tqdm import tqdm


class TrainingProgress:
    def __init__(self, run_dir, total, initial, device, state=None):
        state = state or {}
        self.path = Path(run_dir)/'progress.json'
        self.device, self.total, self.updates = device, total, initial
        self.loss_sum = state.get('loss_sum', 0.)
        self.loss_updates = state.get('loss_updates', 0)
        self.loss = state.get('current_loss')
        self.latest = state.get('latest_evaluation')
        self.previous_elapsed = state.get('elapsed_seconds', 0.)
        self.started = time.monotonic()
        self.bar = tqdm(total=total, initial=initial, desc='S0', unit='update',
                        mininterval=1., maxinterval=5., dynamic_ncols=True,
                        bar_format='{desc} {n_fmt}/{total_fmt} [{percentage:5.1f}%] '
                                   '| {elapsed}<{remaining} | {postfix}')
        self.bar.start_t -= self.previous_elapsed
        self._publish()

    def snapshot(self):
        elapsed = self.previous_elapsed+time.monotonic()-self.started
        memory = {}
        if str(self.device).startswith('cuda'):
            memory = {'vram_allocated_bytes': torch.cuda.memory_allocated(self.device),
                      'vram_reserved_bytes': torch.cuda.memory_reserved(self.device),
                      'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(self.device)}
        return {'updates': self.updates, 'total': self.total,
                'percent': 100*self.updates/self.total, 'elapsed_seconds': elapsed,
                'eta_seconds': elapsed/self.updates*(self.total-self.updates) if self.updates else None,
                'current_loss': self.loss, 'loss_sum': self.loss_sum,
                'loss_updates': self.loss_updates,
                'average_loss': self.loss_sum/self.loss_updates if self.loss_updates else None,
                'latest_evaluation': self.latest, **memory}

    def _publish(self):
        state = self.snapshot()
        fields = {}
        if self.loss is not None:
            fields.update(Loss=f'{self.loss:.4f}', Avg=f'{state["average_loss"]:.4f}')
        if 'vram_reserved_bytes' in state:
            fields['VRAM'] = f'{state["vram_reserved_bytes"]/2**30:.2f}GiB'
        if self.latest:
            fields['Eval'] = (f'Count={self.latest["count_accuracy"]:.0%} '
                              f'TgtP95={self.latest["target_position_p95_mm"]}mm '
                              f'S0={self.latest["s0_gate"]}')
        self.bar.set_postfix(fields, refresh=False)
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
        temporary.replace(self.path)

    def update(self, updates, loss):
        if updates != self.updates+1:
            raise ValueError('progress must advance by one actual optimizer update')
        self.updates = updates
        self.loss, self.loss_sum = loss, self.loss_sum+loss
        self.loss_updates += 1
        self._publish()
        self.bar.update(1)

    def evaluation(self, metrics):
        self.latest = {'count_accuracy': metrics['count_accuracy'], 's0_gate': metrics['s0_gate'],
                       'source_position_p95_mm': metrics['source_position_mm']['p95'],
                       'target_position_p95_mm': metrics['target_position_mm']['p95']}
        self._publish()
        self.bar.refresh()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._publish()
        self.bar.close()
