"""Training-only fixed-feature readouts and disjoint Adam state for fused rows.

The model/state-dict architecture is unchanged. No inference caller imports this
module. A single common readout is fitted on a complete, Train-only partition.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from knolling_dataset.schema import digest
from .contracts import oracle_objects

SLICES = {'xy': slice(0, 2), 'angle': slice(2, 20), 'residual': slice(20, 38)}


def tensor_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        v = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str((v.dtype, tuple(v.shape))).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def feature_hash(model):
    # Readout rows cannot affect the features before the final Linear.
    return tensor_hash({k: v for k, v in model.state_dict().items()
                        if k not in ('layout.pose.output.1.weight', 'layout.pose.output.1.bias')})


@dataclass
class ReadoutData:
    features: torch.Tensor
    xy: torch.Tensor
    bins: torch.Tensor
    weights: torch.Tensor
    mapping: list
    producer: str
    fingerprint: str

    def payload(self):
        return {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else deepcopy(v)
                for k, v in vars(self).items()}

    @property
    def key(self):
        return digest([self.producer, self.fingerprint, self.mapping,
                       tensor_hash({k: v for k, v in vars(self).items() if isinstance(v, torch.Tensor)})])

    def validate(self, model, fingerprint, expected_ids):
        ids = [r[0] for r in self.mapping]
        if (self.producer != feature_hash(model) or self.fingerprint != fingerprint
                or set(ids) != set(expected_ids) or len(set(expected_ids)) != len(expected_ids)
                or len(self.mapping) != len(self.features) or len({tuple(r) for r in self.mapping}) != len(ids)):
            raise ValueError('stale or incomplete readout cache')
        if (self.features.ndim != 2 or self.xy.shape != (len(ids), 2)
                or self.bins.shape != (len(ids),) or self.weights.shape != (len(ids),)
                or self.bins.dtype != torch.long or not ((self.bins >= 0) & (self.bins < 18)).all()
                or not all(torch.isfinite(t).all() for t in (self.features, self.xy, self.weights))):
            raise ValueError('invalid readout tensor contract')
        # Each scene has equal total weight, regardless of object count.
        for sid in expected_ids:
            ix = [i for i, r in enumerate(self.mapping) if r[0] == sid]
            expected = self.weights.new_full((len(ix),), 1 / (len(expected_ids) * len(ix)))
            torch.testing.assert_close(self.weights[ix], expected, rtol=1e-6, atol=1e-8)


@torch.no_grad()
def collect_readout(model, batches, *, expected_ids, fingerprint, check=lambda: None):
    """Hook the *actual* forward path; labels are read only after feature capture."""
    training = model.training
    model.eval()
    rows, seen, mapping = [], set(), []
    captured = []
    hook = model.layout.pose.output[0].register_forward_hook(lambda m, a, o: captured.append(o.detach()))
    try:
        for batch in batches:
            check()
            if any(m['record']['split'] != 'train' for m in batch.metadata):
                raise ValueError('head calibration is Train-only')
            objects = oracle_objects(batch.supervision)
            captured.clear()
            text = model.encode_text(batch.inputs['input_ids'], batch.inputs['attention_mask'])
            out = model.layout(objects, text, batch.inputs['attention_mask'])
            if len(captured) != 1:
                raise ValueError('unexpected feature capture count')
            for i, meta in enumerate(batch.metadata):
                sid = meta['sample_id']; ids = meta['object_ids']; valid = objects.valid[i]
                if sid in seen or sid not in expected_ids or len(ids) != int(valid.sum()) or not len(ids):
                    raise ValueError('duplicate or incomplete calibration scene')
                seen.add(sid)
                n = len(ids); truth = batch.supervision[i]
                bins, _ = model.layout.orientation.encode(truth['target_angle'])
                rows.append((captured[0][i, valid], truth['target_xy'], bins,
                             captured[0].new_full((n,), 1 / (len(expected_ids) * n))))
                mapping.extend([[sid, oid, int(q)] for oid, q in zip(ids, out['query_index'][i, valid])])
        if seen != set(expected_ids) or not rows:
            raise ValueError('calibration omitted expected scenes')
        values = [torch.cat([r[j] for r in rows]).detach() for j in range(4)]
        cache = ReadoutData(*values, mapping, feature_hash(model), fingerprint)
        cache.validate(model, fingerprint, expected_ids)
        check()
        return cache
    finally:
        hook.remove(); model.train(training)


class HeadFitter:
    def __init__(self, model, data, kind, state=None):
        if kind not in ('xy', 'angle'):
            raise ValueError('only XY and orientation logits have calibration solvers')
        self.model, self.kind = model, kind
        linear = model.layout.pose.output[1]; s = SLICES[kind]
        self.weight = nn.Parameter(linear.weight[s].detach().clone())
        self.bias = nn.Parameter(linear.bias[s].detach().clone())
        device = self.weight.device
        self.features, self.xy, self.bins, self.weights = (
            getattr(data, k).to(device) for k in ('features', 'xy', 'bins', 'weights'))
        self.cache_key = data.key
        self.optimizer = torch.optim.LBFGS([self.weight, self.bias], lr=1., max_iter=1,
            max_eval=25, history_size=20, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')
        if state is not None:
            if state['kind'] != kind or state['cache_key'] != self.cache_key:
                raise ValueError('solver cache/kind mismatch')
            self.optimizer.load_state_dict(state['optimizer'])

    def loss(self):
        raw = F.linear(self.features, self.weight, self.bias)
        if self.kind == 'xy':
            xy = raw * raw.new_tensor([.3, .2]) + raw.new_tensor([.3, .2])
            per_object = F.smooth_l1_loss(xy / .010, self.xy / .010, reduction='none').sum(-1)
        else:
            per_object = F.cross_entropy(raw, self.bins, reduction='none')
        return (per_object * self.weights).sum()

    @torch.no_grad()
    def measurements(self):
        raw = F.linear(self.features, self.weight, self.bias)
        if self.kind == 'xy':
            error = ((raw * raw.new_tensor([.3, .2]) + raw.new_tensor([.3, .2]) - self.xy).norm(dim=-1) * 1000).max()
            return dict(value=float(error), passed=bool(error <= 1))
        probability = raw.softmax(-1).gather(-1, self.bins[:, None]).min()
        return dict(value=float(probability), passed=bool(probability >= .9))

    def state_dict(self):
        return dict(kind=self.kind, cache_key=self.cache_key, optimizer=self.optimizer.state_dict())

    @torch.no_grad()
    def apply(self):
        linear = self.model.layout.pose.output[1]; s = SLICES[self.kind]
        linear.weight[s].copy_(self.weight); linear.bias[s].copy_(self.bias)

    def step(self, check, count_closure=lambda: None):
        """Trial failures restore leaf and L-BFGS state; never publish a trial."""
        w, b = self.weight.detach().clone(), self.bias.detach().clone()
        optimizer = deepcopy(self.optimizer.state_dict())
        def closure():
            check(); count_closure(); self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite calibration loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in (self.weight, self.bias)):
                raise FloatingPointError('nonfinite calibration gradient')
            return loss
        try:
            self.optimizer.step(closure); check()
            if not all(torch.isfinite(p).all() for p in (self.weight, self.bias)):
                raise FloatingPointError('nonfinite calibration parameter')
        except BaseException:
            with torch.no_grad():
                self.weight.copy_(w); self.bias.copy_(b)
            self.optimizer.load_state_dict(optimizer)
            raise
        self.apply()
        gradient = torch.cat([p.grad.flatten() for p in (self.weight, self.bias)])
        converged = bool(gradient.abs().max() <= 1e-9 or
                         max(float((self.weight - w).detach().abs().max()), float((self.bias - b).detach().abs().max())) <= 1e-12)
        return float(self.loss().detach()), converged


class PoseAdam:
    """Split optimizer leaves only; the fused model Linear and inference stay intact."""
    def __init__(self, model, lr, weight_decay):
        self.model = model; self.linear = model.layout.pose.output[1]
        self.shared = {n: p for n, p in model.layout.pose.named_parameters()
                       if n not in ('output.1.weight', 'output.1.bias')}
        self.leaves = {}
        for kind, s in SLICES.items():
            for name in ('weight', 'bias'):
                self.leaves[kind + '.' + name] = nn.Parameter(getattr(self.linear, name)[s].detach().clone())
        self.parameters = [*self.shared.values(), *self.leaves.values()]
        self.names = [*self.shared, *self.leaves]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=lr, weight_decay=weight_decay, foreach=False)
        self.activate()

    def activate(self):
        self.model.requires_grad_(False); self.model.zero_grad(set_to_none=True)
        for p in self.shared.values(): p.requires_grad_(True)
        self.model.train()

    @contextmanager
    def forward_context(self):
        def replace(module, args, output):
            return F.linear(args[0], torch.cat([self.leaves[k + '.weight'] for k in SLICES]),
                            torch.cat([self.leaves[k + '.bias'] for k in SLICES]))
        hook = self.linear.register_forward_hook(replace)
        try: yield
        finally: hook.remove()

    @torch.no_grad()
    def sync(self):
        for kind, s in SLICES.items():
            for name in ('weight', 'bias'):
                getattr(self.linear, name)[s].copy_(self.leaves[kind + '.' + name])

    @torch.no_grad()
    def enter_fine(self):
        for n, p in self.leaves.items():
            kind, name = n.split('.')
            p.copy_(getattr(self.linear, name)[SLICES[kind]])
            if kind in ('xy', 'angle'):
                self.optimizer.state.pop(p, None)
        for p in [*self.shared.values(), self.leaves['residual.weight'], self.leaves['residual.bias']]:
            if 'exp_avg' in self.optimizer.state.get(p, {}):
                self.optimizer.state[p]['exp_avg'].zero_()
        for group in self.optimizer.param_groups: group['lr'] = 1e-6
        self.activate()

    def state_dict(self):
        return dict(names=self.names, optimizer=self.optimizer.state_dict(),
                    leaves={n: p.detach().clone() for n, p in self.leaves.items()})

    def load_state_dict(self, state):
        if state['names'] != self.names:
            raise ValueError('Pose Adam parameter mapping mismatch')
        self.optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():
            for n, p in self.leaves.items(): p.copy_(state['leaves'][n])
