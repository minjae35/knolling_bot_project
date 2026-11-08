"""CPU FP64 coordinates for the unchanged, scene-weighted XY Huber objective.

Only training-time calibration imports this module. Accepted FP32 coefficients
are copied into the existing Linear; inference never runs a solver.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .readout import HeadFitter, tensor_hash

SCHEMA = 'xy_svd_huber_v1'


class SVDXYFitter(HeadFitter):
    def __init__(self, model, data, state=None):
        self.model, self.kind, self.cache_key = model, 'xy', data.key
        linear = model.layout.pose.output[1]
        if linear.weight.dtype != torch.float32 or data.features.dtype != torch.float32:
            raise ValueError('SVD calibration requires FP32 model/features')
        self.weight = linear.weight[:2].detach().clone()
        self.bias = linear.bias[:2].detach().clone()
        self.features = data.features.to(self.weight.device)
        self.xy = data.xy.to(self.weight.device)
        self.weights = data.weights.detach().cpu().double()
        self.x = torch.cat((data.features.detach().cpu().double(),
                            torch.ones(len(data.features), 1, dtype=torch.float64)), 1)
        self.target = data.xy.detach().cpu().double()
        self.scale = torch.tensor([.3, .2], dtype=torch.float32).double()
        if (not len(self.x) or self.target.shape != (len(self.x), 2)
                or self.weights.shape != (len(self.x),) or not (self.weights > 0).all()
                or not all(torch.isfinite(t).all() for t in (self.x, self.target, self.weights))):
            raise ValueError('invalid SVD calibration data')
        if state is None:
            self.base = torch.cat((self.weight.cpu().double().T * self.scale,
                (self.bias.cpu().double()*self.scale+self.scale)[None]), 0)
            a = self.weights.sqrt()[:, None] * self.x
            u, singular, vh = torch.linalg.svd(a, full_matrices=False)
            rcond = max(a.shape)*torch.finfo(torch.float32).eps
            rank = int((singular > singular[0]*rcond).sum())
            if not rank: raise ValueError('no numerically usable feature directions')
            self.projection = vh[:rank].T / singular[:rank]
            residual = self.weights.sqrt()[:, None]*(self.target-self.x@self.base)/.010
            self.ls = u[:, :rank].T @ residual
            rank64 = int((singular > singular[0]*max(a.shape)*torch.finfo(torch.float64).eps).sum())
            self.diagnostics = dict(rank=rank, columns=a.shape[1], rows=a.shape[0], rcond=rcond,
                singular_values=singular.tolist(), condition=float(singular[0]/singular[rank-1]),
                retained_ls_rms_mm=float((residual-u[:, :rank]@self.ls).norm()*10),
                full_rank_ls_rms_mm=float((residual-u[:, :rank64]@(u[:, :rank64].T@residual)).norm()*10))
            self.z = nn.Parameter(torch.zeros_like(self.ls))
            self.warm_started = False
        else:
            if (state.get('schema') != SCHEMA or state.get('kind') != 'xy'
                    or state.get('cache_key') != self.cache_key):
                raise ValueError('SVD solver/cache contract mismatch')
            self.base, self.projection, self.ls = [state[k].detach().cpu().clone() for k in ('base', 'projection', 'ls')]
            self.z = nn.Parameter(state['z'].detach().cpu().clone())
            self.warm_started = state['warm_started']
            self.diagnostics = deepcopy(state['diagnostics'])
            rank = self.projection.shape[1]
            if (type(self.warm_started) is not bool or self.base.shape != (self.x.shape[1], 2)
                    or self.projection.shape[0] != self.x.shape[1] or self.z.shape != (rank, 2)
                    or self.ls.shape != self.z.shape or not 0 < rank <= min(self.x.shape)
                    or not all(t.dtype == torch.float64 and torch.isfinite(t).all()
                               for t in (self.base, self.projection, self.ls, self.z))
                    or state['factor_hash'] != self.factor_hash()):
                raise ValueError('invalid SVD factor state')
            a = self.weights.sqrt()[:, None]*(self.x@self.projection)
            if not torch.allclose(a.T@a, torch.eye(rank, dtype=torch.float64), atol=1e-6, rtol=1e-6):
                raise ValueError('SVD factors do not whiten the pinned cache')
            w, b = self.coefficients()
            if not torch.equal(w, self.weight) or not torch.equal(b, self.bias):
                raise ValueError('SVD coordinates and checkpoint head differ')
        self.design = self.x@self.projection
        self.offset = (self.x@self.base-self.target)/.010
        self.optimizer = torch.optim.LBFGS([self.z], lr=1., max_iter=1, max_eval=25,
            history_size=20, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')
        if state is not None: self.optimizer.load_state_dict(state['optimizer'])

    def factor_hash(self):
        return tensor_hash(dict(base=self.base, projection=self.projection, ls=self.ls))

    def coefficients(self):
        physical = self.base + .010*(self.projection@self.z.detach())
        return ((physical[:-1]/self.scale).T.to(self.weight).contiguous(),
                ((physical[-1]-self.scale)/self.scale).to(self.bias))

    def loss(self):
        residual = self.offset+self.design@self.z
        return (F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction='none').sum(-1)*self.weights).sum()

    @torch.no_grad()
    def deployed_xy(self, w, b):
        # Preserve the deployed 38-row GEMM shape; a two-row GEMM can round
        # differently, especially after amplifying weak feature directions.
        linear = self.model.layout.pose.output[1]
        weight, bias = linear.weight.detach().clone(), linear.bias.detach().clone()
        weight[:2].copy_(w); bias[:2].copy_(b)
        return F.linear(self.features, weight, bias)[:, :2]*self.scale.to(w)+self.scale.to(w)

    @torch.no_grad()
    def deployed_loss(self, w, b):
        xy = self.deployed_xy(w, b)
        return (F.smooth_l1_loss(xy/.010, self.xy/.010, reduction='none').sum(-1)*self.weights.to(xy)).sum()

    @torch.no_grad()
    def measurements(self):
        error = (self.deployed_xy(self.weight, self.bias)-self.xy).norm(dim=-1).max()*1000
        return dict(value=float(error), passed=bool(error <= 1), svd=deepcopy(self.diagnostics))

    def state_dict(self):
        return dict(schema=SCHEMA, kind='xy', cache_key=self.cache_key,
            base=self.base.clone(), projection=self.projection.clone(), ls=self.ls.clone(),
            z=self.z.detach().clone(), warm_started=self.warm_started,
            factor_hash=self.factor_hash(), diagnostics=deepcopy(self.diagnostics),
            optimizer=deepcopy(self.optimizer.state_dict()))

    def step(self, check, count_closure=lambda: None):
        """One bounded candidate; the first LS proposal also counts as one update.

        A rejected LS proposal falls back to Huber in these same coordinates.
        No new head reaches the model until deadline/finite/cast checks pass.
        """
        z = self.z.detach().clone(); warm = self.warm_started
        opt = deepcopy(self.optimizer.state_dict())
        before = float(self.loss().detach())
        deployed_before = float(self.deployed_loss(self.weight, self.bias))
        def closure():
            check(); count_closure(); self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite SVD Huber loss')
            loss.backward()
            if not torch.isfinite(self.z.grad).all(): raise FloatingPointError('nonfinite SVD gradient')
            return loss
        try:
            check()
            used_ls = False
            if not self.warm_started:
                count_closure()
                with torch.no_grad(): self.z.copy_(self.ls)
                w, b = self.coefficients()
                if (float(self.loss().detach()) <= before
                        and float(self.deployed_loss(w, b)) <= deployed_before+1e-7):
                    used_ls = True
                else:
                    with torch.no_grad(): self.z.copy_(z)
                self.warm_started = True
            if not used_ls: self.optimizer.step(closure)
            check()
            loss = self.loss()
            gradient, = torch.autograd.grad(loss, self.z)
            w, b = self.coefficients()
            if not all(torch.isfinite(t).all() for t in (loss, gradient, w, b)):
                raise FloatingPointError('nonfinite SVD candidate')
            # Reject cast/rounding regressions instead of silently publishing them.
            if float(self.deployed_loss(w, b)) > deployed_before+1e-7:
                raise FloatingPointError('FP32 SVD candidate increases Huber loss')
            check()
        except BaseException:
            with torch.no_grad(): self.z.copy_(z)
            self.warm_started = warm; self.optimizer.load_state_dict(opt)
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.weight.copy_(w); self.bias.copy_(b); self.apply()
        return float(loss.detach()), bool(gradient.abs().max() <= 1e-9)
