"""Separable log-domain contractions for the continuum heat kernel."""
import math
import os
from functools import partial
import numpy as np
import scipy.special as ssp
import jax
import jax.numpy as jnp
from jax import lax
jax.config.update('jax_enable_x64', True)
NEG = -1e+200
CMAX = 300.0

def _logdiff_ndtr(b, a):
    flip = a + b > 0
    a2 = np.where(flip, -b, a)
    b2 = np.where(flip, -a, b)
    lb = ssp.log_ndtr(b2)
    la = ssp.log_ndtr(a2)
    with np.errstate(divide='ignore'):
        return lb + np.log(-np.expm1(la - lb))

class Kernel1D:

    def __init__(self, s, kind='point', cell=None):
        self.s, self.kind, self.cell = (float(s), kind, cell)
        self.sig = math.sqrt(2 * self.s)

    def ell(self, D):
        D = np.asarray(D, float)
        s = self.s
        if self.kind == 'point':
            return -D * D / (4 * s) - 0.5 * math.log(4 * math.pi * s)
        h = self.cell
        a = (D - h / 2) / self.sig
        b = (D + h / 2) / self.sig
        return _logdiff_ndtr(b, a) - math.log(h)

    def dell(self, D):
        D = np.asarray(D, float)
        s = self.s
        if self.kind == 'point':
            return -D / (2 * s)
        h = self.cell
        a = (D - h / 2) / self.sig
        b = (D + h / 2) / self.sig
        ld = _logdiff_ndtr(b, a)
        lphi = lambda z: -0.5 * z * z - 0.5 * math.log(2 * math.pi)
        return (np.exp(lphi(b) - ld) - np.exp(lphi(a) - ld)) / self.sig

    def width(self):
        w = math.sqrt(4 * self.s * 40.0)
        return w + (self.cell / 2 if self.kind == 'avg' else 0.0)

def _ratio(hin, hout):
    for b in range(1, 65):
        a = b * hout / hin
        if abs(a - round(a)) < 1e-09 * max(1.0, a) and round(a) >= 1:
            return (int(round(a)), b)
    raise ValueError(f'hin/hout = {hin / hout} not a simple ratio')

def _index_map(j, N, bc):
    j = np.asarray(j)
    if bc == 'periodic':
        return (j % N, np.ones(j.shape, bool))
    if bc == 'neumann':
        m = j % (2 * N)
        return (np.where(m >= N, 2 * N - 1 - m, m), np.ones(j.shape, bool))
    ok = (j >= 0) & (j < N)
    return (np.clip(j, 0, N - 1), ok)

def ell_j(D, s, kind, cell):
    if kind == 'point':
        return -D * D / (4 * s) - 0.5 * jnp.log(4 * jnp.pi * s)
    sig = jnp.sqrt(2 * s)
    a = (D - cell / 2) / sig
    b = (D + cell / 2) / sig
    flip = a + b > 0
    a2 = jnp.where(flip, -b, a)
    b2 = jnp.where(flip, -a, b)
    lb = jax.scipy.special.log_ndtr(b2)
    la = jax.scipy.special.log_ndtr(a2)
    return lb + jnp.log(-jnp.expm1(la - lb)) - math.log(cell)

def dell_j(D, s, kind, cell):
    if kind == 'point':
        return -D / (2 * s)
    sig = jnp.sqrt(2 * s)
    a = (D - cell / 2) / sig
    b = (D + cell / 2) / sig
    ld = ell_j(D, s, kind, cell) + math.log(cell)
    c = 0.5 * math.log(2 * math.pi)
    return (jnp.exp(-0.5 * b * b - c - ld) - jnp.exp(-0.5 * a * a - c - ld)) / sig

class Conv1D:

    def __init__(self, kernel, Nin, hin, x0, Nout, hout, X0, bc='neumann', window=np.inf, variants=('v',), s_min=None):
        if isinstance(kernel, Kernel1D):
            kind, cell, s = (kernel.kind, kernel.cell, kernel.s)
        else:
            kind, cell, s = kernel
        self.kind, self.cell, self.s = (kind, cell, float(s))
        s_min = self.s if s_min is None else float(s_min)
        self.variants = tuple(variants)
        self.Nin, self.Nout, self.hin, self.hout, self.bc = (Nin, Nout, hin, hout, bc)
        a, b = _ratio(hin, hout)
        Tmax = math.sqrt(4 * s_min * CMAX) * 0.95
        mult = max(1, int(Tmax // (a * hin)))
        L = Nin * hin
        mult = min(mult, max(1, int(math.ceil(min(2 * L, 2 * window + hin) / (a * hin)))))
        nin, nout = (a * mult, b * mult)
        T = nin * hin
        self.nin, self.nout, self.T = (nin, nout, T)
        nI = -(-Nout // nout)
        self.nI = nI
        j0 = int(math.floor((X0 - x0) / hin))
        c0 = X0 - x0 - j0 * hin
        W = min(window, (Nin + 2) * hin * (2 if bc == 'neumann' else 1) + 2 * T)
        self.W = W
        kmin = int(math.ceil((c0 - W - (nout - 1) * hout) / T)) - 1
        kmax = int(math.floor((c0 + W + (nin - 1) * hin) / T)) + 1
        u = np.arange(nout) * hout
        v = np.arange(nin) * hin
        ks, Ds = ([], [])
        for k in range(kmin, kmax + 1):
            D = c0 - k * T + u[:, None] - v[None, :]
            if np.min(np.abs(D)) > W:
                continue
            ks.append(k)
            Ds.append(D)
        self.ks = np.array(ks)
        self.kmin, self.kmax = (int(self.ks.min()), int(self.ks.max()))
        self.D = jnp.asarray(np.stack(Ds))
        self.inside = jnp.asarray(np.abs(np.stack(Ds)) <= W)
        self.Dc = jnp.asarray(c0 - self.ks * T + u.mean() - v.mean())
        self.du = jnp.asarray(u - u.mean())
        self.dv = jnp.asarray(v - v.mean())
        self.koff = jnp.asarray(self.ks - self.kmin)
        nJ = nI + self.kmax - self.kmin
        jext = j0 + self.kmin * nin + np.arange(nJ * nin)
        idx, ok = _index_map(jext, Nin, bc)
        self.idx = jnp.asarray(idx)
        self.ok = jnp.asarray(ok)
        self.nJ = nJ
        self.npairs = len(ks)
        self.flops = 2 * nI * nout * nin * len(ks) * len(self.variants)

    def __call__(self, a, V=None, s=None):
        s = jnp.asarray(self.s if s is None else s, dtype=jnp.float64)
        return _conv_apply(s, a, V, self.D, self.inside, self.Dc, self.du, self.dv, self.koff, self.idx, self.ok, kind=self.kind, cell=self.cell, variants=self.variants, nI=self.nI, nout=self.nout, nin=self.nin, nJ=self.nJ, Nout=self.Nout)

@partial(jax.jit, static_argnames=('kind', 'cell', 'variants', 'nI', 'nout', 'nin', 'nJ', 'Nout'))
def _conv_apply(s, a, V, D, inside, Dc, du, dv, koff, idx, ok, *, kind, cell, variants, nI, nout, nin, nJ, Nout):
    l0 = ell_j(Dc, s, kind, cell)
    l1 = dell_j(Dc, s, kind, cell)
    alpha_ = l0[:, None] + l1[:, None] * du[None, :]
    beta_ = -l1[:, None] * dv[None, :]
    R = ell_j(D, s, kind, cell) - alpha_[:, :, None] - beta_[:, None, :]
    E = jnp.where(inside, jnp.exp(jnp.minimum(R, 0.0)), 0.0)
    Qs = jnp.stack([E if var == 'v' else E * jnp.where(inside, dell_j(D, s, kind, cell), 0.0) for var in variants], 1)
    R_ = a.shape[0]
    C = 1 if V is None else V.shape[-1]
    ae = jnp.where(ok[None, :], a[:, idx], NEG).reshape(R_, nJ, nin)
    Ve = None if V is None else V[:, idx, :].reshape(R_, nJ, nin, C)
    M = len(variants)

    def step(carry, xs):
        Lr, Sr = carry
        ko, Q, alpha, beta = xs
        at = lax.dynamic_slice_in_dim(ae, ko, nI, axis=1) + beta[None, None, :]
        bmax = jnp.max(at, axis=-1)
        B = jnp.exp(at - bmax[..., None])
        if Ve is None:
            S = jnp.einsum('rIv,muv->rIum', B, Q)[..., None]
        else:
            Vt = lax.dynamic_slice_in_dim(Ve, ko, nI, axis=1)
            S = jnp.einsum('rIvc,muv->rIumc', B[..., None] * Vt, Q)
        L = alpha[None, None, :] + bmax[..., None]
        Ln = jnp.maximum(Lr, L)
        Sr = Sr * jnp.exp(Lr - Ln)[..., None, None] + S * jnp.exp(L - Ln)[..., None, None]
        return ((Ln, Sr), None)
    L0 = jnp.full((R_, nI, nout), 2 * NEG)
    S0 = jnp.zeros((R_, nI, nout, M, C))
    (L, S), _ = lax.scan(step, (L0, S0), (koff, Qs, alpha_, beta_))
    return (L.reshape(R_, nI * nout)[:, :Nout], S.reshape(R_, nI * nout, M, C)[:, :Nout])

def lse_of(L, S):
    return L + jnp.log(S[..., 0, 0])

def sep2d(cx, cy, a, V=None, ycoord=None, s=None):
    Lx, Sx = cx(a, V, s)
    s0 = Sx[..., :1, :1]
    ax = jnp.where(s0[..., 0, 0] > 0, Lx + jnp.log(jnp.where(s0[..., 0, 0] > 0, s0[..., 0, 0], 1.0)), 2 * NEG)
    rat = jnp.where(s0 > 0, Sx / jnp.where(s0 > 0, s0, 1.0), 0.0).reshape(Sx.shape[0], Sx.shape[1], -1)
    if ycoord is not None:
        rat = jnp.concatenate([rat, jnp.broadcast_to(jnp.asarray(ycoord)[:, None, None], rat.shape[:2] + (1,))], -1)
    Ly, Sy = cy(ax.T, jnp.transpose(rat, (1, 0, 2)), s)
    sy0 = Sy[..., :1, :1]
    lv = jnp.where(sy0[..., 0, 0] > 0, Ly + jnp.log(jnp.where(sy0[..., 0, 0] > 0, sy0[..., 0, 0], 1.0)), 2 * NEG)
    Rt = jnp.where(sy0 > 0, Sy / jnp.where(sy0 > 0, sy0, 1.0), 0.0)
    return (lv.T, jnp.transpose(Rt, (1, 0, 2, 3)))
