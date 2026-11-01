"""JAX float64 heat solves, mass-conserving fluxes and particle tracking."""
import functools
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
jax.config.update('jax_enable_x64', True)
from .grids import SphereGrid, PlaneGrid
GL_X, GL_W = np.polynomial.legendre.leggauss(6)

def _eig_block(args):
    import scipy.linalg as sla
    d, off = args
    return sla.eigh_tridiagonal(d, off)

def _sph_spec(ops, f):
    h = ops['Qe'].shape[1]
    G = jnp.fft.rfft(f * ops['sqrtA'][None, :, None], axis=-1)
    G = jnp.transpose(G, (2, 1, 0))
    N, S = (G[:, h:, :], G[:, h - 1::-1, :])
    ce = (N + S) / np.sqrt(2)
    co = (N - S) / np.sqrt(2)
    Qe, Qo = (ops['Qe'], ops['Qo'])

    def fwd(Q, v):
        V = jnp.concatenate([v.real, v.imag], axis=2)
        W = jnp.einsum('mjb,mji->mbi', V, Q)
        W = jnp.swapaxes(W, 1, 2)
        B = v.shape[2]
        return W[:, :, :B] + 1j * W[:, :, B:]
    return (fwd(Qe, ce), fwd(Qo, co))

def _sph_unspec(ops, ce, co, nlon):
    Qe, Qo = (ops['Qe'], ops['Qo'])

    def inv(Q, v):
        V = jnp.concatenate([v.real, v.imag], axis=2)
        W = jnp.einsum('mij,mjb->mib', Q, V)
        B = v.shape[2]
        return W[:, :, :B] + 1j * W[:, :, B:]
    ce, co = (inv(Qe, ce), inv(Qo, co))
    N = (ce + co) / np.sqrt(2)
    S = ((ce - co) / np.sqrt(2))[:, ::-1, :]
    G = jnp.concatenate([S, N], axis=1)
    G = jnp.transpose(G, (2, 1, 0))
    return jnp.fft.irfft(G, n=nlon, axis=-1) / ops['sqrtA'][None, :, None]

def _sph_heat(ops, f, s):
    ce, co = _sph_spec(ops, f)
    ce = ce * jnp.exp(ops['Le'][:, :, None] * s[None, None, :])
    co = co * jnp.exp(ops['Lo'][:, :, None] * s[None, None, :])
    return _sph_unspec(ops, ce, co, f.shape[-1])

def _sph_poisson(ops, r):
    ce, co = _sph_spec(ops, r / ops['area'][None])
    z0 = ops['zero0']
    Le = jnp.where(z0 > 0, 1.0, ops['Le'])
    ce = jnp.where(z0[:, :, None] > 0, 0.0, ce / -Le[:, :, None])
    co = co / -ops['Lo'][:, :, None]
    return _sph_unspec(ops, ce, co, r.shape[-1])

def _dct2(f):
    return jax.scipy.fft.dctn(f, type=2, norm='ortho', axes=(-2, -1))

def _idct2(c):
    return jax.scipy.fft.idctn(c, type=2, norm='ortho', axes=(-2, -1))

def _pl_heat(ops, f, s):
    return _idct2(_dct2(f) * jnp.exp(ops['lam'][None] * s[:, None, None]))

def _pl_poisson(ops, r):
    c = _dct2(r / ops['area'][None])
    lam = ops['lam'].at[0, 0].set(1.0)
    c = (c / (-lam)[None]).at[:, 0, 0].set(0.0)
    return _idct2(c)
_sph_heat_jit = jax.jit(_sph_heat)
TRUNC = 46.0

def _trunc_sizes(Le, Lo, s_min):
    keep = lambda L: L * s_min > -TRUNC
    ce, co = (keep(Le).sum(axis=1), keep(Lo).sum(axis=1))
    c = np.maximum(ce, co)
    mM = int(np.nonzero(c)[0].max()) + 1 if c.any() else 1
    return (mM, max(1, int(c.max())))

def trunc_ops(G, s_min, bucket=True):
    Le, Lo = G.__dict__.setdefault('_L_np', (np.asarray(G.Le), np.asarray(G.Lo)))
    mM, kM = _trunc_sizes(Le, Lo, s_min)
    if bucket:
        mM = min(G.nm, 1 << max(4, int(np.ceil(np.log2(mM)))))
        kM = min(G.h, 1 << max(3, int(np.ceil(np.log2(kM)))))
    if mM * kM * 3 > G.nm * G.h:
        return None
    cache = G.__dict__.setdefault('_trunc_cache', {})
    if (mM, kM) not in cache:
        size = lambda d: 16 * int(np.prod(d['Qe'].shape))
        while cache and (len(cache) >= 6 or sum((size(d) for d in cache.values())) + 16 * mM * kM * G.h > 12 * 2 ** 30):
            cache.pop(next(iter(cache)))
        h = G.h
        o = G.ops
        cache[mM, kM] = dict(Qe=o['Qe'][:mM, :, h - kM:], Qo=o['Qo'][:mM, :, h - kM:], Le=o['Le'][:mM, h - kM:], Lo=o['Lo'][:mM, h - kM:], sqrtA=o['sqrtA'], area=o['area'])
    else:
        cache[mM, kM] = cache.pop((mM, kM))
    return cache[mM, kM]

def _sph_spec_t(ops, f):
    mM, h = (ops['Qe'].shape[0], ops['Qe'].shape[1])
    G = jnp.fft.rfft(f * ops['sqrtA'][None, :, None], axis=-1)[:, :, :mM]
    G = jnp.transpose(G, (2, 1, 0))
    N, S = (G[:, h:, :], G[:, h - 1::-1, :])
    ce = (N + S) / np.sqrt(2)
    co = (N - S) / np.sqrt(2)

    def fwd(Q, v):
        V = jnp.concatenate([v.real, v.imag], axis=2)
        W = jnp.swapaxes(jnp.einsum('mjb,mji->mbi', V, Q), 1, 2)
        B = v.shape[2]
        return W[:, :, :B] + 1j * W[:, :, B:]
    return (fwd(ops['Qe'], ce), fwd(ops['Qo'], co))

def _sph_unspec_t(ops, ce, co, nlon):

    def inv(Q, v):
        V = jnp.concatenate([v.real, v.imag], axis=2)
        W = jnp.einsum('mij,mjb->mib', Q, V)
        B = v.shape[2]
        return W[:, :, :B] + 1j * W[:, :, B:]
    ce, co = (inv(ops['Qe'], ce), inv(ops['Qo'], co))
    N = (ce + co) / np.sqrt(2)
    S = ((ce - co) / np.sqrt(2))[:, ::-1, :]
    G = jnp.transpose(jnp.concatenate([S, N], axis=1), (2, 1, 0))
    G = jnp.pad(G, ((0, 0), (0, 0), (0, nlon // 2 + 1 - G.shape[2])))
    return jnp.fft.irfft(G, n=nlon, axis=-1) / ops['sqrtA'][None, :, None]

def _sph_heat_t(ops, f, s):
    ce, co = _sph_spec_t(ops, f)
    ce = ce * jnp.exp(ops['Le'][:, :, None] * s[None, None, :])
    co = co * jnp.exp(ops['Lo'][:, :, None] * s[None, None, :])
    return _sph_unspec_t(ops, ce, co, f.shape[-1])
_sph_poisson_jit = jax.jit(_sph_poisson)
_pl_heat_jit = jax.jit(_pl_heat)
_pl_poisson_jit = jax.jit(_pl_poisson)

class SphereJ:
    kind = 'sphere'

    def __init__(self, nlon, nlat, workers=None):
        g = SphereGrid(nlon, nlat)
        self.np = g
        self.nlon, self.nlat, self.h = (nlon, nlat, nlat // 2)
        self.nm = nlon // 2 + 1
        A = g.area_row
        h = self.h
        ms = np.arange(self.nm)
        lam_x = 4 * np.sin(ms * g.dlon / 2) ** 2
        off = g.Ty_edge[1:-1] / np.sqrt(A[:-1] * A[1:])
        base = -(g.Ty_edge[:-1] + g.Ty_edge[1:]) / A
        offN, eq = (off[h:], off[h - 1])
        Qe_d = jnp.zeros((self.nm, h, h))
        Qo_d = jnp.zeros((self.nm, h, h))
        put = jax.jit(lambda Abig, Ach, k: lax.dynamic_update_slice(Abig, Ach, (k, 0, 0)), donate_argnums=0)
        ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
        workers = workers or max(1, min(64, ncpu))
        base_n, txa_n = (base[h:], g.Tx_row[h:] / A[h:])
        dev = {}

        def consume(k0_, Qe, Qo):
            dev['e'] = put(dev.get('e', Qe_d), jnp.asarray(Qe), k0_)
            dev['o'] = put(dev.get('o', Qo_d), jnp.asarray(Qo), k0_)
            jax.block_until_ready((dev['e'], dev['o']))
        import scipy.linalg as sla
        from threadpoolctl import threadpool_limits
        Le = np.empty((self.nm, h))
        Lo = np.empty((self.nm, h))
        mc = max(1, min(self.nm, int(400000000.0 // (h * h * 8))))
        with threadpool_limits(1):
            for k0_ in range(0, self.nm, mc):
                ks = list(range(k0_, min(self.nm, k0_ + mc)))
                Qe = np.empty((len(ks), h, h))
                Qo = np.empty((len(ks), h, h))
                for q, k in enumerate(ks):
                    dd = base_n - lam_x[k] * txa_n
                    de = dd.copy()
                    de[0] += eq
                    do = dd.copy()
                    do[0] -= eq
                    Le[k], Qe[q] = sla.eigh_tridiagonal(de, offN)
                    Lo[k], Qo[q] = sla.eigh_tridiagonal(do, offN)
                consume(k0_, Qe, Qo)
        Qe_d, Qo_d = (dev['e'], dev['o'])
        self.Le, self.Lo = (jnp.asarray(Le), jnp.asarray(Lo))
        self.Qe, self.Qo = (Qe_d, Qo_d)
        self.sqrtA = jnp.asarray(np.sqrt(A))
        self.area = jnp.asarray(g.area)
        self.Tx = jnp.asarray(g.Tx_row)[:, None]
        self.Ty = jnp.asarray(g.Ty_edge[1:-1])[:, None]
        self.total_area = g.total_area
        k0 = int(np.argmax(Le[0]))
        self.k0 = k0
        z0 = np.zeros_like(Le)
        z0[0, k0] = 1.0
        self.ops = dict(Qe=self.Qe, Qo=self.Qo, Le=self.Le, Lo=self.Lo, sqrtA=self.sqrtA, area=self.area, zero0=jnp.asarray(z0))
        self.heat_fn, self.poisson_fn = (_sph_heat, _sph_poisson)
        self.heat = lambda f, s: _sph_heat_jit(self.ops, f, s)
        self.poisson = lambda r: _sph_poisson_jit(self.ops, r)

    def _to_spec(self, f):
        return _sph_spec(self.ops, f)

    def _from_spec(self, ce, co):
        return _sph_unspec(self.ops, ce, co, self.nlon)

    def grad_flux(self, f):
        FX = self.Tx[None] * (f - jnp.roll(f, -1, axis=-1))
        FY = self.Ty[None] * (f[:, :-1] - f[:, 1:])
        return (FX, FY)

    def bridge_flux(self, a, b):
        FX = self.Tx[None] * (a * jnp.roll(b, -1, axis=-1) - b * jnp.roll(a, -1, axis=-1))
        FY = self.Ty[None] * (a[:, :-1] * b[:, 1:] - b[:, :-1] * a[:, 1:])
        return (FX, FY)

    def div(self, FX, FY):
        z = jnp.zeros_like(FY[:, :1])
        FYf = jnp.concatenate([z, FY, z], axis=1)
        return FX - jnp.roll(FX, 1, axis=-1) + FYf[:, 1:] - FYf[:, :-1]

class PlaneJ:
    kind = 'plane'

    def __init__(self, nx, ny, Lx=2 * np.pi, Ly=np.pi):
        g = PlaneGrid(nx, ny, Lx, Ly)
        self.np = g
        self.nx, self.ny = (nx, ny)
        self.lam = jnp.asarray(g.lam)
        self.area = jnp.asarray(g.area)
        self.Tx, self.Ty = (g.Tx, g.Ty)
        self.total_area = g.total_area
        self.ops = dict(lam=self.lam, area=self.area)
        self.heat_fn, self.poisson_fn = (_pl_heat, _pl_poisson)
        self.heat = lambda f, s: _pl_heat_jit(self.ops, f, s)
        self.poisson = lambda r: _pl_poisson_jit(self.ops, r)

    def _dct(self, f):
        return _dct2(f)

    def _idct(self, c):
        return _idct2(c)

    def grad_flux(self, f):
        return (self.Tx * (f[:, :, :-1] - f[:, :, 1:]), self.Ty * (f[:, :-1] - f[:, 1:]))

    def bridge_flux(self, a, b):
        FX = self.Tx * (a[:, :, :-1] * b[:, :, 1:] - b[:, :, :-1] * a[:, :, 1:])
        FY = self.Ty * (a[:, :-1] * b[:, 1:] - b[:, :-1] * a[:, 1:])
        return (FX, FY)

    def div(self, FX, FY):
        zx = jnp.zeros_like(FX[:, :, :1])
        zy = jnp.zeros_like(FY[:, :1])
        FXf = jnp.concatenate([zx, FX, zx], axis=2)
        FYf = jnp.concatenate([zy, FY, zy], axis=1)
        return FXf[:, :, 1:] - FXf[:, :, :-1] + FYf[:, 1:] - FYf[:, :-1]

@functools.lru_cache(maxsize=None)
def _sinkhorn_chunk(H, check_every):

    @jax.jit
    def chunk(b, w, ops, r0, s):

        def one(_, b):
            a = r0 / H(ops, b, s)
            bn = 1.0 / H(ops, a, s)
            return jnp.exp((1 - w) * jnp.log(b) + w * jnp.log(bn))
        b = lax.fori_loop(0, check_every, one, b)
        a = r0 / H(ops, b, s)
        bn = 1.0 / H(ops, a, s)
        err = jnp.max(jnp.abs(a * H(ops, bn, s) / r0 - 1))
        return (bn, err)
    return chunk

def sinkhorn(G, rho0, gamma, tol=1e-13, maxit=200000, b0=None, check_every=10, relax=True):
    s = jnp.array([gamma / 2])
    r0 = jnp.asarray(rho0)[None]
    b = jnp.ones_like(r0) if b0 is None else jnp.asarray(b0)[None]
    H, ops = (G.heat_fn, G.ops)
    if G.kind == 'sphere':
        o = trunc_ops(G, gamma / 2)
        if o is not None:
            H, ops = (_sph_heat_t, o)
    heat = jax.jit(H)
    chunk = _sinkhorn_chunk(H, check_every)
    it, w, errs, hist, stalled = (0, 1.0, [], [], False)
    err = np.inf
    while it < maxit:
        b_new, e = chunk(b, jnp.asarray(w), ops, r0, s)
        e = float(e)
        it += check_every + 1
        if not np.isfinite(e) or (w > 1 and errs and (e > errs[-1])):
            w = 1.0 if w <= 1.3 else 1 + (w - 1) / 2
            errs = []
            continue
        b, err = (b_new, e)
        errs.append(e)
        hist.append(e)
        if err < tol:
            break
        if len(hist) > 60 and min(hist[-30:]) > 0.5 * min(hist[:-30]):
            stalled = True
            break
        if relax and w == 1.0 and (len(errs) >= 4) and (errs[-1] < errs[-2] < errs[-3]):
            kap = (errs[-1] / errs[-3]) ** (1.0 / (2 * (check_every + 1)))
            if 0.5 < kap < 1:
                w = min(1.95, 2 / (1 + np.sqrt(1 - kap)))
    a = r0 / heat(ops, b, s)
    return (a[0], b[0], dict(iters=int(it), err=float(min(hist) if hist else err), omega=float(w), stalled=stalled, modes=int(ops['Qe'].shape[0] * ops['Qe'].shape[2]) if 'Qe' in ops else None))

def spec_apply(G, f, fn):
    if G.kind == 'plane':
        return G._idct(G._dct(f) * fn(G.lam[None]))
    ce, co = G._to_spec(f)
    return G._from_spec(ce * fn(G.Le[:, :, None]), co * fn(G.Lo[:, :, None]))

def _phi_int(lam, s0, s1):
    d = s1 - s0
    z = lam * d
    small = jnp.abs(z) < 1e-08
    big = jnp.exp(lam * s0) * jnp.expm1(z) / jnp.where(lam == 0, 1.0, lam)
    return jnp.where(small, jnp.exp(lam * s0) * d * (1 + z / 2), big)

def _tau_hit(ratio, z, d_over_u):
    small = jnp.abs(z) < 1e-06
    near = jnp.abs(z) < 0.5
    ok = (ratio > 0) & ~small
    zs = jnp.where(ok, z, 1.0)
    lg = jnp.where(near, jnp.log1p(jnp.where(near, zs, 0.0)), jnp.log(jnp.where(ok, ratio, 1.0)))
    big = jnp.where(ok, d_over_u * (lg / zs), 0.0)
    ser = d_over_u * (1 - z / 2 + z * z / 3 - z ** 3 / 4)
    return jnp.where(small, ser, big)

def _E(beta, tau):
    z = beta * tau
    small = jnp.abs(z) < 1e-06
    big = jnp.expm1(z) / jnp.where(beta == 0, 1.0, beta)
    return jnp.where(small, tau * (1 + z / 2 + z * z / 6 + z ** 3 / 24), big)

def full_faces(G, FX, FY):
    if G.kind == 'sphere':
        z = jnp.zeros((1, G.nlon))
        return (FX, jnp.concatenate([z, FY, z], axis=0))
    zx = jnp.zeros((G.ny, 1))
    zy = jnp.zeros((1, G.nx))
    return (jnp.concatenate([zx, FX, zx], axis=1), jnp.concatenate([zy, FY, zy], axis=0))

def _wedge_flux(c, cxs, cy, ux, uy, ut):
    cx = c[:, 0] / cy
    cyn = c[:, 1] / cxs
    ct = c[:, 2] / (cxs * cy)
    return jnp.where(cx != 0, cx * ux, 0.0) + jnp.where(cyn != 0, cyn * uy, 0.0) + jnp.where(ct != 0, ct * ut, 0.0)

def _wedge_phys(cn, cxs, cy):
    return jnp.stack([cn[:, 0] * cy, cn[:, 1] * cxs, cn[:, 2] * (cxs * cy)], axis=1)
ITIN_P = 1099511628211
ITIN_H0 = 14695981039346656037

def make_advance(G, jac, polar=True, itinerary=False):
    sph = G.kind == 'sphere'
    if sph:
        nx = G.nlon
        cxs = G.np.dlon
        dz = jnp.asarray(G.np.dz)
    else:
        nx = G.nx
        cxs, cys = (G.np.hx, G.np.hy)

    def faces(FXf, FYf, j, i):
        if sph:
            FW = FXf[j, (i - 1) % nx]
            FE = FXf[j, i]
        else:
            FW = FXf[j, i]
            FE = FXf[j, i + 1]
        return (FW, FE, FYf[j, i], FYf[j + 1, i])

    def step(st, sl):
        j, i, x, y, t, D, act = st[:7]
        wedge = D.shape[2] == 3
        M0, M1, FXf, FYf, t0, t1 = sl
        m0 = M0[j, i]
        md = (M1[j, i] - m0) / (t1 - t0)
        mc = m0 + (t - t0) * md
        left = jnp.maximum(t1 - t, 0.0)
        tau_end = _tau_hit(M1[j, i] / mc, md * left / mc, left / mc)
        FW, FE, FS, FN = faces(FXf, FYf, j, i)
        bx = FE - FW
        ux = FW + bx * x
        by = FN - FS
        uy = FS + by * y
        sx = jnp.where(ux == 0, 1.0, ux)
        sy = jnp.where(uy == 0, 1.0, uy)
        dE = (1 - x) / sx
        dW = -x / sx
        dN = (1 - y) / sy
        dS = -y / sy
        inf = jnp.inf
        tx = jnp.where((ux > 0) & (FE > 0), _tau_hit(FE / sx, bx * dE, dE), inf)
        tx = jnp.where((ux < 0) & (FW < 0), _tau_hit(FW / sx, bx * dW, dW), tx)
        ty = jnp.where((uy > 0) & (FN > 0), _tau_hit(FN / sy, by * dN, dN), inf)
        ty = jnp.where((uy < 0) & (FS < 0), _tau_hit(FS / sy, by * dS, dS), ty)
        tau = jnp.minimum(jnp.minimum(tx, ty), tau_end)
        ex = (tx <= ty) & (tx < tau_end)
        ey = ~ex & (ty < tau_end)
        done = ~(ex | ey)
        D_in = D
        if jac:
            cy = dz[j] if sph else cys
            T = D[:, :, :2]
            dxi = T[:, 0] / cxs
            deta = T[:, 1] / (cy[:, None] if sph else cy)
            dtt = T[:, 2]
            dtau = jnp.where(ex[:, None], -dxi / sx[:, None], jnp.where(ey[:, None], -deta / sy[:, None], -dtt / mc[:, None]))
            ebx, eby, emd = (jnp.exp(bx * tau), jnp.exp(by * tau), jnp.exp(md * tau))
            dxi = ebx[:, None] * (dxi + ux[:, None] * dtau)
            deta = eby[:, None] * (deta + uy[:, None] * dtau)
            dtt = emd[:, None] * (dtt + mc[:, None] * dtau)
            dxi = jnp.where(ex[:, None], 0.0, dxi)
            deta = jnp.where(ey[:, None], 0.0, deta)
            dtt = jnp.where(done[:, None], 0.0, dtt)
            Dn = jnp.stack([dxi * cxs, deta * (cy[:, None] if sph else cy), dtt], axis=1)
            if wedge:
                f = _wedge_flux(D[:, :, 2], cxs, cy, jnp.where(x == 1.0, FE, ux), jnp.where(y == 1.0, FN, uy), mc)
                cn = jnp.stack([jnp.where(ex, f * eby * emd / sx, 0.0), jnp.where(ey, f * ebx * emd / sy, 0.0), jnp.where(done, f * ebx * eby / mc, 0.0)], axis=1)
                Dn = jnp.concatenate([Dn, _wedge_phys(cn, cxs, cy)[:, :, None]], axis=2)
            D = jnp.where(act[:, None, None], Dn, D)
        xn = x + ux * _E(bx, tau)
        yn = y + uy * _E(by, tau)
        tn = jnp.where(done, t1, t + mc * _E(md, tau))
        east = ex & (ux > 0)
        west = ex & (ux < 0)
        north = ey & (uy > 0)
        south = ey & (uy < 0)
        inn = i + east.astype(i.dtype) - west.astype(i.dtype)
        if sph:
            inn = inn % nx
        jn = j + north.astype(j.dtype) - south.astype(j.dtype)
        xn = jnp.where(east, 0.0, jnp.where(west, 1.0, jnp.clip(xn, 0.0, 1.0)))
        yn = jnp.where(north, 0.0, jnp.where(south, 1.0, jnp.clip(yn, 0.0, 1.0)))
        if itinerary:
            code = jnp.where(east, 1, jnp.where(west, 2, jnp.where(north, 3, jnp.where(south, 4, 0))))
        if sph and polar:
            nl = G.nlat
            pol_m = (j == 0) | (j == nl - 1)
            south = j == 0
            Fout = jnp.where(south, FYf[jnp.minimum(1, nl), i], -FYf[jnp.maximum(nl - 1, 0), i])
            M1j = M1[j, i]

            def pstep(xx, yy, tt):
                r = _polar_step(xx, yy, tt, south, FW, FE, Fout, m0, md, M1j, t0, t1)
                return (r[0], r[1], r[2])
            pe = _polar_step(x, y, t, south, FW, FE, Fout, m0, md, M1j, t0, t1)
            pE, pW, pA, pdone = (pe[3], pe[4], pe[5], pe[6])
            if jac:
                cy = dz[j]
                tx_ = D_in[:, 0] / cxs
                ty_ = D_in[:, 1] / cy[:, None]
                tt_ = D_in[:, 2]
                outs = [jax.jvp(pstep, (x, y, t), (tx_[:, k], ty_[:, k], tt_[:, k]))[1] for k in range(2)]
                Dp = jnp.stack([jnp.stack([o[0] * cxs, o[1] * cy, o[2]], axis=1) for o in outs], axis=2)
                if wedge:
                    ptau, rr0, ux0, vr0, rre, pmc = pe[7]
                    ueta = rr0 * (Fout * rr0 - (FE - FW) * (1.0 - rr0))
                    ueta = jnp.where(south, ueta, -ueta)
                    uxi = jnp.where(x == 1.0, FE, ux0) / (2.0 * rr0)
                    f = _wedge_flux(D_in[:, :, 2], cxs, cy, uxi, ueta, pmc)
                    eF = jnp.exp((FE - FW + Fout) * ptau)
                    em = jnp.exp(md * ptau)
                    cn = jnp.stack([jnp.where(pE | pW, f * eF * em * (2.0 * rre) / jnp.where(pE, FE, FW), 0.0), jnp.where(pA, f * eF * em / jnp.where(south, Fout, -Fout), 0.0), jnp.where(pdone, f * eF / pmc, 0.0)], axis=1)
                    Dp = jnp.concatenate([Dp, _wedge_phys(cn, cxs, cy)[:, :, None]], axis=2)
                D = jnp.where((act & pol_m)[:, None, None], Dp, D)
            jp = jnp.where(pA, jnp.where(south, j + 1, j - 1), j)
            ip = (i + pE.astype(i.dtype) - pW.astype(i.dtype)) % nx
            xp = jnp.where(pE, 0.0, jnp.where(pW, 1.0, pe[0]))
            yp = jnp.where(pA, jnp.where(south, 0.0, 1.0), pe[1])
            jn = jnp.where(pol_m, jp, jn)
            inn = jnp.where(pol_m, ip, inn)
            xn = jnp.where(pol_m, xp, xn)
            yn = jnp.where(pol_m, yp, yn)
            tn = jnp.where(pol_m, pe[2], tn)
            done = jnp.where(pol_m, pdone, done)
            if itinerary:
                code = jnp.where(pol_m, jnp.where(pE, 5, jnp.where(pW, 6, jnp.where(pA, 7, 0))), code)
        sel = lambda new, old: jnp.where(act, new, old)
        out = (sel(jn, j), sel(inn, i), sel(xn, x), sel(yn, y), sel(tn, t), D, act & ~done)
        if itinerary:
            h = st[7]
            u = jnp.uint64
            cell = j.astype(u) * u(nx) + i.astype(u)
            hn = h * u(ITIN_P) ^ code.astype(u) + u(8) * cell
            out = out + (sel(hn, h),)
        return out

    @functools.partial(jax.jit, static_argnums=(2,))
    def core(st, sl, maxit):

        def cond(c):
            return jnp.any(c[0][6]) & (c[1] < maxit)

        def body(c):
            return (step(c[0], sl), c[1] + 1)
        st, n = lax.while_loop(cond, body, (st, 0))
        return st + (n,)

    def advance(j, i, x, y, D, sl, h=None):
        t = jnp.full(x.shape, sl[4])
        act = jnp.ones(x.shape, bool)
        st = (j, i, x, y, t, D, act) + ((jnp.asarray(h, jnp.uint64),) if itinerary else ())
        *st, n = core(tuple(st), sl, 6)
        loops = int(n)
        for _ in range(100000):
            idx = np.flatnonzero(np.asarray(st[6]))
            na = idx.size
            if na == 0:
                break
            size = 1 << max(10, int(np.ceil(np.log2(na))))
            sub = jnp.asarray(np.concatenate([idx, np.full(size - na, idx[0])]))
            sact = jnp.asarray(np.arange(size) < na)
            ss = tuple((a[sub] if k != 6 else sact for k, a in enumerate(st)))
            *ss, n = core(ss, sl, 64)
            loops += int(n)
            ii = jnp.asarray(idx)
            st = [a.at[ii].set(b[:na]) for a, b in zip(st, ss)]
        j, i, x, y, t, D, act = st[:7]
        if itinerary:
            return (j, i, x, y, D, st[7], loops)
        return (j, i, x, y, D, loops)
    advance.core = core
    return advance

def locate(G, X, Y):
    g = G.np
    if G.kind == 'sphere':
        i = np.clip(np.floor((X - g.lon_e[0]) / g.dlon).astype(np.int32), 0, g.nlon - 1)
        j = np.clip(np.searchsorted(g.z_e, Y, side='right') - 1, 0, g.nlat - 1).astype(np.int32)
        xi = (X - g.lon_e[i]) / g.dlon
        eta = (Y - g.z_e[j]) / g.dz[j]
    else:
        i = np.clip(np.floor(X / g.hx).astype(np.int32), 0, g.nx - 1)
        j = np.clip(np.floor(Y / g.hy).astype(np.int32), 0, g.ny - 1)
        xi = X / g.hx - i
        eta = Y / g.hy - j
    return (j, i, np.clip(xi, 0, 1), np.clip(eta, 0, 1))

def position(G, j, i, x, y):
    g = G.np
    j, i, x, y = map(np.asarray, (j, i, x, y))
    if G.kind == 'sphere':
        return (g.lon_e[i] + x * g.dlon, g.z_e[j] + y * g.dz[j])
    return ((i + x) * g.hx, (j + y) * g.hy)

def _fix(G, M0, M1, FX, FY, dt, iters=3):
    for _ in range(iters):
        r = -(M1 - M0)[None] / dt - G.div(FX, FY)
        gx, gy = G.grad_flux(G.poisson(r))
        FX, FY = (FX + gx, FY + gy)
    return (FX, FY)
_sph_spec_jit = jax.jit(_sph_spec)
_dct2_jit = jax.jit(_dct2)

def _spec1(G, f):
    f = jnp.asarray(f)[None]
    return _sph_spec_jit(G.ops, f) if G.kind == 'sphere' else _dct2_jit(f)

def _bridge_fns(G):
    fns = G.__dict__.get('_bridge_fns')
    if fns is not None:
        return fns
    sph = G.kind == 'sphere'
    shape = (G.nlat, G.nlon) if sph else (G.ny, G.nx)

    @jax.jit
    def heat_fields(ops, c, sv):
        if sph:
            ce, co = c
            unspec = _sph_unspec_t if ops['Qe'].shape[0] < G.nm else _sph_unspec
            return unspec(ops, ce * jnp.exp(ops['Le'][:, :, None] * sv[None, None, :]), co * jnp.exp(ops['Lo'][:, :, None] * sv[None, None, :]), G.nlon)
        return _idct2(c * jnp.exp(ops['lam'][None] * sv[:, None, None]))

    @jax.jit
    def flux_mass(At, Bt, w, s, area):
        C = At.shape[0] // 7
        At = At.reshape((C, 7) + shape)
        Bt = Bt.reshape((C, 7) + shape)
        fx, fy = G.bridge_flux(At[:, :6].reshape((C * 6,) + shape), Bt[:, :6].reshape((C * 6,) + shape))
        FX = s * jnp.einsum('g,cg...->c...', w, fx.reshape((C, 6) + fx.shape[1:]))
        FY = s * jnp.einsum('g,cg...->c...', w, fy.reshape((C, 6) + fy.shape[1:]))
        return (FX, FY, area[None] * At[:, 6] * Bt[:, 6])

    def sub(c, s_min):
        if not sph:
            return (G.ops, c)
        o = trunc_ops(G, s_min)
        if o is None:
            return (G.ops, c)
        mM, kM = (o['Qe'].shape[0], o['Qe'].shape[2])
        return (o, (c[0][:mM, G.h - kM:], c[1][:mM, G.h - kM:]))

    def fields(ca, cb, ta, w, s):
        tt = jnp.asarray(ta.reshape(-1))
        oa, sa = sub(ca, float(ta.min()) * s)
        ob, sb = sub(cb, float((1 - ta).min()) * s)
        At = heat_fields(oa, sa, tt * s)
        Bt = heat_fields(ob, sb, (1 - tt) * s)
        return flux_mass(At, Bt, jnp.asarray(w), jnp.asarray(s), G.area)

    def node(ca, cb, tn, s):
        oa, sa = sub(ca, tn * s)
        ob, sb = sub(cb, (1 - tn) * s)
        return G.area * heat_fields(oa, sa, jnp.asarray([tn * s]))[0] * heat_fields(ob, sb, jnp.asarray([(1 - tn) * s]))[0]
    G._bridge_fns = (fields, node)
    return G._bridge_fns

def _default_chunk(G):
    return max(1, min(16, int(52000000.0 // G.area.size)))

def bridge_slabs(G, rho0, gamma, t, a, b, chunk=None, reverse=False, fix_iters=1):
    s = gamma / 2
    A = G.area
    chunk = chunk or _default_chunk(G)
    fields, node = _bridge_fns(G)
    ca, cb = (_spec1(G, a), _spec1(G, b))
    M_first = jnp.asarray(rho0) * A
    Mtot = jnp.sum(M_first)
    w = GL_W / 2
    K = len(t) - 1
    chunks = [list(range(c0, min(K, c0 + chunk))) for c0 in range(0, K, chunk)]

    def nodemass(k, raw=None):
        if k == 0:
            return M_first
        if k == K:
            return A * jnp.ones_like(A)
        Mi = node(ca, cb, float(t[k]), s) if raw is None else raw
        return Mi * (Mtot / jnp.sum(Mi))
    M_prev = None
    for idx in chunks[::-1] if reverse else chunks:
        ta = np.array([np.append(t[n] + (GL_X + 1) / 2 * (t[n + 1] - t[n]), t[n + 1]) for n in idx])
        FX, FY, Mraw = fields(ca, cb, ta, w, s)
        M1s = jnp.stack([nodemass(n + 1, Mraw[q]) for q, n in enumerate(idx)])
        M0 = M_prev if M_prev is not None and (not reverse) else nodemass(idx[0])
        M0s = jnp.concatenate([M0[None], M1s[:-1]])
        dts = jnp.asarray([t[n + 1] - t[n] for n in idx])[:, None, None]
        FX, FY = _fix_batch(G, M0s, M1s, FX, FY, dts, iters=fix_iters)
        order = range(len(idx) - 1, -1, -1) if reverse else range(len(idx))
        for q in order:
            n = idx[q]
            if reverse:
                yield (M1s[q], M0s[q], -FX[q], -FY[q], -float(t[n + 1]), -float(t[n]))
            else:
                yield (M0s[q], M1s[q], FX[q], FY[q], float(t[n]), float(t[n + 1]))
        M_prev = M1s[-1]

def _fix_batch(G, M0, M1, FX, FY, dt, iters=3):
    for _ in range(iters):
        r = -(M1 - M0) / dt - G.div(FX, FY)
        gx, gy = G.grad_flux(G.poisson(r))
        FX, FY = (FX + gx, FY + gy)
    return (FX, FY)

def _spec_multi(G, f, mults):
    if G.kind == 'plane':
        c = G._dct(f)
        return jnp.concatenate([G._idct(c * fn(G.lam[None])) for fn in mults])
    ce, co = G._to_spec(f)
    CE = jnp.concatenate([ce * fn(G.Le[:, :, None]) for fn in mults], axis=2)
    CO = jnp.concatenate([co * fn(G.Lo[:, :, None]) for fn in mults], axis=2)
    return G._from_spec(CE, CO)

def diffusion_slabs(G, rho0, s, chunk=None, reverse=False, fix_iters=1):
    A = G.area
    chunk = chunk or _default_chunk(G)
    r0 = jnp.asarray(rho0)[None]
    N = len(s) - 1
    chunks = [list(range(c0, min(N, c0 + chunk))) for c0 in range(0, N, chunk)]

    def tail_slab(M_last):
        sl = float(s[-1])
        tail = lambda L: jnp.where(L == 0, 0.0, jnp.exp(L * sl) / jnp.where(L == 0, 1.0, -L))
        FX, FY = G.grad_flux(spec_apply(G, r0, tail))
        FX, FY = _fix(G, M_last, A * 1.0, FX, FY, 1.0)
        return (M_last, A * 1.0, FX[0], FY[0], sl, sl + 1.0)

    def chunk_slabs(idx, M0):
        mults = [lambda L, a_=float(s[n]), b_=float(s[n + 1]): _phi_int(L, a_, b_) / (b_ - a_) for n in idx]
        mults += [lambda L, b_=float(s[n + 1]): jnp.exp(L * b_) for n in idx]
        if M0 is None:
            mults += [lambda L, a_=float(s[idx[0]]): jnp.exp(L * a_)]
        out = _spec_multi(G, r0, mults)
        C = len(idx)
        I, M1s = (out[:C], out[C:2 * C] * A[None])
        if M0 is None:
            M0 = out[2 * C] * A
        FX, FY = G.grad_flux(I)
        M0s = jnp.concatenate([M0[None], M1s[:-1]])
        dts = jnp.asarray([s[n + 1] - s[n] for n in idx])[:, None, None]
        FX, FY = _fix_batch(G, M0s, M1s, FX, FY, dts, iters=fix_iters)
        return (M0s, M1s, FX, FY)
    if not reverse:
        M_prev = r0[0] * A
        for idx in chunks:
            M0s, M1s, FX, FY = chunk_slabs(idx, M_prev)
            for q, n in enumerate(idx):
                yield (M0s[q], M1s[q], FX[q], FY[q], float(s[n]), float(s[n + 1]))
            M_prev = M1s[-1]
        yield tail_slab(M_prev)
        return
    M_last = spec_apply(G, r0, lambda L: jnp.exp(L * float(s[-1])))[0] * A
    M0_, M1_, FX_, FY_, t0_, t1_ = tail_slab(M_last)
    yield (M1_, M0_, -FX_, -FY_, -t1_, -t0_)
    for idx in chunks[::-1]:
        M0s, M1s, FX, FY = chunk_slabs(idx, r0[0] * A if idx[0] == 0 else None)
        for q in range(len(idx) - 1, -1, -1):
            n = idx[q]
            yield (M1s[q], M0s[q], -FX[q], -FY[q], -float(s[n + 1]), -float(s[n]))

def tracker_backend(backend=None):
    if backend not in (None, 'jax'):
        raise ValueError('Only the JAX tracker is supported.')
    return 'jax'

def run_map(G, slabs, X, Y, jac=True, polar=True, backend=None, itinerary=False, local=False, start=None, ad=False):
    backend = tracker_backend(backend)
    if ad:
        raise ValueError('Automatic-differentiation tracking is not available in this implementation.')
    if start is not None:
        j, i, x, y = (np.asarray(start[0], np.int32), np.asarray(start[1], np.int32), np.asarray(start[2], float), np.asarray(start[3], float))
    else:
        j, i, x, y = locate(G, np.asarray(X, float), np.asarray(Y, float))
    n = x.shape[0]
    j, i, x, y = map(jnp.asarray, (j, i, x, y))
    D = jnp.zeros((n, 3, 3)).at[:, 0, 0].set(1.0).at[:, 1, 1].set(1.0).at[:, 2, 2].set(1.0) if jac else jnp.zeros((n, 3, 2))
    cache = G.__dict__.setdefault('_adv_cache', {})
    key = (jac, polar, bool(itinerary))
    if key not in cache:
        cache[key] = make_advance(G, jac, polar, itinerary=bool(itinerary))
    adv = cache[key]
    h = jnp.full((n,), ITIN_H0, dtype=jnp.uint64) if itinerary else None
    events, cont, mmin = (0, 0.0, np.inf)
    import time as _time
    _t0 = _time.time()
    verbose = os.environ.get('BCART_VERBOSE', '0') == '1'
    t_prep = 0.0
    t_adv = 0.0
    it = iter(slabs)
    ns_ = -1
    while True:
        _a = _time.time()
        try:
            M0, M1, FX, FY, t0, t1 = next(it)
        except StopIteration:
            break
        jax.block_until_ready(FX)
        t_prep += _time.time() - _a
        ns_ += 1
        if verbose and (ns_ < 3 or ns_ % 25 == 0):
            print(f'[bcart] slab {ns_} t={t0:.3g} elapsed {_time.time() - _t0:.1f}s', file=sys.stderr, flush=True)
        mmin = min(mmin, float(jnp.min(M1 / G.area)))
        res = jnp.max(jnp.abs(M1 - M0 + (t1 - t0) * G.div(FX[None], FY[None])[0]) / M1)
        cont = max(cont, float(res))
        FXf, FYf = full_faces(G, FX, FY)
        _b = _time.time()
        sl_ = (M0, M1, FXf, FYf, jnp.asarray(t0), jnp.asarray(t1))
        if itinerary:
            j, i, x, y, D, h, nev = adv(j, i, x, y, D, sl_, h)
        else:
            j, i, x, y, D, nev = adv(j, i, x, y, D, sl_)
        events += int(nev)
        t_adv += _time.time() - _b
    stats = dict(event_loops=events, continuity=cont, min_density=mmin, t_prep=t_prep, t_track=t_adv, tracker=backend)
    if itinerary:
        stats['itinerary'] = np.asarray(h, dtype=np.uint64)
    if local:
        stats['local'] = tuple((np.asarray(a) for a in (j, i, x, y)))
    pos = position(G, j, i, x, y)
    if not jac:
        return (pos, None, stats)
    D = np.asarray(D)
    stats['det'] = D[:, 2, 2].copy()
    return (pos, D[:, :2, :2].copy(), stats)

def _polar_step(xi, eta, t, south, FW, FE, Fout, m0, md, M1, t0, t1):
    rr0 = jnp.where(south, jnp.sqrt(jnp.clip(eta, 0, 1)), jnp.sqrt(jnp.clip(1 - eta, 0, 1)))
    rr0 = jnp.maximum(rr0, 1e-300)
    b = FE - FW
    Ftot = b + Fout
    A = Ftot / 2
    ux0 = FW + b * xi
    mc = m0 + (t - t0) * md
    left = jnp.maximum(t1 - t, 0.0)
    tau_end = _tau_hit(M1 / mc, md * left / mc, left / mc)
    vr0 = A * rr0 - b / 2
    d_arc = 1 - rr0
    t_arc = jnp.where((vr0 > 0) & (Fout > 0), _tau_hit(Fout / 2 / jnp.where(vr0 == 0, 1.0, vr0), A * d_arc / jnp.where(vr0 == 0, 1.0, vr0), d_arc / jnp.where(vr0 == 0, 1.0, vr0)), jnp.inf)

    def edge_time(qm1, cond):
        q = 1 + qm1
        den = b / 2 - qm1 * vr0
        dsafe = jnp.where(den == 0, 1.0, den)
        lin = rr0 * qm1 / dsafe
        dl = A * lin
        L = jnp.where(jnp.abs(dl) < 1e-08, 1 - dl / 2 + dl * dl / 3, jnp.log1p(jnp.where(dl > -1, dl, 0.0)) / jnp.where(dl == 0, 1.0, dl))
        tau_e = lin * L
        ok = cond & (q > 0) & (den != 0) & (dl > -1) & (tau_e > 0) & jnp.isfinite(tau_e)
        return jnp.where(ok, tau_e, jnp.inf)
    uxs = jnp.where(ux0 == 0, 1.0, ux0)
    t_E = edge_time(b * (1 - xi) / uxs, (ux0 > 0) & (FE > 0))
    t_W = edge_time(-b * xi / uxs, (ux0 < 0) & (FW < 0))
    tau = jnp.minimum(jnp.minimum(jnp.minimum(t_E, t_W), t_arc), tau_end)
    eE = (t_E <= tau) & (t_E < tau_end)
    eW = ~eE & (t_W <= tau) & (t_W < tau_end)
    eA = ~eE & ~eW & (t_arc <= tau) & (t_arc < tau_end)
    done = ~(eE | eW | eA)
    EA = _E(A, tau)
    rr = jnp.clip(rr0 + vr0 * EA, 1e-300, 1.0)
    xin = xi + ux0 * EA / (2 * rr)
    xin = jnp.where(eE, 1.0, jnp.where(eW, 0.0, jnp.clip(xin, 0.0, 1.0)))
    tn = jnp.where(done, t1, t + mc * _E(md, tau))
    rr = jnp.where(eA, 1.0, rr)
    etan = jnp.where(south, rr * rr, 1 - rr * rr)
    rrs = jnp.maximum(jnp.where(south, jnp.sqrt(jnp.clip(etan, 0, 1)), jnp.sqrt(jnp.clip(1 - etan, 0, 1))), 1e-300)
    return (xin, etan, tn, eE, eW, eA, done, (tau, rr0, ux0, vr0, rrs, mc))

def reverse_slabs(slabs):
    out = [(M1, M0, -FX, -FY, -t1, -t0) for M0, M1, FX, FY, t0, t1 in slabs]
    return out[::-1]

def bridge_slabs_reversed(G, rho0, gamma, t, a, b, chunk=None):
    return bridge_slabs(G, rho0, gamma, t, a, b, chunk=chunk, reverse=True)

def diffusion_slabs_reversed(G, rho0, s, chunk=None):
    return diffusion_slabs(G, rho0, s, chunk=chunk, reverse=True)

class PathCache:

    def __init__(self, gen, device=False, compact=None):
        self.gen, self.slabs, self.t_copy, self.device = (gen, None, 0.0, device)
        self.compact, self.continuity = (compact, 0.0)
        if compact is not None:
            self._step = jax.jit(lambda M, FX, FY, dt: M - dt * compact.div(FX[None], FY[None])[0])
            self._back = jax.jit(lambda M, FX, FY, dt: M + dt * compact.div(FX[None], FY[None])[0])

    def record(self):
        import time as _time
        if self.compact is not None:
            yield from self._record_compact()
            return
        out, prev = ([], None)
        for sl in self.gen:
            M0, M1, FX, FY, t0, t1 = sl
            tic = _time.time()
            if self.device:
                same = prev is not None and M0 is prev
                out.append((None if same else M0, M1, FX, FY, float(t0), float(t1)))
                prev = M1
            else:
                M0h, M1h = (np.asarray(M0), np.asarray(M1))
                same = prev is not None and np.array_equal(M0h, prev)
                out.append((None if same else M0h, M1h, np.asarray(FX), np.asarray(FY), float(t0), float(t1)))
                prev = M1h
            self.t_copy += _time.time() - tic
            yield sl
        self.slabs = out

    def _record_compact(self):
        G, out, prev = (self.compact, [], None)
        for M0, M1, FX, FY, t0, t1 in self.gen:
            if prev is None:
                self.M_first = prev = M0
            dt = float(t1) - float(t0)
            res = jnp.max(jnp.abs(M1 - M0 + dt * G.div(FX[None], FY[None])[0]) / M1)
            self.continuity = max(self.continuity, float(res))
            M1c = self._step(prev, FX, FY, dt)
            out.append((FX, FY, float(t0), float(t1)))
            yield (prev, M1c, FX, FY, t0, t1)
            prev = M1c
        self.M_last = prev
        self.slabs = out
        self.gen = None

    def nbytes(self):
        if self.compact is not None:
            return sum((int(a.size) * 8 for sl in self.slabs for a in sl[:2])) + 2 * int(self.M_first.size) * 8
        return sum((int(a.size) * 8 for sl in self.slabs for a in sl[:4] if a is not None))

    def _host(self):
        prev = None
        for M0, M1, FX, FY, t0, t1 in self.slabs:
            yield (prev if M0 is None else M0, M1, FX, FY, t0, t1)
            prev = M1

    def forward(self):
        if self.compact is not None:
            M0 = self.M_first
            for FX, FY, t0, t1 in self.slabs:
                M1 = self._step(M0, FX, FY, t1 - t0)
                yield (M0, M1, FX, FY, t0, t1)
                M0 = M1
            return
        for M0, M1, FX, FY, t0, t1 in self._host():
            yield (jnp.asarray(M0), jnp.asarray(M1), jnp.asarray(FX), jnp.asarray(FY), t0, t1)

    def reversed(self):
        if self.compact is not None:
            M1 = self.M_last
            for FX, FY, t0, t1 in self.slabs[::-1]:
                M0 = self._back(M1, FX, FY, t1 - t0)
                yield (M1, M0, -FX, -FY, -t1, -t0)
                M1 = M0
            return
        for M0, M1, FX, FY, t0, t1 in list(self._host())[::-1]:
            yield (jnp.asarray(M1), jnp.asarray(M0), -jnp.asarray(FX), -jnp.asarray(FY), -t1, -t0)

def gauss_points(G, nq):
    g = G.np
    xg, wg = np.polynomial.legendre.leggauss(nq)
    u = (xg + 1) / 2
    wu = wg / 2
    if G.kind == 'sphere':
        xe, ye = (g.lon_e, g.z_e)
    else:
        xe, ye = (g.x_e, g.y_e)
    X = (xe[:-1, None] + np.diff(xe)[:, None] * u[None, :]).ravel()
    Y = (ye[:-1, None] + np.diff(ye)[:, None] * u[None, :]).ravel()
    WX = np.tile(wu, len(xe) - 1)
    WY = np.tile(wu, len(ye) - 1)
    XX, YY = np.meshgrid(X, Y)
    W = np.outer(WY, WX)
    return (XX.ravel(), YY.ravel(), W.ravel())
