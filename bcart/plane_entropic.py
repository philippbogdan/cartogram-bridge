"""Manufactured Brenier maps and continuum bridge flows on a reflecting rectangle."""
import math
import time
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
LX, LY = (2 * np.pi, np.pi)
DIAMETER = float(np.hypot(LX, LY))

class CosinePsi:

    def __init__(self, terms, Lx=LX, Ly=LY, name=''):
        self.terms = [(float(a), int(m), int(n)) for a, m, n in terms]
        self.Lx, self.Ly, self.name = (Lx, Ly, name)

    def derivs(self, X, Y):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        out = {k: np.zeros(np.broadcast(X, Y).shape) for k in ('p', 'x', 'y', 'xx', 'xy', 'yy')}
        for a, m, n in self.terms:
            kx, ky = (m * np.pi / self.Lx, n * np.pi / self.Ly)
            cx, sx, cy, sy = (np.cos(kx * X), np.sin(kx * X), np.cos(ky * Y), np.sin(ky * Y))
            out['p'] += a * cx * cy
            out['x'] += -a * kx * sx * cy
            out['y'] += -a * ky * cx * sy
            out['xx'] += -a * kx ** 2 * cx * cy
            out['xy'] += a * kx * ky * sx * sy
            out['yy'] += -a * ky ** 2 * cx * cy
        return out

    def T0(self, X, Y):
        d = self.derivs(X, Y)
        return (X - d['x'], Y - d['y'])

    def rho0(self, X, Y):
        d = self.derivs(X, Y)
        return (1 - d['xx']) * (1 - d['yy']) - d['xy'] ** 2

    def DT0(self, X, Y):
        d = self.derivs(X, Y)
        return np.stack([np.stack([1 - d['xx'], -d['xy']], -1), np.stack([-d['xy'], 1 - d['yy']], -1)], -2)

    def check(self, n=1201):
        x = np.linspace(0, self.Lx, 2 * n)
        y = np.linspace(0, self.Ly, n)
        X, Y = np.meshgrid(x, y)
        d = self.derivs(X, Y)
        A, B, C = (1 - d['xx'], -d['xy'], 1 - d['yy'])
        lmin = (A + C) / 2 - np.sqrt(((A - C) / 2) ** 2 + B ** 2)
        det = A * C - B * B
        TX, TY = (X - d['x'], Y - d['y'])
        rng = np.random.default_rng(0)
        P = rng.uniform([0, 0], [self.Lx, self.Ly], (200000, 2))
        Q = rng.uniform([0, 0], [self.Lx, self.Ly], (200000, 2))
        TP = np.stack(self.T0(P[:, 0], P[:, 1]), -1)
        TQ = np.stack(self.T0(Q[:, 0], Q[:, 1]), -1)
        mono = np.sum((TP - TQ) * (P - Q), -1) / np.sum((P - Q) ** 2, -1)
        wall = max(np.abs(TX[:, 0]).max(), np.abs(TX[:, -1] - self.Lx).max(), np.abs(TY[0]).max(), np.abs(TY[-1] - self.Ly).max())
        w = np.trapezoid(np.trapezoid(det, x, axis=1), y) / (self.Lx * self.Ly)
        return dict(min_eig_I_minus_D2psi=float(lmin.min()), min_rho0=float(det.min()), max_rho0=float(det.max()), mean_rho0_minus_1=float(w - 1), max_disp=float(np.hypot(d['x'], d['y']).max()), min_monotonicity_ratio=float(mono.min()), wall_residual=float(wall), T0_range=[float(TX.min()), float(TX.max()), float(TY.min()), float(TY.max())])
CASES = {'smooth': CosinePsi([(0.2, 1, 1), (0.06, 2, 0), (0.04, 0, 2)], name='smooth'), 'mixed': CosinePsi([(0.15, 2, 1), (0.12, 1, 0), (0.08, 0, 1), (0.05, 3, 2)], name='mixed'), 'large': CosinePsi([(0.5, 1, 1), (0.15, 2, 0), (0.1, 0, 2)], name='large')}

class Grid:

    def __init__(self, nx, ny, Lx=LX, Ly=LY):
        self.nx, self.ny, self.Lx, self.Ly = (nx, ny, Lx, Ly)
        self.hx, self.hy = (Lx / nx, Ly / ny)
        self.x = (np.arange(nx) + 0.5) * self.hx
        self.y = (np.arange(ny) + 0.5) * self.hy
        self.area = Lx * Ly

    def mesh(self):
        return np.meshgrid(self.x, self.y)

def log_neumann_kernel_1d(xo, xi, L, s, h):
    xo = np.asarray(xo, float)[:, None]
    xi = np.asarray(xi, float)[None, :]
    K = int(math.ceil(math.sqrt(4 * s * 800.0) / (2 * L))) + 2
    ks = np.arange(-K, K + 1)[:, None, None] * 2 * L
    z = np.concatenate([xo - xi + ks, xo + xi + ks], 0)
    e = -z ** 2 / (4 * s)
    m = e.max(0)
    return m + np.log(np.exp(e - m).sum(0)) - 0.5 * np.log(4 * np.pi * s) + np.log(h)
_JIT_CACHE = {}
LSE_CHUNK_ELEMS = 2 ** 20

def _lse_rows(logK, f):
    t = logK[None, :, :] + f[:, None, :]
    m = jnp.max(t, axis=-1)
    return m + jnp.log(jnp.sum(jnp.exp(t - m[..., None]), axis=-1))

def _lse_last(logK, f):
    nr = f.shape[0]
    per = logK.shape[0] * logK.shape[1]
    cs = max(1, min(nr, LSE_CHUNK_ELEMS // per))
    if cs >= nr:
        return _lse_rows(logK, f)
    nc = -(-nr // cs)
    fp = jnp.concatenate([f, jnp.zeros((nc * cs - nr, f.shape[1]), f.dtype)], 0)
    out = jax.lax.map(lambda fc: _lse_rows(logK, fc), fp.reshape(nc, cs, f.shape[1]))
    return out.reshape(nc * cs, -1)[:nr]

def logP_apply(logKx, logKy, f):
    g = _lse_last(logKx, f)
    return _lse_last(logKy, g.T).T

def make_logP(logKx, logKy):
    logKx = jnp.asarray(logKx)
    logKy = jnp.asarray(logKy)
    return lambda f: logP_apply(logKx, logKy, f)

def sinkhorn(grid, logrho0, gamma, w0=None, tol=1e-13, maxit=200000, anderson=8, reg=1e-14, verbose=False, chunk=50, stall=2000, stall_tol=1e-11):
    s = gamma / 2
    logKx = log_neumann_kernel_1d(grid.x, grid.x, grid.Lx, s, grid.hx)
    logKy = log_neumann_kernel_1d(grid.y, grid.y, grid.Ly, s, grid.hy)
    lr = jnp.asarray(logrho0)
    m = max(1, int(anderson))
    shape = (grid.ny, grid.nx)
    n = grid.ny * grid.nx
    use_aa = bool(anderson) and anderson > 0
    Kx, Ky = (jnp.asarray(logKx), jnp.asarray(logKy))

    def body(c, Kx, Ky, lr):
        w, dW, dR, wp, rp, k, err_prev, best_err, best_w, it = c
        g = -logP_apply(Kx, Ky, lr - logP_apply(Kx, Ky, w))
        r = g - w
        r = r - jnp.mean(r)
        err = jnp.max(jnp.abs(r))
        better = err < best_err
        best_w = jnp.where(better, w, best_w)
        best_err = jnp.minimum(err, best_err)
        if use_aa:
            idx = (k - 1) % m
            upd = k > 0
            dW = jnp.where(upd, dW.at[idx].set((w - wp).ravel()), dW)
            dR = jnp.where(upd, dR.at[idx].set((r - rp).ravel()), dR)
            nval = jnp.minimum(k, m)
            mask = jnp.arange(m) < nval
            M = dR @ dR.T
            mm = mask[:, None] & mask[None, :]
            tr = jnp.sum(jnp.where(mask, jnp.diag(M), 0.0))
            M = jnp.where(mm, M, 0.0) + jnp.diag(jnp.where(mask, reg * tr + 1e-300, 1.0))
            b = jnp.where(mask, dR @ r.ravel(), 0.0)
            gam = jnp.linalg.solve(M, b)
            wn = w + r - ((dW + dR).T @ gam).reshape(shape)
            restart = (err > 1000.0 * best_err) & (best_err < 0.01)
            wn = jnp.where(restart, best_w, wn)
            kn = jnp.where(restart, 0, k + 1)
        else:
            wn, kn = (w + r, k + 1)
        wn = jnp.where(err < tol, g, wn)
        return (wn, dW, dR, w, r, kn, err, best_err, best_w, it + 1)
    key = ('sk', shape, m, use_aa, float(tol), float(reg), int(chunk))
    if key not in _JIT_CACHE:

        def run_chunk(c, Kx, Ky, lr):
            it_end = c[-1] + chunk
            return jax.lax.while_loop(lambda c: (c[6] >= tol) & (c[-1] < it_end), lambda c: body(c, Kx, Ky, lr), c)
        _JIT_CACHE[key] = jax.jit(run_chunk)
    run_chunk = _JIT_CACHE[key]
    w = jnp.zeros(shape) if w0 is None else jnp.asarray(w0, dtype=jnp.float64)
    z = jnp.zeros((m, n)) if use_aa else jnp.zeros((1, 1))
    c = (w, z, z, w, jnp.zeros(shape), jnp.asarray(0), jnp.asarray(jnp.inf), jnp.asarray(jnp.inf), w, jnp.asarray(0))
    t0 = time.time()
    hist = []
    stalled = False
    while True:
        c = run_chunk(c, Kx, Ky, lr)
        err, it = (float(c[6]), int(c[-1]))
        hist.append((it, err))
        if verbose:
            print(f'  sinkhorn it {it} res {err:.3e} ({time.time() - t0:.1f}s)', flush=True)
        if err < tol or it >= maxit:
            break
        if len(hist) * chunk >= stall:
            past = min((e for _, e in hist[:-(stall // chunk)])) if len(hist) > stall // chunk else np.inf
            recent = min((e for _, e in hist[-(stall // chunk):]))
            if recent < stall_tol and recent > 0.9 * past:
                stalled = True
                break
    w = c[0]
    P = jax.jit(logP_apply)
    Pw = P(Kx, Ky, w)
    u = lr - Pw
    Pu = P(Kx, Ky, u)
    m1 = float(jnp.max(jnp.abs(jnp.exp(u + Pw - lr) - 1)))
    m2 = float(jnp.max(jnp.abs(jnp.exp(w + Pu) - 1)))
    cst = 0.5 * float(jnp.mean(u) - jnp.mean(w))
    u, w = (u - cst, w + cst)
    return dict(u=np.asarray(u), w=np.asarray(w), marg1=m1, marg2=m2, iters=it, res=err, stalled=stalled, max_abs_w=float(jnp.max(jnp.abs(w))), time=time.time() - t0, hist=hist, logKx=logKx, logKy=logKy)

def coordinate_weight(yc, L, M=None):
    n = len(yc)
    M = n if M is None else M
    m = np.arange(1, M, 2)
    return L / 2 - (4 * L / (np.pi ** 2 * m ** 2) * np.cos(np.outer(yc, m) * np.pi / L)).sum(1)

def barycentric(grid, w, gamma, Xp, Yp):
    s = gamma / 2
    lKx = log_neumann_kernel_1d(Xp, grid.x, grid.Lx, s, grid.hx)
    lKy = log_neumann_kernel_1d(Yp, grid.y, grid.Ly, s, grid.hy)
    wx = coordinate_weight(grid.x, grid.Lx) + grid.Lx
    wy = coordinate_weight(grid.y, grid.Ly) + grid.Ly
    w = jnp.asarray(w)

    @jax.jit
    def f(w, lKx, lKy, wx, wy):
        den = logP_apply(lKx, lKy, w)
        bx = jnp.exp(logP_apply(lKx, lKy, w + jnp.log(wx)[None, :]) - den) - grid.Lx
        by = jnp.exp(logP_apply(lKx, lKy, w + jnp.log(wy)[:, None]) - den) - grid.Ly
        return (bx, by)
    bx, by = f(w, jnp.asarray(lKx), jnp.asarray(lKy), jnp.asarray(wx), jnp.asarray(wy))
    return (np.asarray(bx), np.asarray(by))

class Spectral1D:

    def __init__(self, n, L):
        self.n, self.L = (n, L)
        x = (np.arange(n) + 0.5) * L / n
        m = np.arange(n)
        self.k = m * np.pi / L
        self.wt = np.where(m == 0, np.sqrt(1.0 / n), np.sqrt(2.0 / n))
        self.C = self.wt[:, None] * np.cos(np.outer(self.k, x))
        self.Ct = self.C.T
        self.S = -(self.wt * self.k)[None, :] * np.sin(np.outer(x, self.k))

    def eval_mats(self, xp):
        A = np.outer(xp, self.k)
        c = self.wt[None, :] * np.cos(A)
        s_ = -(self.wt * self.k)[None, :] * np.sin(A)
        cc = -(self.wt * self.k ** 2)[None, :] * np.cos(A)
        return (c, s_, cc)

class Spectral2D:

    def __init__(self, grid):
        self.sx = Spectral1D(grid.nx, grid.Lx)
        self.sy = Spectral1D(grid.ny, grid.Ly)
        j = jnp.asarray
        self.Cx, self.Cy = (j(self.sx.C), j(self.sy.C))
        self.Sx, self.Sy = (j(self.sx.S), j(self.sy.S))
        self.lap = -(j(self.sy.k)[:, None] ** 2 + j(self.sx.k)[None, :] ** 2)

    def analysis(self, f):
        return self.Cy @ f @ self.Cx.T

    def synthesis(self, F):
        return self.Cy.T @ F @ self.Cx

    def grad(self, F):
        return (self.Cy.T @ F @ self.Sx.T, self.Sy @ F @ self.Cx)

    def nonlin(self, F):
        gx, gy = self.grad(F)
        return self.analysis(0.5 * (gx * gx + gy * gy))

def hj_snapshots(sp, F0, gamma, nsteps):
    dt = 1.0 / nsteps
    E = jnp.exp(0.5 * gamma * sp.lap * dt / 2)
    E2 = E * E
    N = sp.nonlin

    def step(Y, _):
        k1 = N(Y)
        k2 = N(E * (Y + 0.5 * dt * k1))
        k3 = N(E * Y + 0.5 * dt * k2)
        k4 = N(E2 * Y + dt * E * k3)
        Yn = E2 * Y + dt / 6 * (E2 * k1 + 2 * E * (k2 + k3) + k4)
        return (Yn, Yn)
    _, Ys = jax.lax.scan(step, F0, None, length=nsteps)
    return jnp.concatenate([F0[None], Ys], 0)

def interp_cos(grid_from, f, grid_to):
    s = Spectral2D(grid_from)
    F = s.analysis(jnp.asarray(f))
    cx, _, _ = s.sx.eval_mats(grid_to.x)
    cy, _, _ = s.sy.eval_mats(grid_to.y)
    return jnp.asarray(cy) @ F @ jnp.asarray(cx).T

def spectrum_tail(grid, f, frac=0.75):
    sp = Spectral2D(grid)
    F = np.abs(np.asarray(jax.jit(lambda f, Cx, Cy: Cy @ f @ Cx.T)(jnp.asarray(f), sp.Cx, sp.Cy)))
    ky, kx = np.meshgrid(np.arange(grid.ny) / grid.ny, np.arange(grid.nx) / grid.nx, indexing='ij')
    return float(F[np.maximum(kx, ky) >= frac].max() / F.max())

def flow_map(grid, u, w, gamma, Xp, Yp, logrho0, nsteps=1000, jac=True, hj_nx=None):
    t0 = time.time()
    g = grid if hj_nx is None else Grid(hj_nx, hj_nx // 2, grid.Lx, grid.Ly)
    epsu, epsw, epslr = (gamma * u, gamma * w, gamma * logrho0)
    if g is not grid:
        ikey = ('ip', grid.nx, grid.ny, g.nx, g.ny, grid.Lx, grid.Ly)
        if ikey not in _JIT_CACHE:
            _JIT_CACHE[ikey] = jax.jit(lambda f, gf=grid, gt=g: interp_cos(gf, f, gt))
        ip = _JIT_CACHE[ikey]
        epsu, epsw, epslr = (ip(jnp.asarray(f)) for f in (epsu, epsw, epslr))
    grid, u, w, logrho0 = (g, None, None, None)
    sp = Spectral2D(grid)

    def core(epsu, epsw, epslr, Xp, Yp, gamma):
        B0 = sp.analysis(jnp.asarray(epsu))
        A1 = sp.analysis(jnp.asarray(epsw))
        Bs = hj_snapshots(sp, B0, gamma, 2 * nsteps)
        As = hj_snapshots(sp, A1, gamma, 2 * nsteps)[::-1]
        A0 = sp.synthesis(As[0])
        B1 = sp.synthesis(Bs[-1])
        A0_sk = jnp.asarray(epslr) - jnp.asarray(epsu)
        cons0 = jnp.max(jnp.abs(A0 - A0_sk))
        cons1 = jnp.max(jnp.abs(B1 + jnp.asarray(epsw)))
        D = As - Bs
        sx, sy = (sp.sx, sp.sy)

        def vel(Dk, X, Y):
            cx, sx_, ccx = (jnp.asarray(a) for a in _eval_mats_j(sx, X))
            cy, sy_, ccy = (jnp.asarray(a) for a in _eval_mats_j(sy, Y))
            Qs = Dk @ sx_.T
            Qc = Dk @ cx.T
            fx = jnp.sum(cy.T * Qs, 0)
            fy = jnp.sum(sy_.T * Qc, 0)
            if not jac:
                return (0.5 * fx, 0.5 * fy, None)
            Qcc = Dk @ ccx.T
            fxx = jnp.sum(cy.T * Qcc, 0)
            fxy = jnp.sum(sy_.T * Qs, 0)
            fyy = jnp.sum(ccy.T * Qc, 0)
            H = 0.5 * jnp.stack([jnp.stack([fxx, fxy], -1), jnp.stack([fxy, fyy], -1)], -2)
            return (0.5 * fx, 0.5 * fy, H)
        dt = 1.0 / nsteps

        def rhs(Dk, X, Y, J):
            vx, vy, H = vel(Dk, X, Y)
            dJ = None if J is None else H @ J
            return (vx, vy, dJ)

        def step(state, k):
            X, Y, J = state
            D0, Dh, D1 = (D[2 * k], D[2 * k + 1], D[2 * k + 2])
            a1 = rhs(D0, X, Y, J)
            a2 = rhs(Dh, X + 0.5 * dt * a1[0], Y + 0.5 * dt * a1[1], None if J is None else J + 0.5 * dt * a1[2])
            a3 = rhs(Dh, X + 0.5 * dt * a2[0], Y + 0.5 * dt * a2[1], None if J is None else J + 0.5 * dt * a2[2])
            a4 = rhs(D1, X + dt * a3[0], Y + dt * a3[1], None if J is None else J + dt * a3[2])
            Xn = X + dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
            Yn = Y + dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
            Jn = None if J is None else J + dt / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
            return ((Xn, Yn, Jn), None)
        J0 = jnp.broadcast_to(jnp.eye(2), (Xp.size, 2, 2)) if jac else None
        (XT, YT, JT), _ = jax.lax.scan(step, (Xp, Yp, J0), jnp.arange(nsteps))
        return (XT, YT, JT, cons0, cons1)
    fkey = ('flow', grid.nx, grid.ny, grid.Lx, grid.Ly, int(nsteps), bool(jac))
    if fkey not in _JIT_CACHE:
        _JIT_CACHE[fkey] = jax.jit(core)
    XT, YT, JT, cons0, cons1 = _JIT_CACHE[fkey](jnp.asarray(epsu), jnp.asarray(epsw), jnp.asarray(epslr), jnp.asarray(Xp, float), jnp.asarray(Yp, float), jnp.asarray(gamma, float))
    XT.block_until_ready()
    return dict(TX=np.asarray(XT), TY=np.asarray(YT), hj_nx=grid.nx, J=None if JT is None else np.asarray(JT), hj_cons_t0=float(cons0), hj_cons_t1=float(cons1), t_total=time.time() - t0)

def _eval_mats_j(s1, xp):
    k = jnp.asarray(s1.k)
    wt = jnp.asarray(s1.wt)
    A = xp[:, None] * k[None, :]
    return (wt * jnp.cos(A), -(wt * k) * jnp.sin(A), -(wt * k ** 2) * jnp.cos(A))
