"""Manufactured McCann maps and continuum bridge flows on the unit sphere."""
import math
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
PI = math.pi

def legendre_heat_kernel(theta, s, lmax=None):
    theta = np.asarray(theta, dtype=float)
    if lmax is None:
        lmax = int(np.ceil(np.sqrt(45.0 / s))) + 10
    x = np.cos(theta)
    p0, p1 = (np.ones_like(x), x.copy())
    tot = p0 * 1.0 / (4 * PI) + 3.0 / (4 * PI) * np.exp(-2 * s) * p1
    for l in range(1, lmax):
        p2 = ((2 * l + 1) * x * p1 - l * p0) / (l + 1)
        L = l + 1
        tot = tot + (2 * L + 1) / (4 * PI) * np.exp(-L * (L + 1) * s) * p2
        p0, p1 = (p1, p2)
    return tot
_GL_CACHE = {}

def _gl(n):
    if n not in _GL_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL_CACHE[n] = (jnp.asarray(0.5 * (x + 1)), jnp.asarray(0.5 * w))
    return _GL_CACHE[n]

def heat_H(th2, s, nquad=64, mmax=2):
    th2 = jnp.minimum(jnp.asarray(th2, dtype=jnp.float64), (PI - 1e-06) ** 2)
    s = jnp.asarray(s, dtype=jnp.float64)
    y, wq = _gl(nquad)
    W = jnp.sqrt(jnp.maximum(PI * PI - th2, 0.0))
    Weff = jnp.minimum(W, jnp.sqrt(160.0 * s))
    w = Weff[..., None] * y
    w2 = w * w
    phi2 = th2[..., None] + w2
    phi = jnp.sqrt(phi2)
    tiny = th2[..., None] <= 1e-08
    th2s = jnp.where(tiny, 1.0, th2[..., None])
    th = jnp.sqrt(th2s)
    phis = jnp.sqrt(th2s + w2)
    a = 0.5 * (phis + th)
    b = 0.5 * w2 / (phis + th)
    sinc = lambda z: jnp.where(jnp.abs(z) < 0.0001, 1 - z * z / 6 + z ** 4 / 120, jnp.sin(z) / jnp.where(jnp.abs(z) < 0.0001, 1.0, z))
    G_reg = sinc(a) * sinc(b)
    wr = jnp.sqrt(w2)
    G0 = sinc(0.5 * wr) ** 2
    sm = wr < 0.01
    wrs = jnp.where(sm, 1.0, wr)
    G1 = jnp.where(sm, -1 / 6 + w2 / 120 - w2 * w2 / 5040, (jnp.sin(wrs) / wrs - 1.0) / (wrs * wrs))
    G2 = jnp.where(sm, 1 / 120 - w2 / 3360, (1 / 12 + (wrs * jnp.cos(wrs) - jnp.sin(wrs)) / (4 * wrs ** 3)) / (wrs * wrs))
    x_ = th2[..., None]
    G = jnp.where(tiny, G0 + x_ * G1 + x_ * x_ * G2, G_reg)
    sx = s[..., None] if s.ndim else s
    R = jnp.ones_like(phi)
    for m in range(1, mmax + 1):
        for sg in (1, -1):
            mm = sg * m
            R = R + (-1) ** m * (1 + 2 * PI * mm / phi) * jnp.exp(-(PI * mm * phi + PI * PI * mm * mm) / sx)
    integrand = jnp.exp(-w2 / (4 * sx)) * R / jnp.sqrt(G)
    I = Weff * jnp.sum(wq * integrand, axis=-1)
    return jnp.log(I) + jnp.log(2.0) + s / 4 - 1.5 * jnp.log(4 * PI * s)

def log_heat_kernel(theta, s, **kw):
    th2 = jnp.asarray(theta) ** 2
    return -th2 / (4 * s) + heat_H(th2, s, **kw)
THETA_C = 2.5
Q_C = 2.0 - 2.0 * math.cos(THETA_C)

def cheb_fit(f, a, b, deg):
    k = np.arange(deg + 1)
    x = np.cos(np.pi * (k + 0.5) / (deg + 1))
    fx = np.asarray(f(0.5 * (b - a) * x + 0.5 * (b + a)), dtype=float)
    c = np.polynomial.chebyshev.chebfit(x, fx, deg)
    return c

def cheb_eval(c, q, a, b):
    x = (2.0 * q - (a + b)) / (b - a)
    b1 = jnp.zeros_like(x)
    b2 = jnp.zeros_like(x)
    for ck in c[:0:-1]:
        b1, b2 = (2.0 * x * b1 - b2 + ck, b1)
    return x * b1 - b2 + c[0]

def cheb_eval_scan(C, q, a, b):
    x = (2.0 * q - (a + b)) / (b - a)
    shp = (C.shape[1],) + x.shape
    bc = lambda c: c.reshape((-1,) + (1,) * x.ndim)

    def body(carry, ck):
        b1, b2 = carry
        return ((2.0 * x * b1 - b2 + bc(ck), b1), None)
    (b1, b2), _ = jax.lax.scan(body, (jnp.zeros(shp), jnp.zeros(shp)), C[:0:-1])
    return x * b1 - b2 + bc(C[0])

def _theta2_over_q(q):
    q = np.asarray(q, dtype=float)
    out = np.ones_like(q)
    m = q > 1e-12
    th = 2 * np.arcsin(np.sqrt(q[m]) / 2)
    out[m] = th * th / q[m]
    return out
_B_COEF = None

def theta2_of_q(q):
    global _B_COEF
    if _B_COEF is None:
        _B_COEF = [float(c) for c in cheb_fit(_theta2_over_q, 0.0, Q_C, 50)]
    qq = jnp.minimum(q, Q_C)
    return q * cheb_eval(_B_COEF, qq, 0.0, Q_C)

class KernelTable:

    def __init__(self, s, deg=48):
        self.s = float(s)
        f = lambda q: np.asarray(heat_H(jnp.asarray(q * _theta2_over_q(q)), self.s))
        self.coef = [float(c) for c in cheb_fit(f, 0.0, Q_C, deg)]
        qt = np.linspace(0, Q_C, 2001)
        err = np.max(np.abs(np.asarray(cheb_eval(self.coef, jnp.asarray(qt), 0.0, Q_C)) - f(qt)))
        self.fit_err = float(err)
        self.ell_derivs(jnp.zeros(1))

    def ell(self, q):
        th2 = theta2_of_q(q)
        H = cheb_eval(self.coef, jnp.minimum(q, Q_C), 0.0, Q_C)
        return -th2 / (4.0 * self.s) + H

    def ell_derivs(self, q):
        global _B_COEF
        theta2_of_q(jnp.zeros(1))
        if not hasattr(self, '_dc'):
            sc = 2.0 / Q_C
            A = np.polynomial.chebyshev.chebmulx(np.asarray(_B_COEF))
            A = 0.5 * Q_C * (A + np.pad(np.asarray(_B_COEF), (0, 1)))
            H = np.asarray(self.coef)
            self._dc = [[float(c) for c in A], [float(c) for c in sc * np.polynomial.chebyshev.chebder(A)], [float(c) for c in sc * sc * np.polynomial.chebyshev.chebder(A, 2)], [float(c) for c in H], [float(c) for c in sc * np.polynomial.chebyshev.chebder(H)], [float(c) for c in sc * sc * np.polynomial.chebyshev.chebder(H, 2)]]
            n = max((len(c) for c in self._dc))
            self._dcm = np.array([np.pad(c, (0, n - len(c))) for c in self._dc]).T
        qq = jnp.minimum(q, Q_C)
        V = cheb_eval_scan(self._dcm, qq, 0.0, Q_C)
        k = 1.0 / (4.0 * self.s)
        far = q > Q_C
        a0 = jnp.where(far, q * V[0] / qq, V[0])
        a1 = jnp.where(far, V[0] / Q_C, V[1])
        a2 = jnp.where(far, 0.0, V[2])
        h1 = jnp.where(far, 0.0, V[4])
        h2 = jnp.where(far, 0.0, V[5])
        return (-k * a0 + V[3], -k * a1 + h1, -k * a2 + h2)

def sph2cart(colat, lon):
    st = np.sin(colat)
    return np.stack([st * np.cos(lon), st * np.sin(lon), np.cos(colat)], axis=-1)

def _sinc(z):
    small = jnp.abs(z) < 0.0001
    zs = jnp.where(small, 1.0, z)
    return jnp.where(small, 1 - z * z / 6 + z ** 4 / 120, jnp.sin(zs) / zs)

def expmap(x, u):
    r2 = jnp.sum(u * u, axis=-1, keepdims=True)
    r = jnp.sqrt(jnp.maximum(r2, 1e-300))
    small = r2 < 1e-16
    c = jnp.where(small, 1 - r2 / 2, jnp.cos(r))
    sc = jnp.where(small, 1 - r2 / 6, jnp.sin(r) / r)
    return c * x + sc * u

def geodist(x, y):
    cr = jnp.cross(x, y)
    return jnp.arctan2(jnp.sqrt(jnp.sum(cr * cr, axis=-1)), jnp.sum(x * y, axis=-1))

def logmap(x, y):
    d = geodist(x, y)
    w = y - jnp.sum(x * y, axis=-1, keepdims=True) * x
    nw = jnp.sqrt(jnp.sum(w * w, axis=-1, keepdims=True))
    return w * (d[..., None] / jnp.maximum(nw, 1e-300))

def tangent_frame(x):
    a = jnp.where(jnp.abs(x[..., 2:3]) < 0.9, jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0]))
    e1 = jnp.cross(a, x)
    e1 = e1 / jnp.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = jnp.cross(x, e1)
    return (e1, e2)

class GLGrid:

    def __init__(self, nlat, nlon=None):
        nlon = 2 * nlat if nlon is None else nlon
        z, wz = np.polynomial.legendre.leggauss(nlat)
        z, wz = (z[::-1], wz[::-1])
        self.nlat, self.nlon = (nlat, nlon)
        self.colat = np.arccos(z)
        self.wz = wz
        self.lon = 2 * np.pi * np.arange(nlon) / nlon
        C, L = np.meshgrid(self.colat, self.lon, indexing='ij')
        self.pts = sph2cart(C, L).reshape(-1, 3)
        self.w = np.repeat(wz * (2 * np.pi / nlon), nlon)
        self.N = nlat * nlon

    @property
    def h(self):
        return float(max(np.max(np.diff(np.concatenate([[0], self.colat, [np.pi]]))), 2 * np.pi / self.nlon))

def fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    lon = np.pi * (1 + 5 ** 0.5) * i
    return sph2cart(np.arccos(z), lon)

def _blocks(X, block):
    n = X.shape[0]
    nb = -(-n // block)
    pad = nb * block - n
    Xp = jnp.concatenate([X, jnp.broadcast_to(X[:1], (pad,) + X.shape[1:])], 0) if pad else X
    return (Xp.reshape((nb, block) + X.shape[1:]), n)

def make_passes(table, block=512):
    ell = table.ell

    def q_of(Xb, Y):
        d = Xb[:, None, :] - Y[None, :, :]
        return jnp.sum(d * d, axis=-1)

    @jax.jit
    def lse(X, Y, f):
        Xb, n = _blocks(X, block)

        def one(xb):
            M = ell(q_of(xb, Y)) + f[None, :]
            m = jnp.max(M, axis=1, keepdims=True)
            return m[:, 0] + jnp.log(jnp.sum(jnp.exp(M - m), axis=1))
        return jax.lax.map(one, Xb).reshape(-1)[:n]

    @jax.jit
    def matvec(X, Y, po, pi, v):
        Xb, n = _blocks(X, block)
        pob, _ = _blocks(po, block)

        def one(args):
            xb, pb = args
            E = jnp.exp(ell(q_of(xb, Y)) + pb[:, None] + pi[None, :])
            return E @ v
        out = jax.lax.map(one, (Xb, pob))
        return out.reshape((-1,) + v.shape[1:])[:n]

    @jax.jit
    def materialize(X, Y, po, pi):
        Xb, n = _blocks(X, block)
        pob, _ = _blocks(po, block)

        def one(args):
            xb, pb = args
            return jnp.exp(ell(q_of(xb, Y)) + pb[:, None] + pi[None, :])
        return jax.lax.map(one, (Xb, pob)).reshape(-1, Y.shape[0])[:n]
    return dict(lse=lse, matvec=matvec, materialize=materialize)

class SphereBridge:

    def __init__(self, grid, logrho0_fn, gamma, deg=48, block=512, materialize_max=16000000):
        self.grid, self.gamma, self.s = (grid, float(gamma), 0.5 * float(gamma))
        self.table = KernelTable(self.s, deg)
        self.P = make_passes(self.table, block)
        self.X = jnp.asarray(grid.pts)
        self.lw = jnp.log(jnp.asarray(grid.w))
        self.logrho0_fn = logrho0_fn
        lr = np.asarray(logrho0_fn(grid.pts))
        mass = np.sum(np.exp(lr) * grid.w)
        self.mass_defect = float(mass / (4 * np.pi) - 1)
        self.lr0 = jnp.asarray(lr - np.log(mass / (4 * np.pi)))
        self.la = self.lr0 + self.lw
        self.materialize_max = materialize_max
        self.alpha = None
        self.beta = None
        self.log = []

    def alpha_of(self, beta):
        return self.lr0 - self.P['lse'](self.X, self.X, beta + self.lw)

    def col_log_marg(self, alpha, beta):
        return beta + self.P['lse'](self.X, self.X, alpha + self.lw)

    def residual(self, alpha, beta):
        return float(jnp.max(jnp.abs(jnp.expm1(self.col_log_marg(alpha, beta)))))

    def solve(self, beta0=None, tol=1e-13, max_newton=40, cg_tol=0.001, cg_max=3000, sinkhorn_pre=0, verbose=True):
        import time
        t0 = time.time()
        N = self.grid.N
        beta = jnp.zeros(N) if beta0 is None else jnp.asarray(beta0)
        X, lw = (self.X, self.lw)
        for _ in range(sinkhorn_pre):
            alpha = self.alpha_of(beta)
            beta = -self.P['lse'](X, X, alpha + lw)
        alpha = self.alpha_of(beta)
        a = jnp.exp(self.la)
        w = jnp.exp(lw)
        mat = N * N <= self.materialize_max
        it_total = 0
        for k in range(max_newton):
            lc = self.col_log_marg(alpha, beta)
            c = w * jnp.exp(lc)
            rel = float(jnp.max(jnp.abs(jnp.expm1(lc))))
            obj = float(jnp.sum(a * alpha) + jnp.sum(w * beta))
            self.log.append(dict(newton=k, rel_marg=rel, obj=obj, cg=it_total, t=time.time() - t0))
            if verbose:
                print(f'  newton {k:2d} max|c/w-1| = {rel:.3e}  cg_total={it_total}  t={time.time() - t0:.1f}s', flush=True)
            if rel < tol:
                break
            rhs = w - c
            po, pi = (alpha + lw, beta + lw)
            if mat:
                Pi = self.P['materialize'](X, X, po, pi)
                Sv = lambda v: c * v - Pi @ v / a @ Pi
            else:
                Sv = lambda v: c * v - self.P['matvec'](X, X, pi, po, self.P['matvec'](X, X, po, pi, v) / a)
            eta = min(cg_tol, max(0.01 * rel, 1e-06))
            d, it = _pcg(Sv, rhs - jnp.mean(rhs), c, eta, cg_max)
            it_total += it
            d = d - jnp.mean(d)
            step = 1.0
            while True:
                bn = beta + step * d
                an = self.alpha_of(bn)
                on = float(jnp.sum(a * an) + jnp.sum(w * bn))
                if np.isfinite(on) and on >= obj - 1e-15 * abs(obj) or step < 0.001:
                    break
                step *= 0.5
            beta, alpha = (bn, an)
            if mat:
                del Pi
        self.alpha, self.beta = (alpha, beta)
        self.solve_time = time.time() - t0
        self.final_rel = rel
        self.newton_iters = k
        self.cg_iters = it_total
        return (alpha, beta)

    def logphi1(self, Z):
        return -self.P['lse'](Z, self.X, self.alpha + self.lw)

    def logphihat0(self, Z):
        return self.logrho0_fn(Z) - self.P['lse'](Z, self.X, self.beta + self.lw)

    def potentials_gamma(self, Z):
        return (self.gamma * self.logphihat0(Z), self.gamma * self.logphi1(Z))

def _pcg(A, b, m, tol, maxit):
    x = jnp.zeros_like(b)
    r = b
    z = r / m
    p = z
    rz = jnp.vdot(r, z)
    nb = float(jnp.sqrt(jnp.vdot(b, b / m)))
    it = 0
    for it in range(1, maxit + 1):
        Ap = A(p)
        al = rz / jnp.vdot(p, Ap)
        x = x + al * p
        r = r - al * Ap
        z = r / m
        rzn = jnp.vdot(r, z)
        if float(jnp.sqrt(rzn)) <= tol * nb:
            break
        p = z + rzn / rz * p
        rz = rzn
    return (x, it)

class ManufacturedMcCann:
    COEF = dict(x1x3=0.8, x1x2=0.5, x3_x1x1_m_x2x2=0.6, x2_4x3x3_m_x1x1_m_x2x2=0.4, x1=0.3, x2x3=-0.35)

    def __init__(self, amp):
        self.amp = float(amp)

    def u_ext(self, x):
        x1, x2, x3 = (x[..., 0], x[..., 1], x[..., 2])
        C = self.COEF
        return self.amp * (C['x1x3'] * x1 * x3 + C['x1x2'] * x1 * x2 + C['x3_x1x1_m_x2x2'] * x3 * (x1 * x1 - x2 * x2) + C['x2_4x3x3_m_x1x1_m_x2x2'] * x2 * (4 * x3 * x3 - x1 * x1 - x2 * x2) + C['x1'] * x1 + C['x2x3'] * x2 * x3)

    def grad_u(self, x):
        g = jax.grad(lambda y: self.u_ext(y))
        G = jnp.vectorize(g, signature='(3)->(3)')(x)
        return G - jnp.sum(G * x, axis=-1, keepdims=True) * x

    def T0_single(self, x):
        xn = x / jnp.linalg.norm(x)
        g = jax.grad(self.u_ext)(xn)
        gt = g - jnp.dot(g, xn) * xn
        return expmap(xn, -gt)

    def T0(self, X):
        return jax.vmap(self.T0_single)(jnp.asarray(X))

    def _jacdet_single(self, x):
        J = jax.jacfwd(self.T0_single)(x)
        y = self.T0_single(x)
        e1, e2 = tangent_frame(x)
        f1, f2 = tangent_frame(y)
        Ex = jnp.stack([e1, e2], 1)
        Ey = jnp.stack([f1, f2], 1)
        return jnp.linalg.det(Ey.T @ J @ Ex)

    def rho0(self, X):
        return jnp.abs(jax.vmap(self._jacdet_single)(jnp.asarray(X)))

    def logrho0(self, X):
        return jnp.log(self.rho0(X))

    def _local_matrix(self, x):
        g = jax.grad(self.u_ext)(x)
        Hs = jax.hessian(self.u_ext)(x)
        e1, e2 = tangent_frame(x)
        E = jnp.stack([e1, e2], 1)
        HessS = E.T @ Hs @ E - jnp.dot(g, x) * jnp.eye(2)
        gt = g - jnp.dot(g, x) * x
        d = jnp.linalg.norm(gt)
        dirv = -(E.T @ gt) / jnp.maximum(d, 1e-300)
        perp = jnp.array([-dirv[1], dirv[0]])
        dcot = jnp.where(d < 1e-06, 1 - d * d / 3, d / jnp.tan(jnp.maximum(d, 1e-300)))
        A = jnp.outer(dirv, dirv) + dcot * jnp.outer(perp, perp)
        return (A - HessS, d)

    def rho0_analytic(self, X):

        def one(x):
            M, d = self._local_matrix(x)
            return jnp.linalg.det(M) * _sinc(d)
        return jax.vmap(one)(jnp.asarray(X))

    def local_min_eig(self, X):
        return jax.vmap(lambda x: jnp.min(jnp.linalg.eigvalsh(self._local_matrix(x)[0])))(jnp.asarray(X))

    def check_c_convexity(self, Xs, Xfine, block=256):
        Xs = jnp.asarray(Xs)
        Xf = jnp.asarray(Xfine)
        Ts = self.T0(Xs)
        us = self.u_ext(Xs)
        uf = self.u_ext(Xf)
        base = 0.5 * geodist(Xs, Ts) ** 2 - us
        worst = []
        for i in range(0, Xs.shape[0], block):
            D = geodist(Xf[None, :, :], Ts[i:i + block, None, :])
            val = 0.5 * D * D - uf[None, :] - base[i:i + block, None]
            worst.append(jnp.min(val, axis=1))
        return jnp.concatenate(worst)

class ZonalReference:

    def __init__(self, a=(0.35, 0.3, -0.2)):
        self.a = tuple((float(v) for v in a))

    def rho0_z(self, z):
        a1, a2, a3 = self.a
        return 1 + a1 * z + a2 * 0.5 * (3 * z * z - 1) + a3 * 0.5 * (5 * z ** 3 - 3 * z)

    def logrho0(self, X):
        return jnp.log(self.rho0_z(jnp.asarray(X)[..., 2]))

    def M0(self, z):
        a1, a2, a3 = self.a
        F = lambda t: t + a1 * t * t / 2 + a2 * 0.5 * (t ** 3 - t) + a3 * 0.5 * (5 * t ** 4 / 4 - 3 * t * t / 2)
        return 0.5 * (F(1.0) - F(z))

    def T0(self, X):
        X = np.asarray(X, dtype=float)
        z = X[:, 2]
        m = self.M0(z)
        north = m <= 0.5
        Th = np.where(north, 2 * np.arcsin(np.sqrt(np.clip(m, 0, 1))), np.pi - 2 * np.arcsin(np.sqrt(np.clip(1 - m, 0, 1))))
        lon = np.arctan2(X[:, 1], X[:, 0])
        return sph2cart(Th, lon)

def _dheat_H(th2, s, nquad=64):
    return jax.jvp(lambda a: heat_H(a, s, nquad=nquad), (th2,), (jnp.ones_like(th2),))[1]

class BridgeFlow:

    def __init__(self, bridge, n_gh=14, chunk=256, newton=2):
        self.B = bridge
        self.gamma = bridge.gamma
        self.n_gh = n_gh
        self.chunk = chunk
        self.newton = newton
        xg, wg = np.polynomial.hermite.hermgauss(n_gh)
        XI = np.stack(np.meshgrid(xg, xg, indexing='ij'), -1).reshape(-1, 2)
        W = np.outer(wg, wg).reshape(-1)
        self.XI = jnp.asarray(XI)
        self.lWg = jnp.asarray(np.log(W) + np.sum(XI ** 2, 1))
        lr0 = bridge.logrho0_fn
        self.D = (bridge.X, bridge.alpha + bridge.lw, bridge.beta + bridge.lw)
        tab = bridge.table

        def lse_vgh(z, f, sign, Xg):
            d = z[None, :] - Xg
            q = jnp.sum(d * d, -1)
            l0, l1, l2 = tab.ell_derivs(q)
            v = l0 + f
            m = jnp.max(v)
            pw = jnp.exp(v - m)
            S = jnp.sum(pw)
            pw = pw / S
            gq = 2.0 * d
            g = jnp.sum((pw * l1)[:, None] * gq, 0)
            Hm = jnp.einsum('j,ja,jb->ab', pw * (l2 + l1 * l1), gq, gq) + 2.0 * jnp.sum(pw * l1) * jnp.eye(3) - jnp.outer(g, g)
            return (sign * (m + jnp.log(S)), sign * g, sign * Hm)

        def psi_vgh_u(u, x, Ex, which, D):
            Xg, fa, fb = D
            Zf = lambda uu: expmap(x, Ex @ uu)
            z = Zf(u)
            J = jax.jacfwd(Zf)(u)
            K2 = jax.jacfwd(jax.jacfwd(Zf))(u)
            if which == 0:
                v, g, Hm = lse_vgh(z, fb, -1.0, Xg)
                lrf = lambda uu: lr0(Zf(uu)[None, :])[0]
                v = v + lrf(u)
                gu = J.T @ g + jax.grad(lrf)(u)
                gl = jax.grad(lrf)
                hd = 0.0001
                Hl = jnp.stack([(gl(u + hd * e) - gl(u - hd * e)) / (2 * hd) for e in jnp.eye(2)], 1)
                Hu = J.T @ Hm @ J + jnp.einsum('a,abc->bc', g, K2) + 0.5 * (Hl + Hl.T)
            else:
                v, g, Hm = lse_vgh(z, fa, -1.0, Xg)
                gu = J.T @ g
                Hu = J.T @ Hm @ J + jnp.einsum('a,abc->bc', g, K2)
            return (v, gu, Hu)
        self.psi_vgh_u = psi_vgh_u
        P = bridge.P
        self.psi_batch = (lambda Z, D: lr0(Z) - P['lse'](Z, D[0], D[2]), lambda Z, D: -P['lse'](Z, D[0], D[1]))
        self._heat = {}
        self._vel = jax.jit(self._velocity_impl)

    def _heat_chunk(self, X, tau, which, D):
        psi_b = self.psi_batch[which]
        e1, e2 = tangent_frame(X)
        E = jnp.stack([e1, e2], -1)

        def K_single(u):
            r2 = jnp.dot(u, u)
            return -r2 / (4 * tau) + heat_H(r2, tau) + jnp.log(_sinc(jnp.sqrt(r2 + 1e-300)))

        def gh_single(u, x, Ex):
            _, gp, Hp = self.psi_vgh_u(u, x, Ex, which, D)
            return (jax.grad(K_single)(u) + gp, jax.hessian(K_single)(u) + Hp)
        gh = jax.vmap(gh_single, in_axes=(0, 0, 0))
        lam_floor = 0.1 / (2 * tau)

        def regH(Hn):
            lam, V = jnp.linalg.eigh(-Hn)
            lam = jnp.maximum(lam, lam_floor)
            return (lam, V)

        def newton_body(_, c):
            g, Hn = gh(c, X, E)
            lam, V = regH(Hn)
            step = jnp.einsum('pab,pb,pcb,pc->pa', V, 1.0 / lam, V, g)
            return c + step
        c = jax.lax.fori_loop(0, self.newton, newton_body, jnp.zeros((X.shape[0], 2)))
        lam, V = regH(gh(c, X, E)[1])
        L = jnp.sqrt(2.0) * (V / jnp.sqrt(lam)[:, None, :]) @ jnp.swapaxes(V, 1, 2)
        logdetL = jnp.log(2.0) - 0.5 * jnp.sum(jnp.log(lam), -1)
        U = c[:, None, :] + jnp.einsum('pab,kb->pka', L, self.XI)
        r2 = jnp.sum(U * U, -1)
        U3 = jnp.einsum('pab,pkb->pka', E, U)
        Z = expmap(X[:, None, :], U3)
        psi = psi_b(Z.reshape(-1, 3), D).reshape(r2.shape)
        lk = -r2 / (4 * tau) + heat_H(r2, tau)
        inside = r2 < (PI - 0.01) ** 2
        r2c = jnp.where(inside, r2, 1.0)
        Fk = jnp.where(inside, lk + jnp.log(_sinc(jnp.sqrt(r2c + 1e-300))) + psi, -jnp.inf) + self.lWg[None, :]
        m = jnp.max(Fk, 1, keepdims=True)
        wk = jnp.exp(Fk - m)
        S = jnp.sum(wk, 1)
        logval = m[:, 0] + jnp.log(S) + logdetL
        fac = 1.0 / (2 * tau) - 2.0 * _dheat_H(r2c, tau)
        g2 = jnp.sum((wk * fac)[..., None] * U, 1) / S[:, None]
        return (logval, jnp.einsum('pab,pb->pa', E, g2))

    def _grad_direct(self, X, which, D):
        e1, e2 = tangent_frame(X)
        E = jnp.stack([e1, e2], -1)
        f = lambda u, x, Ex: self.psi_vgh_u(u, x, Ex, which, D)[1]
        g2 = jax.vmap(f, in_axes=(0, 0, 0))(jnp.zeros((X.shape[0], 2)), X, E)
        return jnp.einsum('pab,pb->pa', E, g2)

    def _grad_log(self, X, tau, which, D):
        return jax.lax.cond(tau > 0, lambda: self._heat_chunk(X, jnp.maximum(tau, 1e-300), which, D)[1], lambda: self._grad_direct(X, which, D))

    def _velocity_impl(self, X, t, D):
        gamma = self.gamma
        g_phi = self._grad_log(X, (1 - t) * gamma / 2, 1, D)
        g_hat = self._grad_log(X, t * gamma / 2, 0, D)
        return 0.5 * gamma * (g_phi - g_hat)

    def velocity(self, X, t):
        X = jnp.asarray(X)
        out = []
        for i in range(0, X.shape[0], self.chunk):
            out.append(self._vel(X[i:i + self.chunk], jnp.asarray(float(t)), self.D))
        return jnp.concatenate(out, 0)

    def log_heat(self, X, tau, which):
        if which not in self._heat:
            self._heat[which] = jax.jit(lambda Xc, ta, D: self._heat_chunk(Xc, ta, which, D))
        f = self._heat[which]
        outs = [f(X[i:i + self.chunk], jnp.asarray(float(tau)), self.D) for i in range(0, X.shape[0], self.chunk)]
        return (jnp.concatenate([o[0] for o in outs]), jnp.concatenate([o[1] for o in outs]))

    def flow(self, X0, nsteps=8, verbose=False):
        from scipy.integrate._ivp import dop853_coefficients as D
        A, Bc, C = (D.A[:12, :12], D.B, D.C[:12])
        X = jnp.asarray(X0)
        h = 1.0 / nsteps
        import time
        t0 = time.time()
        for n in range(nsteps):
            t = n * h
            K = []
            for i in range(12):
                Y = X + h * sum((A[i, j] * K[j] for j in range(i) if A[i, j] != 0)) if i else X
                Y = Y / jnp.linalg.norm(Y, axis=1, keepdims=True)
                K.append(self.velocity(Y, t + C[i] * h))
            X = X + h * sum((Bc[i] * K[i] for i in range(12)))
            X = X / jnp.linalg.norm(X, axis=1, keepdims=True)
            if verbose:
                print(f'    step {n + 1}/{nsteps}  t={time.time() - t0:.1f}s', flush=True)
        return X

def barycentric_maps(bridge, X, n_frechet=8, chunk=128):
    ell = bridge.table.ell

    @jax.jit
    def one(Xc, Xg, fb):
        d = Xc[:, None, :] - Xg[None, :, :]
        Lw = ell(jnp.sum(d * d, -1)) + fb[None, :]
        Lw = Lw - jnp.max(Lw, 1, keepdims=True)
        Wt = jnp.exp(Lw)
        Wt = Wt / jnp.sum(Wt, 1, keepdims=True)
        me = Wt @ Xg
        me = me / jnp.linalg.norm(me, axis=1, keepdims=True)
        m = me
        for _ in range(n_frechet):
            V = logmap(m[:, None, :], Xg[None, :, :])
            m = expmap(m, jnp.einsum('pj,pja->pa', Wt, V))
            m = m / jnp.linalg.norm(m, axis=1, keepdims=True)
        V = logmap(m[:, None, :], Xg[None, :, :])
        resid = jnp.linalg.norm(jnp.einsum('pj,pja->pa', Wt, V), axis=1)
        return (me, m, resid)
    X = jnp.asarray(X)
    Xg, fb = (bridge.X, bridge.beta + bridge.lw)
    outs = [one(X[i:i + chunk], Xg, fb) for i in range(0, X.shape[0], chunk)]
    return tuple((jnp.concatenate([o[k] for o in outs]) for k in range(3)))

def zonal_kernel_matrix(grid, table):
    ell = table.ell
    C = jnp.asarray(grid.colat)
    lon = jnp.asarray(grid.lon)

    @jax.jit
    def row(ci):
        x = jnp.array([jnp.sin(ci), 0.0, jnp.cos(ci)])
        Y = jnp.stack([jnp.sin(C)[:, None] * jnp.cos(lon)[None, :], jnp.sin(C)[:, None] * jnp.sin(lon)[None, :], jnp.broadcast_to(jnp.cos(C)[:, None], (C.shape[0], lon.shape[0]))], -1)
        d = Y - x
        Lk = ell(jnp.sum(d * d, -1))
        m = jnp.max(Lk, 1, keepdims=True)
        return m[:, 0] + jnp.log(jnp.sum(jnp.exp(Lk - m), 1))
    L1 = jax.lax.map(row, C)
    return L1 + math.log(2 * math.pi / grid.nlon)

def solve_zonal(bridge, tol=1e-14, max_newton=60, verbose=True):
    import time
    t0 = time.time()
    g = bridge.grid
    L1 = zonal_kernel_matrix(g, bridge.table)
    lwz = jnp.log(jnp.asarray(g.wz))
    lr = bridge.lr0.reshape(g.nlat, g.nlon)[:, 0]
    a = jnp.exp(lr + lwz)
    wz = jnp.asarray(g.wz)
    beta = jnp.zeros(g.nlat)

    def alpha_of(beta):
        return lr - jax.scipy.special.logsumexp(L1 + (beta + lwz)[None, :], axis=1)
    alpha = alpha_of(beta)
    obj = -np.inf
    for k in range(max_newton):
        Pi = jnp.exp(alpha[:, None] + lwz[:, None] + L1 + beta[None, :] + lwz[None, :])
        c = jnp.sum(Pi, 0)
        rel = float(jnp.max(jnp.abs(c / wz - 1)))
        obj = float(jnp.sum(a * alpha) + jnp.sum(wz * beta))
        if verbose:
            print(f'  zonal newton {k:2d} max|c/w-1| = {rel:.3e}', flush=True)
        if rel < tol:
            break
        S = jnp.diag(c) - Pi.T @ (Pi / a[:, None])
        n = g.nlat
        Sp = S + jnp.ones((n, n)) * jnp.mean(c) / n
        d = jnp.linalg.solve(Sp, wz - c)
        d = d - jnp.mean(d)
        step = 1.0
        while True:
            bn = beta + step * d
            an = alpha_of(bn)
            on = float(jnp.sum(a * an) + jnp.sum(wz * bn))
            if np.isfinite(on) and on >= obj - 1e-15 * abs(obj) or step < 0.0001:
                break
            step *= 0.5
        alpha, beta = (an, bn)
    bridge.alpha = jnp.repeat(alpha, g.nlon)
    bridge.beta = jnp.repeat(beta, g.nlon)
    bridge.solve_time = time.time() - t0
    bridge.final_rel = rel
    bridge.newton_iters = k
    bridge.cg_iters = 0
    return (bridge.alpha, bridge.beta)
