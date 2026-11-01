"""Continuum Neumann-kernel bridge solver for gridded planar densities."""
import math
import time
import numpy as np
import jax
import jax.numpy as jnp
from .ld_core import Kernel1D, Conv1D, sep2d, NEG
from . import engine
jax.config.update('jax_enable_x64', True)
GL_X, GL_W = np.polynomial.legendre.leggauss(6)

def default_r(gamma, h, c=6.0, n=None):
    r = max(1, int(math.floor(math.sqrt(gamma) / (c * h))))
    if n is not None:
        while n % r:
            r -= 1
    return r

class PlaneLD:

    def __init__(self, G, rho0, gamma, r=None, source='avg', window=None, c_target=6.0):
        self.G = G
        g = G.np
        self.nx, self.ny, self.hx, self.hy = (g.nx, g.ny, g.hx, g.hy)
        self.Lx, self.Ly = (g.Lx, g.Ly)
        self.gamma, self.S = (float(gamma), float(gamma) / 2)
        h = max(self.hx, self.hy)
        self.r = default_r(gamma, h, c_target, math.gcd(self.nx, self.ny)) if r is None else r
        r = self.r
        assert self.nx % r == 0 and self.ny % r == 0
        self.nX, self.nY = (self.nx // r, self.ny // r)
        self.HX, self.HY = (r * self.hx, r * self.hy)
        self.source = source
        self.rho0 = np.asarray(rho0, float)
        self.lrho = jnp.asarray(np.log(self.rho0))
        self.logA = math.log(self.hx * self.hy)
        self.logW = math.log(self.HX * self.HY)
        self.xc = (np.arange(self.nx) + 0.5) * self.hx
        self.yc = (np.arange(self.ny) + 0.5) * self.hy
        self.Xt = (np.arange(self.nX) + 0.5) * self.HX
        self.Yt = (np.arange(self.nY) + 0.5) * self.HY
        self.set_window(window)
        self.w = None
        self.u = None
        self.info = {}

    def kern(self, s, axis):
        cell = self.hx if axis == 0 else self.hy
        return Kernel1D(s, 'avg' if self.source == 'avg' else 'point', cell=cell)

    def set_window(self, window):
        if window is None:
            window = (np.inf, np.inf)
        self.window = tuple((float(w) for w in window))
        S = self.S
        wx, wy = self.window
        kx, ky = (self.kern(S, 0), self.kern(S, 1))
        self.t2s = (Conv1D(kx, self.nX, self.HX, self.HX / 2, self.nx, self.hx, self.hx / 2, window=wx), Conv1D(ky, self.nY, self.HY, self.HY / 2, self.ny, self.hy, self.hy / 2, window=wy))
        self.s2t = (Conv1D(kx, self.nx, self.hx, self.hx / 2, self.nX, self.HX, self.HX / 2, window=wx), Conv1D(ky, self.ny, self.hy, self.hy / 2, self.nY, self.HY, self.HY / 2, window=wy))
        self._F = None

    def Gt(self, w):
        lv, _ = sep2d(*self.t2s, w + self.logW)
        return lv

    def Ht(self, u):
        lv, _ = sep2d(*self.s2t, u + self.logA)
        return lv

    def _make(self):
        lr = self.lrho

        def F(w):
            u = lr - self.Gt(w)
            return -self.Ht(u)
        self._F = F

    def solve(self, w0=None, tol=1e-12, maxit=20000, anderson=10, reg=1e-12, verbose=False, check=1):
        if self._F is None:
            self._make()
        F = self._F
        w = jnp.zeros((self.nY, self.nX)) if w0 is None else jnp.asarray(w0)
        t0 = time.time()
        dW, dR = ([], [])
        w_prev = r_prev = None
        best = (np.inf, w)
        hist = []
        err = np.inf
        it = 0
        for it in range(1, maxit + 1):
            g = F(w)
            r = g - w
            r = r - jnp.mean(r)
            err = float(jnp.max(jnp.abs(r)))
            if err < best[0]:
                best = (err, w)
            if it % 25 == 0 or it < 3:
                hist.append((it, err, time.time() - t0))
                if verbose:
                    print(f'    it {it} res {err:.3e} {time.time() - t0:.1f}s', flush=True)
            if err < tol:
                w = g
                break
            if anderson:
                if w_prev is not None:
                    dW.append((w - w_prev).ravel())
                    dR.append((r - r_prev).ravel())
                    if len(dW) > anderson:
                        dW.pop(0)
                        dR.pop(0)
                w_prev, r_prev = (w, r)
                if dW:
                    Rm = jnp.stack(dR, 1)
                    Wm = jnp.stack(dW, 1)
                    M = Rm.T @ Rm
                    M = M + reg * jnp.trace(M) * jnp.eye(M.shape[0])
                    gam = jnp.linalg.solve(M, Rm.T @ r.ravel())
                    w = w + r - ((Wm + Rm) @ gam).reshape(w.shape)
                else:
                    w = w + r
                if err > 100.0 * best[0] and best[0] < 0.001:
                    w = best[1]
                    dW, dR = ([], [])
                    w_prev = r_prev = None
            else:
                w = w + r
        w = jnp.asarray(w)
        u = self.lrho - self.Gt(w)
        marg = float(jnp.max(jnp.abs(jnp.expm1(w + self.Ht(u)))))
        self.w, self.u = (w, u)
        self.info = dict(iters=it, res=err, marg=marg, t_solve=time.time() - t0, hist=hist, r=self.r, window=self.window)
        return self.info

    def _make_newton(self, c_pre):
        lr, logA, logW = (self.lrho, self.logA, self.logW)
        nX, nY = (self.nX, self.nY)
        kx = np.pi * np.arange(nX) / self.Lx
        ky = np.pi * np.arange(nY) / self.Ly
        lam = jnp.asarray(ky[:, None] ** 2 + kx[None, :] ** 2)
        gamma = self.gamma

        def state(w):
            u = lr - self.Gt(w)
            R = w + self.Ht(u)
            return (u, R)

        def Qv(w, u, v):
            V1 = jnp.stack([jnp.ones_like(v), v], -1)
            _, Rt = sep2d(*self.t2s, w + logW, V1)
            z = Rt[..., 0, 1]
            V2 = jnp.stack([jnp.ones_like(z), z], -1)
            _, Rt2 = sep2d(*self.s2t, u + logA, V2)
            return Rt2[..., 0, 1]

        @jax.jit
        def pre(rr):
            c = jax.scipy.fft.dctn(rr, type=2, norm='ortho')
            mu = -jnp.expm1(-c_pre * gamma * lam)
            c = jnp.where(lam == 0, 0.0, c / jnp.where(lam == 0, 1.0, mu))
            return jax.scipy.fft.idctn(c, type=2, norm='ortho')

        def pcg(w, u, R, nu, tol, maxit):
            b = -nu * R
            b = b - jnp.mean(b)
            A = lambda d: nu * (d - Qv(w, u, d))
            x = jnp.zeros_like(b)
            r = b
            z = pre(r / nu)
            p = z
            rz = float(jnp.sum(r * z))
            bn = float(jnp.sqrt(jnp.sum(b * b)))
            k = 0
            while k < maxit and bn > 0 and (float(jnp.sqrt(jnp.sum(r * r))) > tol * bn):
                Ap = A(p)
                pAp = float(jnp.sum(p * Ap))
                if not (pAp > 0 and rz != 0):
                    break
                al = rz / pAp
                x = x + al * p
                r = r - al * Ap
                z = pre(r / nu)
                rzn = float(jnp.sum(r * z))
                p = z + rzn / rz * p
                rz = rzn
                k += 1
            return (x - jnp.mean(x), k)

        @jax.jit
        def dual(w, u):
            return jnp.sum(jnp.exp(lr + logA) * u) + jnp.exp(logW) * jnp.sum(w)
        self._newton = (state, pcg, dual)

    def solve_newton(self, w0=None, tol=1e-12, max_newton=30, cg_tol=0.001, cg_max=2000, c_pre=1.0, sinkhorn_pre=3, verbose=False):
        self._make_newton(c_pre)
        state, pcg, dual = self._newton
        if self._F is None:
            self._make()
        t0 = time.time()
        w = jnp.zeros((self.nY, self.nX)) if w0 is None else jnp.asarray(w0)
        for _ in range(sinkhorn_pre):
            w = self._F(w)
        W = math.exp(self.logW)
        cg_total = 0
        hist = []
        u, R = state(w)
        for k in range(max_newton):
            Rc = R - jnp.mean(R)
            err = float(jnp.max(jnp.abs(jnp.expm1(Rc))))
            hist.append((k, err, cg_total, time.time() - t0))
            if verbose:
                print(f'    newton {k} marg {err:.3e} cg {cg_total} {time.time() - t0:.1f}s', flush=True)
            floor = 64 * np.finfo(float).eps * float(jnp.max(jnp.abs(w)) + jnp.max(jnp.abs(u)))
            if err < max(tol, floor) or (k >= 3 and err > 0.5 * hist[-2][1] and (err < 100 * floor)):
                break
            nu = W * jnp.exp(R)
            eta = min(cg_tol, max(0.01 * err, 1e-07))
            d, nit = pcg(w, u, Rc, nu, eta, cg_max)
            cg_total += int(nit)
            step = 1.0
            obj = float(dual(w, u))
            ok = bool(jnp.all(jnp.isfinite(d)))
            while ok:
                wn = w + step * d
                un, Rn = state(wn)
                on = float(dual(wn, un))
                errn = float(jnp.max(jnp.abs(jnp.expm1(Rn - jnp.mean(Rn)))))
                if np.isfinite(on) and np.isfinite(errn) and (on >= obj - 1e-14 * abs(obj) or errn < err):
                    break
                step *= 0.5
                if step < 0.001:
                    ok = False
            if not ok:
                for _ in range(20):
                    w = self._F(w)
                u, R = state(w)
                hist.append(('sinkhorn_fallback', k))
                continue
            w, u, R = (wn, un, Rn)
        self.w, self.u = (w, u)
        marg = float(jnp.max(jnp.abs(jnp.expm1(R - jnp.mean(R)))))
        self.w = w - jnp.mean(R)
        self.u = self.lrho - self.Gt(self.w)
        marg = float(jnp.max(jnp.abs(jnp.expm1(self.w + self.Ht(self.u)))))
        self.info = dict(newton=k, cg=cg_total, marg=marg, t_solve=time.time() - t0, hist=hist, r=self.r, window=self.window)
        return self.info

    def barycentre(self):
        V = jnp.broadcast_to(jnp.stack([jnp.ones(self.nX), jnp.asarray(self.Xt)], -1)[None], (self.nY, self.nX, 2))
        _, Rt = sep2d(*self.t2s, self.w + self.logW, V, ycoord=jnp.asarray(self.Yt))
        return (np.asarray(Rt[..., 0, 1]), np.asarray(Rt[..., 0, 2]))

    def displacement_window(self, margin_sig=12.0):
        bx, by = self.barycentre()
        X, Y = np.meshgrid(self.xc, self.yc)
        dx = np.max(np.abs(bx - X))
        dy = np.max(np.abs(by - Y))
        m = margin_sig * math.sqrt(2 * self.S) + 2 * max(self.HX, self.HY)
        return (dx + m, dy + m)

    def w_on(self, other):
        from scipy.ndimage import map_coordinates
        g = np.asarray(self.w) * self.gamma
        g = g - g.mean()
        X, Y = np.meshgrid(other.Xt, other.Yt)
        ci = X / self.HX - 0.5
        cj = Y / self.HY - 0.5
        gi = map_coordinates(g, [cj.ravel(), ci.ravel()], order=3, mode='mirror').reshape(X.shape)
        return jnp.asarray(gi / other.gamma)

def solve_chain(G, rho0, gamma_list, source='avg', tol=1e-12, verbose=True, c_target=6.0, method='newton', keep=True, **kw):
    out = []
    prev = None
    for gamma in gamma_list:
        tic = time.time()
        win = None if prev is None else prev.displacement_window()
        P = PlaneLD(G, rho0, gamma, source=source, window=win, c_target=c_target)
        w0 = None if prev is None else prev.w_on(P)
        if method == 'newton':
            info = P.solve_newton(w0, tol=tol, verbose=verbose, **kw)
        else:
            info = P.solve(w0, tol=tol, verbose=verbose, **kw)
        info['t_level'] = time.time() - tic
        if verbose:
            print(f'gamma {gamma:g} r {P.r} ' + ' '.join((f'{k} {v}' for k, v in info.items() if k in ('iters', 'newton', 'cg'))) + f" marg {info['marg']:.2e} win ({P.window[0]:.3f},{P.window[1]:.3f}) {info['t_level']:.1f}s", flush=True)
        if not keep and out:
            out[-1] = None
        out.append(P)
        prev = P
    return out

class PlanePathLD:

    def __init__(self, P, m=1, bin_factor=16.0, margin_sig=10.0, c_switch=1.5, fine_ext=True, c_end=0.5, c_init=0.5, verbose=False, quad='simpson', refine=1, G_assembly=None):
        self.P, self.m = (P, m)
        self.R = int(refine)
        if m % self.R:
            raise ValueError('refine must divide m')
        self.ma = m // self.R
        self.nya, self.nxa = (self.R * P.ny, self.R * P.nx)
        self.hxa, self.hya = (P.hx / self.R, P.hy / self.R)
        self.rho0a = np.repeat(np.repeat(np.asarray(P.rho0), self.R, 0), self.R, 1)
        if G_assembly is not None:
            self.Ga = G_assembly
        else:
            self.Ga = P.G if self.R == 1 else engine.PlaneJ(self.nxa, self.nya, P.G.np.Lx, P.G.np.Ly)
        assert (self.Ga.ny, self.Ga.nx) == (self.nya, self.nxa)
        self.quad = quad
        w = np.zeros(2 * self.ma + 1)
        w[0::2] = 2.0
        w[1::2] = 4.0
        w[0] = w[-1] = 1.0
        self._w = w / w.sum()
        self.verbose = verbose
        self.c_end, self.c_init = (c_end, c_init)
        self.G = P.G
        self.S = P.S
        self.bin_factor, self.margin_sig = (bin_factor, margin_sig)
        bx, by = P.barycentre()
        X, Y = np.meshgrid(P.xc, P.yc)
        self.Dmax = (float(np.max(np.abs(bx - X))), float(np.max(np.abs(by - Y))))
        self.cache = {}
        self.uA = P.u + P.logA
        self.wW = P.w + P.logW
        self.c_switch = c_switch
        self.fine = fine_ext and P.r > 1
        if self.fine:
            kx, ky = (P.kern(P.S, 0), P.kern(P.S, 1))
            win = (self.Dmax[0] + margin_sig * math.sqrt(2 * P.S) + 2 * P.HX, self.Dmax[1] + margin_sig * math.sqrt(2 * P.S) + 2 * P.HY)
            cx = Conv1D(kx, P.nx, P.hx, P.hx / 2, P.nx, P.hx, P.hx / 2, window=win[0])
            cy = Conv1D(ky, P.ny, P.hy, P.hy / 2, P.ny, P.hy, P.hy / 2, window=win[1])
            lv, _ = sep2d(cx, cy, self.uA)
            self.wfA = -lv + P.logA
        self.stats = dict(compiles=0, bins=[])

    def _conv(self, which, tau):
        P, m = (self.P, self.m)
        b = int(math.floor(math.log(tau / self.S) / math.log(self.bin_factor) + 1e-12))
        key = (which, b)
        if key in self.cache:
            return self.cache[key]
        tau_lo = self.S * self.bin_factor ** b
        tau_hi = min(self.S, tau_lo * self.bin_factor)
        frac = tau_hi / self.S
        convs = []
        for ax in (0, 1):
            n, h, L = (P.nx, P.hx, P.Lx) if ax == 0 else (P.ny, P.hy, P.Ly)
            H = P.HX if ax == 0 else P.HY
            if which == 'hat':
                kind, cell, Nin, hin = ('avg' if P.source == 'avg' else 'point', h, n, h)
            elif which == 'phi':
                kind, cell, Nin, hin = ('point', None, n // P.r, H)
            else:
                kind, cell, Nin, hin = ('point', None, n, h)
            win = frac * self.Dmax[ax] + self.margin_sig * math.sqrt(2 * tau_hi) + 2 * hin
            c = Conv1D((kind, cell, tau_hi), Nin, hin, hin / 2, 2 * m * n + 1, h / (2 * m), 0.0, window=win, variants=('v', 'd'), s_min=tau_lo / 1.0001)
            convs.append(c)
        self.cache[key] = tuple(convs)
        self.stats['compiles'] += 2
        self.stats['bins'].append((which, b, convs[0].nin, convs[0].npairs, convs[1].nin, convs[1].npairs))
        return self.cache[key]

    def fields(self, t):
        P = self.P
        tau, taup = (t * self.S, (1 - t) * self.S)
        cx, cy = self._conv('hat', tau)
        lh, Rh = sep2d(cx, cy, self.uA, s=tau)
        gxh, gyh = (Rh[..., 0, 1], Rh[..., 1, 0])
        if math.sqrt(2 * taup) < self.c_end * max(P.HX, P.HY, P.hx, P.hy):
            self.stats['terminal_evals'] = self.stats.get('terminal_evals', 0) + 1
            return (lh, gxh, gyh, -lh - 2 * taup * self._lap(lh), -gxh, -gyh)
        lp, gx, gy = self.phi_fields(t)
        return (lh, gxh, gyh, lp, gx, gy)

    def _lap(self, f):
        dx, dy = (self.P.hx / (2 * self.m), self.P.hy / (2 * self.m))
        g = jnp.pad(f, 1, mode='edge')
        return (g[1:-1, 2:] - 2 * f + g[1:-1, :-2]) / dx ** 2 + (g[2:, 1:-1] - 2 * f + g[:-2, 1:-1]) / dy ** 2

    def _cells(self, f):
        m, ny, nx = (self.ma, self.nya, self.nxa)
        if self.quad != 'simpson':
            return f[1::2, 1::2].reshape(ny, m, nx, m).mean(axis=(1, 3))
        w = self._w
        acc = 0.0
        for a in range(2 * m + 1):
            row = f[a:a + 2 * m * ny:2 * m][:ny]
            col = sum((w[b] * row[:, b:b + 2 * m * nx:2 * m][:, :nx] for b in range(2 * m + 1)))
            acc = acc + w[a] * col
        return acc

    def _face_x(self, J):
        m, ny, nx = (self.ma, self.nya, self.nxa)
        Jf = J[:, 2 * m:2 * m * nx:2 * m]
        if self.quad != 'simpson':
            return Jf[1::2].reshape(ny, m, nx - 1).mean(1)
        return sum((self._w[k] * Jf[k:k + 2 * m * ny:2 * m][:ny] for k in range(2 * m + 1)))

    def _face_y(self, J):
        m, ny, nx = (self.ma, self.nya, self.nxa)
        Jf = J[2 * m:2 * m * ny:2 * m]
        if self.quad != 'simpson':
            return Jf[:, 1::2].reshape(ny - 1, nx, m).mean(2)
        return sum((self._w[k] * Jf[:, k:k + 2 * m * nx:2 * m][:, :nx] for k in range(2 * m + 1)))

    def in_initial(self, t):
        h = min(self.hxa, self.hya)
        return math.sqrt(2 * t * self.S) < self.c_init * h

    def initial_coeffs(self, t):
        P, m = (self.P, self.ma)
        ny, nx = (self.nya, self.nxa)
        lp, gx, gy = self.phi_fields(t)
        vx = P.gamma * gx[1::2, 2 * m:2 * m * nx:2 * m].reshape(ny, m, nx - 1).mean(1)
        vy = P.gamma * gy[2 * m:2 * m * ny:2 * m, 1::2].reshape(ny - 1, nx, m).mean(2)
        return (vx, vy, P.gamma / 2 / math.sqrt(4 * math.pi * t * self.S))

    def initial_substeps(self, M0, vx, vy, dif, dt, cfl=0.25, nmax=4000):
        ny, nx = (self.nya, self.nxa)
        A = self.hxa * self.hya
        Lx, Ly = (self.hya, self.hxa)
        rate = ((jnp.concatenate([jnp.abs(vx), jnp.zeros((ny, 1))], 1) + jnp.concatenate([jnp.zeros((ny, 1)), jnp.abs(vx)], 1)) * Lx + (jnp.concatenate([jnp.abs(vy), jnp.zeros((1, nx))], 0) + jnp.concatenate([jnp.zeros((1, nx)), jnp.abs(vy)], 0)) * Ly) / A
        n = int(min(nmax, max(1, math.ceil(dt * float(jnp.max(rate)) / cfl))))
        h = dt / n
        r0 = M0 / A
        oX = Lx * dif * (r0[:, :-1] - r0[:, 1:])
        oY = Ly * dif * (r0[:-1] - r0[1:])
        M = M0
        FXs = 0.0
        FYs = 0.0
        for _ in range(n):
            r = M / A
            FX = Lx * vx * jnp.where(vx > 0, r[:, :-1], r[:, 1:]) + oX
            FY = Ly * vy * jnp.where(vy > 0, r[:-1], r[1:]) + oY
            FX, FY = _limit_plane(M, FX, FY, h)
            M = M - h * self.Ga.div(FX[None], FY[None])[0]
            FXs = FXs + FX / n
            FYs = FYs + FY / n
        self.stats['substeps'] = self.stats.get('substeps', 0) + n
        return (M, FXs, FYs)

    def mass_flux(self, t, need_mass=True):
        P, m = (self.P, self.ma)
        e2 = P.gamma / 2
        ny, nx = (self.nya, self.nxa)
        hx, hy = (self.hxa, self.hya)
        if self.in_initial(t):
            self.stats['initial_evals'] = self.stats.get('initial_evals', 0) + 1
            lp, gx, gy = self.phi_fields(t)
            r0 = jnp.asarray(self.rho0a)
            jx = gx[1::2, 2 * m:2 * m * nx:2 * m].reshape(ny, m, nx - 1).mean(1)
            jy = gy[2 * m:2 * m * ny:2 * m, 1::2].reshape(ny - 1, nx, m).mean(2)
            dif = e2 / math.sqrt(4 * math.pi * t * self.S)
            FX = hy * (2 * e2 * jnp.where(jx > 0, r0[:, :-1], r0[:, 1:]) * jx + dif * (r0[:, :-1] - r0[:, 1:]))
            FY = hx * (2 * e2 * jnp.where(jy > 0, r0[:-1], r0[1:]) * jy + dif * (r0[:-1] - r0[1:]))
            return (r0 * (hx * hy) if need_mass else None, FX, FY)
        lh, gxh, gyh, lp, gx, gy = self.fields(t)
        rho = jnp.exp(lh + lp)
        Jx = e2 * rho * (gx - gxh)
        Jy = e2 * rho * (gy - gyh)
        FX = hy * self._face_x(Jx)
        FY = hx * self._face_y(Jy)
        M = hx * hy * self._cells(rho) if need_mass else None
        return (M, FX, FY)

    def phi_fields(self, t):
        P = self.P
        taup = (1 - t) * self.S
        if self.fine and math.sqrt(2 * taup) < self.c_switch * max(P.HX, P.HY):
            cx, cy = self._conv('phif', taup)
            lp, Rp = sep2d(cx, cy, self.wfA, s=taup)
        else:
            cx, cy = self._conv('phi', taup)
            lp, Rp = sep2d(cx, cy, self.wW, s=taup)
        return (lp, Rp[..., 0, 1], Rp[..., 1, 0])

def _limit_plane(M0, FX, FY, dt, frac=0.5):
    zx = jnp.zeros_like(FX[:, :1])
    zy = jnp.zeros_like(FY[:1])
    out = (jnp.concatenate([jnp.maximum(FX, 0), zx], 1) + jnp.concatenate([zx, jnp.maximum(-FX, 0)], 1) + jnp.concatenate([jnp.maximum(FY, 0), zy], 0) + jnp.concatenate([zy, jnp.maximum(-FY, 0)], 0)) * dt
    f = jnp.minimum(1.0, frac * M0 / jnp.maximum(out, 1e-300))
    FX = jnp.where(FX > 0, FX * f[:, :-1], FX * f[:, 1:])
    FY = jnp.where(FY > 0, FY * f[:-1], FY * f[1:])
    return (FX, FY)

def bridge_slabs_ld(G, rho0, gamma, t, P=None, reverse=False, m=1, stats=None, path_kw=None, refine=1, **chain_kw):
    if reverse:
        fwd = [tuple((np.asarray(x) if hasattr(x, 'shape') else x for x in sl)) for sl in bridge_slabs_ld(G, rho0, gamma, t, P=P, m=m, stats=stats, path_kw=path_kw, refine=refine, **chain_kw)]
        for M0, M1, FX, FY, t0, t1 in fwd[::-1]:
            yield (jnp.asarray(M1), jnp.asarray(M0), -jnp.asarray(FX), -jnp.asarray(FY), -t1, -t0)
        return
    if P is None:
        gamma_list = chain_kw.pop('gamma_list', None) or default_chain(gamma)
        P = solve_chain(G, rho0, gamma_list, verbose=chain_kw.pop('verbose', False), **chain_kw)[-1]
    path_kw = dict(path_kw or {})
    nGL = path_kw.pop('nGL', 4)
    path = PlanePathLD(P, m=m, refine=refine, **path_kw)
    G = path.Ga
    A = G.area
    M_prev = jnp.asarray(path.rho0a) * A
    GLx, GLw = np.polynomial.legendre.leggauss(nGL)
    w = GLw / 2
    K = len(t) - 1
    res = []
    mass_def = []
    min_ratio = np.inf
    total = jnp.sum(M_prev)
    tic = time.time()
    for n in range(K):
        t0, t1 = (float(t[n]), float(t[n + 1]))
        dt = t1 - t0
        FX = 0.0
        FY = 0.0
        for x, wq in zip(GLx, w) if not (n < K - 1 and path.in_initial(t1)) else ():
            _, fx, fy = path.mass_flux(t0 + (x + 1) / 2 * dt, need_mass=False)
            FX = FX + wq * fx
            FY = FY + wq * fy
        if n < K - 1 and path.in_initial(t1):
            vx = 0.0
            vy = 0.0
            dif = 0.0
            for x, wq in zip(GLx, w):
                a_, b_, c_ = path.initial_coeffs(t0 + (x + 1) / 2 * dt)
                vx = vx + wq * a_
                vy = vy + wq * b_
                dif = dif + wq * c_
            M1, FX, FY = path.initial_substeps(M_prev, vx, vy, dif, dt)
        else:
            M1 = path.mass_flux(t1)[0] if n < K - 1 else A * 1.0
        if n < K - 1:
            mass_def.append(float(jnp.sum(M1) / total - 1))
            M1 = M1 * (total / jnp.sum(M1))
        min_ratio = min(min_ratio, float(jnp.min(M1 / A)))
        FXb, FYb = (FX[None], FY[None])
        r = -(M1 - M_prev)[None] / dt - G.div(FXb, FYb)
        res.append(float(jnp.max(jnp.abs(r)) / (jnp.max(jnp.abs(G.div(FXb, FYb))) + 1e-300)))
        FXb, FYb = engine._fix(G, M_prev, M1, FXb, FYb, dt)
        if getattr(path, 'verbose', False) and (n % 8 == 0 or n == K - 1):
            print(f'    slab {n}/{K} t={t1:.4g} res {res[-1]:.2e} {time.time() - tic:.1f}s', flush=True)
        yield (M_prev, M1, FXb[0], FYb[0], t0, t1)
        M_prev = M1
    if stats is not None:
        stats.update(path.stats)
        stats['pre_fix_residual'] = res
        stats['min_mass_ratio'] = min_ratio
        stats['mass_defect_max'] = float(np.max(np.abs(mass_def))) if mass_def else 0.0
        stats['t_path'] = time.time() - tic
        stats['P_info'] = {k: v for k, v in P.info.items() if k != 'hist'}

def default_chain(gamma, start=0.1, factor=3.0):
    out = [gamma]
    while out[-1] * factor < start * 1.0001:
        out.append(out[-1] * factor)
    if out[-1] < start * 0.9999 and gamma < start:
        out.append(start)
    return out[::-1]

def nodes(gamma, h, K=48):
    from .nodes import bridge_nodes
    return bridge_nodes(gamma, h, K, 0.25)

def solve(G, rho0, gamma, warm=None, verbose=False, **kw):
    if warm is None:
        return solve_chain(G, rho0, default_chain(gamma), verbose=verbose, **kw)[-1]
    P = PlaneLD(G, rho0, gamma, window=warm.displacement_window(), **{k: v for k, v in kw.items() if k in ('source', 'c_target', 'r')})
    P.solve_newton(warm.w_on(P), verbose=verbose, **{k: v for k, v in kw.items() if k in ('tol', 'max_newton', 'cg_tol', 'cg_max', 'c_pre')})
    return P
