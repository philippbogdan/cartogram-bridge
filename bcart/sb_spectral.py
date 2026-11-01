"""Deviation-form bridge potentials and relative velocity errors for large gamma."""
import numpy as np

class SpectralBridge:

    def __init__(self, dom, rho0, gamma, tol=1e-16, maxit=5000, naive=False, proj=None):
        self.dom, self.gamma, self.T = (dom, gamma, gamma / 2.0)
        self.proj = proj
        mean = dom.integrate(rho0) / dom.area
        self.rho0 = rho0 / mean
        self.R = self._an(self.rho0 - 1.0)
        self.R[0, 0] = 0.0
        self.qT = dom.heat_factor(self.T)
        self.naive = naive
        self._solve(tol, maxit)

    def _an(self, f):
        C = self.dom.analysis(f)
        return C if self.proj is None else C * self.proj

    def _mean(self, f):
        return self.dom.integrate(f) / self.dom.area

    def _solve(self, tol, maxit):
        d = self.dom
        if self.naive:
            return self._solve_naive(tol, maxit)
        A = self.R.copy()
        hist = []
        best = None
        for it in range(maxit):
            A, B, mw, k1 = self._sweep(A)
            if it == 0:
                Aprev = self.R
            du = np.max(np.abs((A - Aprev) * self.qT)) / np.max(np.abs(A * self.qT))
            dA = np.max(np.abs(A - Aprev)) / np.max(np.abs(A))
            Aprev = A
            hist.append(max(du, dA))
            if best is None or hist[-1] < best[0]:
                best = (hist[-1], A.copy(), B.copy(), 1.0 - mw, k1)
            if hist[-1] <= tol or (it > 20 and min(hist[-10:]) >= min(hist[:-10])):
                break
        _, self.A, self.B, self.kappa2, self.kappa1 = best
        self.iters, self.hist = (it + 1, hist)
        self.conv = best[0]
        self.B, _ = self._beta(self.A)
        self.residuals()

    def _beta(self, A):
        d = self.dom
        Cu = A * self.qT
        u = d.synthesis(Cu)
        rem = u * u / (1.0 + u)
        mw = -self._mean(rem)
        B = (-Cu + self._an(rem)) / (1.0 - mw)
        B[0, 0] = 0.0
        return (B, mw)

    def _sweep(self, A):
        d = self.dom
        B, mw = self._beta(A)
        v = d.synthesis(B * self.qT)
        h = self.rho0 * (v / (1.0 + v))
        k1 = 1.0 - self._mean(h)
        An = (self.R - self._an(h)) / k1
        An[0, 0] = 0.0
        return (An, B, mw, k1)

    def _solve_naive(self, tol, maxit):
        d = self.dom
        ph0 = self.rho0.copy()
        for it in range(maxit):
            phi1 = 1.0 / d.synthesis(self._an(ph0) * self.qT)
            ph0n = self.rho0 / d.synthesis(self._an(phi1) * self.qT)
            ch = np.max(np.abs(ph0n - ph0)) / np.max(np.abs(ph0n))
            ph0 = ph0n
            if ch <= tol or it > 400:
                break
        phi1 = 1.0 / d.synthesis(self._an(ph0) * self.qT)
        a, c = (self._mean(ph0), self._mean(phi1))
        self.A = self._an(ph0 / a - 1.0)
        self.A[0, 0] = 0.0
        self.B = self._an(phi1 / c - 1.0)
        self.B[0, 0] = 0.0
        self.iters, self.hist, self.conv = (it + 1, [], ch)
        self.residuals()

    def residuals(self):
        d = self.dom
        u = d.synthesis(self.A * self.qT)
        v = d.synthesis(self.B * self.qT)
        al = d.synthesis(self.A)
        be = d.synthesis(self.B)
        mw = self._mean(u / (1 + u))
        k = 1.0 - mw
        r2 = k * (be + u + be * u) - mw
        r1 = k * (1 + al) * (1 + v) / self.rho0 - 1.0
        self.res = (float(np.max(np.abs(r1))), float(np.max(np.abs(r2))))
        self.res2_rel = float(np.max(np.abs(r2)) / max(np.max(np.abs(u)), 1e-300))
        return self.res

    def V(self, s):
        d = self.dom
        cy = self.A * d.heat_factor(s)
        cx = self.B * d.heat_factor(self.T - s)
        y, x = (d.synthesis(cy), d.synthesis(cx))
        gy, gx = (d.grad(cy), d.grad(cx))
        return tuple((gxi / (1 + x) - gyi / (1 + y) for gxi, gyi in zip(gx, gy)))

    def U(self, heatfac):
        d = self.dom
        cz = self.R * heatfac
        z = d.synthesis(cz)
        g = d.grad(cz)
        return (tuple((-gi / (1 + z) for gi in g)), 1 + z)

def time_change(kind, s, gamma, l1, psi0=0.0):
    if kind == 'linear':
        return (-l1 * s, 1.0)
    a = l1 * (gamma - 2 * s)
    b = l1 * (gamma - s)
    core = -np.expm1(-a)
    if kind == 'canonical':
        g = core + np.exp(-b)
        dnum = 1 + np.exp(-a)
    elif kind == 'densmatch':
        g = core
        dnum = 1 + np.exp(-a)
    elif kind == 'balanced':
        q = np.exp(-l1 * gamma / 2)
        extra = psi0 * q * -np.expm1(-l1 * s) / (1 - q) * np.exp(l1 * s)
        g = core + np.exp(-b) + extra
        dnum = 1 + np.exp(-a) - psi0 * q / (1 - q)
    else:
        raise ValueError(kind)
    logf = -l1 * s + np.log(g) if g > 0 else -np.inf
    return (logf, dnum / g if g > 0 else np.inf)

def heatfac_from_logf(dom, logf, l1):
    if not np.isfinite(logf):
        return np.zeros_like(dom.lam)
    lam = dom.lam
    hf = np.exp(lam / l1 * logf)
    if dom.kind == 'sphere':
        hf = np.where(dom.mask, hf, 0.0)
    return hf

def _vnorm2(dom, F, w=None):
    q = sum((fi * fi for fi in F))
    return dom.integrate(q if w is None else w * q)

def error_at(br, s, kind, l1=None, psi0=0.0):
    dom = br.dom
    l1 = dom.l1 if l1 is None else l1
    logf, dsig = time_change(kind, s, br.gamma, l1, psi0)
    sigma = -logf / l1
    V = br.V(s)
    U, rhoH = br.U(heatfac_from_logf(dom, logf, l1))
    nU = _vnorm2(dom, U)
    if nU == 0.0:
        return dict(s=s, sigma=sigma, dsigma=dsig, E_area=np.nan, E_rho=np.nan)
    D = tuple((vi / dsig - ui for vi, ui in zip(V, U)))
    return dict(s=float(s), sigma=float(sigma), dsigma=float(dsig), E_area=float(np.sqrt(_vnorm2(dom, D) / nU)), E_rho=float(np.sqrt(_vnorm2(dom, D, rhoH) / _vnorm2(dom, U, rhoH))))

def free_error(br, s, l1=None):
    from scipy.optimize import minimize_scalar
    dom = br.dom
    l1 = dom.l1 if l1 is None else l1
    V = br.V(s)

    def err(logf):
        U, _ = br.U(heatfac_from_logf(dom, logf, l1))
        nU = _vnorm2(dom, U)
        vu = dom.integrate(sum((a * b for a, b in zip(V, U))))
        vv = _vnorm2(dom, V)
        c = vu / vv
        return np.sqrt(max(_vnorm2(dom, tuple((c * a - b for a, b in zip(V, U)))), 0.0) / nU)
    T = br.T
    lo, hi = (-l1 * 4 * T - 5, 0.0)
    grid = np.linspace(lo, hi, 81)
    vals = [err(g) for g in grid]
    i = int(np.argmin(vals))
    r = minimize_scalar(err, bracket=None, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 80)]), method='bounded', options=dict(xatol=1e-12))
    return (float(min(r.fun, vals[i])), float(-r.x / l1))

def first_mode_G(dom, R, l1):
    m = np.isclose(dom.lam, l1)
    C = np.where(m, R, 0.0)
    ell = dom.synthesis(C)
    gl = dom.grad(C)
    g2 = tuple((2 * ell * gi for gi in gl))
    den = _vnorm2(dom, gl)
    if den == 0.0:
        return (0.0, C)
    return (float(np.sqrt(_vnorm2(dom, g2) / den)), C)

def predicted_E(dom, Cl, l1, tau, c):
    ell = dom.synthesis(Cl)
    C2 = dom.analysis(ell * ell)
    gl = dom.grad(Cl)
    C = C2 * dom.heat_factor(tau) / np.cosh(l1 * tau) - c * C2
    return 0.5 * float(np.sqrt(_vnorm2(dom, dom.grad(C)) / _vnorm2(dom, gl)))

def _unit(v):
    n = sum((t * t for t in v)) ** 0.5
    return tuple((t / n for t in v))
N1 = _unit((0.6, -0.3, 0.74))
N2 = _unit((-0.5, 0.8, -0.2))

def density(name, X, Y, Z=None, exp=np.exp, cos=np.cos):
    if name == 'sph_generic':
        d1 = N1[0] * X + N1[1] * Y + N1[2] * Z
        d2 = N2[0] * X + N2[1] * Y + N2[2] * Z
        return exp(2 * d1) + 0.6 * exp(3 * d2) + 0.2
    if name == 'sph_zonal':
        return exp(2.5 * Z) + 0.3 * exp(-1.5 * Z) + 0.2
    if name == 'pl_generic':
        return exp(cos(X / 2) + 0.5 * cos(X) * cos(Y) + 0.7 * cos(2 * Y) + 0.4 * cos(Y))
    if name == 'pl_nogen':
        return exp(cos(X) + 0.7 * cos(2 * Y) + 0.5 * cos(X) * cos(Y))
    raise ValueError(name)

def make_density(dom, name):
    if dom.kind == 'sphere':
        X, Y, Z = dom.cart()
        return density(name, X, Y, Z)
    X, Y = dom.mesh()
    return density(name, X, Y)

def errors_multi(br, s, specs):
    dom = br.dom
    V = br.V(s)
    out = {}
    for lab, (kind, l1, psi0) in specs.items():
        logf, dsig = time_change(kind, s, br.gamma, l1, psi0)
        if not np.isfinite(logf):
            out[lab] = (np.nan, np.nan, np.inf, np.inf)
            continue
        U, rhoH = br.U(heatfac_from_logf(dom, logf, l1))
        D = tuple((vi / dsig - ui for vi, ui in zip(V, U)))
        nU, nUr = (_vnorm2(dom, U), _vnorm2(dom, U, rhoH))
        out[lab] = (float(np.sqrt(_vnorm2(dom, D) / nU)), float(np.sqrt(_vnorm2(dom, D, rhoH) / nUr)), float(-logf / l1), float(dsig))
    return out

def s_grid(T, n_end=60, n_mid=40, tmin=1e-06):
    tau = np.geomspace(tmin, T / 2, n_end)
    s = np.concatenate([[0.0, T], tau, T - tau, np.linspace(0, T, n_mid + 2)[1:-1]])
    return np.unique(np.clip(s, 0, T))
