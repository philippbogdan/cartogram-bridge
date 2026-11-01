"""Compare float64 velocity errors with independent multiprecision Sinkhorn."""
import argparse, json, os, sys, time
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bcart import sb_spectral as sb
from bcart.spectral import SphereSpec, PlaneSpec

def odot(A, B):
    return np.dot(A, B)

class MPSphereZonal:
    l1 = 2

    def __init__(self, L, n):
        self.L, self.n = (L, n)
        x0, _ = np.polynomial.legendre.leggauss(n)
        xs = []
        for xi in x0:
            x = mp.mpf(xi)
            for _ in range(8):
                p, dp = self._pdp(x, n)
                x = x - p / dp
            xs.append(x)
        self.x = xs
        self.w = [2 / ((1 - x * x) * self._pdp(x, n)[1] ** 2) for x in xs]
        self.st = [mp.sqrt(1 - x * x) for x in xs]
        P = np.empty((L + 1, n), dtype=object)
        D = np.empty((L + 1, n), dtype=object)
        for j, x in enumerate(xs):
            p0, p1 = (mp.mpf(1), x)
            vals, ders = ([p0, p1], [mp.mpf(0), mp.mpf(1)])
            for l in range(2, L + 1):
                p0, p1 = (p1, ((2 * l - 1) * x * p1 - (l - 1) * p0) / l)
                vals.append(p1)
            for l in range(2, L + 1):
                ders.append(l * (x * vals[l] - vals[l - 1]) / (x * x - 1))
            for l in range(L + 1):
                c = mp.sqrt((2 * l + 1) / (4 * mp.pi))
                P[l, j], D[l, j] = (c * vals[l], c * ders[l])
        self.P, self.D = (P, D)
        self.lam = [l * (l + 1) for l in range(L + 1)]
        self.W = np.array([2 * mp.pi * w for w in self.w], dtype=object)

    @staticmethod
    def _pdp(x, n):
        p0, p1 = (mp.mpf(1), x)
        for k in range(2, n + 1):
            p0, p1 = (p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k)
        return (p1, n * (x * p1 - p0) / (x * x - 1))

    def grid(self, name):
        return np.array([sb.density(name, 0, 0, x, exp=mp.exp) for x in self.x], dtype=object)

    def analysis(self, f):
        return odot(self.P, f * self.W)

    def heat(self, c, s):
        return np.array([c[l] * mp.exp(-self.lam[l] * s) for l in range(self.L + 1)], dtype=object)

    def val_grad(self, c):
        v = odot(c, self.P)
        d = odot(c, self.D)
        return (v, [-self.st[j] * d[j] for j in range(self.n)])

    def integ(self, f):
        return mp.fsum(f * self.W)

class MPPlane:
    l1 = mp.mpf(1) / 4

    def __init__(self, Nx, Ny):
        self.Nx, self.Ny = (Nx, Ny)
        Lx, Ly = (2 * mp.pi, mp.pi)
        self.xc = [(i + mp.mpf(1) / 2) * Lx / Nx for i in range(Nx)]
        self.yc = [(j + mp.mpf(1) / 2) * Ly / Ny for j in range(Ny)]
        self.kx = [mp.pi * k / Lx for k in range(Nx)]
        self.ky = [mp.pi * l / Ly for l in range(Ny)]
        M = lambda pts, ks, fn: np.array([[fn(k * p) for k in ks] for p in pts], dtype=object)
        self.Cx, self.Sx = (M(self.xc, self.kx, mp.cos), M(self.xc, self.kx, mp.sin))
        self.Cy, self.Sy = (M(self.yc, self.ky, mp.cos), M(self.yc, self.ky, mp.sin))
        self.lam = np.array([[a * a + b * b for b in self.ky] for a in self.kx], dtype=object)
        self.cell = Lx * Ly / (Nx * Ny)
        ex = np.array([mp.mpf(1) / Nx] + [mp.mpf(2) / Nx] * (Nx - 1), dtype=object)
        ey = np.array([mp.mpf(1) / Ny] + [mp.mpf(2) / Ny] * (Ny - 1), dtype=object)
        self.ex, self.ey = (ex, ey)
        self.KX = np.array(self.kx, dtype=object)[:, None]
        self.KY = np.array(self.ky, dtype=object)[None, :]

    def grid(self, name):
        return np.array([[sb.density(name, x, y, exp=mp.exp, cos=mp.cos) for y in self.yc] for x in self.xc], dtype=object)

    def analysis(self, f):
        return odot(odot(self.Cx.T, f), self.Cy) * self.ex[:, None] * self.ey[None, :]

    def heat(self, c, s):
        return c * np.vectorize(lambda l: mp.exp(-l * s), otypes=[object])(self.lam)

    def val_grad(self, c):
        v = odot(odot(self.Cx, c), self.Cy.T)
        gx = -odot(odot(self.Sx, c * self.KX), self.Cy.T)
        gy = -odot(odot(self.Cx, c * self.KY), self.Sy.T)
        return (v, (gx, gy))

    def integ(self, f):
        return mp.fsum(f.ravel()) * self.cell

def mp_errors(D, name, gamma, svals, kind_list):
    gamma = mp.mpf(gamma)
    T = gamma / 2
    rho = D.grid(name)
    rho = rho / (D.integ(rho) / D.integ(rho * 0 + 1))
    R = D.analysis(rho)
    ph0 = rho.copy()
    tol = mp.mpf(10) ** (-(mp.mp.dps - 8))
    for it in range(2000):
        phi1 = 1 / D.val_grad(D.heat(D.analysis(ph0), T))[0]
        new = rho / D.val_grad(D.heat(D.analysis(phi1), T))[0]
        ch = max((abs(a - b) for a, b in zip(np.ravel(new), np.ravel(ph0))))
        ph0 = new
        if ch < tol:
            break
    phi1 = 1 / D.val_grad(D.heat(D.analysis(ph0), T))[0]
    A, B = (D.analysis(ph0), D.analysis(phi1))
    l1 = D.l1
    out = []
    sphere = isinstance(D, MPSphereZonal)
    for s in svals:
        s = mp.mpf(s)
        ph, gph = D.val_grad(D.heat(A, s))
        p, gp = D.val_grad(D.heat(B, T - s))
        comps = [gp] if sphere else list(gp)
        hcomps = [gph] if sphere else list(gph)
        V = [np.array(g1, dtype=object) / p - np.array(g2, dtype=object) / ph for g1, g2 in zip(comps, hcomps)]
        res = {}
        for kind in kind_list:
            if kind == 'linear':
                sig, dsig = (s, mp.mpf(1))
            else:
                f = mp.exp(-l1 * s) - mp.exp(-l1 * (gamma - s)) + mp.exp(-l1 * gamma)
                sig, dsig = (-mp.log(f) / l1, (mp.exp(-l1 * s) + mp.exp(-l1 * (gamma - s))) / f)
            z, gz = D.val_grad(D.heat(R, sig))
            Ucomps = [gz] if sphere else list(gz)
            U = [-np.array(g, dtype=object) / z for g in Ucomps]
            Dif = [v / dsig - u for v, u in zip(V, U)]
            n2 = lambda F, w=None: D.integ(sum((f * f for f in F)) * (1 if w is None else w))
            res[kind] = (float(mp.sqrt(n2(Dif) / n2(U))), float(mp.sqrt(n2(Dif, z) / n2(U, z))))
        out.append((float(s), res))
    return (out, it + 1)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--domain', choices=['sphere', 'plane'], required=True)
    ap.add_argument('--density', required=True)
    ap.add_argument('--gamma', type=float, nargs='+', required=True)
    ap.add_argument('--Lmp', type=int, default=0)
    ap.add_argument('--L64', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time()
    if a.domain == 'sphere':
        Lmp = a.Lmp or 40
        dom64 = SphereSpec(a.L64 or 40)
    else:
        Lmp = a.Lmp or 32
        dom64 = PlaneSpec(2 * (a.L64 or 48), a.L64 or 48)
    rho64 = sb.make_density(dom64, a.density)
    results = []
    for gamma in a.gamma:
        l1 = dom64.l1
        mp.mp.dps = 30 + int(l1 * gamma / 2.3)
        D = MPSphereZonal(Lmp, Lmp + 16) if a.domain == 'sphere' else MPPlane(2 * Lmp, Lmp)
        T = gamma / 2
        svals = sorted({0.0, T / 2, max(T - 1 / l1, 0.0), T - 0.05 * T, T})
        ref, its = mp_errors(D, a.density, gamma, svals, ['linear', 'canonical'])
        br = sb.SpectralBridge(dom64, rho64, gamma)
        spec = {'linear': ('linear', l1, 0.0), 'canonical': ('canonical', l1, 0.0)}
        for s, r in ref:
            f64 = sb.errors_multi(br, s, spec)
            row = dict(gamma=gamma, s=s, tau=T - s, mp_dps=mp.mp.dps, mp_iters=its)
            for k in spec:
                row[k] = dict(mp_area=r[k][0], f64_area=f64[k][0], rel_area=abs(f64[k][0] - r[k][0]) / r[k][0], mp_rho=r[k][1], f64_rho=f64[k][1], rel_rho=abs(f64[k][1] - r[k][1]) / r[k][1])
            results.append(row)
            print(f"gamma={gamma:6.1f} s={s:8.4f} canon mp={r['canonical'][0]:.12e} f64={f64['canonical'][0]:.12e} rel={row['canonical']['rel_area']:.1e} | lin rel={row['linear']['rel_area']:.1e}", flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(dict(Lmp=Lmp, runtime_s=time.time() - t0, rows=results), f, indent=1)
    print('wrote', a.out)
if __name__ == '__main__':
    main()
