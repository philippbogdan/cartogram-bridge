"""Density paths whose cell masses and face fluxes satisfy discrete continuity."""
import numpy as np
GL_X, GL_W = np.polynomial.legendre.leggauss(6)

def normalise(grid, mass):
    rho = mass / grid.area
    return rho / (np.sum(mass) / grid.total_area)

def sinkhorn(grid, rho0, gamma, tol=1e-13, maxit=100000, b0=None, verbose=False):
    s = gamma / 2
    b = np.ones_like(rho0) if b0 is None else b0.copy()
    for it in range(maxit):
        a = rho0 / grid.heat(b, s)
        Ka = grid.heat(a, s)
        b = 1.0 / Ka
        err = np.max(np.abs(a * grid.heat(b, s) / rho0 - 1))
        if verbose and it % 50 == 0:
            print(it, err)
        if err < tol:
            break
    return (a, b, dict(iters=it + 1, err=err))

def poisson_fix(grid, FX, FY, resid):
    rhs = resid / grid.area
    if grid.kind == 'plane':
        c = grid.to_spec(rhs)
        lam = grid.lam.copy()
        lam[0, 0] = 1.0
        c = c / -lam
        c[..., 0, 0] = 0.0
        chi = grid.from_spec(c)
    else:
        e = grid.eig()
        ce, co = grid.to_spec(rhs)
        Le = e['Le'].copy()
        k0 = int(np.argmax(Le[0]))
        Le[0, k0] = 1.0
        ce = ce / -Le
        ce[..., 0, k0] = 0.0
        co = co / -e['Lo']
        chi = grid.from_spec(ce, co)
    gx, gy = grid.grad_flux(chi)
    return (FX + gx, FY + gy)

class Path:

    def __init__(self, grid, t, M, FX, FY, meta):
        self.grid, self.t, self.M, self.FX, self.FY, self.meta = (grid, t, M, FX, FY, meta)

    def continuity_residual(self):
        g = self.grid
        r = 0.0
        for n in range(len(self.t) - 1):
            d = self.M[n + 1] - self.M[n] + (self.t[n + 1] - self.t[n]) * g.div(self.FX[n], self.FY[n])
            r = max(r, np.max(np.abs(d) / self.M[n + 1]))
        return r

def _assemble(grid, t, dens_at, flux_at, meta):
    A = grid.area
    M = [dens_at(tk) * A for tk in t]
    FXs, FYs = ([], [])
    for n in range(len(t) - 1):
        t0, t1 = (t[n], t[n + 1])
        dt = t1 - t0
        FX = 0.0
        FY = 0.0
        for x, w in zip(GL_X, GL_W):
            fx, fy = flux_at(t0 + (x + 1) / 2 * dt)
            FX = FX + w / 2 * fx
            FY = FY + w / 2 * fy
        resid = -(M[n + 1] - M[n]) / dt - grid.div(FX, FY)
        FX, FY = poisson_fix(grid, FX, FY, resid)
        FXs.append(FX)
        FYs.append(FY)
    return Path(grid, np.asarray(t), M, FXs, FYs, meta)

def bridge_path(grid, rho0, gamma, t, a=None, b=None, **kw):
    info = {}
    if a is None:
        a, b, info = sinkhorn(grid, rho0, gamma, **kw)
    s = gamma / 2

    def fields(tk):
        return (grid.heat(a, tk * s), grid.heat(b, (1 - tk) * s))

    def dens_at(tk):
        if tk == 0:
            return rho0
        if tk == 1:
            return np.ones_like(rho0)
        at, bt = fields(tk)
        return at * bt

    def flux_at(tk):
        at, bt = fields(tk)
        fx, fy = grid.bridge_flux(at, bt)
        return (s * fx, s * fy)
    return (_assemble(grid, t, dens_at, flux_at, dict(kind='bridge', gamma=gamma, **info)), (a, b))

def diffusion_path(grid, rho0, s_nodes):

    def dens_at(sk):
        if np.isinf(sk):
            return np.ones_like(rho0)
        return grid.heat(rho0, sk)
    finite = np.asarray(s_nodes, float)
    if np.isinf(finite[-1]):
        pass

    def flux_at(sk):
        return grid.grad_flux(grid.heat(rho0, sk))
    if not np.isinf(finite[-1]):
        return _assemble(grid, finite, dens_at, flux_at, dict(kind='diffusion'))
    p = _assemble(grid, finite[:-1], dens_at, flux_at, dict(kind='diffusion'))
    sl = finite[-2]
    rho_l = grid.heat(rho0, sl)
    r = (rho_l - 1.0) * grid.area
    FX0 = np.zeros_like(p.FX[0])
    FY0 = np.zeros_like(p.FY[0])
    FX, FY = poisson_fix(grid, FX0, FY0, r)
    p.t = np.append(p.t, p.t[-1] + 1.0)
    p.M.append(grid.area * 1.0)
    p.FX.append(FX)
    p.FY.append(FY)
    p.meta['tail_from'] = sl
    return p
