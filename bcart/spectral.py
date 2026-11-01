"""Continuum heat semigroups on the reflecting rectangle and unit sphere."""
import numpy as np
import scipy.fft as sfft
import scipy.special as ssp

def gauss_legendre(n):
    x, _ = ssp.roots_legendre(n)
    x = 0.5 * (x - x[::-1])

    def pd(x):
        p0, p1 = (np.ones_like(x), x.copy())
        for k in range(2, n + 1):
            p0, p1 = (p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k)
        return (p1, n * (x * p1 - p0) / (x * x - 1))
    for _ in range(3):
        p, dp = pd(x)
        x = x - p / dp
    x = 0.5 * (x - x[::-1])
    _, dp = pd(x)
    w = 2.0 / ((1 - x * x) * dp * dp)
    return (x, 0.5 * (w + w[::-1]))

class SphereSpec:
    kind = 'sphere'

    def __init__(self, L, nlat=None, nlon=None):
        self.L = L
        self.nlat = nlat if nlat is not None else 2 * L + 2
        self.nlon = nlon if nlon is not None else 4 * L + 4
        assert self.nlon >= 2 * L + 2 and self.nlat >= L + 1
        x, w = gauss_legendre(self.nlat)
        self.x, self.w = (x, w)
        self.sint = np.sqrt((1.0 - x) * (1.0 + x))
        self.theta = np.arccos(x)
        self.phi = 2 * np.pi * np.arange(self.nlon) / self.nlon
        self.qw = np.outer(w, np.full(self.nlon, 2 * np.pi / self.nlon))
        self.area = 4 * np.pi
        self.l1, self.l2 = (2.0, 6.0)
        ls = np.arange(L + 1)
        ms = np.arange(L + 1)
        self.mask = ls[:, None] >= ms[None, :]
        self.lam = np.where(self.mask, (ls * (ls + 1.0))[:, None], 0.0)
        self._tables()

    def _tables(self):
        L, x, st = (self.L, self.x, self.sint)
        P = np.zeros((L + 1, L + 1, self.nlat))
        D = np.zeros_like(P)
        pmm = np.full(self.nlat, 1.0 / np.sqrt(4 * np.pi))
        dmm = np.zeros(self.nlat)
        for m in range(L + 1):
            if m > 0:
                pmm = np.sqrt((2 * m + 1) / (2.0 * m)) * st * pmm
                dmm = m * (x / st) * pmm
            P[m, m], D[m, m] = (pmm, dmm)
            if m + 1 <= L:
                a = np.sqrt(2 * m + 3.0)
                P[m, m + 1] = a * x * pmm
                D[m, m + 1] = a * (-st * pmm + x * dmm)
            for l in range(m + 2, L + 1):
                a = np.sqrt((4.0 * l * l - 1) / (l * l - m * m))
                ap = np.sqrt((4.0 * (l - 1) ** 2 - 1) / ((l - 1) ** 2 - m * m))
                P[m, l] = a * (x * P[m, l - 1] - P[m, l - 2] / ap)
                D[m, l] = a * (-st * P[m, l - 1] + x * D[m, l - 1] - D[m, l - 2] / ap)
        fac = np.where(np.arange(L + 1) == 0, 1.0, np.sqrt(2.0))[:, None, None]
        self.P = P * fac
        self.D = D * fac
        self.Q = self.P / st
        self.Pw = self.P * self.w

    def analysis(self, f):
        F = sfft.rfft(f, axis=1)[:, :self.L + 1] * (2 * np.pi / self.nlon)
        C = np.einsum('mlj,jm->lm', self.Pw, F)
        return np.where(self.mask, C, 0.0)

    def _synth(self, C, T):
        g = np.einsum('lm,mlj->jm', C, T)
        n = self.nlon
        c = np.zeros((self.nlat, n // 2 + 1), dtype=complex)
        c[:, 0] = n * g[:, 0].real
        c[:, 1:self.L + 1] = 0.5 * n * g[:, 1:]
        return sfft.irfft(c, n=n, axis=1)

    def synthesis(self, C):
        return self._synth(C, self.P)

    def grad(self, C):
        m = np.arange(self.L + 1)[None, :]
        return (self._synth(C, self.D), self._synth(1j * m * C, self.Q))

    def heat_factor(self, s):
        return np.where(self.mask, np.exp(-s * self.lam), 0.0)

    def integrate(self, f):
        return np.sum(self.qw * f)

    def mean_coef(self, C):
        return C[0, 0].real / np.sqrt(4 * np.pi)

    def unit_coef(self):
        C = np.zeros((self.L + 1, self.L + 1), complex)
        C[0, 0] = np.sqrt(4 * np.pi)
        return C

    def norm2(self, C):
        return np.sum(np.abs(C) ** 2)

    def gradnorm2(self, C):
        return np.sum(self.lam * np.abs(C) ** 2)

    def tail(self, C):
        return np.max(np.abs(C), axis=1)

    def cart(self):
        st, ct = (self.sint[:, None], self.x[:, None])
        cp, sp = (np.cos(self.phi)[None, :], np.sin(self.phi)[None, :])
        return (st * cp, st * sp, ct * np.ones_like(cp))

class PlaneSpec:
    kind = 'plane'

    def __init__(self, Nx, Ny, Lx=2 * np.pi, Ly=np.pi):
        self.Nx, self.Ny, self.Lx, self.Ly = (Nx, Ny, Lx, Ly)
        self.xc = (np.arange(Nx) + 0.5) * Lx / Nx
        self.yc = (np.arange(Ny) + 0.5) * Ly / Ny
        self.kx = np.pi * np.arange(Nx) / Lx
        self.ky = np.pi * np.arange(Ny) / Ly
        self.lam = self.kx[:, None] ** 2 + self.ky[None, :] ** 2
        ev = sorted(set(np.round(self.lam.ravel(), 12)))
        self.l1, self.l2 = (ev[1], ev[2])
        self.area = Lx * Ly
        self.cell = Lx / Nx * (Ly / Ny)
        ex = np.where(np.arange(Nx) == 0, 1.0, 0.5)
        ey = np.where(np.arange(Ny) == 0, 1.0, 0.5)
        self.nw = Lx * Ly * ex[:, None] * ey[None, :]
        self.Cx = np.cos(np.outer(self.xc, self.kx))
        self.Sx = np.sin(np.outer(self.xc, self.kx))
        self.Cy = np.cos(np.outer(self.yc, self.ky))
        self.Sy = np.sin(np.outer(self.yc, self.ky))

    def analysis(self, f):
        X = sfft.dctn(f, type=2)
        c = X / (self.Nx * self.Ny)
        c[0, :] *= 0.5
        c[:, 0] *= 0.5
        return c

    def synthesis(self, c):
        y = c * 0.25
        y[0, :] *= 2.0
        y[:, 0] *= 2.0
        return sfft.dctn(y, type=3)

    def grad(self, c):
        gx = -self.Sx @ (self.kx[:, None] * c) @ self.Cy.T
        gy = -self.Cx @ (c * self.ky[None, :]) @ self.Sy.T
        return (gx, gy)

    def heat_factor(self, s):
        return np.exp(-s * self.lam)

    def integrate(self, f):
        return np.sum(f) * self.cell

    def mean_coef(self, c):
        return c[0, 0]

    def unit_coef(self):
        c = np.zeros((self.Nx, self.Ny))
        c[0, 0] = 1.0
        return c

    def norm2(self, c):
        return np.sum(self.nw * c ** 2)

    def gradnorm2(self, c):
        return np.sum(self.nw * self.lam * c ** 2)

    def tail(self, c):
        K = np.maximum(np.arange(self.Nx)[:, None] / self.Nx, np.arange(self.Ny)[None, :] / self.Ny)
        idx = np.floor(K * self.Nx).astype(int)
        out = np.zeros(self.Nx)
        np.maximum.at(out, idx.ravel(), np.abs(c).ravel())
        return out

    def mesh(self):
        return np.meshgrid(self.xc, self.yc, indexing='ij')
