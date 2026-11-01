"""Area weights and finite-volume Laplacians on structured grids."""
import numpy as np
import scipy.fft as sfft
import scipy.linalg as sla

class SphereGrid:
    kind = 'sphere'

    def __init__(self, nlon, nlat):
        assert nlat % 2 == 0 and nlon % 2 == 0
        self.nlon, self.nlat = (nlon, nlat)
        self.dlon = 2 * np.pi / nlon
        self.dlat = np.pi / nlat
        j = np.arange(nlat + 1)
        self.lat_e = -np.pi / 2 + self.dlat * j
        self.z_e = np.sin(self.lat_e)
        self.z_e[0], self.z_e[-1] = (-1.0, 1.0)
        self.z_e[nlat // 2] = 0.0
        self.lat_c = 0.5 * (self.lat_e[1:] + self.lat_e[:-1])
        self.lon_e = -np.pi + self.dlon * np.arange(nlon + 1)
        self.lon_c = 0.5 * (self.lon_e[1:] + self.lon_e[:-1])
        self.dz = np.diff(self.z_e)
        self.area_row = self.dlon * self.dz
        self.area = np.repeat(self.area_row[:, None], nlon, axis=1)
        self.total_area = 4 * np.pi
        self.Tx_row = self.dlat / (np.cos(self.lat_c) * self.dlon)
        self.Ty_edge = np.cos(self.lat_e) * self.dlon / self.dlat
        self.Ty_edge[0] = self.Ty_edge[-1] = 0.0
        self.diameter = np.pi
        self._eig = None

    def eig(self):
        if self._eig is not None:
            return self._eig
        n, h = (self.nlat, self.nlat // 2)
        A = self.area_row
        ms = np.arange(self.nlon // 2 + 1)
        lam_x = 4 * np.sin(ms * self.dlon / 2) ** 2
        off = self.Ty_edge[1:-1] / np.sqrt(A[:-1] * A[1:])
        base = -(self.Ty_edge[:-1] + self.Ty_edge[1:]) / A
        offN = off[h:]
        eq = off[h - 1]
        Le = np.empty((len(ms), h))
        Lo = np.empty((len(ms), h))
        Qe = np.empty((len(ms), h, h))
        Qo = np.empty((len(ms), h, h))
        for k, m in enumerate(ms):
            d = base[h:] - lam_x[k] * self.Tx_row[h:] / A[h:]
            de = d.copy()
            de[0] += eq
            do = d.copy()
            do[0] -= eq
            Le[k], Qe[k] = sla.eigh_tridiagonal(de, offN)
            Lo[k], Qo[k] = sla.eigh_tridiagonal(do, offN)
        self._eig = dict(Le=Le, Lo=Lo, Qe=Qe, Qo=Qo)
        return self._eig

    def to_spec(self, f):
        e = self.eig()
        h = self.nlat // 2
        g = f * np.sqrt(self.area_row)[:, None]
        G = sfft.rfft(g, axis=-1)
        G = np.swapaxes(G, -1, -2)
        N, S = (G[..., h:], G[..., :h][..., ::-1])
        ce = (N + S) / np.sqrt(2)
        co = (N - S) / np.sqrt(2)
        ce = np.einsum('mji,...mj->...mi', e['Qe'], ce)
        co = np.einsum('mji,...mj->...mi', e['Qo'], co)
        return (ce, co)

    def from_spec(self, ce, co):
        e = self.eig()
        ce = np.einsum('mij,...mj->...mi', e['Qe'], ce)
        co = np.einsum('mij,...mj->...mi', e['Qo'], co)
        N = (ce + co) / np.sqrt(2)
        S = ((ce - co) / np.sqrt(2))[..., ::-1]
        G = np.concatenate([S, N], axis=-1)
        G = np.swapaxes(G, -1, -2)
        g = sfft.irfft(G, n=self.nlon, axis=-1)
        return g / np.sqrt(self.area_row)[:, None]

    def heat(self, f, s):
        e = self.eig()
        ce, co = self.to_spec(f)
        s = np.asarray(s)[..., None, None]
        return self.from_spec(ce * np.exp(s * e['Le']), co * np.exp(s * e['Lo']))

    def eigenvalues(self):
        e = self.eig()
        nm = self.nlon // 2 + 1
        mult = np.full(nm, 2)
        mult[0] = 1
        mult[-1] = 1
        ev = [np.repeat(e[k], mult, axis=0).ravel() for k in ('Le', 'Lo')]
        return np.sort(np.concatenate(ev))[::-1]

    def grad_flux(self, f):
        FX = self.Tx_row[:, None] * (f - np.roll(f, -1, axis=1))
        FY = np.zeros((self.nlat + 1, self.nlon))
        FY[1:-1] = self.Ty_edge[1:-1, None] * (f[:-1] - f[1:])
        return (FX, FY)

    def bridge_flux(self, a, b):
        FX = self.Tx_row[:, None] * (a * np.roll(b, -1, axis=1) - b * np.roll(a, -1, axis=1))
        FY = np.zeros((self.nlat + 1, self.nlon))
        FY[1:-1] = self.Ty_edge[1:-1, None] * (a[:-1] * b[1:] - b[:-1] * a[1:])
        return (FX, FY)

    def div(self, FX, FY):
        return FX - np.roll(FX, 1, axis=1) + FY[1:] - FY[:-1]

    def laplacian(self, f):
        FX, FY = self.grad_flux(f)
        return -self.div(FX, FY) / self.area

class PlaneGrid:
    kind = 'plane'

    def __init__(self, nx, ny, Lx=2 * np.pi, Ly=np.pi):
        self.nx, self.ny, self.Lx, self.Ly = (nx, ny, Lx, Ly)
        self.hx, self.hy = (Lx / nx, Ly / ny)
        self.x_e = self.hx * np.arange(nx + 1)
        self.y_e = self.hy * np.arange(ny + 1)
        self.x_c = 0.5 * (self.x_e[1:] + self.x_e[:-1])
        self.y_c = 0.5 * (self.y_e[1:] + self.y_e[:-1])
        self.area = np.full((ny, nx), self.hx * self.hy)
        self.total_area = Lx * Ly
        self.Tx = self.hy / self.hx
        self.Ty = self.hx / self.hy
        kx = np.arange(nx)
        ky = np.arange(ny)
        self.lam = -(4 / self.hx ** 2 * np.sin(np.pi * kx / (2 * nx)) ** 2)[None, :] - (4 / self.hy ** 2 * np.sin(np.pi * ky / (2 * ny)) ** 2)[:, None]
        self.diameter = np.hypot(Lx, Ly)

    def to_spec(self, f):
        return sfft.dctn(f, type=2, norm='ortho', axes=(-2, -1))

    def from_spec(self, c):
        return sfft.idctn(c, type=2, norm='ortho', axes=(-2, -1))

    def heat(self, f, s):
        s = np.asarray(s)[..., None, None]
        return self.from_spec(self.to_spec(f) * np.exp(s * self.lam))

    def eigenvalues(self):
        return np.sort(self.lam.ravel())[::-1]

    def grad_flux(self, f):
        FX = np.zeros((self.ny, self.nx + 1))
        FY = np.zeros((self.ny + 1, self.nx))
        FX[:, 1:-1] = self.Tx * (f[:, :-1] - f[:, 1:])
        FY[1:-1] = self.Ty * (f[:-1] - f[1:])
        return (FX, FY)

    def bridge_flux(self, a, b):
        FX = np.zeros((self.ny, self.nx + 1))
        FY = np.zeros((self.ny + 1, self.nx))
        FX[:, 1:-1] = self.Tx * (a[:, :-1] * b[:, 1:] - b[:, :-1] * a[:, 1:])
        FY[1:-1] = self.Ty * (a[:-1] * b[1:] - b[:-1] * a[1:])
        return (FX, FY)

    def div(self, FX, FY):
        return FX[:, 1:] - FX[:, :-1] + FY[1:] - FY[:-1]

    def laplacian(self, f):
        FX, FY = self.grad_flux(f)
        return -self.div(FX, FY) / self.area
