"""Flat-map cartograms. Input mass rows run north to south; output points use radians."""
import os
import sys
import time
import numpy as np
VERBOSE = os.environ.get('BCART_VERBOSE', '0') == '1'

def _log(msg):
    if VERBOSE:
        print(f"[bcart {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)
import jax.numpy as jnp
from . import engine
from .paths import normalise
from .nodes import bridge_nodes, diffusion_nodes
from .measure import distortion, summary, geodesic, vertex_points, graticule_jacobian, edge_points, edge_jacobian

class Result:

    def __init__(self, cart, kind, gamma, X, Y, x, y, D, meta):
        self.cart, self.kind, self.gamma = (cart, kind, gamma)
        self.X, self.Y, self.x, self.y, self.D, self.meta = (X, Y, x, y, D, meta)
        self.det = getattr(cart, '_det', None) if D is not None else None

    def _where(self, k):
        g = self.cart.G.np
        j, i = np.unravel_index(k, g.area.shape)
        if self.cart.domain == 'sphere':
            return dict(lat=float(np.degrees(g.lat_c[j])), lon=float(np.degrees(g.lon_c[i])))
        x, y = (float(g.x_c[i]), float(g.y_c[j]))
        return dict(x=x, y=y, lat=float(np.degrees(y) - 90), lon=float(np.degrees(x) - 180))

    def lonlat(self):
        if self.cart.domain == 'sphere':
            return (np.degrees(self.x), np.degrees(np.arcsin(np.clip(self.y, -1, 1))))
        return (self.x, self.y)

    def measures(self):
        c = self.cart
        out = dict(self.meta)
        if self.D is not None:
            dist = distortion(c.domain, self.X, self.Y, self.x, self.y, self.D)
            det = dist['area_ratio']
            if self.meta.get('centres'):
                w_area = c.G.np.area.ravel()
                w_mass = (c.rho0 * c.G.np.area).ravel()
                dev_D = np.abs(det / c.rho0.ravel() - 1)
                dev = np.abs(self.det / c.rho0.ravel() - 1) if self.det is not None else dev_D
                k = int(np.argmax(dev))
                out['equalisation_max_dev'] = float(dev[k])
                out['equalisation_median_dev'] = float(np.median(dev))
                out['equalisation_worst'] = dict(self._where(k), rho0=float(c.rho0.ravel()[k]))
                out['equalisation_source'] = 'wedge' if self.det is not None else 'detD'
                out['equalisation_max_dev_detD'] = float(dev_D.max())
                out['equalisation_median_dev_detD'] = float(np.median(dev_D))
            else:
                w_area = np.ones_like(det)
                w_mass = w_area
            out.update(summary(dist, w_area, w_mass))
            if self.meta.get('centres'):
                d2 = dist['displacement'] ** 2
                out['cost_midpoint'] = float(0.5 * np.sum(w_mass * d2) / np.sum(w_mass))
            if getattr(self, 'V', None) is not None:
                Dg, Xc, Yc, xc, yc = graticule_jacobian(c.domain, c.G.np, *self.V)
                dg = distortion(c.domain, Xc, Yc, xc, yc, Dg)
                gs = summary(dg, w_area, w_mass)
                out.update({'grat_' + k: v for k, v in gs.items() if k.startswith(('rotation', 'anisotropy'))})
                rot, an = (np.abs(dg['rotation']), dg['anisotropy'])
                for key, v in (('rotation', rot), ('anisotropy', an)):
                    k = int(np.nanargmax(v))
                    out[f'grat_{key}_worst'] = dict(self._where(k), value=float(v[k]))
                for thr in (1.0, 10.0):
                    out[f'grat_rotation_area_above_{thr:g}deg'] = float(np.sum(w_area * (rot > thr)) / np.sum(w_area))
            if getattr(self, 'E', None) is not None:
                De = edge_jacobian(c.domain, c.G.np, *self.E)
                de = distortion(c.domain, self.X, self.Y, self.x, self.y, De)
                es = summary(de, w_area, w_mass)
                out.update({'edge_' + k: v for k, v in es.items() if k.startswith(('rotation', 'anisotropy'))})
        return out

def _keep(keep_path, nslabs, ncell):
    if keep_path is True or keep_path is False or keep_path is None:
        return bool(keep_path)
    if keep_path == 'ld':
        return False
    limit = float(os.environ.get('BCART_PATH_CACHE_GB', '0.25')) * 2 ** 30
    return nslabs * 3 * ncell * 8 <= limit

class Cartogram:

    def __init__(self, masses, domain='plane', Lx=2 * np.pi, Ly=np.pi):
        if domain != 'plane':
            raise ValueError('The cartogram API accepts the flat rectangle only.')
        m = np.asarray(masses, float)[::-1]
        if m.ndim != 2 or min(m.shape) < 2 or (not np.all(np.isfinite(m))) or np.any(m <= 0):
            raise ValueError('Supply a finite, strictly positive mass array with at least two rows and columns.')
        ny, nx = m.shape
        self.domain = domain
        tic = time.time()
        _log(f'grid setup {domain} {nx}x{ny}')
        self.G = engine.SphereJ(nx, ny) if domain == 'sphere' else engine.PlaneJ(nx, ny, Lx, Ly)
        self.setup_time = time.time() - tic
        _log(f'grid ready in {self.setup_time:.1f}s')
        g = self.G.np
        self.rho0 = normalise(g, m)
        if domain == 'sphere':
            self.h = g.dlat
            self.l1 = self._l1_sphere()
        else:
            self.h = max(g.hx, g.hy)
            self.l1 = float(min(-g.lam[0, 1], -g.lam[1, 0]))
        self._warm = None

    def _run(self, slabs, X, Y, jac, centres, edges=0, G_track=None):
        Gt = self.G if G_track is None else G_track
        R = 1 if Gt is self.G else Gt.ny // self.G.ny
        if not centres:
            (x, y), D, st = engine.run_map(Gt, slabs, X, Y, jac)
            self._det = st.get('det')
            return ((x, y), D, st, None)
        VX, VY, vshape = vertex_points(self.domain, self.G.np)
        parts = [(X, Y), (VX, VY)]
        if edges:
            EX, EY, lay = edge_points(self.domain, self.G.np, edges)
            parts.append((EX, EY))
        if R > 1 and jac:
            g = self.G.np
            off = (np.arange(R) + 0.5) / R - 0.5
            SX = np.concatenate([X + a * g.hx for a in off for b in off])
            SY = np.concatenate([Y + b * g.hy for a in off for b in off])
            parts.append((SX, SY))
        AX = np.concatenate([p[0] for p in parts])
        AY = np.concatenate([p[1] for p in parts])
        n, nv = (X.size, VX.size)
        ne = parts[2][0].size if edges else 0
        (x, y), D, st = engine.run_map(Gt, slabs, AX, AY, jac)
        if R > 1 and jac:
            s0 = n + nv + ne
            Ds = D[s0:].reshape(R * R, n, 2, 2)
            ds = st['det'][s0:].reshape(R * R, n)
            k = np.argmax(np.abs(ds / self.rho0.ravel()[None] - 1), axis=0)
            self._det = np.take_along_axis(ds, k[None], 0)[0]
            D = Ds.mean(0)
            st['det_subcells'] = R * R
        else:
            self._det = st['det'][:n] if 'det' in st else None
            if D is not None:
                D = D[:n]
        V = (x[n:n + nv], y[n:n + nv], vshape)
        self._edges = (x[n + nv:n + nv + ne], y[n + nv:n + nv + ne], lay) if edges else None
        return ((x[:n], y[:n]), D, st, V)

    def _l1_sphere(self):
        Le, Lo = (np.asarray(self.G.Le), np.asarray(self.G.Lo))
        ev = np.sort(np.concatenate([Le.ravel(), Lo.ravel()]))[::-1]
        return float(-ev[1])

    def centres(self):
        g = self.G.np
        if self.domain == 'sphere':
            LON, Z = np.meshgrid(g.lon_c, np.sin(g.lat_c))
        else:
            LON, Z = np.meshgrid(g.x_c, g.y_c)
        return (LON.ravel(), Z.ravel())

    def _points(self, points):
        if points is None:
            X, Y = self.centres()
            return (X, Y, True)
        X, Y = points
        if self.domain == 'sphere':
            X, Y = (np.radians(X), np.sin(np.radians(Y)))
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if X.ndim != 1 or X.shape != Y.shape or X.size == 0 or not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError('Point coordinates must be nonempty, matching finite one-dimensional arrays.')
        if np.any((X < 0) | (X > self.G.np.Lx) | (Y < 0) | (Y > self.G.np.Ly)):
            raise ValueError('Point coordinates must lie inside the rectangle.')
        return X, Y, False
    FV_MIN = {'sphere': 0.03, 'plane': 0.14}

    @property
    def gamma_fv(self):
        return max(self.FV_MIN[self.domain], 15 * self.h)

    def bridge(self, gamma, K=None, points=None, jac=True, tol=1e-12, maxit=500000, r=1.1, edges=0, solver='auto', keep_path=False):
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError('gamma must be finite and positive.')
        if solver not in ('auto', 'fv', 'ld'):
            raise ValueError('solver must be auto, fv or ld.')
        X, Y, centres = self._points(points)
        if solver == 'auto':
            solver = 'fv' if gamma >= self.gamma_fv else 'ld'
        if solver == 'ld':
            return self._bridge_ld(gamma, X, Y, centres, jac, edges, keep_path)
        b0 = None
        if self._warm is not None:
            e0, bw = self._warm
            b0 = np.exp(np.log(bw) * (e0 / gamma))
        t0 = time.time()
        _log(f'sinkhorn gamma={gamma:g}')
        a, b, info = engine.sinkhorn(self.G, self.rho0, gamma, tol=tol, maxit=maxit, b0=b0)
        b.block_until_ready()
        t1 = time.time()
        _log(f'sinkhorn done {t1 - t0:.1f}s {info}')
        self._warm = (gamma, np.asarray(b))
        t = bridge_nodes(gamma, self.h, K, self.l1, r=r)
        _log(f'path+map: {len(t) - 1} slabs')
        gen = engine.bridge_slabs(self.G, self.rho0, gamma, t, a, b)
        keep_path = _keep(keep_path, len(t) - 1, self.rho0.size)
        pc = engine.PathCache(gen) if keep_path else None
        (x, y), D, st, V = self._run(pc.record() if keep_path else gen, X, Y, jac, centres, edges)
        t2 = time.time()
        meta = dict(kind='bridge', solver='fv', gamma=gamma, sinkhorn_iters=info['iters'], sinkhorn_err=info['err'], t_sinkhorn=t1 - t0, t_map=t2 - t1, t_total=t2 - t0, slabs=len(t) - 1, node_ratio=r, continuity=st['continuity'], event_loops=st['event_loops'], centres=centres, min_density=st['min_density'], valid=bool(info['err'] < 1e-10 and st['min_density'] > 0 and (st['continuity'] < 1e-11)))
        if keep_path:
            meta['t_keep_path'] = pc.t_copy
        res = Result(self, 'bridge', gamma, X, Y, x, y, D if jac else None, meta)
        res.V = V
        res.E = self._edges if edges else None
        G_, r0_ = (self.G, self.rho0)
        res.paths = (pc.forward, pc.reversed) if keep_path else (lambda: engine.bridge_slabs(G_, r0_, gamma, t, a, b), lambda: engine.bridge_slabs_reversed(G_, r0_, gamma, t, a, b))
        return res

    def _bridge_ld(self, gamma, X, Y, centres, jac, edges, keep_path=False):
        import importlib
        ld = importlib.import_module(f'.ld_{self.domain}', __package__)
        t0 = time.time()
        warm = getattr(self, '_ld_warm', None)
        if warm is not None and (not 0.2 < warm.gamma / gamma < 5):
            warm = None
        if warm is None and self._warm is not None and hasattr(ld, 'FVWarm') and (0.2 < self._warm[0] / gamma < 5):
            warm = ld.FVWarm(*self._warm)
        _log(f'ld static gamma={gamma:g} warm={(None if warm is None else warm.gamma)}')
        P = ld.solve(self.G, self.rho0, gamma, warm=warm)
        t1 = time.time()
        self._ld_warm = P
        t = ld.nodes(gamma, self.h)
        _log(f'ld path+map: {len(t) - 1} slabs')
        R = self.plane_refine(gamma)
        kw = {}
        Gt, G_path, r0_path, P_path = (self.G, self.G, self.rho0, P)
        marg_path = None
        if R > 1:
            cache = self.__dict__.setdefault('_G_track', {})
            if R not in cache:
                g = self.G.np
                cache[R] = engine.PlaneJ(R * g.nx, R * g.ny, g.Lx, g.Ly)
            Gt = cache[R]
            if os.environ.get('BCART_PLANE_REFINE_MODE', self.PLANE_REFINE_MODE) == 'split':
                r0_path = np.repeat(np.repeat(self.rho0, R, 0), R, 1)
                tic = time.time()
                P_path = ld.solve(Gt, r0_path, gamma, warm=P)
                t1 += time.time() - tic
                G_path = Gt
                marg_path = float(P_path.info.get('marg', np.nan))
                kw = dict(path_kw=dict(c_init=0.5 * R))
            else:
                kw = dict(m=R, refine=R, path_kw=dict(G_assembly=Gt))
        gen = ld.bridge_slabs_ld(G_path, r0_path, gamma, t, P_path, **kw)
        keep_path = bool(keep_path)
        dev = os.environ.get('BCART_PATH_CACHE_DEVICE', '1') == '1'
        pc = engine.PathCache(gen, device=dev or R > 1, compact=Gt if R > 1 else None) if keep_path else None
        slabs = pc.record() if keep_path else gen
        if keep_path and R > 1:
            for _ in slabs:
                pass
            del gen, slabs
            P_path = None
            slabs = pc.forward()
        (x, y), D, st, V = self._run(slabs, X, Y, jac, centres, edges, G_track=Gt)
        t2 = time.time()
        cont = max(st['continuity'], pc.continuity if keep_path else 0.0)
        marg = float(P.info.get('marg', np.nan)) if hasattr(P, 'info') else np.nan
        if marg_path is not None:
            marg = max(marg, marg_path)
        meta = dict(kind='bridge', solver='ld', gamma=gamma, sinkhorn_iters=P.info.get('newton') if hasattr(P, 'info') else None, sinkhorn_err=marg, t_sinkhorn=t1 - t0, t_map=t2 - t1, t_total=t2 - t0, slabs=len(t) - 1, continuity=cont, event_loops=st['event_loops'], centres=centres, min_density=st['min_density'], tracking_refine=R, valid=bool(marg < 1e-09 and st['min_density'] > 0 and (cont < 1e-11)))
        if keep_path:
            meta['t_keep_path'] = pc.t_copy
        res = Result(self, 'bridge', gamma, X, Y, x, y, D if jac else None, meta)
        res.V = V
        res.E = self._edges if edges else None
        res.G_track = Gt
        G_, r0_, P_ = (G_path, r0_path, P_path)
        res.paths = (pc.forward, pc.reversed) if keep_path else (lambda: ld.bridge_slabs_ld(G_, r0_, gamma, t, P_, **kw), lambda: ld.bridge_slabs_ld(G_, r0_, gamma, t, P_, reverse=True, **kw))
        return res
    PLANE_REFINE_EPS = 0.0013
    PLANE_REFINE_MIN_EPS_H2 = 8.0
    PLANE_REFINE_MODE = 'split'

    def plane_refine(self, gamma):
        if self.domain != 'plane':
            return 1
        lim = float(os.environ.get('BCART_PLANE_REFINE_EPS', self.PLANE_REFINE_EPS))
        return 2 if gamma <= lim and gamma >= self.PLANE_REFINE_MIN_EPS_H2 * self.h ** 2 else 1

    def diffusion(self, K=None, points=None, jac=True, r=1.1, edges=0, keep_path=False):
        X, Y, centres = self._points(points)
        t0 = time.time()
        s = diffusion_nodes(self.h, self.l1, K, dev=float(self.rho0.max()), r=r)
        gen = engine.diffusion_slabs(self.G, self.rho0, s)
        keep_path = _keep(keep_path, len(s) + 1, self.rho0.size)
        pc = engine.PathCache(gen) if keep_path else None
        (x, y), D, st, V = self._run(pc.record() if keep_path else gen, X, Y, jac, centres, edges)
        t1 = time.time()
        meta = dict(kind='diffusion', gamma=np.inf, t_total=t1 - t0, slabs=len(s), s_max=float(s[-1]), continuity=st['continuity'], event_loops=st['event_loops'], centres=centres, min_density=st['min_density'])
        if keep_path:
            meta['t_keep_path'] = pc.t_copy
        res = Result(self, 'diffusion', np.inf, X, Y, x, y, D if jac else None, meta)
        res.V = V
        res.E = self._edges if edges else None
        G_, r0_ = (self.G, self.rho0)
        res.paths = (pc.forward, pc.reversed) if keep_path else (lambda: engine.diffusion_slabs(G_, r0_, s), lambda: engine.diffusion_slabs_reversed(G_, r0_, s))
        return res
