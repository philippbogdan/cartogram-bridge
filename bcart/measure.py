"""Jacobian, displacement and density diagnostics for a computed map."""
import numpy as np

def unit_vec(lon, z):
    c = np.sqrt(np.clip(1 - z * z, 0, None))
    return np.stack([c * np.cos(lon), c * np.sin(lon), z], axis=-1)

def en_frame(lon, z):
    E = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=-1)
    c = np.sqrt(np.clip(1 - z * z, 0, None))
    N = np.stack([-z * np.cos(lon), -z * np.sin(lon), c], axis=-1)
    return (E, N)

def geodesic(lon0, z0, lon1, z1):
    a, b = (unit_vec(lon0, z0), unit_vec(lon1, z1))
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))

def physical_jacobian(kind, X, Y, x, y, D):
    if kind == 'plane':
        return D.copy()
    ci = np.sqrt(1 - Y * Y)
    co = np.sqrt(1 - y * y)
    M = D.copy()
    M[:, 0, :] *= co[:, None]
    M[:, 1, :] /= co[:, None]
    M[:, :, 0] /= ci[:, None]
    M[:, :, 1] *= ci[:, None]
    return M

def transport_frame(X, Y, x, y):
    p, q = (unit_vec(X, Y), unit_vec(x, y))
    E0, N0 = en_frame(X, Y)
    E1, N1 = en_frame(x, y)
    den = (1 + np.sum(p * q, axis=-1))[:, None]

    def tr(v):
        return v - np.sum(v * q, axis=-1)[:, None] / den * (p + q)
    tE, tN = (tr(E0), tr(N0))
    P = np.empty((len(X), 2, 2))
    P[:, 0, 0] = np.sum(tE * E1, -1)
    P[:, 1, 0] = np.sum(tE * N1, -1)
    P[:, 0, 1] = np.sum(tN * E1, -1)
    P[:, 1, 1] = np.sum(tN * N1, -1)
    return P

def distortion(kind, X, Y, x, y, D):
    M = physical_jacobian(kind, X, Y, x, y, D)
    if kind == 'sphere':
        P = transport_frame(X, Y, x, y)
        M = np.einsum('nji,njk->nik', P, M)
        disp = geodesic(X, Y, x, y)
    else:
        disp = np.hypot(x - X, y - Y)

    def angle(M):
        return np.degrees(np.arctan2(M[:, 1, 0] - M[:, 0, 1], M[:, 0, 0] + M[:, 1, 1]))
    rot = angle(M)
    rot_en = angle(physical_jacobian(kind, X, Y, x, y, D)) if kind == 'sphere' else rot
    a, b, c, d = (M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1])
    s = np.linalg.svd(M, compute_uv=False)
    return dict(rotation=rot, rotation_en=rot_en, anisotropy=s[:, 0] / s[:, 1], area_ratio=a * d - b * c, displacement=disp)

def wquantile(v, w, q):
    o = np.argsort(v)
    v, w = (v[o], w[o])
    c = np.cumsum(w)
    c = c / c[-1]
    return np.interp(q, c, v)

def summary(dist, w_area, w_mass):
    out = {}
    for name, w in (('area', w_area), ('mass', w_mass)):
        for key in ('rotation', 'rotation_en', 'anisotropy'):
            v = dist[key] if key == 'anisotropy' else np.abs(dist[key])
            out[f'{key}_median_{name}'] = float(wquantile(v, w, 0.5))
            out[f'{key}_p95_{name}'] = float(wquantile(v, w, 0.95))
    out['rotation_max'] = float(np.max(np.abs(dist['rotation'])))
    out['anisotropy_max'] = float(np.max(dist['anisotropy']))
    out['disp_max'] = float(np.max(dist['displacement']))
    out['disp_median_mass'] = float(wquantile(dist['displacement'], w_mass, 0.5))
    return out

def vertex_points(kind, g, nudge=1e-09):
    if kind == 'sphere':
        lon = g.lon_e[:-1] + nudge * g.dlon
        z = g.z_e.copy()
        dz = np.concatenate([g.dz, g.dz[-1:]])
        z = np.where(np.arange(len(z)) < len(z) - 1, z + nudge * dz, z - nudge * dz)
        LON, Z = np.meshgrid(lon, z)
    else:
        x = g.x_e.copy()
        y = g.y_e.copy()
        x[:-1] += nudge * g.hx
        x[-1] -= nudge * g.hx
        y[:-1] += nudge * g.hy
        y[-1] -= nudge * g.hy
        LON, Z = np.meshgrid(x, y)
    return (LON.ravel(), Z.ravel(), LON.shape)

def graticule_jacobian(kind, g, vx, vy, shape):
    VX = vx.reshape(shape)
    VY = vy.reshape(shape)
    if kind == 'sphere':
        nl, nx = (g.nlat, g.nlon)
        sw_x, sw_y = (VX[:-1], VY[:-1])
        se_x, se_y = (np.roll(VX[:-1], -1, axis=1), np.roll(VY[:-1], -1, axis=1))
        nw_x, nw_y = (VX[1:], VY[1:])
        ne_x, ne_y = (np.roll(VX[1:], -1, axis=1), np.roll(VY[1:], -1, axis=1))
        wrap = lambda d: (d + np.pi) % (2 * np.pi) - np.pi
        dxl = wrap(se_x - sw_x) + wrap(ne_x - nw_x)
        dyl = se_y - sw_y + (ne_y - nw_y)
        dxz = wrap(nw_x - sw_x) + wrap(ne_x - se_x)
        dyz = nw_y - sw_y + (ne_y - se_y)
        hx = g.dlon * np.ones((nl, nx))
        hy = g.dz[:, None] * np.ones((nl, nx))
        LON, Z = np.meshgrid(g.lon_c, np.sin(g.lat_c))
        from .measure import unit_vec as _uv
        P = _uv(sw_x, sw_y) + _uv(se_x, se_y) + _uv(nw_x, nw_y) + _uv(ne_x, ne_y)
        P /= np.linalg.norm(P, axis=-1, keepdims=True)
        xc = np.arctan2(P[..., 1], P[..., 0])
        yc = P[..., 2]
    else:
        sw_x, sw_y = (VX[:-1, :-1], VY[:-1, :-1])
        se_x, se_y = (VX[:-1, 1:], VY[:-1, 1:])
        nw_x, nw_y = (VX[1:, :-1], VY[1:, :-1])
        ne_x, ne_y = (VX[1:, 1:], VY[1:, 1:])
        dxl = se_x - sw_x + (ne_x - nw_x)
        dyl = se_y - sw_y + (ne_y - nw_y)
        dxz = nw_x - sw_x + (ne_x - se_x)
        dyz = nw_y - sw_y + (ne_y - se_y)
        hx = g.hx * np.ones(sw_x.shape)
        hy = g.hy * np.ones(sw_x.shape)
        LON, Z = np.meshgrid(g.x_c, g.y_c)
        xc = (sw_x + se_x + nw_x + ne_x) / 4
        yc = (sw_y + se_y + nw_y + ne_y) / 4
    D = np.empty(hx.shape + (2, 2))
    D[..., 0, 0] = dxl / (2 * hx)
    D[..., 1, 0] = dyl / (2 * hx)
    D[..., 0, 1] = dxz / (2 * hy)
    D[..., 1, 1] = dyz / (2 * hy)
    return (D.reshape(-1, 2, 2), LON.ravel(), Z.ravel(), xc.ravel(), yc.ravel())

def edge_points(kind, g, m):
    u, w = np.polynomial.legendre.leggauss(m)
    u = (u + 1) / 2
    w = w / 2
    if kind == 'sphere':
        xe, ye = (g.lon_e, g.z_e)
        nx = g.nlon
        hx_ = np.full(nx, g.dlon)
        vx_cols = xe[:-1]
    else:
        xe, ye = (g.x_e, g.y_e)
        nx = g.nx
        hx_ = np.full(nx, g.hx)
        vx_cols = xe
    ny = len(ye) - 1
    HX = (xe[:-1][None, :, None] + hx_[None, :, None] * u[None, None, :]) * np.ones((ny + 1, 1, 1))
    HY = ye[:, None, None] * np.ones((1, nx, m))
    dy = np.diff(ye)
    VX = vx_cols[None, :, None] * np.ones((ny, 1, m))
    VY = (ye[:-1][:, None, None] + dy[:, None, None] * u[None, None, :]) * np.ones((1, len(vx_cols), 1))
    X = np.concatenate([HX.ravel(), VX.ravel()])
    Y = np.concatenate([HY.ravel(), VY.ravel()])
    return (X, Y, dict(m=m, w=w, ny=ny, nx=nx, nh=HX.size, hshape=HX.shape, vshape=VX.shape))

def edge_jacobian(kind, g, x, y, lay):
    w = lay['w']
    nh = lay['nh']
    ny, nx = (lay['ny'], lay['nx'])
    hx = x[:nh].reshape(lay['hshape'])
    hy = y[:nh].reshape(lay['hshape'])
    vx = x[nh:].reshape(lay['vshape'])
    vy = y[nh:].reshape(lay['vshape'])
    if kind == 'sphere':
        ref_h = g.lon_c[None, :, None] * np.ones(hx.shape)
        hx = ref_h + (hx - ref_h + np.pi) % (2 * np.pi) - np.pi
        ref_v = g.lon_e[:-1][None, :, None] * np.ones(vx.shape)
        vx = ref_v + (vx - ref_v + np.pi) % (2 * np.pi) - np.pi
        dX = g.dlon
        dY = g.dz[:, None]
    else:
        dX = g.hx
        dY = g.hy
    Ih_x = np.sum(hx * w, -1)
    Ih_y = np.sum(hy * w, -1)
    Iv_x = np.sum(vx * w, -1)
    Iv_y = np.sum(vy * w, -1)
    if kind == 'sphere':
        Iv_xE = np.roll(Iv_x, -1, axis=1)
        Iv_yE = np.roll(Iv_y, -1, axis=1)
        Iv_xE[:, -1] += 2 * np.pi
        Iv_xW, Iv_yW = (Iv_x, Iv_y)
    else:
        Iv_xW, Iv_yW = (Iv_x[:, :-1], Iv_y[:, :-1])
        Iv_xE, Iv_yE = (Iv_x[:, 1:], Iv_y[:, 1:])
    D = np.empty((ny, nx, 2, 2))
    D[..., 0, 0] = (Iv_xE - Iv_xW) / dX
    D[..., 1, 0] = (Iv_yE - Iv_yW) / dX
    D[..., 0, 1] = (Ih_x[1:] - Ih_x[:-1]) / dY
    D[..., 1, 1] = (Ih_y[1:] - Ih_y[:-1]) / dY
    return D.reshape(-1, 2, 2)
