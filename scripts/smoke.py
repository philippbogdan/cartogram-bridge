"""Exercise both planar solvers, density preservation and the diffusion-end clock."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bcart import Cartogram, sb_spectral as sb
from bcart.spectral import PlaneSpec


def main():
    x = (np.arange(16) + 0.5) * 2 * np.pi / 16
    y = (np.arange(8) + 0.5) * np.pi / 8
    X, Y = np.meshgrid(x, y)
    mass = 1 + 0.2 * np.cos(X / 2) + 0.1 * np.cos(Y)
    c = Cartogram(mass[::-1], 'plane')
    for solver, gamma in [('fv', 1.0), ('ld', 0.08)]:
        r = c.bridge(gamma, solver=solver)
        m = r.measures()
        assert np.isfinite(r.x).all() and np.isfinite(r.y).all()
        assert m['equalisation_max_dev'] < 1e-8, m
        assert r.meta['sinkhorn_err'] < 1e-8, r.meta
        assert r.meta['min_density'] > 0
        assert np.max(np.hypot(r.x - r.X, r.y - r.Y)) > 1e-3
        assert np.all((r.x >= 0) & (r.x <= 2 * np.pi))
        assert np.all((r.y >= 0) & (r.y <= np.pi))
        print(solver, 'density deviation', m['equalisation_max_dev'])
    d = c.diffusion()
    assert d.measures()['equalisation_max_dev'] < 1e-8
    u = Cartogram(np.ones((8, 16)), 'plane').bridge(1.0, solver='fv')
    assert np.max(np.hypot(u.x - u.X, u.y - u.Y)) < 1e-12
    dom = PlaneSpec(24, 12)
    density = sb.make_density(dom, 'pl_generic')
    b = sb.SpectralBridge(dom, density, 100)
    natural = sb.error_at(b, b.T, 'linear')['E_area']
    canonical = sb.error_at(b, b.T, 'canonical')['E_area']
    assert abs(natural - 1) < 1e-8
    assert canonical < 1e-5
    logf, derivative = sb.time_change('canonical', b.T, b.gamma, dom.l1)
    assert abs(-logf / dom.l1 - b.gamma) < 1e-10
    assert np.isclose(derivative, 2 * np.exp(dom.l1 * b.gamma / 2))
    print('Planar maps, uniform identity and both clock comparisons passed.')


if __name__ == '__main__':
    main()
