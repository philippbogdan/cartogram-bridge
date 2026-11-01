"""Time grids that resolve the initial and terminal layers of the density path."""
import numpy as np

def _geom(a, b, r):
    if b <= a:
        return np.array([b])
    n = max(1, int(np.ceil(np.log(b / a) / np.log(r))))
    return np.geomspace(a, b, n + 1)

def bridge_nodes(gamma, h, K=None, l1=2.0, r=1.2):
    S = gamma / 2
    s0 = min(h * h / 50, S / 10000.0)
    head = _geom(s0, S / 2, r)
    e0 = min(0.001 / l1, S / 10000.0)
    tail = S - _geom(e0, S / 2, r)[::-1]
    s = np.unique(np.concatenate([[0.0], head, tail, [S]]))
    return s / S

def diffusion_nodes(h, l1=2.0, K=None, dev=1000.0, r=1.2):
    s0 = h * h / 50
    s_max = (np.log(dev) + 36) / l1
    return np.concatenate([[0.0], _geom(s0, s_max, r)])
