"""Check the manufactured spherical reference on finite evaluation sets."""
import argparse
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax.numpy as jnp
from bcart import sphere_entropic as se
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out', default='computed/transport_end_sphere/mccann_check.json')
ap.add_argument('--amp', type=float, default=0.08)
ap.add_argument('--nfine', type=int, default=400)
ap.add_argument('--nsample', type=int, default=4000)
a = ap.parse_args()
M = se.ManufacturedMcCann(a.amp)
G = se.GLGrid(a.nfine)
X = jnp.asarray(G.pts)
mins = []
dets, deta, disp = ([], [], [])
for i in range(0, X.shape[0], 20000):
    Xi = X[i:i + 20000]
    mins.append(float(M.local_min_eig(Xi).min()))
    r = np.asarray(M.rho0(Xi))
    ra = np.asarray(M.rho0_analytic(Xi))
    dets.append(r.min())
    deta.append(float(np.max(np.abs(r - ra))))
    disp.append(float(np.max(np.linalg.norm(np.asarray(M.grad_u(Xi)), axis=1))))
Xs = jnp.asarray(se.fibonacci_points(a.nsample))
gap = np.asarray(M.check_c_convexity(Xs, X, block=64))
out = dict(amp=a.amp, nfine_points=int(X.shape[0]), nsample=a.nsample, local_min_eig=min(mins), min_jacobian=float(min(dets)), max_ad_vs_jacobi_field=max(deta), max_displacement=max(disp), global_min_gap=float(gap.min()), note='Finite-set minimum cost gap for the manufactured map. A positive sampled gap does not establish global optimality.')
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
json.dump(out, open(a.out, 'w'), indent=1)
print(out)
