"""Recompute spherical flow and barycentric maps against exact reference maps."""
import argparse
import json
import math
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
import jax.numpy as jnp
from bcart import sphere_entropic as se

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=['manufactured', 'zonal'], default='manufactured')
    ap.add_argument('--gamma', type=float, nargs='+', required=True)
    ap.add_argument('--amp', type=float, default=0.08)
    ap.add_argument('--kappa', type=float, default=6.0, help='nlat = ceil(kappa/sqrt(gamma))')
    ap.add_argument('--nlat_min', type=int, default=24)
    ap.add_argument('--n_gh', type=int, default=12)
    ap.add_argument('--nsteps', type=int, default=8)
    ap.add_argument('--neval', type=int, default=24, help='evaluation grid GL(neval) x 2 neval')
    ap.add_argument('--block', type=int, default=512)
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--no_flow', action='store_true')
    ap.add_argument('--flow_max_gamma', type=float, default=1.0, help='flow map only for gamma <= this')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--outdir', default='computed/transport_end_sphere')
    args = ap.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    base = f'{outdir}/{args.case}_{args.tag}'
    if args.case == 'manufactured':
        ref = se.ManufacturedMcCann(args.amp)
        T0fn = lambda X: np.asarray(ref.T0(jnp.asarray(X)))
    else:
        ref = se.ZonalReference()
        T0fn = ref.T0
    logrho0 = ref.logrho0
    ev = se.GLGrid(args.neval)
    Xe = ev.pts
    we = ev.w
    T0e = T0fn(Xe)
    rho_e = np.exp(np.asarray(logrho0(jnp.asarray(Xe))))

    def errs(T):
        d = np.asarray(se.geodist(jnp.asarray(T), jnp.asarray(T0e)))
        return dict(max_over_pi=float(d.max() / math.pi), l2_rho0_over_pi=float(np.sqrt(np.sum(we * rho_e * d * d) / np.sum(we * rho_e)) / math.pi), argmax=[float(v) for v in Xe[int(np.argmax(d))]])
    out = dict(case=args.case, args=vars(args), results=[])
    if args.case == 'manufactured':
        out['reference'] = dict(type='McCann exp_x(-grad u)', amp=args.amp, coef=se.ManufacturedMcCann.COEF)
    else:
        out['reference'] = dict(type='zonal monotone rearrangement', a=ref.a)
    maps = {'X': Xe, 'T0': T0e}
    prev = []
    for gamma in sorted(args.gamma, reverse=True):
        t0 = time.time()
        nlat = max(args.nlat_min, int(math.ceil(args.kappa / math.sqrt(gamma))))
        nlat += nlat % 2
        g = se.GLGrid(nlat)
        B = se.SphereBridge(g, logrho0, gamma, block=args.block)
        rec = dict(gamma=gamma, nlat=nlat, N=g.N, h=g.h, h_over_sqrt_gamma=g.h / math.sqrt(gamma), table_fit_err=B.table.fit_err, mass_defect=B.mass_defect)
        print(f'gamma={gamma:.4g} nlat={nlat} N={g.N} table_err={B.table.fit_err:.1e}', flush=True)
        if args.case == 'zonal':
            se.solve_zonal(B)
            rec['rel_marg_2d'] = B.residual(B.alpha, B.beta)
        else:
            beta0 = None
            Xg = jnp.asarray(g.pts)
            if prev:
                gs = [(e, Bp.gamma * Bp.logphi1(Xg)) for e, Bp in prev[-2:]]
                if len(gs) == 2:
                    (e1, g1), (e2, g2) = gs
                    gnew = g2 + (g2 - g1) * (gamma - e2) / (e2 - e1)
                else:
                    gnew = gs[0][1]
                beta0 = gnew / gamma
            B.solve(beta0=beta0, sinkhorn_pre=0 if prev else 20)
            rec['rel_marg_2d'] = B.final_rel
        rec.update(solve_time=B.solve_time, newton=B.newton_iters, cg=B.cg_iters)
        t1 = time.time()
        me, mf, resid = se.barycentric_maps(B, Xe)
        rec['bary_time'] = time.time() - t1
        rec['bary_extrinsic'] = errs(np.asarray(me))
        rec['bary_frechet'] = errs(np.asarray(mf))
        rec['frechet_resid_max'] = float(jnp.max(resid))
        maps[f'bary_ext_{gamma:.6g}'] = np.asarray(me)
        maps[f'bary_fre_{gamma:.6g}'] = np.asarray(mf)
        print(f"  bary ext {rec['bary_extrinsic']['max_over_pi']:.3e}  fre {rec['bary_frechet']['max_over_pi']:.3e}", flush=True)
        if not args.no_flow and gamma <= args.flow_max_gamma:
            t1 = time.time()
            Fl = se.BridgeFlow(B, n_gh=args.n_gh, chunk=args.chunk)
            T = Fl.flow(Xe, nsteps=args.nsteps, verbose=True)
            rec['flow_time'] = time.time() - t1
            rec['flow'] = errs(np.asarray(T))
            maps[f'flow_{gamma:.6g}'] = np.asarray(T)
            print(f"  flow {rec['flow']['max_over_pi']:.3e}  L2 {rec['flow']['l2_rho0_over_pi']:.3e}", flush=True)
        rec['total_time'] = time.time() - t0
        out['results'].append(rec)
        prev.append((gamma, B))
        prev = prev[-2:]
        with open(base + '.json', 'w') as f:
            json.dump(out, f, indent=1)
        np.savez(base + '.npz', **maps)
    print('done', flush=True)
if __name__ == '__main__':
    main()
