"""Recompute planar flow and barycentric maps against manufactured transport maps."""
import argparse
import json
import math
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from bcart import plane_entropic as pe

def grid_n(gamma, c, nmin, mult=32):
    nx = pe.LX / (c * math.sqrt(gamma))
    nx = max(nmin, int(math.ceil(nx / mult)) * mult)
    return (nx, nx // 2)

def interp_to(grid_from, f, grid_to):
    import jax
    return np.asarray(jax.jit(lambda f: pe.interp_cos(grid_from, f, grid_to))(f))

def solve_one(case, gamma, nx, ny, nsteps, Xl, Yl, w_init, tol, anderson, verbose, hj_nx):
    g = pe.Grid(nx, ny)
    X, Y = g.mesh()
    logrho0 = np.log(case.rho0(X, Y))
    w0 = None
    if w_init is not None:
        gprev, gamma_prev, wprev = w_init
        w0 = interp_to(gprev, gamma_prev * wprev, g) / gamma
    sk = pe.sinkhorn(g, logrho0, gamma, w0=w0, tol=tol, anderson=anderson, verbose=verbose)
    t1 = time.time()
    XP, YP = np.meshgrid(Xl, Yl)
    fm = pe.flow_map(g, sk['u'], sk['w'], gamma, XP.ravel(), YP.ravel(), logrho0, nsteps=nsteps, hj_nx=hj_nx)
    sk['tail_w'] = pe.spectrum_tail(g, gamma * sk['w'], 0.5)
    bx, by = pe.barycentric(g, sk['w'], gamma, Xl, Yl)
    return (g, sk, fm, (bx, by), time.time() - t1)

def errors(case, Xl, Yl, TX, TY):
    XP, YP = np.meshgrid(Xl, Yl)
    T0x, T0y = case.T0(XP, YP)
    d = np.hypot(TX.reshape(XP.shape) - T0x, TY.reshape(XP.shape) - T0y)
    rho = case.rho0(XP, YP)
    l2 = math.sqrt(np.sum(rho * d ** 2) / np.sum(rho))
    D = pe.DIAMETER
    return dict(max=float(d.max() / D), l2_rho0=float(l2 / D), argmax=[float(XP.ravel()[d.argmax()]), float(YP.ravel()[d.argmax()])])

def fit_rate(gamma, err):
    gamma = np.asarray(gamma)
    err = np.asarray(err)
    if len(gamma) < 2:
        return None
    A = np.stack([np.log(gamma), np.ones_like(gamma)], 1)
    coef, res, *_ = np.linalg.lstsq(A, np.log(err), rcond=None)
    n = len(gamma)
    if n > 2:
        r = np.log(err) - A @ coef
        s2 = r @ r / (n - 2)
        cov = s2 * np.linalg.inv(A.T @ A)
        se = float(math.sqrt(cov[0, 0]))
    else:
        se = None
    return dict(slope=float(coef[0]), slope_se=se, prefactor=float(math.exp(coef[1])), gamma_range=[float(gamma.min()), float(gamma.max())], n=n)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', default=['smooth'], nargs='+', choices=sorted(pe.CASES))
    ap.add_argument('--gamma', type=float, nargs='+', default=[0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001])
    ap.add_argument('--c', type=float, default=0.5, help='grid: h <= c sqrt(gamma)')
    ap.add_argument('--nmin', type=int, default=256)
    ap.add_argument('--hj-nx', type=int, default=256)
    ap.add_argument('--nsteps', type=int, default=400)
    ap.add_argument('--lattice', type=int, nargs=2, default=[64, 32])
    ap.add_argument('--tol', type=float, default=1e-13)
    ap.add_argument('--anderson', type=int, default=8)
    ap.add_argument('--refine', choices=['none', 'all'], default='all', help='also solve with finer grid (c/1.4) and 2x time steps')
    ap.add_argument('--fit-max-gamma', type=float, default=0.02)
    ap.add_argument('--out', default='computed/transport_end_plane')
    ap.add_argument('--tag', default='')
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    cases = a.case
    for name in cases:
        a.case = name
        run_case(a)

def run_case(a):
    import jax
    case = pe.CASES[a.case]
    os.makedirs(a.out, exist_ok=True)
    nlx, nly = a.lattice
    Xl = (np.arange(nlx) + 0.5) * pe.LX / nlx
    Yl = (np.arange(nly) + 0.5) * pe.LY / nly
    out = dict(case=a.case, psi_terms=case.terms, psi_check=case.check(), diameter=pe.DIAMETER, lattice=[nlx, nly], args=vars(a), runs=[])
    print(json.dumps(out['psi_check']), flush=True)
    gamma_list = sorted(a.gamma, reverse=True)
    warm = None
    warm_f = None
    for gamma in gamma_list:
        nx, ny = grid_n(gamma, a.c, a.nmin)
        t0 = time.time()
        g, sk, fm, (bx, by), tdyn = solve_one(case, gamma, nx, ny, a.nsteps, Xl, Yl, warm, a.tol, a.anderson, a.verbose, a.hj_nx)
        warm = (g, gamma, sk['w'])
        rec = dict(gamma=gamma, nx=nx, ny=ny, nsteps=a.nsteps, sinkhorn=dict(iters=sk['iters'], marg1=sk['marg1'], marg2=sk['marg2'], time=sk['time'], spectrum_tail_gamma_w=sk['tail_w'], residual=sk['res'], stalled_at_roundoff=sk['stalled'], max_abs_log_phi1=sk['max_abs_w']), hj_nx=fm['hj_nx'], hj_cons_t0=fm['hj_cons_t0'], hj_cons_t1=fm['hj_cons_t1'], t_dyn=tdyn, t_total=time.time() - t0)
        rec['flow'] = errors(case, Xl, Yl, fm['TX'], fm['TY'])
        rec['bary'] = errors(case, Xl, Yl, bx.ravel(), by.ravel())
        XP, YP = np.meshgrid(Xl, Yl)
        detJ = np.linalg.det(fm['J']).reshape(XP.shape)
        rec['jac_det_over_rho0_minus1'] = float(np.max(np.abs(detJ / case.rho0(XP, YP) - 1)))
        T0x, T0y = case.T0(XP, YP)
        if a.refine == 'all':
            nx2, ny2 = grid_n(gamma, a.c / 1.4, int(a.nmin * 1.4 + 31) // 32 * 32)
            g2, sk2, fm2, (bx2, by2), _ = solve_one(case, gamma, nx2, ny2, 2 * a.nsteps, Xl, Yl, (g, gamma, sk['w']), a.tol, a.anderson, a.verbose, int(a.hj_nx * 1.5) // 32 * 32)
            dflow = np.hypot(fm2['TX'] - fm['TX'], fm2['TY'] - fm['TY']).max() / pe.DIAMETER
            dbary = np.hypot(bx2 - bx, by2 - by).max() / pe.DIAMETER
            fm3 = pe.flow_map(g, sk['u'], sk['w'], gamma, XP.ravel(), YP.ravel(), np.log(case.rho0(*g.mesh())), nsteps=2 * a.nsteps, jac=False, hj_nx=a.hj_nx)
            dtime = np.hypot(fm3['TX'] - fm['TX'], fm3['TY'] - fm['TY']).max() / pe.DIAMETER
            rec['refine'] = dict(nx=nx2, ny=ny2, nsteps=2 * a.nsteps, hj_nx=fm2['hj_nx'], hj_cons_refined=[fm2['hj_cons_t0'], fm2['hj_cons_t1']], jac_refined=float(np.max(np.abs(np.linalg.det(fm2['J']).reshape(XP.shape) / case.rho0(XP, YP) - 1))), flow_change_grid_and_time=float(dflow), flow_change_time_only=float(dtime), bary_change_grid=float(dbary), flow_err_refined=errors(case, Xl, Yl, fm2['TX'], fm2['TY']), marg_refined=[sk2['marg1'], sk2['marg2']], iters_refined=sk2['iters'])
        out['runs'].append(rec)
        print(json.dumps({k: v for k, v in rec.items()}), flush=True)
        np.savez_compressed(os.path.join(a.out, f'fields_{a.case}{a.tag}_gamma{gamma:g}.npz'), Xl=Xl, Yl=Yl, TX=fm['TX'].reshape(XP.shape), TY=fm['TY'].reshape(XP.shape), BX=bx, BY=by, T0X=T0x, T0Y=T0y, detJ=detJ)
        with open(os.path.join(a.out, f'{a.case}{a.tag}.json'), 'w') as f:
            json.dump(out, f, indent=1)
    R = [r for r in out['runs'] if r['gamma'] <= a.fit_max_gamma]
    E = [r['gamma'] for r in R]
    out['fit'] = dict(flow_max=fit_rate(E, [r['flow']['max'] for r in R]), flow_l2=fit_rate(E, [r['flow']['l2_rho0'] for r in R]), bary_max=fit_rate(E, [r['bary']['max'] for r in R]), bary_l2=fit_rate(E, [r['bary']['l2_rho0'] for r in R]))
    allE = [r['gamma'] for r in out['runs']]
    fe = [r['flow']['max'] for r in out['runs']]
    out['local_slopes_flow_max'] = [float(np.log(fe[i + 1] / fe[i]) / np.log(allE[i + 1] / allE[i])) for i in range(len(fe) - 1)]
    be = [r['bary']['max'] for r in out['runs']]
    out['local_slopes_bary_max'] = [float(np.log(be[i + 1] / be[i]) / np.log(allE[i + 1] / allE[i])) for i in range(len(be) - 1)]
    below = [r['gamma'] for r in out['runs'] if r['flow']['max'] < 1e-06]
    out['largest_measured_gamma_flow_max_below_1e-6'] = max(below) if below else None
    with open(os.path.join(a.out, f'{a.case}{a.tag}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out['fit'], indent=1))
    print('local slopes flow', out['local_slopes_flow_max'])
    print('local slopes bary', out['local_slopes_bary_max'])
if __name__ == '__main__':
    main()
