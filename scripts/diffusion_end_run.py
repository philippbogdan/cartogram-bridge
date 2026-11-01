"""Recompute spectral diffusion-end sweeps and their sampled time curves."""
import argparse, json, sys, time, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bcart.spectral import SphereSpec, PlaneSpec
from bcart import sb_spectral as sb

def make_dom(domain, L):
    return SphereSpec(L) if domain == 'sphere' else PlaneSpec(2 * L, L)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--domain', choices=['sphere', 'plane'], required=True)
    ap.add_argument('--density', required=True)
    ap.add_argument('--L', type=int, required=True, help='sphere: degree; plane: Ny (Nx = 2 Ny)')
    ap.add_argument('--Lcheck', type=int, default=0, help='second resolution for cross-check (0: none)')
    ap.add_argument('--gamma-min', type=float, required=True)
    ap.add_argument('--gamma-max', type=float, required=True)
    ap.add_argument('--ngamma', type=int, default=44)
    ap.add_argument('--l1', type=float, default=None, help='eigenvalue used in the time changes')
    ap.add_argument('--proj-even-x', action='store_true', help='plane: project onto cos(kx/2), k even')
    ap.add_argument('--psi-scan-every', type=int, default=3)
    ap.add_argument('--naive-at', type=float, nargs='*', default=[], help='gamma values for a naive-Sinkhorn comparison')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time()
    dom = make_dom(a.domain, a.L)
    l1 = dom.l1 if a.l1 is None else a.l1

    def proj_for(d):
        if not a.proj_even_x:
            return None
        return (np.arange(d.Nx)[:, None] % 2 == 0) * np.ones((1, d.Ny))
    rho = sb.make_density(dom, a.density)
    proj = proj_for(dom)
    br0 = sb.SpectralBridge(dom, rho, 1.0, proj=proj)
    G, Cl = sb.first_mode_G(dom, br0.R, l1)
    psi_opt = G / (2 + G)
    tails = dom.tail(br0.R).tolist()
    rho0n = br0.rho0
    info = dict(domain=a.domain, density=a.density, L=a.L, grid=list(rho.shape), l1_used=l1, l1_domain=dom.l1, l2_domain=dom.l2, G=G, psi_opt_pred=psi_opt, rho0_min=float(rho0n.min()), rho0_max=float(rho0n.max()), coef_tail=tails, proj_even_x=a.proj_even_x)
    if a.Lcheck:
        domc = make_dom(a.domain, a.Lcheck)
        rhoc = sb.make_density(domc, a.density)
        info['Lcheck'] = a.Lcheck
    specs = {'linear': ('linear', l1, 0.0), 'canonical': ('canonical', l1, 0.0), 'densmatch': ('densmatch', l1, 0.0), 'balanced': ('balanced', l1, psi_opt)}
    gammas = np.geomspace(a.gamma_min, a.gamma_max, a.ngamma)
    rows, curves = ([], {})
    naive_idx = {int(np.argmin(np.abs(np.log(gammas / en)))) for en in a.naive_at}
    for ie, gamma in enumerate(gammas):
        T = gamma / 2
        br = sb.SpectralBridge(dom, rho, gamma, proj=proj)
        ss = sb.s_grid(T)
        E = {k: np.array([np.nan] * len(ss) * 4).reshape(len(ss), 4) for k in specs}
        for i, s in enumerate(ss):
            r = sb.errors_multi(br, s, specs)
            for k in specs:
                E[k][i] = r[k]
        row = dict(gamma=float(gamma), iters=br.iters, conv=br.conv, res_eq1=br.res[0], res_eq2=br.res[1], res_eq2_rel_u=br.res2_rel)
        for k in specs:
            ea, er = (E[k][:, 0], E[k][:, 1])
            ok = np.isfinite(ea)
            i = int(np.nanargmax(ea))
            row[k] = dict(sup_area=float(np.nanmax(ea)), s_at_sup=float(ss[i]), tau_at_sup=float(T - ss[i]), sup_rho=float(np.nanmax(er)), E_T_area=float(ea[-1]), E_T_rho=float(er[-1]), E_0_area=float(ea[0]), E_last_finite_area=float(ea[ok][-1]), s_last_finite=float(ss[ok][-1]))
            curves[f'{k}_{ie}'] = E[k]
        curves[f's_{ie}'] = ss
        q = np.exp(-l1 * gamma / 2)
        row['pred_canonical_T'] = float(q * sb.predicted_E(dom, Cl, l1, 0.0, q))
        row['pred_balanced_sup'] = float(q * psi_opt)
        fe = {}
        for tau in [0.0, min(0.5 / l1, T)]:
            e, sig = sb.free_error(br, T - tau, l1)
            fe[f'{tau:.4g}'] = dict(E=e, sigma=sig)
        row['free_optimum'] = fe
        if ie % a.psi_scan_every == 0 or ie == len(gammas) - 1:
            sc = sb.s_grid(T, n_end=25, n_mid=15)
            psis = np.linspace(0.0, 0.8, 17)
            sups = []
            for p in psis:
                sp = {'b': ('balanced', l1, p)}
                sups.append(max((sb.errors_multi(br, s, sp)['b'][0] for s in sc)))
            j = int(np.argmin(sups))
            row['psi_scan'] = dict(psi=psis.tolist(), sup_area=sups, best_psi=float(psis[j]), best_sup=float(sups[j]), canonical_sup_coarse=float(sups[0]))
        if a.Lcheck:
            brc = sb.SpectralBridge(domc, rhoc, gamma, proj=proj_for(domc))
            idx = np.unique(np.concatenate([np.linspace(0, len(ss) - 1, 25).astype(int), [int(np.nanargmax(E[k][:, 0])) for k in specs]]))
            dmax, dabs = ({}, {})
            for k in ['linear', 'canonical', 'densmatch', 'balanced']:
                d = da = 0.0
                for i in idx:
                    rc = sb.errors_multi(brc, ss[i], {k: specs[k]})[k]
                    if np.isfinite(rc[0]) and np.isfinite(E[k][i, 0]):
                        da = max(da, abs(rc[0] - E[k][i, 0]))
                        if E[k][i, 0] > 1e-11:
                            d = max(d, abs(rc[0] - E[k][i, 0]) / E[k][i, 0])
                dmax[k], dabs[k] = (d, da)
            row['rescheck_maxrel'] = dmax
            row['rescheck_maxabs'] = dabs
        if ie in naive_idx:
            if True:
                bn = sb.SpectralBridge(dom, rho, gamma, naive=True, proj=proj)
                rn = sb.errors_multi(bn, T, {'canonical': specs['canonical']})['canonical']
                row['naive_canonical_E_T'] = rn[0]
        rows.append(row)
        c = row['canonical']
        print(f"gamma={gamma:8.3f} it={br.iters:3d} canon sup={c['sup_area']:.3e} at tau={c['tau_at_sup']:.2e} E(T)={c['E_T_area']:.3e} pred={row['pred_canonical_T']:.3e} lin sup={row['linear']['sup_area']:.4f} bal sup={row['balanced']['sup_area']:.3e} " + (f"res={max(row['rescheck_maxrel'].values()):.1e}/{max(row['rescheck_maxabs'].values()):.1e}" if a.Lcheck else ''), flush=True)
    out = dict(info=info, runtime_s=time.time() - t0, rows=rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(a.out.replace('.json', '_curves.npz'), gamma=gammas, **curves)
    print('wrote', a.out, f'{time.time() - t0:.1f}s')
if __name__ == '__main__':
    main()
