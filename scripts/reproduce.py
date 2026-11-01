"""Recalculate the reported fits from included arrays and check their recorded errors."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def agree(actual, expected, name, rtol=1e-6, atol=2e-15):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
        raise ValueError(f"Stored evidence disagrees with its arrays: {name}")


def regression(gamma, error, correction=False, exponential=False):
    gamma, error = np.asarray(gamma), np.asarray(error)
    if np.any(gamma <= 0) or np.any(error <= 0):
        raise ValueError("Fits require positive finite noise levels and errors.")
    x = gamma if exponential else np.log(gamma)
    columns = [np.ones(len(x)), x]
    if correction:
        columns.append(gamma)
    A = np.column_stack(columns)
    y = np.log(error)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    residual = y - A @ coef
    covariance = (residual @ residual) / (len(x) - A.shape[1]) * np.linalg.inv(A.T @ A)
    ans = dict(exponent=float(-coef[1] if exponential else coef[1]),
               se=float(np.sqrt(covariance[1, 1])), prefactor=float(np.exp(coef[0])),
               n=len(x), gamma_range=[float(gamma.min()), float(gamma.max())],
               decades=float(np.log10(gamma.max() / gamma.min())))
    if correction:
        ans['linear_correction'] = float(coef[2])
    return ans


def cross(gamma, error, level, increasing, log_gamma):
    order = np.argsort(gamma)
    if not increasing:
        order = order[::-1]
    g, e = np.asarray(gamma)[order], np.asarray(error)[order]
    above = np.flatnonzero(e >= level)
    if not len(above) or above[-1] == len(e) - 1:
        return None
    i = above[-1]
    weight = (np.log(level) - np.log(e[i])) / (np.log(e[i + 1]) - np.log(e[i]))
    x = np.log(g) if log_gamma else g
    value = x[i] + weight * (x[i + 1] - x[i])
    return dict(interpolated=float(np.exp(value) if log_gamma else value),
                bracket_gamma=[float(g[i]), float(g[i + 1])],
                bracket_error=[float(e[i]), float(e[i + 1])],
                measured_below=float(g[i + 1]))


def geodesic(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def plane(results):
    out = {}
    for case in ['smooth', 'mixed', 'large']:
        directory = results / 'transport_end_plane'
        record = read(directory / f'{case}.json')
        g, errors, floor = [], [], []
        for row in record['runs']:
            gamma = row['gamma']
            with np.load(directory / f'fields_{case}_gamma{gamma:g}.npz', allow_pickle=False) as z:
                e = np.hypot(z['TX'] - z['T0X'], z['TY'] - z['T0Y']).max() / (math.pi * math.sqrt(5))
                b = np.hypot(z['BX'] - z['T0X'], z['BY'] - z['T0Y']).max() / (math.pi * math.sqrt(5))
            agree(e, row['flow']['max'], f'{case}: flow at {gamma}')
            agree(b, row['bary']['max'], f'{case}: barycentre at {gamma}')
            g.append(gamma)
            errors.append(float(e))
            floor.append(max(row['refine']['flow_change_grid_and_time'], row['refine']['flow_change_time_only']))
        g, errors, floor = map(np.asarray, (g, errors, floor))
        mask = (g <= 0.02) & (errors > 100 * floor)
        out[case] = dict(fit=regression(g[mask], errors[mask]), gamma=g.tolist(),
                         flow_max=errors.tolist(), refinement_change=floor.tolist(), fit_mask=mask.tolist())
    return out


def sphere(results):
    directory = results / 'transport_end_sphere'
    record = read(directory / 'manufactured_main.json')
    g, errors, bary = [], [], []
    with np.load(directory / 'manufactured_main.npz', allow_pickle=False) as z:
        for row in record['results']:
            if 'flow' not in row:
                continue
            gamma = row['gamma']
            e = geodesic(z[f'flow_{gamma:.6g}'], z['T0']).max() / math.pi
            b = geodesic(z[f'bary_fre_{gamma:.6g}'], z['T0']).max() / math.pi
            agree(e, row['flow']['max_over_pi'], f'sphere: flow at {gamma}')
            agree(b, row['bary_frechet']['max_over_pi'], f'sphere: barycentre at {gamma}')
            g.append(gamma)
            errors.append(float(e))
            bary.append(float(b))
        refinement = {}
        for tag in ['refine_grid', 'refine_gh_steps']:
            with np.load(directory / f'manufactured_{tag}.npz', allow_pickle=False) as other:
                refinement[tag] = {k: float(geodesic(z[k], other[k]).max() / math.pi)
                                   for k in other.files if k.startswith('flow_') and k in z}
    g, errors, bary = map(np.asarray, (g, errors, bary))
    mask = g <= 0.01
    order = np.argsort(g[mask])
    slopes = np.diff(np.log(errors[mask][order])) / np.diff(np.log(g[mask][order]))
    with np.load(directory / 'zonal_main.npz', allow_pickle=False) as z:
        zonal = {k: float(geodesic(z[k], z['T0']).max() / math.pi)
                 for k in z.files if k.startswith('flow_')}
    return dict(two_term_fit=regression(g[mask], errors[mask], correction=True),
                plain_fit=regression(g[mask], errors[mask]), local_slopes=slopes.tolist(),
                barycentric_fit=regression(g[g <= 0.02], bary[g <= 0.02]),
                crossing_1e_minus_6=cross(g, errors, 1e-6, increasing=False, log_gamma=True),
                gamma=g.tolist(), flow_max=errors.tolist(), bary_max=bary.tolist(),
                fit_mask=mask.tolist(), refinement=refinement, zonal_error= zonal)


def diffusion(results):
    out = {}
    directory = results / 'diffusion_end'
    for case in ['sph_generic', 'sph_zonal', 'pl_generic', 'pl_nogen_l1eff', 'pl_nogen_l1dom', 'pl_nogen_l1eff_noproj']:
        record = read(directory / f'{case}.json')
        g, canonical, natural, endpoint_factors = [], [], [], []
        with np.load(directory / f'{case}_curves.npz', allow_pickle=False) as z:
            for i, row in enumerate(record['rows']):
                gamma = row['gamma']
                agree(gamma, z['gamma'][i], f'{case}: noise coordinate')
                agree(z[f's_{i}'][0], 0, f'{case}: initial time')
                agree(z[f's_{i}'][-1], gamma / 2, f'{case}: terminal time')
                e = float(np.nanmax(z[f'canonical_{i}'][:, 0]))
                l = float(z[f'linear_{i}'][-1, 0])
                agree(e, row['canonical']['sup_area'], f'{case}: maximum error at {gamma}')
                agree(l, row['linear']['E_T_area'], f'{case}: terminal error at {gamma}')
                agree(z[f'canonical_{i}'][-1, 2], gamma, f'{case}: terminal diffusion time')
                factor = z[f'canonical_{i}'][-1, 3] * np.exp(-record['info']['l1_used'] * gamma / 2)
                agree(factor, 2, f'{case}: normalised terminal clock derivative')
                endpoint_factors.append(float(factor))
                g.append(gamma)
                canonical.append(e)
                natural.append(l)
        g, e = np.asarray(g), np.asarray(canonical)
        mask = (e > 1e-13) & (e < 1e-3) & np.isfinite(e)
        info = record['info']
        out[case] = dict(gamma=g.tolist(), canonical_max=e.tolist(), natural_terminal_error=natural,
                         lambda_clock=info['l1_used'], lambda_domain=info['l1_domain'],
                         lambda_next_distinct=info['l2_domain'], fit_mask=mask.tolist(),
                         normalised_terminal_clock_coefficient=float(np.median(endpoint_factors)),
                         decay_fit=regression(g[mask], e[mask], exponential=True) if mask.sum() >= 3 else None,
                         crossing_1e_minus_8=cross(g, e, 1e-8, increasing=True, log_gamma=False))
    absolute = []
    digits = []
    for case in ['pl_generic', 'sph_zonal']:
        for row in read(directory / f'mpcheck_{case}.json')['rows']:
            digits.append(row['mp_dps'])
            for clock in ['linear', 'canonical']:
                for norm in ['area', 'rho']:
                    absolute.append(abs(row[clock][f'mp_{norm}'] - row[clock][f'f64_{norm}']))
    out['multiprecision'] = dict(max_absolute_difference=max(absolute), digits_range=[min(digits), max(digits)])
    return out


def compute(results):
    d = diffusion(results)
    factor = d['sph_generic']['normalised_terminal_clock_coefficient']
    return dict(transport_plane=plane(results), transport_sphere=sphere(results), diffusion=d,
                terminal_potential_factor=factor,
                terminal_factor_meaning='The clock data recover the coefficient 2 by sigma_prime(S)*exp(-lambda_1*gamma/2). The exact potential identity and the asymptotic heat-velocity ratio are derived in docs/limits.md.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results', help='Evidence directory to read.')
    parser.add_argument('--write', type=Path, help='Also save the recalculated values as JSON.')
    args = parser.parse_args()
    output = compute(args.results)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    for name, value in output['transport_plane'].items():
        f = value['fit']
        print(f"Plane {name}: p = {f['exponent']:.9f} +/- {f['se']:.9f}, n={f['n']}")
    s = output['transport_sphere']
    for key in ['two_term_fit', 'plain_fit', 'barycentric_fit']:
        f = s[key]
        print(f"Sphere {key}: p = {f['exponent']:.9f} +/- {f['se']:.9f}, n={f['n']}")
    print(f"Sphere fitted range: {s['plain_fit']['decades']:.6f} decades")
    for name in ['sph_generic', 'pl_generic']:
        d = output['diffusion'][name]
        f = d['decay_fit']
        prediction = min(d['lambda_domain'], d['lambda_next_distinct'] - d['lambda_domain']) / 2
        print(f"{name}: decay = {f['exponent']:.9f} +/- {f['se']:.9f}; predicted {prediction:g}")
        c = d['crossing_1e_minus_8']
        print(f"  Error 1e-8 crossing: interpolated gamma={c['interpolated']:.9g}; sampled below at {c['measured_below']:.9g}")
        print(f"  Natural-clock terminal relative error at largest gamma: {d['natural_terminal_error'][-1]:.12g}")
    print(f"Terminal coefficient from the clock data: {output['terminal_potential_factor']:g}; the heat-velocity ratio approaches 2 at large gamma.")
    print(f"Multiprecision maximum absolute difference: {output['diffusion']['multiprecision']['max_absolute_difference']:.9g}")
    print('Stored map errors and time-curve maxima agree with the included arrays.')


if __name__ == '__main__':
    main()
