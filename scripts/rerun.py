"""List or execute the numerical experiments serially, writing separate results."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands(group, out):
    result = []
    if group in ('diffusion', 'all'):
        specifications = [
            ('sph_generic', 'sphere', 'sph_generic', 40, 64, 0.5, 32, 44, []),
            ('sph_zonal', 'sphere', 'sph_zonal', 40, 56, 0.5, 32, 44, []),
            ('pl_generic', 'plane', 'pl_generic', 48, 64, 2, 280, 44, []),
            ('pl_nogen_l1eff', 'plane', 'pl_nogen', 48, 64, 2, 100, 30, ['--l1', '1', '--proj-even-x']),
            ('pl_nogen_l1dom', 'plane', 'pl_nogen', 48, 0, 2, 160, 20, ['--proj-even-x']),
            ('pl_nogen_l1eff_noproj', 'plane', 'pl_nogen', 48, 0, 2, 100, 20, ['--l1', '1']),
        ]
        for name, domain, density, degree, check, low, high, count, extra in specifications:
            result.append(['scripts/diffusion_end_run.py', '--domain', domain, '--density', density,
                           '--L', str(degree), '--Lcheck', str(check), '--gamma-min', str(low),
                           '--gamma-max', str(high), '--ngamma', str(count), *extra,
                           '--out', str(out / 'diffusion_end' / (name + '.json'))])
    if group in ('precision', 'all'):
        for domain, density, gamma in [('sphere', 'sph_zonal', [2, 8, 16, 24, 32]),
                                       ('plane', 'pl_generic', [20, 60, 120, 200])]:
            result.append(['scripts/diffusion_end_mpcheck.py', '--domain', domain, '--density', density,
                           '--gamma', *map(str, gamma), '--out', str(out / 'diffusion_end' / ('mpcheck_' + density + '.json'))])
    if group in ('plane', 'all'):
        for case in ['smooth', 'mixed', 'large']:
            stored = json.loads((ROOT / 'results/transport_end_plane' / (case + '.json')).read_text())
            args = stored['args']
            cmd = ['scripts/transport_end_plane.py', '--case', case, '--gamma',
                   *[str(row['gamma']) for row in stored['runs']]]
            for key in ['c', 'nmin', 'hj_nx', 'nsteps', 'tol', 'anderson', 'refine', 'fit_max_gamma']:
                cmd += ['--' + key.replace('_', '-'), str(args[key])]
            cmd += ['--lattice', *map(str, args['lattice']), '--out', str(out / 'transport_end_plane')]
            result.append(cmd)
    if group in ('sphere', 'all'):
        for name in ['manufactured_main', 'manufactured_refine_grid', 'manufactured_refine_gh_steps', 'zonal_main']:
            stored = json.loads((ROOT / 'results/transport_end_sphere' / (name + '.json')).read_text())
            args = stored['args']
            cmd = ['scripts/transport_end_sphere.py', '--gamma', *[str(row['gamma']) for row in stored['results']]]
            for key in ['case', 'amp', 'kappa', 'nlat_min', 'n_gh', 'nsteps', 'neval', 'block', 'chunk', 'flow_max_gamma', 'tag']:
                cmd += ['--' + key, str(args[key])]
            cmd += ['--outdir', str(out / 'transport_end_sphere')]
            result.append(cmd)
        result.append(['scripts/check_mccann_sphere.py', '--out', str(out / 'transport_end_sphere/mccann_check.json')])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('group', choices=['diffusion', 'precision', 'plane', 'sphere', 'all'])
    p.add_argument('--out', type=Path, default=ROOT / 'computed')
    p.add_argument('--run', action='store_true', help='Execute the listed commands; otherwise only print them.')
    a = p.parse_args()
    out = a.out.resolve()
    evidence = (ROOT / 'results').resolve()
    if out == evidence or evidence in out.parents or out in evidence.parents:
        p.error('Use a separate output directory to preserve the included evidence.')
    if a.run:
        targets = {'diffusion': ['diffusion_end'], 'precision': ['diffusion_end'], 'plane': ['transport_end_plane'],
                   'sphere': ['transport_end_sphere'], 'all': ['diffusion_end', 'transport_end_plane', 'transport_end_sphere']}
        for folder in targets[a.group]:
            (out / folder).mkdir(parents=True, exist_ok=True)
    for command in commands(a.group, out):
        full = [sys.executable, *command]
        print(shlex.join(full), flush=True)
        if a.run:
            subprocess.run(full, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
