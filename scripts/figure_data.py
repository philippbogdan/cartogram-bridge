"""Export compact, plot-free evidence for the two supporting figures."""
import argparse
import json
from pathlib import Path
import shutil

from reproduce import compute, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    numbers = compute(ROOT / 'results')
    (a.out / 'transport.json').write_text(json.dumps(dict(plane=numbers['transport_plane'],
                                                        sphere=numbers['transport_sphere']), indent=2) + '\n')
    (a.out / 'clocks.json').write_text(json.dumps({k: numbers['diffusion'][k] for k in ['sph_generic', 'pl_generic']}, indent=2) + '\n')
    for name in ['sph_generic', 'pl_generic']:
        shutil.copyfile(ROOT / f'results/diffusion_end/{name}_curves.npz', a.out / f'{name}_curves.npz')
    print('Exported transport fits, error samples and the two complete clock sweeps.')


if __name__ == '__main__':
    main()
