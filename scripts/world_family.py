"""Compute a coarse flat world-map illustration from the included public inputs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bcart import Cartogram
from bcart.data import load_raw, masses
from bcart.grids import PlaneGrid


def polylines(coast_path):
    import shapefile
    coast = []
    with shapefile.Reader(str(coast_path)) as reader:
        for shape in reader.shapes():
            points = np.radians(np.asarray(shape.points, float)) + [np.pi, np.pi / 2]
            parts = list(shape.parts) + [len(points)]
            for a, b in zip(parts[:-1], parts[1:]):
                part = points[a:b]
                cuts = np.flatnonzero(np.abs(np.diff(part[:, 0])) > np.pi) + 1
                for segment in np.split(part, cuts):
                    coast.extend([segment, np.full((1, 2), np.nan)])
    graticule = []
    for lon in range(-180, 181, 15):
        lat = np.linspace(-90, 90, 721)
        graticule.extend([np.column_stack([np.full(len(lat), np.radians(lon) + np.pi), np.radians(lat) + np.pi / 2]),
                         np.full((1, 2), np.nan)])
    for lat in range(-90, 91, 15):
        lon = np.linspace(-180, 180, 1441)
        graticule.extend([np.column_stack([np.radians(lon) + np.pi, np.full(len(lon), np.radians(lat) + np.pi / 2)]),
                         np.full((1, 2), np.nan)])
    return np.concatenate(coast), np.concatenate(graticule)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'computed/world')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    p.add_argument('--coast', type=Path, default=ROOT / 'data/natural_earth/ne_50m_coastline.zip')
    p.add_argument('--factor', type=int, default=10, help='10 gives one-degree cells.')
    p.add_argument('--gamma-min', type=float, default=0.002)
    p.add_argument('--gamma-max', type=float, default=30)
    p.add_argument('--levels', type=int, default=40)
    a = p.parse_args()
    if a.factor < 1 or 1800 % a.factor or 3600 % a.factor:
        p.error('factor must divide 1800 and 3600.')
    if a.levels < 2 or not 0 < a.gamma_min < a.gamma_max:
        p.error('Choose at least two increasing, positive noise levels.')
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Choose an empty output directory to preserve existing computations.')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'LICENSE.md').write_text((ROOT / 'LICENSES/CC-BY-4.0.md').read_text())
    (a.out / 'ATTRIBUTION.md').write_text((ROOT / 'data/hero/ATTRIBUTION.md').read_text())
    grid = PlaneGrid(3600 // a.factor, 1800 // a.factor)
    mass, treatment = masses(grid, a.factor, directory=a.data_dir)
    population, land = load_raw(a.factor, directory=a.data_dir)
    cart = Cartogram(mass[::-1], 'plane')
    centres = np.column_stack(cart.centres())
    coast, grat = polylines(a.coast)
    good_coast, good_grat = np.isfinite(coast).all(axis=1), np.isfinite(grat).all(axis=1)
    inputs = np.concatenate([centres, coast[good_coast], grat[good_grat]])
    n, nc = len(centres), int(good_coast.sum())
    levels = np.geomspace(a.gamma_min, a.gamma_max, a.levels)
    metadata = dict(domain='plane', rectangle=[[0, float(2 * np.pi)], [0, float(np.pi)]],
                    grid=[grid.nx, grid.ny], coordinate_unit='radian', treatment=treatment,
                    interpretation='Coarse illustration; finite positive gamma is not the exact transport map.',
                    licence='CC-BY-4.0', attribution='GHS-POP R2023A, epoch 2020; Schiavina, Freire and MacManus, European Commission JRC (2023). Natural Earth is public domain.',
                    changes='One-degree aggregation, mean-land-density ocean fill, 1% land floor, normalisation and geometric mapping.', levels=[])
    for index in list(range(a.levels - 1, -1, -1)) + [a.levels]:
        gamma = None if index == a.levels else float(levels[index])
        r = cart.diffusion(points=(inputs[:, 0], inputs[:, 1])) if gamma is None else cart.bridge(gamma, points=(inputs[:, 0], inputs[:, 1]))
        mapped = np.column_stack([r.x, r.y])
        coast_map, grat_map = np.full_like(coast, np.nan), np.full_like(grat, np.nan)
        coast_map[good_coast] = mapped[n:n + nc]
        grat_map[good_grat] = mapped[n + nc:]
        deviation = float(np.max(np.abs(r.det[:n] / cart.rho0.ravel() - 1)))
        if not np.isfinite(mapped).all() or r.meta['min_density'] <= 0:
            raise RuntimeError(f'Invalid map at level {index}.')
        entry = dict(index=index, file=f'level_{index:02d}.npz', gamma=gamma,
                     kind=r.kind, solver=r.meta.get('solver', 'fv'),
                     largest_cell_density_deviation=deviation,
                     sinkhorn_error=r.meta.get('sinkhorn_err'), continuity=r.meta['continuity'],
                     minimum_density=r.meta['min_density'], tracking_refine=r.meta.get('tracking_refine', 1))
        np.savez_compressed(a.out / entry['file'], cell_input=centres, cell_mapped=mapped[:n],
                            population=population.ravel(), land_fraction=land.ravel(), mass=mass.ravel(),
                            coast_input=coast, coast_mapped=coast_map,
                            graticule_input=grat, graticule_mapped=grat_map,
                            licence=np.array('CC-BY-4.0'), attribution=np.array(metadata['attribution']),
                            changes=np.array(metadata['changes']))
        metadata['levels'].append(entry)
        metadata['levels'].sort(key=lambda row: row['index'])
        (a.out / 'levels.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        print(json.dumps(entry), flush=True)


if __name__ == '__main__':
    main()
