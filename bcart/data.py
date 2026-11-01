"""Population counts, land fractions and the planar ocean-fill convention."""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / 'data'


def load_raw(factor=10, directory=DATA):
    """Return population and land fraction with row zero at the south edge.

    Counts are summed. Land fractions retain spherical-area weighting during
    aggregation, including when the resulting cells are used on the flat map.
    """
    if not isinstance(factor, int) or factor < 1 or 1800 % factor or 3600 % factor:
        raise ValueError('factor must divide both 1800 and 3600.')
    directory = Path(directory)
    with np.load(directory / 'pop_0p1deg.npz', allow_pickle=False) as z:
        pop = z['pop'][::-1]
    with np.load(directory / 'landfrac_0p1deg.npz', allow_pickle=False) as z:
        land = z['land_frac'][::-1]
    if factor > 1:
        ny, nx = pop.shape[0] // factor, pop.shape[1] // factor
        shape = (ny, factor, nx, factor)
        pop = pop.reshape(shape).sum(axis=(1, 3))
        lat = np.linspace(-np.pi / 2, np.pi / 2, land.shape[0] + 1)
        weight = np.broadcast_to(np.diff(np.sin(lat))[:, None], land.shape)
        land = (land * weight).reshape(shape).sum(axis=(1, 3)) / weight.reshape(shape).sum(axis=(1, 3))
    return pop, land


def masses(grid, factor=10, floor=0.01, directory=DATA):
    """Fill ocean area at mean land density and add a floor over every land cell."""
    pop, land = load_raw(factor, directory)
    if grid.area.shape != pop.shape:
        raise ValueError('The grid and aggregated input shapes differ.')
    rho_land = pop.sum() / np.sum(land * grid.area)
    mass = pop + ((1 - land) + floor * land) * rho_land * grid.area
    return mass, dict(pop_total=float(pop.sum()), rho_land=float(rho_land), floor=floor)
