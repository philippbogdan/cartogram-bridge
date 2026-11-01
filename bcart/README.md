# Make a flat-map cartogram

Turn your own cell masses into a density-equalising flat map, choosing a bridge noise level or the diffusion endpoint through the same `Cartogram` interface. The input is a strictly positive cell-mass array, with row zero at the north edge. The map sends the reflecting rectangle $`[0,2\pi]\times[0,\pi]`$ to itself. Outputs use radians and flatten the grid with the south row first.

```python
import numpy as np
from bcart import Cartogram

x = (np.arange(32) + 0.5) * 2 * np.pi / 32
y = (np.arange(16) + 0.5) * np.pi / 16
X, Y = np.meshgrid(x, y)
masses_south_to_north = 1 + 0.2 * np.cos(X / 2) + 0.1 * np.cos(Y)
c = Cartogram(masses_south_to_north[::-1], "plane")
bridge = c.bridge(gamma=1.0)
diffusion = c.diffusion()
xy = np.column_stack([bridge.x, bridge.y])
density_error = bridge.measures()["equalisation_max_dev"]
```

Pass `points=(x, y)` to map other points in the same plane coordinates. Only finite points are accepted; split coastline separators before mapping. Default points are cell centres. The reported centre density deviation is $`\max|\det DT/\rho_0-1|`$, using the propagated determinant. It checks the discrete map's density constraint, not its distance from continuum optimal transport.

`solver="auto"` chooses a finite-volume walk at larger noise and a log-domain continuum Neumann kernel at smaller noise. Both paths use JAX in float64. The default streams the density path without retaining it. Check solver residuals and positive densities before using an output; changing a grid changes the approximation. No speed guarantee is supplied.

The [computation guide](../REPRODUCE.md) installs dependencies and lists the tests. `spectral.py` and `sb_spectral.py` implement the large-noise checks with NumPy and SciPy. `plane_entropic.py` and `sphere_entropic.py` implement the smooth manufactured transport checks. The public cartogram API is restricted to the flat map; the spherical modules here serve those controlled experiments.
