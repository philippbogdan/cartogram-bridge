# Included world maps

Forty bridge maps at noise 0.002 to 30 and one separate diffusion map reproduce the world illustration without solving again. All use the same 360 by 180 flat grid.

| File | Contents |
|---|---|
| `common.npz` | Invariant `cell_input`, `population`, `land_fraction`, `mass`, `coast_input`, `graticule_input`, plus `licence`, `attribution`, `changes` strings |
| `level_00.npz` to `level_39.npz` | Bridge `cell_mapped`, `coast_mapped`, `graticule_mapped`, in increasing noise order |
| `level_40.npz` | The same mapped fields for diffusion; its noise is `null`, not zero |
| `levels.json` | Index-to-file mapping, noise, solver, diagnostics, source hashes and rounding errors |

Static numerical inputs remain exact float64 copies. Mapped coordinates are float32, with a maximum recorded change of $`2.385\times10^{-7}`$ radians, less than 0.00012 poster pixels. Diagnostics are retained from the original float64 computation; they were not recomputed from rounded coordinates and do not certify the rendered geometry.

Coordinates are plane radians. The 64,800 centres run west to east within each row and south to north between rows. Population is the original number of people in each input cell; `mass` includes artificial ocean fill and the land floor. NaN rows separate coastline and graticule polylines. Loading one map means combining its three mapped arrays with `common.npz`. The manifest's `source_sha256` values identify the original full-precision files, while the render manifest hashes the included packed files.

[Reproduction commands](../../REPRODUCE.md#render-the-included-world-maps) render these maps or rebuild the pack. [World-map details](../../docs/world-map.md) explain the computation and nearest-centre colour field used in the movie, inline GIF and poster. These population-derived arrays retain [CC BY 4.0](LICENSE.md); keep the standalone [attribution and changes](ATTRIBUTION.md) with copies. Natural Earth data are public domain.
