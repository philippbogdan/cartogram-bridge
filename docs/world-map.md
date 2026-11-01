# The coarse world-map illustration

The included [map pack](../data/hero/README.md) contains 40 bridge maps at noise 0.002 to 30 and a separate diffusion map on a one-degree flat grid. The [movie](../figures/world-bridge.mp4), [inline GIF](../figures/world-bridge.gif) and [poster](../figures/world-bridge-poster.png) show a nearest-centre population colour field with mapped coastlines and graticule. The smooth test problems in [results/](../results/README.md) supply the measured endpoint rates.

## Calculation and output

The domain is the plate carrée rectangle $`[0,2\pi]\times[0,\pi]`$, where longitude and latitude become rectangular coordinates: $`x=\mathrm{longitude}+\pi`$ and $`y=\mathrm{latitude}+\pi/2`$, in radians. All four edges are reflecting walls, including the two longitude edges. This is a flat-map calculation.

`scripts/world_family.py` aggregates the included inputs to 360 by 180 cell masses and calls `bcart.Cartogram(masses, "plane")`. It passes masses with the north row first. For each positive noise level, the producer maps the cell centres, Natural Earth 50m coastlines and a 15-degree graticule through the same map. It computes a separate Gastner-Newman diffusion map as well.

From the repository root, the computation can be rerun with:

```sh
python scripts/world_family.py --out computed/world --factor 10 --levels 40 --gamma-min 0.002 --gamma-max 30
```

This uses the included family's noise grid, `np.geomspace(0.002, 30, 40)`, and a separate diffusion endpoint. Installation and rebuilding the geographic inputs are covered in [REPRODUCE.md](../REPRODUCE.md#rebuild-the-world-illustration).

The producer writes each completed level to `level_NN.npz` and updates `levels.json` after it finishes. A partial manifest records partial progress. The included family has all 41 unique indices and files: 0 to 39 are bridge maps, and 40 is diffusion. Indices 0 to 19 use the log-domain continuum Neumann kernel, at noise 0.002 to 0.2165389; indices 20 to 39 use the finite-volume walk, at 0.2770864 to 30. Diffusion uses the finite-volume solver.

The family was computed in consecutive blocks of five levels. Within each invocation, noise decreases and the first log-domain solve starts cold; warm starts do not cross block boundaries. The resulting maps have been compared at every adjacent pair, including block and solver joins. This records the observed agreement, not proof that cold starts yield an identical continuum solution.

The full producer files contain original and mapped centres, population, land fractions, filled masses, coastlines and graticules. The included compact pack stores invariant arrays once in `common.npz`, unchanged in float64, and the three mapped coordinate arrays in each `level_NN.npz` as float32. The maximum recorded rounding is less than $`2.385\times10^{-7}`$ radians, or 0.00012 pixels on the poster. `levels.json` retains the float64 solver diagnostics, source hashes and rounding errors; [the pack guide](../data/hero/README.md) gives the schema. Coordinates are radians, cell rows run south to north, and NaN rows separate polylines.

## Source raster preparation

Both raster arrays have shape `(1800, 3600)`. Row zero spans 90°N to 89.9°N; column zero spans 180°W to 179.9°W. Rows run north to south and columns west to east. Loading with `bcart.data.load_raw` reverses the rows for internal plane coordinates.

Changes: the 30-arcsecond raster is snapped to its nearest 1/120-degree lattice, then summed in 12 by 12 blocks in float64. Source column `c` maps to fine column `c-1`, and source row `r` to fine row `r+108`. The two overhanging longitude columns wrap across the antimeridian. Rows outside the source's coverage near the poles are padded with zero population and missing-data fraction one. Negative, NaN or declared missing values contribute zero population. Ocean is zero in the source raster, not a missing-data marker; `nodata_frac` must not be used as an ocean mask.

The Natural Earth 10m land polygons were rasterised at 1/120 degree using pixel centres and averaged in 12 by 12 blocks. The 50m coastline is separate.

## Data treatment

The population input is GHS-POP R2023A, epoch 2020. Its 0.1-degree cells are summed in 10 by 10 blocks. Natural Earth land fractions are averaged using spherical area during aggregation, matching the supplied data preparation, even though the subsequent map uses planar area. With coarse-cell population $`p_i`$, land fraction $`l_i`$ and planar cell area $`A_i`$, set

```math
\bar\rho_L=\frac{\sum_i p_i}{\sum_i l_iA_i},\qquad
m_i=p_i+\bar\rho_L A_i[(1-l_i)+0.01l_i].
```

Ocean area receives mean land density. The floor is added over **all** land area, including populated cells; zero-population land receives only that floor. These masses are normalised to a density of mean one before solving. Raw population counts remain separate from the artificial fill. The preceding raster alignment and aggregation are recorded above; source attribution remains in the [input provenance](../data/README.md).

## What the diagnostics mean

Automatic solver selection switches between two discrete approximations to the reference process. The finite-volume walk evolves mass between grid cells; the log-domain method evaluates the continuum heat kernel with reflecting walls. Their difference at fixed resolution can affect the map independently of the solver residual.

The included records give these ranges over all 41 maps:

| Diagnostic | Minimum | Maximum |
|---|---:|---:|
| Largest cell density deviation | $`1.0436\times10^{-14}`$ | $`3.5528\times10^{-14}`$ |
| Continuity residual | $`1.8126\times10^{-14}`$ | $`4.4676\times10^{-14}`$ |
| Minimum path density, mean-one units | 0.00983335 | 0.00996705 |
| Sinkhorn residual, bridge maps only | $`2.2204\times10^{-16}`$ | $`9.2593\times10^{-13}`$ |

`tracking_refine` is 1 throughout. The diffusion Sinkhorn field is `null` because that solve has no Sinkhorn iteration. The largest cell density deviation is $`\max_i|\det DT(x_i)/\rho_0(x_i)-1|`$ at the original cell centres. The determinant is propagated through the flow; it is not the area of a polygon drawn between those centres. Small residuals check the discrete calculation, not continuum-map accuracy, injectivity of a reconstructed mesh or distortion rankings.

Adjacent displacement is measured between corresponding coordinates and divided by the rectangle diameter $`\pi\sqrt{5}`$. At the solver join, indices 19 to 20, the 95th-percentile centre displacement is $`5.41\times10^{-4}`$, compared with $`4.42\times10^{-4}`$ before and $`3.79\times10^{-4}`$ after. The join has extra cell-scale variation, although the paired previews look similar at 830 pixels. Local maxima also rise at block joins 14 to 15 and 24 to 25; the coastline has an isolated maximum at 33 to 34. Their 95th percentiles follow the surrounding trend. These observations do not isolate cold starts as the cause. The complete comparison is in [the render manifest](../figures/render-manifest.json).

The largest small-noise changes occur in near-empty land at the 1% floor. The one-degree grid at noise 0.002 has not reached a demonstrated transport asymptotic regime. Above roughly 0.3 the maps change little at display scale; noise 30 and the separate diffusion map have centre displacement $`1.06\times10^{-6}`$ at the 95th percentile and $`2.21\times10^{-5}`$ at maximum. This is an endpoint comparison, not a continuum error estimate.

Straight triangles between neighbouring mapped centres are not a valid transported mesh at this resolution. Splitting each regular input quad along its southwest-to-northeast diagonal gives 3,650 negative signed areas out of 128,522 triangles at noise 0.002, in both the original float64 and packed coordinates. Curved flow lines can differ from straight chords, so this test alone does not establish a folded continuous flow; it does rule out treating that straight mesh as faithful cell boundaries. The nearest-centre reconstruction is described below.

## Rendering

Each map pixel takes the original population category and colour of its closest mapped centre in flat coordinates. All 64,800 centres participate, including ocean samples. The lookup is evaluated at pixel centres for every frame, using squared Euclidean distance; equal-distance ties use the lowest original row-major index. The field is not blurred or smoothed. Graticules and pale coastlines are drawn above it, with a dark keyline beneath the coastlines. These display regions are not transported cell polygons, their areas do not measure population, and the reconstruction does not establish mass conservation.

Counts from one to the global maximum of 38,536,875.913529 people share one fixed logarithmic scale. The inferno palette starts at position 0.4549, the lowest sampled colour with at least 3:1 contrast against the background, and ends at its upper endpoint. Positive counts below one person use purple `#4E4272`, zero-population land uses grey `#33363B`, and zero-population ocean uses background `#0B0C0E`. The category counts are 14,936, 530, 10,093 and 39,241 respectively. The 122 positive-population centres with zero land fraction keep their population colours. Colour represents original people per input cell, never artificial ocean fill or the land floor.

The [render manifest](../figures/render-manifest.json) records the lookup and sampling checks. Brute-force comparison over 86,000 sampled pixels found no mismatches; no exact ties or coincident centres were recorded. At movie resolution, up to 5,776 centres in a bridge frame are nearest to no pixel, including up to 1,228 centres with at least one person. The poster omits 2,973 centres through this sampling, while the geographic frames represent every centre. These are limits of raster display resolution; all centres remain in every lookup. Near the top centre, short Kara Sea coastline traces from roughly 78 to 79°E and 72 to 73°N lie inside the populated fill. That local overlap is retained and illustrates why the display cannot certify valid transported cell boundaries.

The MP4 has 477 frames at 24 fps, lasting 19.875 seconds. It holds geography for one second, cuts to bridge index 0 at noise 0.002 for two seconds, then interpolates coordinates between neighbouring computed bridges. Ordinary intervals below 0.3 use 12 frames; those above use three. The solver join uses half-second holds on either side of a cut. Noise 30 gets a further one-second hold, then a cut leads to three seconds of the separate diffusion map. Every bridge level appears; gamma labels interpolate logarithmically. No motion is magnified. The poster uses computed index 0 without parameter interpolation.

This sequence varies noise, not bridge time. Intermediate display frames are not extra solved maps. Geography is an unmapped reference, and the smallest positive-noise map is not an exact optimal-transport map. The one-degree illustration does not validate the smooth-data rate or complete a 0.1-degree world-family study.

The MP4 is H.264 High, 1920 by 960, with a 1728 by 864 map rectangle; the poster is 3320 by 1660, with 300 dpi metadata. The inline GIF is a 1000 by 500 transcode of the MP4 at approximately 12 fps, with 239 frames and a 192-colour palette. It samples the movie's existing frames; it does not render additional maps. DejaVu Sans and encoder details are recorded in [THIRD_PARTY.md](../THIRD_PARTY.md), and the render and transcode commands are in [REPRODUCE.md](../REPRODUCE.md#render-the-included-world-maps). Population-derived arrays and visuals retain [CC BY 4.0](../LICENSES/CC-BY-4.0.md), with standalone [attribution and changes](../data/hero/ATTRIBUTION.md). Natural Earth is public domain.
