# Check the result

From this directory, with NumPy installed:

```sh
python scripts/reproduce.py
```

This command reads only included data. It recalculates map errors from point arrays, checks them against the JSON records, recalculates sampled time maxima, fits the reported exponents and decay rates, and compares the stored multiprecision errors. It does not run a bridge solver. To save the calculation, add `--write computed/headlines.json`.

Expected rounded outputs are plane exponents 1.999419, 1.993079 and 1.992858; sphere two-term exponent 2.018957 and plain exponent 1.968892; diffusion decay rates 0.999965 and 0.124994. The meaning of each fit and its uncertainty is in [results/README.md](results/README.md).

## Rebuild the supporting figures

Install the figure dependencies, extract the plot data from the included results, then render both figures:

```sh
python -m pip install -e '.[figures]'
python scripts/figure_data.py --out computed/figure-data
python scripts/render_figures.py --data computed/figure-data
```

The renderer writes PNG, SVG and `render-manifest.json` into `figures/`. Use `--out computed/figures` to keep the included exports, or `--only transport` or `--only clocks` to rebuild one figure. The manifest records input hashes, fit checks, display choices and output hashes; the [figure guide](docs/figures.md) explains the encodings and numerical limits.

## Render the included world maps

The compact family in `data/hero/` needs no solver rerun. Install the figure dependencies above and FFmpeg with libx264 support, then run:

```sh
python scripts/render_figures.py --hero data/hero --only hero --out computed/hero-figures
```

This writes `world-bridge.mp4`, `world-bridge-poster.png` and a manifest. The renderer fills each map pixel with the colour of its nearest mapped cell centre, using all 64,800 centres and a fixed population scale. Use `--ffmpeg /path/to/ffmpeg` if the encoder is not on PATH. Omitting `--out` replaces the included exports in `figures/`. With `--data computed/figure-data --hero data/hero` and no `--only`, the renderer builds all three visuals; repeated `--only` selects a subset. Other figure entries in an existing output manifest are retained.

The supplied movie used FFmpeg 8.1.1, libx264 with CRF 21 and the slow preset, H.264 High profile, yuv420p and limited-range BT.709 colour. It has 477 frames at 24 fps, lasts 19.875 seconds and uses DejaVu Sans. Different encoder or package versions can change output bytes. See [the map guide](docs/world-map.md) for the schedule, data precision and limits.

The inline GIF is a transcode of the MP4, not a separate map render. To reproduce the included 1000 by 500, 12 fps GIF from the included movie:

```sh
ffmpeg -i figures/world-bridge.mp4 -vf "fps=12,scale=1000:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" figures/world-bridge.gif
```

Use `computed/hero-figures/` in place of `figures/` for both paths to transcode a freshly rendered movie separately. The included GIF was made with FFmpeg 8.1.1 and contains 239 frames; GIF delays alternate between 80 and 90 ms to approximate 12 fps.

## Run the numerical solvers

Use Python 3.12 or later in a virtual environment. The geographic extras are needed only for the world illustration and for rebuilding input rasters.

```sh
python -m pip install -e '.[geography,figures]'
python scripts/smoke.py
```

The smoke check exercises nonuniform planar maps with both solvers, density preservation, a uniform-map identity and both diffusion clocks. Float64 JAX execution works on a CPU; a compatible single accelerator is optional. Compilation and full sweeps can take substantially longer than reading the stored evidence. The spherical solver uses blocked contractions when a dense matrix would exceed 16 million entries.

List the complete computation without executing it:

```sh
python scripts/rerun.py all
```

Run it serially into a separate directory, then apply exactly the same analysis:

```sh
python scripts/rerun.py all --out computed --run
python scripts/reproduce.py --results computed --write computed/headlines.json
```

Individual groups are `plane`, `sphere`, `diffusion` and `precision`. Settings come from the included experiment records. The rerun uses completed noise levels, not extra levels present only in requested settings. Analytic densities and exact reference maps are defined in `bcart/plane_entropic.py`, `bcart/sphere_entropic.py` and `bcart/sb_spectral.py`; these experiments need no geographic input. The rerun includes spatial and temporal refinement, multiprecision comparisons and the sampled spherical reference check. Full results can differ in their rounding and fit errors between platforms.

The packaged smoke test, stored-evidence calculation and bounded plane and sphere driver checks passed on Linux CPU with Python 3.12, JAX 0.11.1, NumPy 2.4.6, SciPy 1.18.0 and PyShp 3.1.6. The returned arrays were checked for finite coordinates, valid geometry and agreement with the recorded map errors. The headline values match the included results to every printed digit.

The driver checks used one noise level each and smaller evaluation sets than the stored experiments. They establish execution of the port, not reproduction of the full sweeps; those measurements predate this packaging. To repeat these bounded checks:

```sh
python scripts/transport_end_plane.py --case smooth --gamma 0.02 --nmin 64 --hj-nx 64 --nsteps 100 --lattice 16 8 --refine all --out computed/port-plane
python scripts/transport_end_sphere.py --case manufactured --gamma 0.04 --kappa 4 --nlat_min 24 --n_gh 12 --nsteps 8 --neval 6 --block 64 --chunk 32 --tag port_check --outdir computed/port-sphere
```

The plane's fit summary is `null` because a single noise level cannot determine a slope. The observed local analysis versions are recorded in [requirements-observed.txt](requirements-observed.txt); they are an environment record, not a tested installation lock.

## Rebuild the world illustration

The compact 0.1-degree population and land inputs are included, with their [provenance and treatment](data/README.md). The included 40-map, one-degree family and its separate diffusion endpoint can be recomputed with:

```sh
python scripts/world_family.py --out computed/world --factor 10 --levels 40 --gamma-min 0.002 --gamma-max 30
```

This writes mapped centres, population, land fraction, coastlines and graticules, plus solver diagnostics for each completed level. It also computes the diffusion endpoint. It does not produce an exact zero-noise map or validate the smooth-data rate for the world raster. The included family was computed in blocks of five; this single invocation uses the same noise grid but can have different warm starts and rounding. Its schema, diagnostics and limitations are in [docs/world-map.md](docs/world-map.md).

Pack the completed outputs and render them without replacing the included family:

```sh
python scripts/world_hero.py --source computed/world --out computed/hero
python scripts/render_figures.py --hero computed/hero --only hero --out computed/hero-figures
```

The packer stores unchanged inputs once in `common.npz`, converts mapped coordinates to float32 and records source hashes and rounding errors in `levels.json`. It also writes standalone attribution and licence notices. Passing `--out data/hero` to the packer replaces the included pack; use that only when intentionally updating it. The [pack guide](data/hero/README.md) gives the index and array mapping.

To regenerate the input arrays from public sources, reserve disk space for the much larger original raster, install the geographic dependencies, and download:

```sh
python scripts/fetch_inputs.py --out downloads
```

The population archive contains the GeoTIFF identified in [data/README.md](data/README.md). The land archive can be read directly by the shapefile reader. Rebuilding the arrays is explicit:

```sh
python -m zipfile -e downloads/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip downloads/population
mkdir -p computed
python scripts/prepare_population.py --raster downloads/population/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif --out computed/pop_0p1deg.npz
python scripts/prepare_landfrac.py --land-shp downloads/ne_10m_land.zip --out computed/landfrac_0p1deg.npz
python scripts/world_family.py --data-dir computed --coast downloads/ne_50m_coastline.zip --out computed/world-rebuilt
```

For these outputs, retain the attribution and processing notice in `data/README.md` and `LICENSES/CC-BY-4.0.md`. The GHS-POP CC BY 4.0 terms and Natural Earth public-domain terms were verified on 1 November 2026. Raw-input downloads and rebuilding have not been rerun for this distribution.
