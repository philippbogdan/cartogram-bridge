# Measurement definitions and limits

The [evidence table](../results/README.md) lists the recalculated values and their source files. The static sphere comparison uses a **Fréchet barycentre**, the point minimising the average squared geodesic distance to the possible destinations of each source point; the flow map follows the bridge's net mass velocity.

All ± values are ordinary least-squares standard errors conditional on the chosen fit model and range, not confidence in a theorem or an estimate of discretisation error. Map errors are maxima over evaluation points, divided by domain diameter. These smooth manufactured examples support the quadratic rate derived under hypothesis (H); they do not establish that hypothesis for arbitrary population data. The [derivation](limits.md) states the assumptions and distinguishes a uniform-in-time bound from sampled numerical evidence.

## Transport measurements

Plane errors are maxima over a fixed 64 by 32 lattice, divided by $`\pi\sqrt5`$. Fits use $`\gamma\le0.02`$ and retain errors greater than 100 times the larger change under grid-plus-time and time-only refinement. This leaves 10, 11 and 11 points. The smallest retained noise levels are 0.0003, 0.0002 and 0.0002 respectively. Every excluded point remains in the data.

Sphere errors are geodesic distances divided by $`\pi`$, evaluated at 1,152 points. The flow fit uses $`0.0025\le\gamma\le0.01`$: five points spanning 0.60206 decades. The two-term model is $`\log E=\log C+p\log\gamma+a\gamma`$. Consecutive local slopes range from 1.938761 to 1.989470. The static-map fit uses $`\gamma\le0.02`$.

The sphere's $10^{-6}$ crossing at $`\gamma=0.00545884`$ is **log-log interpolated**. The adjacent observations are above at 0.007 and below at 0.005. Refinement arrays check the flow at 0.005, not at every small noise level. The `args.gamma` lists retain requested levels; only entries in `results` were completed. The zonal case is a separate symmetry check, whose flow map is independent of noise and therefore does not measure a quadratic rate.

Plane NPZ files contain lattice axes `Xl`, `Yl`, transport coordinates `T0X`, `T0Y`, bridge coordinates `TX`, `TY`, barycentric coordinates `BX`, `BY` and `detJ`. Sphere NPZ files contain unit vectors `X`, `T0`, and `flow_*`, `bary_ext_*`, `bary_fre_*` keyed by noise level.

## Diffusion measurements

The plotted error is relative $L^2$ velocity error with area measure. Time maxima use the stored nonuniform sample sets, including both endpoints. The uniform-in-time statement belongs to the derivation; a sampled maximum alone cannot establish it. Fits use $`10^{-13}<\max_s E<10^{-3}`$ and regress $`\log E`$ against $`\gamma`$.

The $10^{-8}$ crossings are **log-linear interpolated**: 17.2642 on the sphere and 141.321 on the plane. The next sampled levels below the threshold are 17.9112 and 157.619. The natural-clock terminal relative error approaches one from above in the resolved large-noise samples: approximately $`1+4.2\times10^{-10}`$ at $`\gamma=11.0434`$ on the sphere and $`1+3.8\times10^{-9}`$ at $`\gamma=79.0949`$ on the plane. It rounds to one at all stored levels from 19.7301 and 157.6189 respectively.

In each `*_curves.npz`, `gamma` is the sweep, `s_i` is heat time, and `linear_i` or `canonical_i` has columns `[E_area, E_rho, sigma, dsigma_ds]`. `E_rho` uses diffusion-density weighting. JSON `null` denotes undefined values in auxiliary clock comparisons; it never means zero. Curve files retain only the two clocks shown.

The independent [sphere](../results/diffusion_end/mpcheck_sph_zonal.json) and [plane](../results/diffusion_end/mpcheck_pl_generic.json) multiprecision calculations differ from the stored float64 errors by at most $`1.810\times10^{-15}`$ in absolute error across both clocks and norms. Precision varies with noise; this is not a fixed-30-digit computation or a relative accuracy bound.

The non-generic plane files retain the symmetry counterexample, its corrected active-eigenvalue clock and the effect of leaving the symmetry unprojected. No world-population comparison is used in these fits.
