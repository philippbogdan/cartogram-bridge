# Evidence for the two limits

The stored maps show a quadratic approach to optimal transport; the velocity curves show why matching diffusion needs a change of clock. Every reported fit can be recalculated from these arrays with [one command](../REPRODUCE.md). [headlines.json](headlines.json) is its generated output, not an independent measurement.

| Measurement | Recalculated value | Evidence |
|---|---|---|
| Plane flow exponent, smooth | 1.999419 ± 0.000189 | [smooth.json](transport_end_plane/smooth.json) and matching `fields_smooth_gamma*.npz` |
| Plane flow exponent, mixed | 1.993079 ± 0.002235 | [mixed.json](transport_end_plane/mixed.json) and matching fields |
| Plane flow exponent, large | 1.992858 ± 0.002685 | [large.json](transport_end_plane/large.json) and matching fields |
| Sphere flow exponent, two-term fit | 2.018957 ± 0.001821 | [manufactured_main.json](transport_end_sphere/manufactured_main.json), [point arrays](transport_end_sphere/manufactured_main.npz) |
| Sphere flow exponent, plain power law | 1.968892 ± 0.005854 | Same five points |
| Sphere static Fréchet exponent | 0.984404 ± 0.002039 | Same arrays, seven points |
| Sphere diffusion decay | 0.999965 ± 0.000013; predicted 1 | [sph_generic.json](diffusion_end/sph_generic.json), [time curves](diffusion_end/sph_generic_curves.npz) |
| Plane diffusion decay | 0.124994 ± 0.000002; predicted 1/8 | [pl_generic.json](diffusion_end/pl_generic.json), [time curves](diffusion_end/pl_generic_curves.npz) |

The ± values are regression standard errors. [Measurement definitions, fit masks and numerical limits](../docs/measurements.md) explain how to read the arrays; the [figure guide](../docs/figures.md) explains the plotted marks.
