# Reading the supporting figures

The [transport figure](../figures/transport-rate.png) compares computed maps with known transport maps. The [clock figure](../figures/clocks.png) compares velocities along the bridge with diffusion velocities. Their inputs come from the included [results](../results/README.md); [rebuild commands](../REPRODUCE.md#rebuild-the-supporting-figures) extract those inputs and recreate the exports.

## Transport-rate encodings

All 18 noise levels for each planar case and all nine spherical levels remain visible. Filled markers identify points admitted to the stored fit; open markers identify excluded points. Thin lines join deterministic samples. Pale bands trace the fitted flow models over their admitted ranges, not uncertainty intervals. Dotted planar lines show the change under numerical refinement. The neutral reference segments are exactly $`E=\gamma^2`$, with scale factor one.

The panels share error limits and the same horizontal and vertical distance per decade. Only three spherical noise ticks are labelled to keep them legible at the display width. The planar cases use circles, squares and triangles as well as colour. The spherical flow uses circles and a solid line; the static Fréchet average uses diamonds and a dashed line. Seven static markers are filled, matching its stored fit, although no static fit curve is drawn.

The spherical flow's two-term fit includes a correction in noise; its exponent 2.019 is not the uncorrected log-log slope. The plain power-law fit to the same five points gives 1.969 over about 0.6 decades. The planar refinement exclusion, full fit equations and uncertainty definitions are in [the measurement guide](measurements.md#transport-measurements).

## Clock encodings and precision

Each panel uses 44 curves with 161 nonuniform time samples per curve, including both endpoints. The horizontal coordinate is bridge time $`t=2s/\gamma`$; the vertical coordinate is relative area-weighted $L^2$ velocity error. Colour identifies noise in spectral units, $`\lambda_1\gamma`$, with one logarithmic scale shared by the four panels. Lines join sampled times, and circles at $t=1$ mark endpoints at or above the display threshold. A line between samples does not establish the continuous-time bound in the derivation.

The export uses a common logarithmic axis from $10^{-15}$ to $10$. Samples at or above $`2\times10^{-15}`$ are drawn at their measured heights, with lines only between adjacent retained samples and dots for isolated ones. Smaller values, including stored zeros, are drawn as downward triangles at the threshold and their actual times; the triangle height indicates a category, not a measured error. There are 269 zeros in each spherical clock panel and 322 in each planar clock panel. Endpoints below the threshold use triangles instead of circles. These zeros are numerical outputs, not evidence of exact agreement.

Some early-time errors reach roughly $10^{-15}$ and then fall to much smaller values or zero. Treat values below $`2\times10^{-15}`$ as below a conservative display-resolution threshold: they do not establish accuracy at that scale. The threshold rounds up the largest observed absolute discrepancy, $`1.810\times10^{-15}`$, in the separate multiprecision comparisons. Those checks cover selected cases and noise levels, not every plotted sample, so this threshold is neither a certified error bound nor a proof that every value above it is resolved. No stored values, fit masks or fitted decay rates are changed by this interpretation. The [precision checks](measurements.md#diffusion-measurements) retain their exact scope.

## Files and notices

The PNG files are opaque RGB images, 3320 pixels wide with 300 dpi metadata; the intended display width is 830 pixels. SVG companions preserve glyph outlines. [render-manifest.json](../figures/render-manifest.json) records source hashes, fit checks, encodings and output hashes. DejaVu Sans glyphs retain their [font notice](../LICENSES/DejaVu.txt); the original supporting figures are CC0 1.0.
