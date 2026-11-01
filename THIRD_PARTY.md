# Third-party material

The population inputs and their transformations carry the [GHS-POP attribution](LICENSES/CC-BY-4.0.md). Natural Earth land and coastline data are [public domain](https://www.naturalearthdata.com/about/terms-of-use/). No code or result numbers are imported from the historical-cartogram repository; it is cited as the preceding transport construction.

The GHS-POP copyright notice and Natural Earth terms were verified online on 1 November 2026. GHS-POP permits reuse under CC BY 4.0 with appropriate credit and an indication of changes. Natural Earth's raster and vector data are public domain.

The dependencies below are installed separately, not bundled. Their own licence notices continue to apply. The package terms were checked against installed metadata on 1 November 2026; Rasterio, Pillow and setuptools were also checked against their installed licence text.

| Dependency | Terms | Use |
|---|---|---|
| NumPy | BSD-3-Clause, with 0BSD, MIT, Zlib and CC0-1.0 components | Arrays |
| SciPy | BSD | Transforms and numerical integration |
| JAX, JAXlib | Apache-2.0 | Numerical kernels |
| mpmath | BSD | Multiprecision reference |
| threadpoolctl | BSD-3-Clause | Numerical thread limits |
| PyShp | MIT | Coastlines and land geometry |
| Rasterio | BSD-3-Clause | Source raster preparation |
| Matplotlib | PSF-based Matplotlib licence | Figures and movie frames |
| Pillow | MIT-CMU | Image handling |
| setuptools | MIT | Package build |

The sphere integrator reads DOP853 coefficients from SciPy at runtime; they are not copied into this tree. The source distribution includes no dependency binaries. Rasterio and Pillow installations can contain libraries with separate notices, retained by those installations. Importing these packages does not change the CC0 dedication of this repository's own code.

The supporting figures and world movie use DejaVu Sans, including its mathematical glyphs. SVG files preserve glyph outlines as paths; PNG files and video frames contain rasterised text. The unmodified [DejaVu notice](LICENSES/DejaVu.txt), read from the installed Matplotlib distribution, retains the Bitstream Vera and Arev terms and the public-domain status of DejaVu changes. These third-party glyph rights remain separate from the CC0 dedication of the original figures.

The included movie was encoded with FFmpeg 8.1.1 and libx264, using CRF 21, the slow preset, H.264 High profile, yuv420p and limited-range BT.709 colour. The inline GIF was transcoded from that MP4 with FFmpeg 8.1.1; its command is in [REPRODUCE.md](REPRODUCE.md#render-the-included-world-maps). That FFmpeg build enables GPL and version-3 components; libx264 is GPL-2.0-or-later. These are encoder software terms, not a licence on the encoded frames: the population-derived movie, GIF and poster retain [CC BY 4.0](LICENSES/CC-BY-4.0.md). No font or encoder binary is bundled. The encoder is identified for reproducibility; its software licence does not require an additional notice inside the video.
