# Geographic inputs

Build the coarse world illustration from compact population counts, land fractions and coastline geometry, without downloading the original global raster. These geographic inputs do not enter the transport-rate or diffusion-clock fits.

| File | Content | Terms |
|---|---|---|
| `pop_0p1deg.npz` | `pop`, people per 0.1-degree cell; `nodata_frac`; provenance strings | CC BY 4.0 |
| `landfrac_0p1deg.npz` | `land_frac`, fraction covered by Natural Earth 10m land | Natural Earth public domain |
| `natural_earth/ne_50m_coastline.zip` | Coastline geometry for the illustration | Natural Earth public domain |
| [hero/](hero/README.md) | 40 bridge maps and diffusion, with shared inputs and diagnostics | CC BY 4.0 |

Population source: Schiavina, M., Freire, S. and MacManus, K. (2023), *GHS-POP R2023A: GHS population grid multitemporal (1975-2030)*, European Commission, Joint Research Centre. [Dataset DOI](https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE), [source archive](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The epoch is 2020, WGS84, 30 arcseconds, V1.0. The JRC copyright notice, verified on 1 November 2026, identifies European Union copyright and permits reuse under CC BY 4.0 with attribution and changes indicated.

Changes: the source raster was aligned and summed onto 0.1-degree cells, with missing population values treated as zero. [Exact raster alignment and aggregation](../docs/world-map.md#source-raster-preparation) records the processing; `nodata_frac` is not an ocean mask.

Land is from [Natural Earth](https://www.naturalearthdata.com/about/terms-of-use/), whose terms were verified on 1 November 2026: its raster and vector data are public domain. The 10m land polygons were rasterised at 1/120 degree using pixel centres, then averaged in 12 by 12 blocks. The 50m coastline is bundled separately. The source 10m land archive is downloaded only when rebuilding inputs.

Further changes for the illustration are the one-degree aggregation, ocean fill, land floor, normalisation and geometric mapping described in [world-map.md](../docs/world-map.md). Keep this attribution and the [licence notice](../LICENSES/CC-BY-4.0.md) with all population-derived outputs. No endorsement by the data providers is implied.
