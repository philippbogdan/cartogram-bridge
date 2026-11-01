"""Aggregate the public GHS-POP raster onto the stated global grid."""
import argparse, sys, numpy as np, rasterio
from pathlib import Path
FINE = 120
K = 12
NX, NY = (3600, 1800)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raster', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--source-url', default='https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip')
    ap.add_argument('--meta', action='append', default=[], help='key=value metadata strings')
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ds = rasterio.open(a.raster)
    t = ds.transform
    px = 1.0 / FINE
    assert abs(t.a - px) < 1e-09 and abs(-t.e - px) < 1e-09 and (t.b == 0) and (t.d == 0)
    c0 = (t.c + 180.0) / px
    r0 = (90.0 - t.f) / px
    ci, ri = (int(round(c0)), int(round(r0)))
    print(f'size {ds.width}x{ds.height} col offset {c0:.4f}->{ci}, row offset {r0:.4f}->{ri}, nodata={ds.nodata}')
    assert abs(c0 - ci) < 0.1 and abs(r0 - ri) < 0.1
    assert ci + ds.width >= NX * K and ci <= 0 and (ri >= 0) and (ri + ds.height <= NY * K)
    nd_val = ds.nodata
    pop = np.zeros((NY, NX))
    nod = np.zeros((NY, NX))
    nodata_count = 0
    neg_count = 0
    neg_sum = 0.0
    S = 20
    for s in range(0, NY, S):
        f0, f1 = (s * K, (s + S) * K)
        fine_pop = np.zeros((f1 - f0, NX * K))
        fine_nd = np.ones((f1 - f0, NX * K), bool)
        r_lo, r_hi = (max(f0, ri), min(f1, ri + ds.height))
        if r_hi > r_lo:
            blk = ds.read(1, window=((r_lo - ri, r_hi - ri), (0, ds.width))).astype(np.float64)
            bad = np.isnan(blk)
            if nd_val is not None:
                bad |= blk == nd_val
            neg = ~bad & (blk < 0)
            neg_count += int(neg.sum())
            neg_sum += float(blk[neg].sum())
            bad |= neg
            nodata_count += int(bad.sum())
            v = np.where(bad, 0.0, blk)
            assert ci == -1 and ds.width == NX * K + 2
            rs = slice(r_lo - f0, r_hi - f0)
            sub = fine_pop[rs]
            subn = fine_nd[rs]
            sub[:] = v[:, 1:NX * K + 1]
            subn[:] = bad[:, 1:NX * K + 1]
            sub[:, -1] += v[:, 0]
            subn[:, -1] &= bad[:, 0]
            sub[:, 0] += v[:, -1]
            subn[:, 0] &= bad[:, -1]
        h = f1 - f0
        pop[s:s + S] = fine_pop.reshape(S, K, NX, K).sum(axis=(1, 3))
        nod[s:s + S] = fine_nd.reshape(S, K, NX, K).mean(axis=(1, 3))
    print(f'NoData/neg/NaN pixels within raster: {nodata_count}; negative pixels {neg_count} (sum {neg_sum})')
    i, j = np.unravel_index(np.argmax(pop), pop.shape)
    lat_n = 90 - i * 0.1
    lon_w = -180 + j * 0.1
    print(f'total population {pop.sum():.1f}')
    print(f'max cell {pop.max():.1f} at rows/cols ({i},{j}) lat {lat_n - 0.1:.1f}..{lat_n:.1f} lon {lon_w:.1f}..{lon_w + 0.1:.1f}')
    print(f'cells with pop>0: {(pop > 0).sum()}; cells with nodata_frac>0: {(nod > 0).sum()}; ==1: {(nod == 1).sum()}')
    meta = dict((m.split('=', 1) for m in a.meta))
    meta.update(licence='CC-BY-4.0', citation='Schiavina, Freire and MacManus (2023), GHS-POP R2023A, European Commission JRC. DOI: 10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE', source_raster=a.raster.split('/')[-1], source_url=a.source_url, grid='3600 lon x 1800 lat, 0.1 deg, row0 = 90N..89.9N, col0 = 180W..179.9W, rows N->S, cols W->E', processing='sum of 12x12 30-arcsec pixels; raster snapped to 1/120-deg lattice (col-1,row+108), extra edge columns wrapped across antimeridian; |lat|>89.1 padded NoData; NoData/NaN/negative = 0 pop')
    np.savez_compressed(a.out, pop=pop, nodata_frac=nod, **{k: np.array(v) for k, v in meta.items()})
if __name__ == '__main__':
    main()
