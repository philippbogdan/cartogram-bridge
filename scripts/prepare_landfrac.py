"""Rasterise Natural Earth land polygons onto the documented population grid."""
import argparse

import numpy as np
import rasterio.features
from rasterio.transform import from_origin
import shapefile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--land-shp', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    with shapefile.Reader(a.land_shp) as reader:
        geometries = [shape.__geo_interface__ for shape in reader.shapes()]
    land = np.zeros((1800, 3600))
    for row in range(0, 1800, 20):
        transform = from_origin(-180, 90 - row * 0.1, 1 / 120, 1 / 120)
        mask = rasterio.features.rasterize([(g, 1) for g in geometries], out_shape=(240, 43200),
                                          transform=transform, fill=0, all_touched=False, dtype='uint8')
        land[row:row + 20] = mask.reshape(20, 12, 3600, 12).mean(axis=(1, 3))
    np.savez_compressed(a.out, land_frac=land,
                        source=np.array('Natural Earth 10m land, public domain.'),
                        grid=np.array('0.1 degree; rows north to south, columns west to east.'),
                        processing=np.array('Pixel-centre rasterisation at 1/120 degree, averaged in 12 by 12 blocks.'))


if __name__ == '__main__':
    main()
