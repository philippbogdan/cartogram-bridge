"""Download the public rasters and polygons needed to rebuild the geographic inputs."""
import argparse
from pathlib import Path
import urllib.request

INPUTS = {
    'population': ('GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip',
                   'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip'),
    'land': ('ne_10m_land.zip', 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip'),
    'coast': ('ne_50m_coastline.zip', 'https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('downloads'))
    p.add_argument('--only', choices=sorted(INPUTS), nargs='+', default=list(INPUTS))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.only:
        filename, url = INPUTS[name]
        dest = a.out / filename
        if dest.exists():
            raise FileExistsError(f'Preserving existing input: {dest}')
        temporary = dest.with_suffix(dest.suffix + '.partial')
        print(url, flush=True)
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open('xb') as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        temporary.rename(dest)


if __name__ == '__main__':
    main()
