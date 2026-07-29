"""
add_population_worldpop.py
=============================
Joins WorldPop population into a state's stats.json. WorldPop publishes one
raster per country (not per state), so this downloads the national Nigeria
raster once and reuses it for every state you add - only the zonal-stats
step repeats per state.

Usage:
    python add_population_worldpop.py lagos
    python add_population_worldpop.py ogun

Requires:
    pip install rasterio requests shapely
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.mask import mask
from shapely.geometry import shape, mapping

APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
RASTER_PATH = APP_DIR / "nga_ppp_2020_UNadj.tif"  # shared across all states

WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/NGA/"
    "nga_ppp_2020_UNadj.tif"
)


def download_worldpop_raster():
    if RASTER_PATH.exists():
        print(f"Using cached {RASTER_PATH}")
        return
    print(f"Downloading WorldPop Nigeria raster (national, shared across all states) ...")
    with requests.get(WORLDPOP_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(RASTER_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print("Download complete.")


def zonal_population_sum(src, geom):
    try:
        out_image, _ = mask(src, [mapping(geom)], crop=True, nodata=0)
    except ValueError:
        return 0.0
    data = out_image[0]
    data = np.where(data < 0, 0, data)
    return float(data.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_slug", help="Slug from data/states.json, e.g. 'lagos'")
    args = parser.parse_args()

    state_dir = DATA_DIR / args.state_slug
    stats_path = state_dir / "stats.json"
    lga_path = state_dir / "lgas.geojson"
    if not stats_path.exists():
        raise SystemExit(f"{stats_path} missing - run build_state_flood_stats.py '{args.state_slug}' first")

    download_worldpop_raster()

    stats = json.loads(stats_path.read_text())
    geoms = {f["properties"]["name"]: shape(f["geometry"]) for f in json.loads(lga_path.read_text())["features"]}

    with rasterio.open(RASTER_PATH) as src:
        for row in stats:
            geom = geoms.get(row["name"])
            if geom is None:
                continue
            total_pop = zonal_population_sum(src, geom)
            row["population_estimate"] = round(total_pop)

            months_flooded = row["recurrence_months"]
            exposed_share = min(0.85, months_flooded / 40)
            row["population_exposed_pct"] = round(exposed_share * 100, 1)
            row["population_exposed_est"] = round(total_pop * exposed_share)

            print(f"{row['name']:20s} pop={total_pop:>12,.0f}  exposed≈{row['population_exposed_est']:>10,}")

    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nUpdated {stats_path}")


if __name__ == "__main__":
    main()
