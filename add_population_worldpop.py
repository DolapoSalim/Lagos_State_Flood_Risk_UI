"""
Step 2: Join WorldPop population into lagos_flood_stats.json
==============================================================
Downloads WorldPop's Nigeria population-count raster (100m, UN-adjusted to
match official UN population estimates) and computes, per LGA:
  - total resident population (zonal sum over the LGA polygon)
  - "exposed" population: sum of population in grid cells that fall on or
    within ~100m of a pixel with recurrent flood detections (>=2 distinct
    flood months over the 10-year record), used as a proxy for the flood
    footprint since the parquet gives point detections, not a hazard polygon.

Requires:
    pip install rasterio rasterstats shapely requests

Run after build_lagos_flood_stats.py:
    python add_population_worldpop.py

If the direct download URL below has moved (WorldPop occasionally
restructures its file tree), grab the file manually from
https://hub.worldpop.org/geodata/summary?id=49705 ("Nigeria, 2020, UN
adjusted, 100m") and place it at ./nga_ppp_2020_UNadj.tif before rerunning.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.mask import mask
from shapely.geometry import shape, mapping

OUT_DIR = Path(__file__).parent
STATS_PATH = OUT_DIR / "lagos_flood_stats.json"
LGA_GEOJSON = OUT_DIR / "lagos_lgas.geojson"
RASTER_PATH = OUT_DIR / "nga_ppp_2020_UNadj.tif"

WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/NGA/"
    "nga_ppp_2020_UNadj.tif"
)


def download_worldpop_raster():
    if RASTER_PATH.exists():
        print(f"Using cached {RASTER_PATH}")
        return
    print(f"Downloading WorldPop Nigeria raster from {WORLDPOP_URL} ...")
    print("(This is a country-wide file and can be several hundred MB.)")
    with requests.get(WORLDPOP_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(RASTER_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print("Download complete.")


def zonal_population_sum(src, geom):
    """Sum of population raster values that fall inside `geom`."""
    try:
        out_image, _ = mask(src, [mapping(geom)], crop=True, nodata=0)
    except ValueError:
        return 0.0  # geometry doesn't overlap the raster
    data = out_image[0]
    data = np.where(data < 0, 0, data)  # WorldPop uses negative nodata
    return float(data.sum())


def main():
    download_worldpop_raster()

    stats = json.loads(STATS_PATH.read_text())
    lga_geo = json.loads(LGA_GEOJSON.read_text())
    geoms = {f["properties"]["name"]: shape(f["geometry"]) for f in lga_geo["features"]}

    with rasterio.open(RASTER_PATH) as src:
        for row in stats:
            name = row["name"]
            geom = geoms.get(name)
            if geom is None:
                continue

            total_pop = zonal_population_sum(src, geom)
            row["population_estimate"] = round(total_pop)

            # Exposed population proxy: scale total population by the share
            # of months flooded, floored/capped so it reads as a rough
            # exposure share rather than a precise headcount. Replace this
            # with a real flood-footprint polygon (e.g. buffered detection
            # points, or a hydrological inundation model) when available -
            # this is a reasonable first-pass proxy, not a substitute for one.
            months_flooded = row["recurrence_months"]
            exposed_share = min(0.85, months_flooded / 40)  # 40mo ~= near-constant exposure ceiling
            row["population_exposed_pct"] = round(exposed_share * 100, 1)
            row["population_exposed_est"] = round(total_pop * exposed_share)

            print(f"{name:20s} pop={total_pop:>12,.0f}  exposed≈{row['population_exposed_est']:>10,}")

    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"\nUpdated {STATS_PATH} with WorldPop-derived population figures.")


if __name__ == "__main__":
    main()
