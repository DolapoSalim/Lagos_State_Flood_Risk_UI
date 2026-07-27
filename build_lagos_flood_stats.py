"""
Lagos Flood Recurrence Pipeline
================================
Downloads the AI4G flood detection tile covering Lagos State (N06E003),
clips it to Lagos LGA boundaries, applies the recommended false-positive
filters, and aggregates flood recurrence + exposure statistics per LGA.

Output: lagos_flood_stats.json  ->  drop this straight into the dashboard
(lagos_flood_dashboard.html) using the "Load updated data" button, or
replace the embedded PLACEHOLDER_STATS block in that file directly.

Requirements:
    pip install pandas numpy rasterio shapely geopandas huggingface_hub pyarrow

Run:
    python build_lagos_flood_stats.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from shapely.geometry import Point, shape

REPO_ID = "ai-for-good-lab/ai4g-flood-dataset"
REPO_TYPE = "dataset"

# Lagos falls entirely inside a single 3x3 degree tile once rounded down
# (6.583N, 3.750E) -> N06E003. Confirm your area of interest with the
# "Finding Your Area of Interest" section of the dataset README if you
# expand this to other states.
TILE_ID = "N06E003"

OUT_DIR = Path(__file__).parent
LGA_GEOJSON = OUT_DIR / "lagos_lgas.geojson"  # shipped alongside this script
STATS_OUT = OUT_DIR / "lagos_flood_stats.json"

# Same recommended filters as the dataset README, tuned for long-period
# aggregation (we're summarizing 10 years, not a single event).
def apply_recommended_filters(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df.dem_metric_2 < 10)
        & (df.soil_moisture_sca > 1)
        & (df.soil_moisture_zscore > 1)
        & (df.soil_moisture > 20)
        & (df.temp > 0)
        & (df.land_cover != 60)
        & (df.edge_false_positives == 0)
    ]


def download_tile_parquet() -> pd.DataFrame:
    lat_band = TILE_ID[:3]  # e.g. "N06"
    remote_path = f"{lat_band}/{TILE_ID}/{TILE_ID}-post-processing.parquet"
    local_path = hf_hub_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, filename=remote_path
    )
    return pd.read_parquet(local_path)


def load_lga_polygons():
    gj = json.loads(LGA_GEOJSON.read_text())
    polys = {}
    for feat in gj["features"]:
        name = feat["properties"]["name"]
        polys[name] = {
            "geometry": shape(feat["geometry"]),
            "area_km2": feat["properties"]["area_km2"],
        }
    return polys


def assign_lga(lat: float, lon: float, polys: dict):
    pt = Point(lon, lat)
    for name, info in polys.items():
        if info["geometry"].contains(pt):
            return name
    return None


def risk_tier_from_recurrence(months: int) -> str:
    if months >= 24:
        return "Severe"
    if months >= 14:
        return "High"
    if months >= 7:
        return "Moderate"
    return "Low"


def main():
    print(f"Downloading tile {TILE_ID} from {REPO_ID} ...")
    df = download_tile_parquet()
    print(f"  raw rows: {len(df):,}")

    df = apply_recommended_filters(df)
    print(f"  after recommended filters: {len(df):,}")

    polys = load_lga_polygons()

    print("Assigning detections to Lagos LGAs (this is the slow step) ...")
    # Vectorized-ish spatial join using shapely's STRtree would be faster for
    # large tiles; a simple point-in-polygon loop is shown here for clarity.
    from shapely.strtree import STRtree

    names = list(polys.keys())
    geoms = [polys[n]["geometry"] for n in names]
    tree = STRtree(geoms)
    geom_to_name = {id(g): n for g, n in zip(geoms, names)}

    def match(lat, lon):
        pt = Point(lon, lat)
        candidates = tree.query(pt)
        for idx in candidates:
            g = geoms[idx] if isinstance(idx, (int, np.integer)) else idx
            if g.contains(pt):
                return geom_to_name.get(id(g))
        return None

    df["lga"] = [match(la, lo) for la, lo in zip(df["lat"], df["lon"])]
    df = df.dropna(subset=["lga"])
    print(f"  detections falling inside Lagos LGAs: {len(df):,}")

    results = []
    for name, info in polys.items():
        sub = df[df["lga"] == name]
        if sub.empty:
            recurrence_months = 0
            yearly = {str(y): 0.0 for y in range(2015, 2025)}
        else:
            sub = sub.assign(ym=sub["year"].astype(str) + "-" + sub["month"].astype(str))
            recurrence_months = sub["ym"].nunique()
            yearly = {}
            for y in range(2015, 2025):
                yearly[str(y)] = round(
                    sub[sub["year"] == y]["ym"].nunique(), 1
                )

        results.append(
            {
                "name": name,
                "risk_tier": risk_tier_from_recurrence(recurrence_months),
                "recurrence_months": int(recurrence_months),
                "yearly_flood_events": yearly,
                "detections_total": int(len(sub)),
                "area_km2": info["area_km2"],
                # Population figures are NOT in this dataset. Join your own
                # population source (e.g. WorldPop, NPC estimates) here by
                # LGA name to populate population_estimate / exposed counts.
                "population_estimate": None,
                "population_exposed_pct": None,
                "population_exposed_est": None,
            }
        )

    STATS_OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {STATS_OUT}")


if __name__ == "__main__":
    main()
