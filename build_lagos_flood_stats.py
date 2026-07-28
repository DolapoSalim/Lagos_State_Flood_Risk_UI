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

    # Full monthly index Oct 2014 - Sep 2024 (dataset's native coverage).
    # This is what forecast_timesfm.py consumes as historical context -
    # 120 monthly points is a far better input to a time-series foundation
    # model than the 10 yearly points alone.
    month_index = pd.period_range("2014-10", "2024-09", freq="M")

    results = []
    for name, info in polys.items():
        sub = df[df["lga"] == name]
        if not sub.empty:
            sub = sub.assign(period=pd.to_datetime(
                sub["year"].astype(str) + "-" + sub["month"].astype(str) + "-01"
            ).dt.to_period("M"))
            # 1 if that LGA had >=1 detection in that calendar month, 0 otherwise.
            # (Switch to sub.groupby("period").size() instead of .any() if you
            # want detection *counts* per month rather than a flood/no-flood flag.)
            monthly_flag = sub.groupby("period").size()
        else:
            monthly_flag = pd.Series(dtype=int)

        monthly_series = [int(monthly_flag.get(p, 0) > 0) for p in month_index]

        yearly = {}
        for y in range(2015, 2025):
            months_this_year = [
                monthly_series[i] for i, p in enumerate(month_index) if p.year == y
            ]
            yearly[str(y)] = int(sum(months_this_year))

        recurrence_months = int(sum(monthly_series))

        results.append(
            {
                "name": name,
                "risk_tier": risk_tier_from_recurrence(recurrence_months),
                "recurrence_months": recurrence_months,
                "yearly_flood_events": yearly,  # months flooded per calendar year, 0-12
                "monthly_series": monthly_series,  # 120 points, Oct 2014 - Sep 2024, 0/1 per month
                "monthly_series_start": "2014-10",
                "detections_total": int(len(sub)),
                "area_km2": info["area_km2"],
                # Population figures are NOT in this dataset - populated by
                # add_population_worldpop.py in a later step.
                "population_estimate": None,
                "population_exposed_pct": None,
                "population_exposed_est": None,
            }
        )

    STATS_OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {STATS_OUT}")


if __name__ == "__main__":
    main()
