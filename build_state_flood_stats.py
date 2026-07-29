"""
build_state_flood_stats.py
============================
Downloads every AI4G tile a state's bounding box touches, merges them,
applies the recommended filters, and aggregates flood recurrence per LGA -
same logic as the original Lagos-only script, generalized to N tiles and
any state registered via extract_state_lgas.py.

Usage:
    python build_state_flood_stats.py lagos
    python build_state_flood_stats.py ogun

Requires:
    pip install pandas numpy shapely huggingface_hub pyarrow
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

REPO_ID = "ai-for-good-lab/ai4g-flood-dataset"
REPO_TYPE = "dataset"

APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"


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


def download_tile(tile_id: str) -> pd.DataFrame:
    lat_band = tile_id[:3]
    remote_path = f"{lat_band}/{tile_id}/{tile_id}-post-processing.parquet"
    local_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=remote_path)
    return pd.read_parquet(local_path)


def risk_tier_from_recurrence(months: int) -> str:
    if months >= 24:
        return "Severe"
    if months >= 14:
        return "High"
    if months >= 7:
        return "Moderate"
    return "Low"


def load_state_meta(slug: str) -> dict:
    registry = json.loads((DATA_DIR / "states.json").read_text())
    for s in registry["states"]:
        if s["slug"] == slug:
            return s
    raise SystemExit(f"'{slug}' not found in data/states.json - run extract_state_lgas.py first")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_slug", help="Slug from data/states.json, e.g. 'lagos'")
    args = parser.parse_args()

    meta = load_state_meta(args.state_slug)
    state_dir = DATA_DIR / args.state_slug
    lga_geojson_path = state_dir / "lgas.geojson"
    if not lga_geojson_path.exists():
        raise SystemExit(f"{lga_geojson_path} missing - run extract_state_lgas.py '{meta['name']}' first")

    print(f"State: {meta['name']}  |  tiles to pull: {meta['tiles']}")

    frames = []
    for tile_id in meta["tiles"]:
        print(f"  downloading {tile_id} ...")
        try:
            frames.append(download_tile(tile_id))
        except Exception as e:
            print(f"  WARNING: could not fetch {tile_id} ({e}) - skipping. "
                  f"If this tile is mostly ocean/outside Nigeria it may not exist, which is fine.")
    if not frames:
        raise SystemExit("No tiles downloaded successfully - nothing to process.")

    df = pd.concat(frames, ignore_index=True)
    print(f"  combined raw rows: {len(df):,}")

    df = apply_recommended_filters(df)
    print(f"  after recommended filters: {len(df):,}")

    lga_geo = json.loads(lga_geojson_path.read_text())
    polys = {
        f["properties"]["name"]: {
            "geometry": shape(f["geometry"]),
            "area_km2": f["properties"]["area_km2"],
        }
        for f in lga_geo["features"]
    }
    names = list(polys.keys())
    geoms = [polys[n]["geometry"] for n in names]
    tree = STRtree(geoms)
    geom_to_name = {id(g): n for g, n in zip(geoms, names)}

    def match(lat, lon):
        pt = Point(lon, lat)
        for idx in tree.query(pt):
            g = geoms[idx] if isinstance(idx, (int, np.integer)) else idx
            if g.contains(pt):
                return geom_to_name.get(id(g))
        return None

    print("Assigning detections to LGAs (slow step for large states) ...")
    df["lga"] = [match(la, lo) for la, lo in zip(df["lat"], df["lon"])]
    df = df.dropna(subset=["lga"])
    print(f"  detections falling inside {meta['name']} LGAs: {len(df):,}")

    month_index = pd.period_range("2014-10", "2024-09", freq="M")

    results = []
    for name, info in polys.items():
        sub = df[df["lga"] == name]
        if not sub.empty:
            sub = sub.assign(period=pd.to_datetime(
                sub["year"].astype(str) + "-" + sub["month"].astype(str) + "-01"
            ).dt.to_period("M"))
            monthly_flag = sub.groupby("period").size()
        else:
            monthly_flag = pd.Series(dtype=int)

        monthly_series = [int(monthly_flag.get(p, 0) > 0) for p in month_index]
        yearly = {}
        for y in range(2015, 2025):
            months_this_year = [monthly_series[i] for i, p in enumerate(month_index) if p.year == y]
            yearly[str(y)] = int(sum(months_this_year))
        recurrence_months = int(sum(monthly_series))

        results.append({
            "name": name,
            "risk_tier": risk_tier_from_recurrence(recurrence_months),
            "recurrence_months": recurrence_months,
            "yearly_flood_events": yearly,
            "monthly_series": monthly_series,
            "monthly_series_start": "2014-10",
            "detections_total": int(len(sub)),
            "area_km2": info["area_km2"],
            "population_estimate": None,
            "population_exposed_pct": None,
            "population_exposed_est": None,
        })

    stats_path = state_dir / "stats.json"
    stats_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {stats_path}")

    # flip the registry flag so the dashboard knows this state has real data
    registry_path = DATA_DIR / "states.json"
    registry = json.loads(registry_path.read_text())
    for s in registry["states"]:
        if s["slug"] == args.state_slug:
            s["has_flood_stats"] = True
    registry_path.write_text(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
