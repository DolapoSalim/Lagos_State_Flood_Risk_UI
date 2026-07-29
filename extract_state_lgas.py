"""
extract_state_lgas.py
======================
Adds a new state to the app: pulls its LGA boundaries out of the national
Nigeria LGA boundary file, simplifies them for the web, works out which
AI4G flood-dataset tiles it spans, and registers it in data/states.json
so it shows up in the dashboard's state picker.

This only needs to be run ONCE per state (boundaries don't change).
build_state_flood_stats.py / add_population_worldpop.py / forecast_timesfm.py
are what you re-run to refresh a state's flood/population/forecast numbers.

Usage:
    python extract_state_lgas.py "Lagos"
    python extract_state_lgas.py "Ogun"
    python extract_state_lgas.py "Kano"

State names must match NAME_1 in the source boundary file - run with
--list to see all 37 available (36 states + FCT).

Requires:
    pip install shapely requests
"""

import argparse
import json
import math
from pathlib import Path

import requests
from shapely.geometry import shape, mapping

APP_DIR = Path(__file__).parent.parent  # pipeline/ -> app root
DATA_DIR = APP_DIR / "data"
NATIONAL_GEOJSON_URL = (
    "https://raw.githubusercontent.com/qedsoftware/geojson_data/main/nigeria-lga.geojson"
)
NATIONAL_GEOJSON_CACHE = APP_DIR / "_nigeria-lga-source.geojson"

NAME_FIXES = {
    "Ajeromi/Ifelodun": "Ajeromi-Ifelodun",
    "Amuwo Odofin": "Amuwo-Odofin",
    "Badagary": "Badagry",
    "Ibeju/Lekki": "Ibeju-Lekki",
    "Ifako/Ijaye": "Ifako-Ijaiye",
    "LagosIsland": "Lagos Island",
    "Mainland": "Lagos Mainland",
    "Oshodi/Isolo": "Oshodi-Isolo",
}


def slugify(name: str) -> str:
    return name.lower().replace(" ", "-").replace("/", "-")


def load_national_geojson() -> dict:
    if NATIONAL_GEOJSON_CACHE.exists():
        return json.loads(NATIONAL_GEOJSON_CACHE.read_text())
    print("Downloading national LGA boundary file (one-time, ~10MB) ...")
    r = requests.get(NATIONAL_GEOJSON_URL, timeout=60)
    r.raise_for_status()
    NATIONAL_GEOJSON_CACHE.write_text(r.text)
    return json.loads(r.text)


def tile_id(lat_floor: int, lon_floor: int) -> str:
    lat_part = f"N{abs(lat_floor):02d}" if lat_floor >= 0 else f"S{abs(lat_floor):02d}"
    lon_part = f"E{abs(lon_floor):03d}" if lon_floor >= 0 else f"W{abs(lon_floor):03d}"
    return f"{lat_part}{lon_part}"


def tiles_for_bbox(min_lat, max_lat, min_lon, max_lon):
    tiles = []
    lat = math.floor(min_lat / 3) * 3
    while lat <= max_lat:
        lon = math.floor(min_lon / 3) * 3
        while lon <= max_lon:
            tiles.append(tile_id(lat, lon))
            lon += 3
        lat += 3
    return tiles


def update_states_registry(entry: dict):
    registry_path = DATA_DIR / "states.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"states": []}
    registry["states"] = [s for s in registry["states"] if s["slug"] != entry["slug"]]
    registry["states"].append(entry)
    registry["states"].sort(key=lambda s: s["name"])
    registry_path.write_text(json.dumps(registry, indent=2))
    print(f"Registered '{entry['name']}' in {registry_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state", nargs="?", help="State name, e.g. 'Lagos', 'Ogun', 'Kano'")
    parser.add_argument("--list", action="store_true", help="List available state names and exit")
    args = parser.parse_args()

    national = load_national_geojson()

    if args.list:
        names = sorted({f["properties"]["NAME_1"] for f in national["features"]})
        print("\n".join(names))
        return

    if not args.state:
        parser.error("state name required (or pass --list)")

    matches = [f for f in national["features"] if f["properties"]["NAME_1"].lower() == args.state.lower()]
    if not matches:
        available = sorted({f["properties"]["NAME_1"] for f in national["features"]})
        print(f"No LGAs found for '{args.state}'. Available names:\n" + "\n".join(available))
        return

    slug = slugify(args.state)
    out_dir = DATA_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    out_features = []
    all_lats, all_lons = [], []
    for f in matches:
        geom = shape(f["geometry"])
        simplified = geom.simplify(0.0015, preserve_topology=True)
        centroid = geom.centroid
        area_km2 = round(geom.area * 111 * 111 * 0.85, 1)
        raw_name = f["properties"]["NAME_2"]
        clean_name = NAME_FIXES.get(raw_name, raw_name)

        minx, miny, maxx, maxy = geom.bounds
        all_lons.extend([minx, maxx])
        all_lats.extend([miny, maxy])

        out_features.append({
            "type": "Feature",
            "properties": {
                "name": clean_name,
                "centroid": [round(centroid.x, 4), round(centroid.y, 4)],
                "area_km2": area_km2,
            },
            "geometry": mapping(simplified),
        })

    state_bbox = [min(all_lons), min(all_lats), max(all_lons), max(all_lats)]  # [minlon,minlat,maxlon,maxlat]
    tiles = tiles_for_bbox(state_bbox[1], state_bbox[3], state_bbox[0], state_bbox[2])
    state_centroid = [
        round((state_bbox[0] + state_bbox[2]) / 2, 4),
        round((state_bbox[1] + state_bbox[3]) / 2, 4),
    ]

    lgas_path = out_dir / "lgas.geojson"
    lgas_path.write_text(json.dumps({"type": "FeatureCollection", "features": out_features}))
    print(f"Wrote {len(out_features)} LGAs -> {lgas_path}")
    print(f"State spans tile(s): {tiles}  <- build_state_flood_stats.py will need to pull each of these")

    update_states_registry({
        "name": args.state,
        "slug": slug,
        "centroid": state_centroid,
        "bbox": state_bbox,
        "tiles": tiles,
        "lga_count": len(out_features),
        "has_flood_stats": (out_dir / "stats.json").exists(),
    })


if __name__ == "__main__":
    main()
