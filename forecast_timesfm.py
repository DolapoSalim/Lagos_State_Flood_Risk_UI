"""
forecast_timesfm.py
======================
Runs Google's TimesFM zero-shot on a state's per-LGA monthly flood history
(120 months) to forecast the next 12 months, with a 10th/90th percentile
uncertainty band.

Usage:
    python forecast_timesfm.py lagos
    python forecast_timesfm.py ogun

Requires:
    pip install timesfm[torch]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import timesfm

APP_DIR = Path(__file__).parent.parent
DATA_DIR = APP_DIR / "data"
HORIZON = 12


def load_model():
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_slug", help="Slug from data/states.json, e.g. 'lagos'")
    args = parser.parse_args()

    stats_path = DATA_DIR / args.state_slug / "stats.json"
    if not stats_path.exists():
        raise SystemExit(f"{stats_path} missing - run build_state_flood_stats.py '{args.state_slug}' first")

    stats = json.loads(stats_path.read_text())

    series_list, names = [], []
    for row in stats:
        monthly = row.get("monthly_series")
        if not monthly:
            print(f"Skipping {row['name']}: no monthly_series found")
            continue
        series_list.append(np.array(monthly, dtype=np.float32))
        names.append(row["name"])

    if not series_list:
        print("Nothing to forecast.")
        return

    print("Loading TimesFM 2.5 (200M) ...")
    model = load_model()

    print(f"Forecasting {HORIZON} months ahead for {len(series_list)} LGAs ...")
    point_forecast, quantile_forecast = model.forecast(horizon=HORIZON, inputs=series_list)

    by_name = {row["name"]: row for row in stats}
    for i, name in enumerate(names):
        point = [min(max(p, 0), 1) for p in point_forecast[i].tolist()]
        q10 = quantile_forecast[i, :, 1].tolist()
        q90 = quantile_forecast[i, :, 9].tolist()

        by_name[name]["forecast_next_12mo"] = {
            "point": [round(p, 3) for p in point],
            "q10": [round(q, 3) for q in q10],
            "q90": [round(q, 3) for q in q90],
        }
        by_name[name]["predicted_months_flooded_next_year"] = round(sum(point), 1)
        print(f"  {name:20s} predicted months flooded next 12mo ≈ {round(sum(point), 1)}")

    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nUpdated {stats_path}")


if __name__ == "__main__":
    main()
