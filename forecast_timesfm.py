"""
Step 3: Forecast next-year flood recurrence with TimesFM
==========================================================
Uses Google Research's TimesFM (2.5, 200M) foundation model to forecast the
next 12 months of monthly flood activity per LGA, zero-shot, from the
120-month historical series produced by build_lagos_flood_stats.py.

TimesFM forecasts a continuous signal, not the 0/1 flood flag directly - so
we forecast the flag series as-is (it behaves like a low sparse-rate signal)
and read off both the point forecast and the 10th/90th percentile band,
which gives you a defensible "how confident is this" range rather than a
single number administrators might over-trust.

Requires:
    pip install timesfm[torch]
    (see https://github.com/google-research/timesfm for backend-specific
    install notes - CPU works fine for a series this short, no GPU needed)

Run after build_lagos_flood_stats.py (and, optionally, after
add_population_worldpop.py):
    python forecast_timesfm.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import timesfm

OUT_DIR = Path(__file__).parent
STATS_PATH = OUT_DIR / "lagos_flood_stats.json"
HORIZON = 12  # months ahead to forecast


def load_model():
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
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
    stats = json.loads(STATS_PATH.read_text())

    series_list = []
    names = []
    for row in stats:
        monthly = row.get("monthly_series")
        if not monthly:
            print(f"Skipping {row['name']}: no monthly_series found "
                  f"(re-run build_lagos_flood_stats.py first)")
            continue
        series_list.append(np.array(monthly, dtype=np.float32))
        names.append(row["name"])

    if not series_list:
        print("Nothing to forecast - no monthly_series data present.")
        return

    print(f"Loading TimesFM 2.5 (200M) ...")
    model = load_model()

    print(f"Forecasting {HORIZON} months ahead for {len(series_list)} LGAs ...")
    point_forecast, quantile_forecast = model.forecast(
        horizon=HORIZON, inputs=series_list
    )
    # point_forecast: (n_series, HORIZON)
    # quantile_forecast: (n_series, HORIZON, 10) -> mean, then q10..q90

    by_name = {row["name"]: row for row in stats}
    for i, name in enumerate(names):
        point = point_forecast[i].tolist()
        q10 = quantile_forecast[i, :, 1].tolist()   # 10th percentile
        q90 = quantile_forecast[i, :, 9].tolist()   # 90th percentile

        # Clip to [0,1] since the underlying series is a flood/no-flood flag;
        # summing the clipped point forecast gives an expected number of
        # flooded months next year rather than a literal 0/1 prediction.
        point_clipped = [min(max(p, 0), 1) for p in point]
        predicted_months_next_year = round(sum(point_clipped), 1)

        by_name[name]["forecast_next_12mo"] = {
            "point": [round(p, 3) for p in point_clipped],
            "q10": [round(q, 3) for q in q10],
            "q90": [round(q, 3) for q in q90],
        }
        by_name[name]["predicted_months_flooded_next_year"] = predicted_months_next_year
        print(f"  {name:20s} predicted months flooded next 12mo ≈ {predicted_months_next_year}")

    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"\nUpdated {STATS_PATH} with TimesFM forecasts.")


if __name__ == "__main__":
    main()
