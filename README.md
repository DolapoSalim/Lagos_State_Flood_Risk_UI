# Nigeria Flood Recurrence Atlas

Static web app for visualizing 10-year satellite flood recurrence, WorldPop
population exposure, and a 12-month TimesFM forecast, per LGA, per state.
No backend required — it's a state picker + a folder of small JSON files.

## Run it locally

```bash
cd nigeria-flood-atlas
python3 -m http.server 8000
```
Open http://localhost:8000 in a browser. (It must be served over http:// —
opening index.html directly via file:// will block the data fetches due to
browser CORS rules on local files.)

## Host it for real (no backend needed)

This is 100% static files, so any of these work with zero config:
- **GitHub Pages**: push this folder to a repo, enable Pages on it
- **Netlify / Vercel**: drag-and-drop the folder in their dashboard, or `netlify deploy`
- A shared drive + a colleague running the `python -m http.server` command works too, for something quick and internal

## Folder structure

```
nigeria-flood-atlas/
  index.html              <- the whole app (map, charts, table)
  data/
    states.json            <- registry of which states have data, and where
    lagos/
      lgas.geojson          <- LGA boundaries (real, from public admin geodata)
      stats.json             <- flood recurrence + population + forecast per LGA
    ogun/                   <- (add more states the same way)
      ...
  pipeline/
    extract_state_lgas.py    <- Step 0: pull a state's LGA boundaries, register it
    build_state_flood_stats.py <- Step 1: Sentinel-1 flood detections per LGA
    add_population_worldpop.py <- Step 2: WorldPop population + exposure
    forecast_timesfm.py        <- Step 3: TimesFM 12-month forecast
```

## Adding a new state

Four commands, run from `pipeline/`, each building on the last:

```bash
cd pipeline
pip install shapely requests pandas numpy huggingface_hub pyarrow rasterio "timesfm[torch]"

python extract_state_lgas.py "Ogun"              # pulls boundaries, registers the state
python build_state_flood_stats.py ogun           # Sentinel-1 detections, all tiles the state spans
python add_population_worldpop.py ogun           # WorldPop population + exposure
python forecast_timesfm.py ogun                  # TimesFM 12-month forecast
```

Then just reload the app — the new state appears in the dropdown automatically,
because `extract_state_lgas.py` registers it in `data/states.json`.

Not sure of the exact state name to pass? Run:
```bash
python extract_state_lgas.py --list
```

Note: `extract_state_lgas.py` figures out which AI4G tile(s) a state spans
automatically from its bounding box — this handles states that cross more
than one 3°×3° tile (most states larger than Lagos will).

## Updating an existing state's data

Just re-run steps 1–3 for that slug (`build_state_flood_stats.py` →
`add_population_worldpop.py` → `forecast_timesfm.py`) whenever the source
dataset updates, or as more Sentinel-1 history accumulates. Boundaries
(`lgas.geojson`) don't need re-running unless LGA boundaries themselves change.

## The "Load updated data" button

If you'd rather hand someone a `stats.json` file directly instead of them
running the pipeline (e.g. a colleague who processed a state on their own
machine), the dropdown-selected state's dashboard will accept it via the
"Load updated data" button in the header — it overrides the fetched data
client-side without touching the files on disk.
