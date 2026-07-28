# Federated datasets — local dashboard

Interactive viewer for **every federated dataset** under `data/raw/` (pipeline layout:
`train|val|test|test_label/*.npy` + `clusters.json` + `metadata.json`). One server process
serves them all and you **switch dataset from the header dropdown** — no relaunch:
- **`wsd_fed`** — real: **31 univariate KPI clients**, each the longest **NaN-free window** of a WSD KPI,
  split at 50% into `train` + `val` (both clean) and `test` (with anomalies), grouped into **4 similarity
  clusters** computed on the train portion only (no test leakage). `val` = last 1 period (1 day) of the
  first half, held out for model selection. Selection drops **10 near-duplicate KPIs** as a standard step.
- **`toy_fed_uni` / `toy_fed_uni_scarce` / `toy_fed_uni_scarcer`** — synthetic univariate benchmark
  (C=1), clients `uni_*` in machine-type clusters `M1_rotary..M6_drive`; the `_scarce*` variants only
  shrink `train_length` (the FL scarcity knob).

The WSD-only bits (original-CSV timestamps, `manifest.csv` windowing, the 210-KPI overview tab)
gate themselves off automatically for the synthetic datasets and degrade gracefully.

## Run
```bash
# from the dashboard dir, with the project env — serves ALL datasets at once
python server.py
# then open, and pick the dataset from the header dropdown:
http://localhost:8765
```
`DASH_DATASET=toy_fed_uni python server.py` sets only the **initial** selection (default `wsd_fed`);
you can still switch to any dataset in-page. Change the port with `DASH_PORT=9000 python server.py`.
The current dataset is kept in the URL (`?ds=<name>`), so a refresh stays put.
Stdlib-only (no pip installs); uPlot is vendored under `static/vendor/` so it works offline.

## What you can do
- **Folders by cluster** — left sidebar lists the 4 shape clusters as folders → the KPIs inside each.
  Filter by KPI id with the search box.
- **Zoom freely** — drag a range on the chart, scroll to zoom in/out, double-click to reset.
  Buttons `full / train / test` jump to those regions; they stay in sync with manual zoom.
- **Full length ⟷** — prints the entire client series long (~4 samples/px) in a horizontally-scrollable
  strip, so you can read it at high resolution instead of fitting it into the window.
- **Legend** — bottom of the sidebar: colours, split line, and the KPI badge
  (`0.11% · 27a` = % of test points that are anomalous · number of anomalous points; `a` = anomalies)
  and the `NaN Nt/Ne` badge (missing points in train / test — 0 everywhere in this NaN-free dataset).
- **Train / val / test / labels** — `train` (anomaly-free) is shaded green, `val` (anomaly-free,
  model selection) amber, the `test` region slate; the dashed line is the (train+val)|test split at
  50%; red bands are the labelled anomalies. Each layer has a toggle.
- **Provenance** — the info panel shows the original source file `data/WSD/real-world/<id>.csv`
  (WSD is **210 separate KPI series**, lengths 30,737–36,471), the NaN-free window used
  (`original[win_start:win_end]`), the split, train/val/test lengths, anomaly count/%, NaN counts, and timespan.
- **Overview tab (all 210 in context)** — a sparkline grid of the whole WSD population; the **31 frozen**
  clients are drawn in cluster colour with their **window highlighted** inside the full original series
  (grey context + green train + split + red anomalies); the **179 excluded** ones are dimmed grey and
  tagged with why they were dropped (no clean window / no anomaly / near-duplicate / anomaly-dominated /
  no-period), read from `selection.csv`. Filter all / frozen / excluded / has-NaN (empty — dataset is
  NaN-free); click any frozen tile to open it in Detail.
  Data: `dashboard/overview.json` (built by `scripts/build_overview.py`).

## API
Every endpoint below is dataset-scoped via `?ds=<name>` (default = `DASH_DATASET`).
- `GET /api/datasets` — the list of servable datasets `{name, n_clients, n_clusters, has_overview, noun}` + `default`.
- `GET /api/meta?ds=` — dataset metadata.
- `GET /api/manifest?ds=` — clusters → clients tree.
- `GET /api/series/<id>?ds=` — `{value[], anomaly_spans[[s,e]], split, n, win_start, win_end, orig_n, nan_*, ts_*, source, ...}`.
- `GET /api/overview?ds=` — WSD only: all 210 with frozen membership, windows, and sparklines (empty for synthetic).

Data is read read-only from the frozen snapshot; the dashboard never modifies it.
Files: `server.py` (stdlib HTTP + JSON API), `static/index.html`, `static/app.js`, `static/vendor/uPlot.*`.
