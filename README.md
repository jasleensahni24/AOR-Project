# Reverse Logistics Optimizer

A Streamlit decision-support application for planning the collection of refillable glass-bottle crates from retail stores. It combines priority-based greedy routing, flexible time-slot scheduling, feasibility-preserving local search, and independently recalculated operational KPIs.

## Problem statement

Retail stores hold empty reusable-bottle crates until collection. Store capacity, current inventory, pickup requests, vehicle capacity, travel time, and available collection slots constrain the collection plan. The system prioritizes stores with higher inventory filling rates and pickup requests, constructs depot-return routes, assigns those routes to flexible-duration slots, and improves the routes without violating the model's constraints.

This is a heuristic decision-support implementation inspired by the 2022 refillable-glass-bottle vehicle-routing study; it does not reproduce or claim optimality equivalent to that paper's integer-linear programming model.

## Project objectives

- Prioritize stores using filling rate and pickup-request status.
- Produce feasible routes while respecting vehicle capacity and route duration.
- Schedule routes into as few feasible flexible time slots as the configured best-fit heuristic permits.
- Improve travel and collection duration through relocate, swap, and 2-opt neighborhoods.
- Report coverage, resource utilization, route metrics, infeasibilities, and downloadable outputs.

## System architecture

```text
Streamlit UI (app.py)
  ├── data_processing.py  validation, sample loading, and coordinate distances
  ├── priority.py         filling-rate and request-based store ranking
  ├── routing.py          priority-tier greedy route construction
  ├── scheduling.py       greedy best-fit route-to-slot assignment
  ├── local_search.py     feasible same-slot relocate, swap, and 2-opt
  ├── evaluation.py       independent route and KPI recalculation
  ├── visualization.py    maps and space-time figures
  ├── scenario_analysis.py scenario runs
  └── export_results.py   export data-frame/report builders
```

The web app is self-contained at the repository root. Inputs are loaded relative to `app.py` with `pathlib`; downloadable results are created in memory rather than relying on persistent cloud-server storage.

## Algorithm methodology

1. **Priority calculation:** filling rate is `current crate demand / store storage capacity`; score is `0.70 × filling rate + 0.30 × pickup request`. Equal scores retain input order.
2. **Route generation:** begin at the central depot, consider stores in descending priority tiers, and insert a feasible store with the least incremental travel time. Capacity and route-duration limits are checked. Routes start and end at the depot; stores that cannot be collected are reported.
3. **Time-slot scheduling:** routes are processed in input order. Each route is assigned to the open slot leaving the least remaining duration, with earlier slots winning ties. A new slot opens only if no existing slot can fit the route. Insufficient slots produce a partial schedule and explicit unscheduled route IDs.
4. **Local search:** evaluate feasible relocate, swap, and 2-opt candidates, restricted to each route's existing time slot. Accept only the greatest improvement meeting the configured threshold; stop at a local optimum or iteration cap.
5. **KPI evaluation:** independently recompute distances, travel/service/operational time, crates and bottles collected, store coverage, vehicle capacity utilization, missed high-volume stores, and time-slot utilization. The configured available slot count is used as the time-resource denominator.

## Folder structure

```text
Reverse_Logistics_Optimizer/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── stores.csv
│   ├── distances.csv
│   ├── sample_data.csv
│   └── parameters.json
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── priority.py
│   ├── routing.py
│   ├── scheduling.py
│   ├── local_search.py
│   ├── evaluation.py
│   ├── export_results.py
│   ├── scenario_analysis.py
│   └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_heuristic.py
│   ├── test_data_processing.py
│   ├── test_evaluation.py
│   ├── test_local_search.py
│   ├── test_priority.py
│   ├── test_routing.py
│   ├── test_scheduling.py
│   ├── conftest.py
│   └── test_report.md
└── outputs/
    └── .gitkeep
```

The extra source and test modules are required by the existing app and regression suite; they are retained rather than duplicated or removed. `data/parameters.json` contains the sample operating configuration needed to initialize the application.

## Datasets

- **`data/stores.csv`**: 20 synthetic store records, including IDs/names, coordinates, current crate demand, storage capacity, pickup requests, filling rate, and priority score. Values are illustrative sample data.
- **`data/distances.csv`**: symmetric labeled distance matrix in kilometres. It includes a zero diagonal, the `DEPOT`, and all store IDs. Sample distances are straight-line/geodesic, not road distances.
- **`data/sample_data.csv`**: header-only CSV template for uploading stores.
- **`data/parameters.json`**: sample warehouse location, vehicle, crate, service, route-duration, and flexible-slot settings.

Store upload columns:

| Column | Meaning |
|---|---|
| `Store ID` | Unique, non-empty route key. |
| `Store name` | Display name. |
| `Latitude`, `Longitude` | Geographic coordinates; latitude [-90, 90], longitude [-180, 180]. |
| `Current empty bottle crates (demand)` | Non-negative integer crate demand. |
| `Maximum crate storage capacity` | Positive integer store capacity; demand cannot exceed it. |
| `Pickup request` | Binary request indicator, 0 or 1. |

Uploaded distance matrices use a `NodeID` column and one numeric column per node. The matrix must be complete, symmetric, and have a zero diagonal. Alternatively, the app can construct straight-line distances from valid coordinates.

## Installation

Python 3.11 or later is recommended. From this directory:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

## Local execution

Run from the project root:

```bash
streamlit run app.py
```

The app loads its bundled sample data automatically. No preparatory script, notebook, or VS Code extension is needed.

## Testing

Run all unit and heuristic tests:

```bash
pytest -v
```

Run the integration/heuristic cases only:

```bash
pytest tests/test_heuristic.py -v
```

The session writes the executed results to `tests/test_report.md`. Test data is generated in fixtures and does not modify production CSVs.

## GitHub repository setup

From the project directory:

```bash
git init
git add .
git commit -m "Prepare reverse logistics optimizer"
git branch -M main
git remote add origin https://github.com/<your-account>/<your-repository>.git
git push -u origin main
```

The `.gitignore` excludes virtual environments, caches, temporary outputs, secrets, and machine-specific editor files while retaining the sample CSVs. Do not commit credentials or Streamlit secrets.

## Streamlit Community Cloud deployment

1. Push this folder's contents to a GitHub repository (the repository root must contain `app.py` and `requirements.txt`).
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the entry point.
3. Deploy. The application reads data under `data/` relative to its own file, so it does not depend on the launch working directory.
4. Use the app's CSV/text download buttons to save generated outputs locally; files written to a cloud instance are not treated as durable storage.

No screenshots are currently committed.

## Known limitations

- The route engine models one vehicle making sequential depot-return trips, not a simultaneous multi-vehicle fleet.
- Straight-line/geodesic distances do not represent road-network paths, traffic, or actual drive time.
- Time slots are flexible duration capacities, not store-specific clock-time windows.
- Local search preserves the scheduled time slot of each store and route; it cannot reassign work across slots.
- The priority weights and sample inputs are project choices; the heuristic is not an exact solver and provides no global optimality guarantee.
