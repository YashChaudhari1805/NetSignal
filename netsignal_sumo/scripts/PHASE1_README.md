# NetSignal — Phase 1: Data Collection

Complete guide to generating, validating and packaging the training dataset
for the BiLSTM incident detector (Phase 2).

---

## Overview

Phase 1 produces **labelled time-series data** by running hundreds of SUMO
simulation episodes.  Each episode either runs clean (baseline) or injects a
traffic incident at a random time and location.  Three CSV files are written
per episode, then merged into numpy arrays for model training.

```
netsignal_sumo/
├── scripts/
│   ├── collect_data.py      ← Step 1 — run SUMO episodes, write CSVs
│   ├── validate_data.py     ← Step 2 — check data quality
│   ├── prepare_dataset.py   ← Step 3 — normalise + export .npy arrays
│   └── run_phase1.sh        ← Runs all three steps in one command
│
└── output/
    ├── data_collection_summary.csv   ← one row per episode
    ├── episodes/
    │   ├── ep_0001_state.csv         ← per-node state, every timestep
    │   ├── ep_0001_incident.csv      ← ground-truth incident labels
    │   ├── ep_0001_metrics.csv       ← network-wide summary metrics
    │   └── ...
    └── dataset/
        ├── X_train.npy               ← (N, T, 36, 7)  float32 features
        ├── y_train.npy               ← (N, T, 36)     int8   labels
        ├── X_val.npy
        ├── y_val.npy
        ├── X_test.npy
        ├── y_test.npy
        ├── feature_stats.json        ← normalisation constants
        └── manifest.json             ← split metadata + shapes
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| SUMO        | ≥ 1.15  | `sumo` on PATH |
| Python      | ≥ 3.10  | |
| traci       | any     | `pip install traci eclipse-sumo` |
| numpy       | ≥ 1.24  | `pip install numpy` |
| matplotlib  | ≥ 3.7   | Optional — only needed for `--plot` |

Network must already be generated:
```bash
python scripts/generate_network.py
python scripts/fix_flows.py
```

---

## Quick Start

### Option A — one command (recommended)

```bash
bash scripts/run_phase1.sh --episodes 500
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 500 | Total simulation episodes |
| `--baseline-ratio` | 0.20 | Fraction without incidents |
| `--workers` | 1 | Parallel workers (Linux/macOS only) |
| `--downsample` | 1 | Keep every N-th timestep |
| `--seed` | 42 | Random seed |

### Option B — step by step

```bash
# Step 1 — collect data (takes the most time)
python scripts/collect_data.py --episodes 500 --baseline-ratio 0.2

# Step 2 — validate (fast, ~30s)
python scripts/validate_data.py --plot

# Step 3 — prepare dataset (fast, ~1 min)
python scripts/prepare_dataset.py --downsample 1
```

---

## Step 1: collect_data.py

Runs N SUMO episodes in headless mode. Each episode:

1. Launches `sumo` via TraCI on a fresh port
2. Runs the fixed-time baseline controller for 3600 steps (1 simulated hour)
3. At `incident_time`, blocks a randomly chosen lane for 300 s
4. Logs state + incident + metrics CSVs

### Arguments

```
--episodes        int    Total episodes (default: 500)
--baseline-ratio  float  Fraction with no incident, e.g. 0.2 (default: 0.2)
--baseline-only          All episodes are clean runs
--incident-only          All episodes have an incident
--workers         int    Parallel processes (default: 1)
--output-dir      path   Where to write episode CSVs
--port            int    Base TraCI port (default: 8813)
--seed            int    Random seed (default: 42)
--dry-run                Print plan without running
```

### Output CSVs

**ep_XXXX_state.csv** — one row per (step, node):

| Column | Type | Description |
|--------|------|-------------|
| `episode` | int | Episode ID |
| `step` | int | Simulation second (0–3599) |
| `node_id` | str | SUMO junction ID (e.g. `C2`) |
| `waiting` | int | Vehicles halted at this node |
| `queue` | int | Same as waiting (alias) |
| `avg_speed` | float | Mean speed on controlled lanes (m/s) |
| `vehicle_count` | int | Vehicles in controlled lanes |
| `current_phase` | int | TLS phase (0=EW green, 1=EW yellow, 2=NS green, 3=NS yellow) |
| `ev_nearby` | 0/1 | Emergency vehicle within 2 hops |
| `preempted` | 0/1 | Node is on active EV preemption corridor |

**ep_XXXX_incident.csv** — one row per (step, node):

| Column | Type | Description |
|--------|------|-------------|
| `incident_active` | 0/1 | Incident is happening right now |
| `incident_lane` | str | SUMO lane ID being blocked ("" if none) |
| `incident_distance` | int | Manhattan hops from node to incident node (999 if none) |

**ep_XXXX_metrics.csv** — one row per step (network-wide):

| Column | Type | Description |
|--------|------|-------------|
| `total_waiting` | int | All waiting vehicles across network |
| `total_vehicles` | int | All on-road vehicles |
| `network_avg_speed` | float | Mean speed (m/s) |
| `ev_active` | 0/1 | Emergency vehicle on road |
| `incident_active` | 0/1 | Incident currently injected |

### Time estimate

| Episodes | Workers | Approx. wall time |
|----------|---------|-------------------|
| 100 | 1 | ~1.5 h |
| 500 | 1 | ~7.5 h |
| 500 | 4 | ~2 h (Linux/macOS) |

> Each episode runs 3600 simulation steps. SUMO headless typically runs
> 60–120× real-time, so each episode takes ~30–60 s of wall time.

---

## Step 2: validate_data.py

Scans all episode CSVs and reports:
- Missing or corrupted files
- Truncated episodes (< 95% of expected rows)
- Unexpected phase values or negative speeds
- Per-episode and dataset-level statistics
- Recommendations for BiLSTM training

```bash
python scripts/validate_data.py --plot --verbose
```

| Argument | Description |
|----------|-------------|
| `--episode-dir` | Path to episodes folder |
| `--max-episodes` | Inspect first N only (for quick checks) |
| `--plot` | Save `phase1_dataset_overview.png` |
| `--verbose` | Print issues even for passing episodes |

### What to look for

- **All episodes passed** → proceed to Step 3
- **Short state file** → re-run that episode or exclude it
- **Inc/baseline ratio < 1.5** → collect more incident episodes
- **< 500 000 state rows** → collect more episodes for robust training

---

## Step 3: prepare_dataset.py

Converts CSVs into numpy arrays, normalises features, and splits into
train / val / test sets.

```bash
python scripts/prepare_dataset.py \
  --train-ratio 0.70 \
  --val-ratio   0.15 \
  --downsample  1
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-ratio` | 0.70 | Fraction of episodes for training |
| `--val-ratio` | 0.15 | Fraction for validation (remainder = test) |
| `--downsample` | 1 | Keep every N-th timestep (reduces memory) |
| `--exclude-episodes` | [] | Space-separated IDs to drop |
| `--seed` | 42 | Reproducible split |

### Output array shapes

```
X_train  (N_train, T, 36, 7)   float32   normalised features
y_train  (N_train, T, 36)      int8      binary incident labels

  N = number of episodes in split
  T = timesteps per episode (3600 / downsample)
  36 = nodes (6×6 grid)
  7  = features: waiting, queue, avg_speed, vehicle_count,
                 current_phase, ev_nearby, preempted
```

### Loading in Phase 2

```python
import numpy as np, json

X_train = np.load("output/dataset/X_train.npy")   # (N, T, 36, 7)
y_train = np.load("output/dataset/y_train.npy")   # (N, T, 36)

with open("output/dataset/feature_stats.json") as f:
    stats = json.load(f)

with open("output/dataset/manifest.json") as f:
    manifest = json.load(f)

print("Train shape :", X_train.shape)
print("Label density:", y_train.mean())   # fraction of positive samples
print("Features     :", stats["feature_names"])
```

---

## Troubleshooting

### `sumo: command not found`
Add SUMO bin to PATH:
```bash
# Linux
export PATH=$PATH:/usr/share/sumo/bin

# macOS (Homebrew)
export PATH=$PATH:/opt/homebrew/bin

# Windows — restart terminal after SUMO installer
```

### `TraCI connection failed`
Port conflict. Try a different port:
```bash
python scripts/collect_data.py --port 9000
```

### Episode files are tiny (< 1 KB)
The network XML is missing or flow file has route errors.
Run `fix_flows.py` again:
```bash
python scripts/fix_flows.py
python scripts/collect_data.py --episodes 5 --dry-run   # verify plan
```

### Out of memory during prepare_dataset.py
Use `--downsample 5` or `--downsample 10` to reduce the timestep dimension:
```bash
python scripts/prepare_dataset.py --downsample 5
# X shape becomes (N, 720, 36, 7) instead of (N, 3600, 36, 7)
```

### Parallel mode hangs on Windows
Use `--workers 1` (serial mode) on Windows:
```bash
python scripts/collect_data.py --episodes 500 --workers 1
```

---

## Dataset Recommendations for BiLSTM

| Goal | Suggested collection |
|------|----------------------|
| Quick prototype | 100 episodes, `--downsample 10` |
| Standard training | 500 episodes, `--downsample 1` |
| High-performance | 1000+ episodes, `--downsample 1` |

Aim for an **incident/baseline ratio of 3:1 to 4:1** — enough baseline
episodes for the model to learn "normal", but mostly incident episodes
for anomaly detection signal.

The `--baseline-ratio 0.20` default (20% baseline, 80% incident) is a
good starting point.  If your BiLSTM overfits to incident patterns,
increase the baseline ratio to 0.30–0.40.
