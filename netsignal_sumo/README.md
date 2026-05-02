# NetSignal — SUMO Environment

A 6×6 urban grid traffic simulation environment for the NetSignal research project.
Designed as a baseline and data-collection platform for adaptive signal control and
incident detection using Graph Attention Networks (GAT), PPO, and BiLSTM.

---

## Project structure

```
netsignal_sumo/
├── config/
│   ├── grid_nodes.txt        Junction ID → grid coordinate mapping
│   ├── netsignal.sumocfg     SUMO configuration
│   └── netsignal.view.xml    SUMO-GUI visual theme (white background)
├── flows/
│   └── flow.xml              Vehicle flows and emergency vehicle definition
├── network/
│   └── grid6x6.net.xml       Compiled road network (36 intersections, 300 m spacing)
├── output/                   Auto-created on first run
│   ├── state_log_<ts>.csv    Per-intersection state — BiLSTM training input
│   ├── metrics_log_<ts>.csv  Network-wide metrics per control step
│   ├── incident_log_<ts>.csv Ground-truth incident labels
│   ├── summary_plot.png      Waiting-vehicle time series with stat cards
│   └── tripinfo.xml          Per-vehicle travel time (SUMO output)
└── scripts/
    ├── generate_network.py   Step 1 — generate grid6x6.net.xml via netgenerate
    ├── fix_flows.py          Step 2 — regenerate flow.xml from real edge IDs
    └── traciloop.py          Step 3 — main TraCI simulation loop  ← start here
```

---

## Setup

### 1. Install SUMO

**Windows** — download the `.msi` installer from https://sumo.dlr.de/docs/Downloads.php.
The installer adds `sumo`, `sumo-gui`, and `netgenerate` to PATH automatically.

**macOS**
```bash
brew install sumo
```

**Ubuntu / Debian**
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update && sudo apt-get install sumo sumo-tools
```

Verify:
```bash
sumo --version
netgenerate --version
```

### 2. Install Python dependencies

```bash
pip install traci eclipse-sumo numpy pandas matplotlib networkx
```

### 3. Generate the network

```bash
python scripts/generate_network.py
```

### 4. Regenerate flow routes

Only needed after changing the network. Reads real edge IDs from the compiled
`.net.xml` and rewrites `flows/flow.xml` and `config/grid_nodes.txt`.

```bash
python scripts/fix_flows.py
```

---

## Running the simulation

| Command | Effect |
|---------|--------|
| `python scripts/traciloop.py` | Headless — fast, suitable for training loops |
| `python scripts/traciloop.py --gui` | GUI mode — watch vehicles move |
| `python scripts/traciloop.py --gui --incident` | GUI + inject incident at t=600 s |
| `python scripts/traciloop.py --incident --incident-time 300 --incident-lane e_1_2_1_3_0` | Custom incident |

**GUI behaviour:** SUMO-GUI opens with a white, clean visual theme. The simulation
auto-pauses at **t=900 s** when the emergency vehicle spawns. Press play to watch
signal preemption along the EV corridor.

---

## Console dashboard

The simulation prints a live status table every 30 seconds:

```
┌──────────────────────────────────────────────────────────────────┐
│  NetSignal  ·  t =   900s        [ EV ACTIVE ]  [            ]  │
├──────────────────────────────────────────────────────────────────┤
│  waiting:   87 veh    on-road:   412 veh    avg speed:  8.1 m/s  │
│  EV route  A0 -> B0 -> C0 -> C1 -> C2 -> D2 -> D3               │
├──────────────────────────────────────────────────────────────────┤
│  Node       Waiting  Vehicles   Avg Speed  Phase                 │
│  ────────   ───────  ────────  ──────────  ─────────             │
│  A0               3         8    8.2 m/s  EW green               │
│  C2               6        14    6.1 m/s  NS green               │
│  D3               2         5   11.3 m/s  EW green               │
│  F5               4         9    7.8 m/s  NS green               │
│  D1               5        11    9.0 m/s  EW green               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Output files

| File | Contents | Primary use |
|------|----------|-------------|
| `state_log_<ts>.csv` | Per-intersection state every timestep | BiLSTM training input |
| `metrics_log_<ts>.csv` | Network-wide waiting, speed, vehicle count | Baseline evaluation |
| `incident_log_<ts>.csv` | Ground-truth incident labels | BiLSTM supervision |
| `summary_plot.png` | Time-series chart + avg/peak/final stat cards | Quick run overview |
| `tripinfo.xml` | Per-vehicle travel time (SUMO native) | Throughput analysis |

---

## Network parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Grid size | 6×6 (36 intersections) | Large enough for multi-intersection coordination; fast enough on a laptop |
| Road length | 300 m | Realistic urban block size |
| Lanes | 2 per direction | Arterial-grade roads |
| Speed limit | 50 km/h (13.89 m/s) | Urban arterial standard |
| Max vehicles | 1 500 | Prevents gridlock while maintaining congestion pressure |
| Control cycle | 5 s | Standard for adaptive signal research |
| EV departs | t=900 s | Gives 15 minutes for traffic to reach steady state |
| Hospital node | D3 (col 3, row 3) | Central destination for shortest-path preemption |

---

## Extending the simulation

### Replace the fixed-time controller with a GNN

`FixedTimeController.get_actions` returns `{junction_id: phase_index}`.
Swap it out with your GAT + PPO model using the same interface:

```python
class GNNController:
    def get_actions(self, state: dict) -> dict[str, int]:
        # state keys: waiting, queue, avg_speed, vehicle_count, current_phase
        ...
        return {node_id: predicted_phase, ...}
```

Pass your controller instance into `run_simulation` (or swap the instantiation
line in `traciloop.py`).

### Generate BiLSTM training data

```bash
for i in $(seq 1 500); do
    python scripts/traciloop.py --incident --incident-time $((RANDOM % 3000 + 300))
done
```

Each run produces a `state_log_<ts>.csv` and `incident_log_<ts>.csv` pair.

### Collect a fixed-time baseline

```bash
python scripts/traciloop.py   # headless — completes in ~30 s
# Check output/metrics_log_*.csv for avg_waiting — this is your baseline to beat
```

---

## Troubleshooting

**`sumo: command not found`**
SUMO is not on PATH. On Windows, restart your terminal after installation.
On Linux: `export PATH=$PATH:/usr/share/sumo/bin`

**`TraCI connection failed`**
SUMO did not start in time. In `traciloop.py`, locate `traci.start(...)` and
add `time.sleep(4)` immediately before it.

**Route error warnings**
Expected during development. The config has `ignore-route-errors=true`.
To eliminate them, run `python scripts/fix_flows.py` to regenerate routes
from the actual compiled network edge IDs.

**Simulation is slow in GUI mode**
Drag the speed slider in the SUMO-GUI bottom toolbar to maximum, or use
headless mode for any run that does not require visual inspection.

**Vehicles not appearing**
The `network/grid6x6.net.xml` may be missing or stale.
Run `python scripts/generate_network.py` then `python scripts/fix_flows.py`.
