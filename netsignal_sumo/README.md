# NetSignal — SUMO Environment

A 6×6 urban grid traffic simulation environment built for the NetSignal research project.

## Project Structure

```
netsignal_sumo/
├── network/
│   └── grid6x6.net.xml     ← Road network (36 intersections, 300m spacing)
├── flows/
│   └── flow.xml            ← Vehicle flows + emergency vehicle definition
├── config/
│   └── netsignal.sumocfg   ← SUMO configuration file
├── output/                 ← Auto-created: logs, plots, trip data
└── scripts/
    └── traciloop.py        ← Main Python TraCI control loop ← START HERE
```

---

## Step 1 — Install SUMO

Download from: https://sumo.dlr.de/docs/Installing

**Windows:** Download the installer (.msi) — it adds sumo and sumo-gui to your PATH automatically.

**macOS:**
```bash
brew install sumo
```

**Linux (Ubuntu/Debian):**
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

Verify installation:
```bash
sumo --version
```

---

## Step 2 — Install Python dependencies

```bash
pip install traci eclipse-sumo numpy pandas matplotlib networkx
```

---

## Step 3 — Run the simulation

**GUI mode** (watch the simulation — start here):
```bash
cd netsignal_sumo
python scripts/traciloop.py --gui
```

**Headless mode** (fast, for training loops):
```bash
python scripts/traciloop.py
```

**GUI + inject a traffic incident at t=600s:**
```bash
python scripts/traciloop.py --gui --incident
```

**Custom incident time and lane:**
```bash
python scripts/traciloop.py --incident --incident-time 300 --incident-lane e_1_2_1_3_0
```

---

## What you'll see

When running with `--gui`, SUMO-GUI opens showing the 6×6 grid. Vehicles will
start flowing. At **t=900s** the simulation automatically pauses (breakpoint set
in the config) — this is when the emergency vehicle (EV_ambulance_001) spawns.
Press play to watch the corridor preemption in action.

The console prints a live dashboard every 30 seconds:

```
═════════════════════════════════════════════════════════════════
  NetSignal Dashboard  |  Step:   900s  |  🔴 EV: ACTIVE
─────────────────────────────────────────────────────────────────
  Network total waiting:   87 vehicles
  EV corridor: n_0_0 → n_0_1 → n_0_2 → n_1_2 → n_2_2 → n_3_3
─────────────────────────────────────────────────────────────────
  Intersection  Waiting  Vehicles  Avg Speed  Phase
  ──────────── ─────────  ────────  ─────────  ──────
  n_0_0               3         8    8.2 m/s  EW 🟢
  n_2_2               6        14    6.1 m/s  NS 🟢
  n_3_3               2         5   11.3 m/s  EW 🟢
  n_5_5               4         9    7.8 m/s  NS 🟢
  n_1_3               5        11    9.0 m/s  EW 🟢
═════════════════════════════════════════════════════════════════
```

---

## Output files

After running, check the `output/` folder:

| File | Contents |
|------|----------|
| `state_log_*.csv` | Per-intersection state every timestep — **input for BiLSTM training** |
| `metrics_log_*.csv` | Network-wide waiting, speed, vehicle count per timestep |
| `incident_log_*.csv` | Ground truth incident labels (when `--incident` is used) |
| `summary_plot.png` | Waiting vehicles over time with EV and incident markers |
| `tripinfo.xml` | SUMO trip statistics (travel time per vehicle) |
| `queue.xml` | Queue lengths per lane |

---

## Network design decisions

| Parameter | Value | Reason |
|-----------|-------|--------|
| Grid size | 6×6 (36 intersections) | Big enough to stress multi-intersection coordination, fast enough to run on a laptop |
| Road length | 300m | Realistic urban block size |
| Lanes | 2 per direction | Arterial-grade roads |
| Speed limit | 50 km/h (13.89 m/s) | Urban arterial standard |
| Max vehicles | 1,500 | Prevents gridlock while maintaining congestion pressure |
| Control cycle | 5 seconds | Standard for adaptive signal research |
| Vehicle cap | 1,500 | Keeps sim fast on modest hardware |

---

## Next steps (what to build on top of this)

1. **Replace `FixedTimeController`** with your GAT + PPO model
   - `FixedTimeController.get_actions()` returns `{intersection_id: phase_index}`
   - Your GNN replaces this method — same interface, smarter decisions

2. **Run 500 incident episodes** for BiLSTM training data:
   ```bash
   for i in $(seq 1 500); do
     python scripts/traciloop.py --incident --incident-time $((RANDOM % 3000 + 300))
   done
   ```

3. **Collect baseline metrics** (run once, save the numbers):
   ```bash
   python scripts/traciloop.py   # headless, fast
   # Check output/metrics_log_*.csv for avg waiting time
   # This is your fixed-time baseline to beat
   ```

---

## Troubleshooting

**"sumo: command not found"**
→ SUMO is not on your PATH. On Windows, restart your terminal after installation.
On Linux: `export PATH=$PATH:/usr/share/sumo/bin`

**"TraCI connection failed"**
→ SUMO didn't start in time. Increase the `time.sleep(2)` in `traciloop.py` to `time.sleep(4)`.

**"route error" warnings in console**
→ Normal during development. The config has `ignore-route-errors=true`. 
Fix by running `duarouter` to validate routes, or generate the network with `netgenerate`.

**Simulation is too slow in GUI**
→ In SUMO-GUI, drag the speed slider (bottom toolbar) to maximum, or 
use headless mode (`python traciloop.py` without `--gui`).

**Vehicles not appearing**
→ The network XML needs to be compiled by SUMO's `netconvert` tool first if using
the raw XML. Alternatively, generate it directly:
```bash
netgenerate --grid --grid.number=6 --grid.length=300 \
            --default.lanenumber=2 --output-file=network/grid6x6.net.xml
```
Then rerun with the generated file.
