#!/usr/bin/env python3
"""
collect_data.py
===============
Phase 1 — Automated data collection for NetSignal.

Runs multiple SUMO simulation episodes in headless mode, each producing:
    - ep_NNNN_state.csv    : Per-intersection state every timestep
    - ep_NNNN_incident.csv : Ground-truth incident labels
    - ep_NNNN_metrics.csv  : Network-wide summary metrics

Usage:
    python scripts/collect_data.py                              # 500 mixed episodes, serial
    python scripts/collect_data.py --episodes 100               # custom count
    python scripts/collect_data.py --workers 4                  # parallel (Windows + Linux/macOS)
    python scripts/collect_data.py --sim-duration 1800          # 30-min episodes (2x faster)
    python scripts/collect_data.py --write-every 5              # write every 5th step (5x less I/O)
    python scripts/collect_data.py --baseline-only              # no incidents
    python scripts/collect_data.py --incident-only              # all incidents
    python scripts/collect_data.py --output-dir data/raw        # custom output path

Parallel mode works on Windows, Linux, and macOS.
On Windows it uses multiprocessing with the 'spawn' start method explicitly
to avoid issues with the default 'fork' behaviour under some environments.

Output structure:
    output/
        data_collection_summary.csv   — episode-level summary
        episodes/
            ep_0001_state.csv
            ep_0001_incident.csv
            ep_0001_metrics.csv
            ep_0002_state.csv
            ...
"""

import argparse
import csv
import multiprocessing
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import traci                   # type: ignore[import-untyped]
    import traci.constants as tc   # type: ignore[import-untyped]
except ImportError:
    sys.exit(
        "ERROR: traci is not installed.\n"
        "Run:  pip install traci eclipse-sumo"
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = Path(__file__).parent.resolve()
_PROJECT_DIR = _SCRIPT_DIR.parent
CONFIG_FILE  = _PROJECT_DIR / "config" / "netsignal.sumocfg"
_NODE_MAP_FILE = _PROJECT_DIR / "config" / "grid_nodes.txt"

# ---------------------------------------------------------------------------
# Constants (match traciloop.py)
# ---------------------------------------------------------------------------

GRID_SIZE          = 6
SIM_DURATION       = 3600          # seconds — overridden by --sim-duration flag
CONTROL_FREQ       = 5
EV_PREFIX          = "EV_"
DEFAULT_GREEN      = 30
DEFAULT_YELLOW     = 3
WRITE_EVERY        = 1             # write every N steps — overridden by --write-every flag

PHASE_EW_GREEN     = 0
PHASE_EW_YELLOW    = 1
PHASE_NS_GREEN     = 2
PHASE_NS_YELLOW    = 3

# Incident defaults
MIN_INCIDENT_START = 300
MAX_INCIDENT_START = 3000
INCIDENT_DURATION  = 300          # seconds

# State CSV columns
STATE_COLUMNS = [
    "episode", "step", "node_id",
    "waiting", "queue", "avg_speed", "vehicle_count", "current_phase",
    "ev_nearby",          # 1 if EV is within 2 hops, else 0
    "preempted",          # 1 if this node is under EV preemption
]

# Incident CSV columns
INCIDENT_COLUMNS = [
    "episode", "step", "node_id",
    "incident_active",    # 1 if any incident affects this node's upstream
    "incident_lane",      # lane ID being blocked ("" if none)
    "incident_distance",  # hops from blocked lane's node to this node (999 if none)
]

# Metrics CSV columns
METRICS_COLUMNS = [
    "episode", "step",
    "total_waiting", "total_vehicles",
    "network_avg_speed",
    "ev_active", "incident_active",
    "incident_start", "incident_end",
    "incident_lane",
]

# ---------------------------------------------------------------------------
# Node map loader
# ---------------------------------------------------------------------------

def load_node_map() -> dict[str, str]:
    node_map: dict[str, str] = {}
    if not _NODE_MAP_FILE.exists():
        print(f"  WARNING: Node map not found at {_NODE_MAP_FILE}")
        print("  Run fix_flows.py first.")
        return node_map
    for line in _NODE_MAP_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        node_map[key.strip()] = value.strip()
    return node_map


def get_all_intersections(node_map: dict) -> list[str]:
    return [
        node_map[f"NODE_{col}_{row}"]
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
        if f"NODE_{col}_{row}" in node_map
    ]


def get_node_coords(node_map: dict) -> dict[str, tuple[int, int]]:
    """Return {junction_id: (col, row)} for grid coordinate lookups."""
    coords = {}
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            key = f"NODE_{col}_{row}"
            if key in node_map:
                coords[node_map[key]] = (col, row)
    return coords


def hop_distance(
    node_a: str,
    node_b: str,
    node_coords: dict[str, tuple[int, int]],
) -> int:
    """Manhattan distance in grid hops between two junction IDs."""
    if node_a not in node_coords or node_b not in node_coords:
        return 999
    ca, ra = node_coords[node_a]
    cb, rb = node_coords[node_b]
    return abs(ca - cb) + abs(ra - rb)


# ---------------------------------------------------------------------------
# Fixed-time controller (same as traciloop.py)
# ---------------------------------------------------------------------------

class FixedTimeController:
    def __init__(self, green: int = DEFAULT_GREEN, yellow: int = DEFAULT_YELLOW) -> None:
        self.green_dur  = green
        self.yellow_dur = yellow
        self.cycle_len  = 2 * (green + yellow)
        self._step      = 0

    def get_actions(self, intersections: list[str]) -> dict[str, int]:
        actions: dict[str, int] = {}
        stagger = 0
        for node_id in intersections:
            cycle_pos = (self._step + stagger) % self.cycle_len
            if cycle_pos < self.green_dur:
                phase = PHASE_EW_GREEN
            elif cycle_pos < self.green_dur + self.yellow_dur:
                phase = PHASE_EW_YELLOW
            elif cycle_pos < 2 * self.green_dur + self.yellow_dur:
                phase = PHASE_NS_GREEN
            else:
                phase = PHASE_NS_YELLOW
            actions[node_id] = phase
            stagger = (stagger + 2) % 8
        self._step += 1
        return actions


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    episode_id:       int,
    output_dir:       Path,
    has_incident:     bool,
    incident_time:    int | None,
    incident_lane:    str | None,
    port:             int,
    seed:             int,
    sim_duration:     int = 3600,
    write_every:      int = 1,
) -> dict:
    """
    Run one complete SUMO simulation episode and write CSV logs.

    Args:
        episode_id:    Integer episode number.
        output_dir:    Directory to write CSV files.
        has_incident:  Whether to inject a traffic incident.
        incident_time: Simulation step to start the incident.
        incident_lane: SUMO lane ID to block.
        port:          TraCI port number (must be unique per parallel worker).
        seed:          Random seed passed to SUMO.
        sim_duration:  Number of simulation steps to run (default 3600).
        write_every:   Write a row every N steps to reduce I/O (default 1).

    Returns:
        Summary dict with episode metadata and aggregate statistics.
    """
    node_map         = load_node_map()
    intersections    = get_all_intersections(node_map)
    node_coords      = get_node_coords(node_map)
    hospital_node    = node_map.get("HOSPITAL_NODE", "D3")

    ep_str      = f"{episode_id:04d}"
    state_path  = output_dir / f"ep_{ep_str}_state.csv"
    inc_path    = output_dir / f"ep_{ep_str}_incident.csv"
    met_path    = output_dir / f"ep_{ep_str}_metrics.csv"

    incident_end  = (incident_time + INCIDENT_DURATION) if has_incident else None
    incident_node = _lane_to_node(incident_lane) if incident_lane else None

    # ---- SUMO launch ----
    sumo_cmd = [
        "sumo",
        "-c", str(CONFIG_FILE),
        "--seed", str(seed),
        "--ignore-route-errors",
        "--no-step-log",
        "--no-warnings",
        "--time-to-teleport", "300",
    ]

    try:
        traci.start(sumo_cmd, port=port)
    except Exception as exc:
        return {
            "episode": episode_id,
            "status": f"FAILED_LAUNCH: {exc}",
            "steps_completed": 0,
        }

    controller      = FixedTimeController()
    incident_active = False
    affected_vehs:  set[str] = set()

    waiting_history: list[float] = []
    speed_history:   list[float] = []

    # ---- Subscribe to lane variables once (avoids per-step individual calls) ----
    all_lane_ids: set[str] = set()
    for node_id in intersections:
        try:
            for lane_id in traci.trafficlight.getControlledLanes(node_id):
                all_lane_ids.add(lane_id)
        except Exception:
            pass
    for lane_id in all_lane_ids:
        try:
            traci.lane.subscribe(lane_id, [
                tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                tc.LAST_STEP_VEHICLE_NUMBER,
                tc.LAST_STEP_MEAN_SPEED,
                tc.LAST_STEP_VEHICLE_ID_LIST,
            ])
        except Exception:
            pass

    # Write buffers — flushed every BATCH_SIZE rows to reduce I/O calls
    BATCH_SIZE = 200
    state_buf:    list[dict] = []
    incident_buf: list[dict] = []
    metrics_buf:  list[dict] = []

    try:
        with (
            open(state_path, "w", newline="") as sf,
            open(inc_path,   "w", newline="") as inf_f,
            open(met_path,   "w", newline="") as mf,
        ):
            state_writer   = csv.DictWriter(sf,    fieldnames=STATE_COLUMNS)
            incident_writer = csv.DictWriter(inf_f, fieldnames=INCIDENT_COLUMNS)
            metrics_writer = csv.DictWriter(mf,    fieldnames=METRICS_COLUMNS)

            state_writer.writeheader()
            incident_writer.writeheader()
            metrics_writer.writeheader()

            for step in range(sim_duration):
                try:
                    traci.simulationStep()
                except traci.exceptions.FatalTraCIError:
                    break

                # ---- read network state using subscription results ----
                sub_results = traci.lane.getAllSubscriptionResults()
                node_states, ev_vehicles, total_waiting = _read_state_from_subscriptions(
                    intersections, sub_results
                )
                ev_active = bool(ev_vehicles)

                # ---- compute EV-related flags ----
                ev_nearby_set = _get_ev_nearby_nodes(ev_vehicles, node_coords, intersections, hop_distance)
                preempted_set = _get_preempted_nodes(ev_vehicles, intersections, node_coords)

                # ---- handle incident ----
                if has_incident and incident_time is not None and incident_lane is not None:
                    incident_active, affected_vehs = _step_incident(
                        step, incident_time, incident_end,
                        incident_lane, incident_active, affected_vehs,
                    )

                # ---- apply fixed-time control ----
                if step % CONTROL_FREQ == 0:
                    actions = controller.get_actions(intersections)
                    for node_id, phase in actions.items():
                        try:
                            current = traci.trafficlight.getPhase(node_id)
                            if current != phase and current not in (PHASE_EW_YELLOW, PHASE_NS_YELLOW):
                                traci.trafficlight.setPhase(node_id, phase)
                        except traci.exceptions.TraCIException:
                            pass

                # ---- accumulate metrics every step for history, write every write_every steps ----
                speeds = [s["avg_speed"] for s in node_states.values() if s["avg_speed"] > 0]
                net_speed = float(np.mean(speeds)) if speeds else 0.0
                total_vehicles = sum(s["vehicle_count"] for s in node_states.values())
                waiting_history.append(total_waiting)
                speed_history.append(net_speed)

                # ---- buffer rows (only on sampled steps) ----
                if step % write_every != 0:
                    continue

                for node_id in intersections:
                    ns = node_states.get(node_id, _empty_node_state())
                    state_buf.append({
                        "episode":       episode_id,
                        "step":          step,
                        "node_id":       node_id,
                        "waiting":       ns["waiting"],
                        "queue":         ns["queue"],
                        "avg_speed":     ns["avg_speed"],
                        "vehicle_count": ns["vehicle_count"],
                        "current_phase": ns["current_phase"],
                        "ev_nearby":     1 if node_id in ev_nearby_set else 0,
                        "preempted":     1 if node_id in preempted_set else 0,
                    })

                    if has_incident and incident_active and incident_node:
                        dist = hop_distance(node_id, incident_node, node_coords)
                        inc_active_flag = 1
                    else:
                        dist            = 999
                        inc_active_flag = 0

                    incident_buf.append({
                        "episode":           episode_id,
                        "step":              step,
                        "node_id":           node_id,
                        "incident_active":   inc_active_flag,
                        "incident_lane":     incident_lane if (has_incident and incident_active) else "",
                        "incident_distance": dist,
                    })

                metrics_buf.append({
                    "episode":           episode_id,
                    "step":              step,
                    "total_waiting":     total_waiting,
                    "total_vehicles":    total_vehicles,
                    "network_avg_speed": round(net_speed, 3),
                    "ev_active":         1 if ev_active else 0,
                    "incident_active":   1 if incident_active else 0,
                    "incident_start":    incident_time if has_incident else "",
                    "incident_end":      incident_end  if has_incident else "",
                    "incident_lane":     incident_lane if has_incident else "",
                })

                # ---- flush buffers in bulk ----
                if len(metrics_buf) >= BATCH_SIZE:
                    state_writer.writerows(state_buf);    state_buf.clear()
                    incident_writer.writerows(incident_buf); incident_buf.clear()
                    metrics_writer.writerows(metrics_buf);   metrics_buf.clear()

            # ---- flush remaining buffered rows ----
            if state_buf:
                state_writer.writerows(state_buf)
            if incident_buf:
                incident_writer.writerows(incident_buf)
            if metrics_buf:
                metrics_writer.writerows(metrics_buf)
                speed_history.append(net_speed)

    except Exception:
        traceback.print_exc()
    finally:
        try:
            traci.close()
        except Exception:
            pass

    return {
        "episode":         episode_id,
        "status":          "OK",
        "steps_completed": len(waiting_history),
        "has_incident":    has_incident,
        "incident_time":   incident_time,
        "incident_lane":   incident_lane,
        "avg_waiting":     round(float(np.mean(waiting_history)), 2) if waiting_history else 0,
        "peak_waiting":    int(max(waiting_history)) if waiting_history else 0,
        "avg_speed":       round(float(np.mean(speed_history)), 3)   if speed_history  else 0,
        "state_file":      str(state_path),
        "incident_file":   str(inc_path),
        "metrics_file":    str(met_path),
    }


# ---------------------------------------------------------------------------
# Helper: read network state
# ---------------------------------------------------------------------------

def _read_state(intersections: list[str]) -> tuple[dict, dict, int]:
    state: dict = {}
    ev_vehicles: dict = {}
    total_waiting = 0

    for node_id in intersections:
        try:
            lane_ids = list(set(traci.trafficlight.getControlledLanes(node_id)))
            waiting       = 0
            vehicle_count = 0
            speed_sum     = 0.0
            speed_n       = 0

            for lane_id in lane_ids:
                try:
                    waiting       += traci.lane.getLastStepHaltingNumber(lane_id)
                    vehicle_count += traci.lane.getLastStepVehicleNumber(lane_id)
                    speed_sum     += traci.lane.getLastStepMeanSpeed(lane_id)
                    speed_n       += 1
                    for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                        if vid.startswith(EV_PREFIX):
                            ev_vehicles[vid] = lane_id
                except traci.exceptions.TraCIException:
                    pass

            try:
                phase = traci.trafficlight.getPhase(node_id)
            except traci.exceptions.TraCIException:
                phase = -1

            state[node_id] = {
                "waiting":       waiting,
                "queue":         waiting,
                "avg_speed":     round(speed_sum / speed_n, 3) if speed_n else 0.0,
                "vehicle_count": vehicle_count,
                "current_phase": phase,
            }
            total_waiting += waiting

        except traci.exceptions.TraCIException:
            state[node_id] = _empty_node_state()

    return state, ev_vehicles, total_waiting


def _empty_node_state() -> dict:
    return {"waiting": 0, "queue": 0, "avg_speed": 0.0, "vehicle_count": 0, "current_phase": -1}


def _read_state_from_subscriptions(
    intersections: list[str],
    sub_results:   dict,
) -> tuple[dict, dict, int]:
    """
    Build network state from pre-fetched TraCI subscription results.

    Replaces _read_state() with a single bulk dict lookup instead of
    hundreds of individual TraCI round-trips per step.

    Args:
        intersections: List of junction IDs.
        sub_results:   Dict of {lane_id: {var_id: value}} from
                       traci.lane.getAllSubscriptionResults().

    Returns:
        (state dict, emergency_vehicles dict, total_waiting int)
    """
    state: dict = {}
    ev_vehicles: dict = {}
    total_waiting = 0

    for node_id in intersections:
        try:
            lane_ids = list(set(traci.trafficlight.getControlledLanes(node_id)))
        except Exception:
            state[node_id] = _empty_node_state()
            continue

        waiting       = 0
        vehicle_count = 0
        speed_sum     = 0.0
        speed_n       = 0

        for lane_id in lane_ids:
            res = sub_results.get(lane_id, {})
            if not res:
                continue
            waiting       += res.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
            vehicle_count += res.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
            speed_sum     += res.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
            speed_n       += 1
            for vid in res.get(tc.LAST_STEP_VEHICLE_ID_LIST, []):
                if vid.startswith(EV_PREFIX):
                    ev_vehicles[vid] = lane_id

        try:
            phase = traci.trafficlight.getPhase(node_id)
        except Exception:
            phase = -1

        state[node_id] = {
            "waiting":       waiting,
            "queue":         waiting,
            "avg_speed":     round(speed_sum / speed_n, 3) if speed_n else 0.0,
            "vehicle_count": vehicle_count,
            "current_phase": phase,
        }
        total_waiting += waiting

    return state, ev_vehicles, total_waiting


# ---------------------------------------------------------------------------
# Helper: EV-related flags
# ---------------------------------------------------------------------------

def _get_ev_nearby_nodes(
    ev_vehicles:   dict,
    node_coords:   dict,
    intersections: list[str],
    hop_fn,
    max_hops:      int = 2,
) -> set[str]:
    nearby: set[str] = set()
    if not ev_vehicles:
        return nearby
    for ev_lane in ev_vehicles.values():
        ev_node = _lane_to_node(ev_lane)
        if not ev_node:
            continue
        for node_id in intersections:
            if hop_fn(node_id, ev_node, node_coords) <= max_hops:
                nearby.add(node_id)
    return nearby


def _get_preempted_nodes(
    ev_vehicles:   dict,
    intersections: list[str],
    node_coords:   dict,
    lookahead:     int = 3,
) -> set[str]:
    """Simple heuristic: nodes within ``lookahead`` hops ahead of EV."""
    preempted: set[str] = set()
    if not ev_vehicles:
        return preempted
    for ev_lane in ev_vehicles.values():
        ev_node = _lane_to_node(ev_lane)
        if not ev_node:
            continue
        for node_id in intersections:
            if 0 < hop_distance(node_id, ev_node, node_coords) <= lookahead:
                preempted.add(node_id)
    return preempted


def _lane_to_node(lane_id: str | None) -> str | None:
    """Extract source junction ID from a lane or edge ID."""
    if not lane_id:
        return None
    edge_id = lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id
    if edge_id.startswith(":"):
        return None
    parts = edge_id.split("-")
    return parts[0] if parts else None


# ---------------------------------------------------------------------------
# Helper: incident stepping
# ---------------------------------------------------------------------------

def _step_incident(
    step:          int,
    start:         int,
    end:           int | None,
    lane_id:       str,
    was_active:    bool,
    affected:      set[str],
) -> tuple[bool, set[str]]:
    """Apply or release a traffic incident. Returns (is_active, affected_set)."""
    if end is None:
        end = start + INCIDENT_DURATION

    if start <= step <= end:
        try:
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                if vid not in affected:
                    try:
                        traci.vehicle.setSpeed(vid, 0.0)
                        affected.add(vid)
                    except traci.exceptions.TraCIException:
                        pass
        except traci.exceptions.TraCIException:
            pass
        return True, affected

    if was_active and step > end:
        for vid in affected:
            try:
                traci.vehicle.setSpeed(vid, -1)
            except traci.exceptions.TraCIException:
                pass
        return False, set()

    return False, affected


# ---------------------------------------------------------------------------
# Episode scheduler
# ---------------------------------------------------------------------------

def build_episode_plan(
    n_episodes:     int,
    baseline_ratio: float,
    seed_base:      int,
    node_map:       dict,
) -> list[dict]:
    """
    Build a list of episode configurations.

    Args:
        n_episodes:     Total number of episodes.
        baseline_ratio: Fraction of episodes without incidents (0.0–1.0).
        seed_base:      Base random seed; each episode gets seed_base + episode_id.
        node_map:       Node map for picking valid incident lanes.

    Returns:
        List of dicts, one per episode, with all parameters needed by run_episode.
    """
    rng = random.Random(seed_base)
    candidate_lanes = _get_candidate_incident_lanes(node_map)

    n_baseline = int(n_episodes * baseline_ratio)
    plan       = []

    for ep in range(n_episodes):
        is_baseline = ep < n_baseline

        if is_baseline:
            cfg = {
                "episode_id":   ep + 1,
                "has_incident": False,
                "incident_time": None,
                "incident_lane": None,
            }
        else:
            inc_time = rng.randint(MIN_INCIDENT_START, MAX_INCIDENT_START)
            inc_lane = rng.choice(candidate_lanes) if candidate_lanes else "e_2_2_2_3_0"
            cfg = {
                "episode_id":    ep + 1,
                "has_incident":  True,
                "incident_time": inc_time,
                "incident_lane": inc_lane,
            }

        cfg["seed"] = seed_base + ep
        plan.append(cfg)

    rng.shuffle(plan)
    for idx, cfg in enumerate(plan):
        cfg["episode_id"] = idx + 1

    return plan


def _get_candidate_incident_lanes(node_map: dict) -> list[str]:
    """
    Return plausible lane IDs derived from the node map for incident injection.

    These are constructed edge IDs following SUMO's naming convention for
    the 6×6 netgenerate grid.
    """
    lanes = []
    for row in range(GRID_SIZE - 1):
        for col in range(GRID_SIZE):
            node_a = node_map.get(f"NODE_{col}_{row}")
            node_b = node_map.get(f"NODE_{col}_{row+1}")
            if node_a and node_b:
                lanes.append(f"{node_a}{node_b}_0")
                lanes.append(f"{node_a}{node_b}_1")
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE - 1):
            node_a = node_map.get(f"NODE_{col}_{row}")
            node_b = node_map.get(f"NODE_{col+1}_{row}")
            if node_a and node_b:
                lanes.append(f"{node_a}{node_b}_0")
    return lanes


# ---------------------------------------------------------------------------
# Serial runner
# ---------------------------------------------------------------------------

def run_serial(
    plan:         list[dict],
    output_dir:   Path,
    base_port:    int = 8813,
    sim_duration: int = 3600,
    write_every:  int = 1,
) -> list[dict]:
    """Run all episodes sequentially in the current process."""
    results = []

    for cfg in plan:
        ep   = cfg["episode_id"]
        n    = len(plan)
        mode = "INCIDENT " if cfg["has_incident"] else "BASELINE"

        print(
            f"  [{ep:>4}/{n}]  {mode}  "
            + (f"t={cfg['incident_time']}s  lane={cfg['incident_lane']}" if cfg["has_incident"] else "")
        )

        t0 = time.time()
        result = run_episode(
            episode_id    = ep,
            output_dir    = output_dir,
            has_incident  = cfg["has_incident"],
            incident_time = cfg.get("incident_time"),
            incident_lane = cfg.get("incident_lane"),
            port          = base_port,
            seed          = cfg["seed"],
            sim_duration  = sim_duration,
            write_every   = write_every,
        )
        elapsed = time.time() - t0

        status = result.get("status", "?")
        if status == "OK":
            print(
                f"          OK  {result['steps_completed']}s  "
                f"avg_wait={result['avg_waiting']:.1f}  "
                f"peak_wait={result['peak_waiting']}  "
                f"({elapsed:.1f}s wall)"
            )
        else:
            print(f"          FAILED: {status}")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Parallel runner — Windows + Linux/macOS via multiprocessing (spawn-safe)
# ---------------------------------------------------------------------------

def _episode_worker(kwargs: dict) -> dict:
    """Top-level picklable worker — required for multiprocessing on Windows."""
    return run_episode(**kwargs)


def run_parallel(
    plan:         list[dict],
    output_dir:   Path,
    n_workers:    int,
    base_port:    int = 8820,
    sim_duration: int = 3600,
    write_every:  int = 1,
) -> list[dict]:
    """
    Run episodes in parallel using multiprocessing.Pool with explicit 'spawn'.

    'spawn' is the safe start method on Windows (avoids WinAPI/TraCI socket
    conflicts that occur with the default 'fork' on some environments).
    Each worker gets a unique TraCI port so SUMO instances don't collide.
    """
    ctx = multiprocessing.get_context("spawn")

    job_list = []
    for idx, cfg in enumerate(plan):
        port = base_port + (idx % n_workers)
        job_list.append({
            "episode_id":    cfg["episode_id"],
            "output_dir":    output_dir,
            "has_incident":  cfg["has_incident"],
            "incident_time": cfg.get("incident_time"),
            "incident_lane": cfg.get("incident_lane"),
            "port":          port,
            "seed":          cfg["seed"],
            "sim_duration":  sim_duration,
            "write_every":   write_every,
        })

    results: list[dict] = []
    n = len(plan)

    with ctx.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_episode_worker, job_list):
            ep     = result.get("episode", "?")
            status = result.get("status", "?")
            if status == "OK":
                print(
                    f"  [{ep:>4}/{n}]  OK  "
                    f"avg_wait={result['avg_waiting']:.1f}  "
                    f"peak_wait={result['peak_waiting']}"
                )
            else:
                print(f"  [{ep:>4}/{n}]  FAILED: {status}")
            results.append(result)

    results.sort(key=lambda r: r.get("episode", 0))
    return results

    results.sort(key=lambda r: r.get("episode", 0))
    return results


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(results: list[dict], output_dir: Path) -> Path:
    summary_path = output_dir.parent / "data_collection_summary.csv"

    fieldnames = [
        "episode", "status", "has_incident", "incident_time", "incident_lane",
        "steps_completed", "avg_waiting", "peak_waiting", "avg_speed",
        "state_file", "incident_file", "metrics_file",
    ]

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    return summary_path


# ---------------------------------------------------------------------------
# Progress bar (no dependencies)
# ---------------------------------------------------------------------------

def print_progress(current: int, total: int, width: int = 40) -> None:
    pct   = current / total if total else 0
    filled = int(width * pct)
    bar   = "█" * filled + "░" * (width - filled)
    print(f"\r  [{bar}] {current}/{total} ({pct*100:.0f}%)", end="", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NetSignal Phase 1 — automated SUMO data collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--episodes",       type=int,   default=500,
                        help="Total number of simulation episodes (default: 500)")
    parser.add_argument("--baseline-ratio", type=float, default=0.2,
                        help="Fraction of episodes without incidents (default: 0.2 = 20%%)")
    parser.add_argument("--baseline-only",  action="store_true",
                        help="Run all episodes without incidents")
    parser.add_argument("--incident-only",  action="store_true",
                        help="Run all episodes with incidents")
    parser.add_argument("--workers",        type=int,   default=1,
                        help="Number of parallel workers (default: 1 = serial). Works on Windows.")
    parser.add_argument("--sim-duration",   type=int,   default=3600,
                        help="Simulation steps per episode (default: 3600 = 1 hr). "
                             "Use 1800 for 2x speedup.")
    parser.add_argument("--write-every",    type=int,   default=1,
                        help="Write a CSV row every N steps (default: 1 = every step). "
                             "Use 5 for 5x less I/O.")
    parser.add_argument("--output-dir",     type=str,   default=None,
                        help="Output directory (default: <project>/output/episodes)")
    parser.add_argument("--port",           type=int,   default=8813,
                        help="Base TraCI port (default: 8813)")
    parser.add_argument("--seed",           type=int,   default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Print the episode plan without running simulations")

    args = parser.parse_args()

    sim_duration = max(1, args.sim_duration)
    write_every  = max(1, args.write_every)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _PROJECT_DIR / "output" / "episodes"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate config
    if not CONFIG_FILE.exists():
        sys.exit(
            f"ERROR: SUMO config not found at {CONFIG_FILE}\n"
            "Run generate_network.py and fix_flows.py first."
        )

    node_map = load_node_map()
    if not node_map:
        sys.exit(
            "ERROR: Node map is empty. Run fix_flows.py first."
        )

    # Determine baseline ratio
    if args.baseline_only and args.incident_only:
        sys.exit("ERROR: --baseline-only and --incident-only are mutually exclusive.")
    if args.baseline_only:
        baseline_ratio = 1.0
    elif args.incident_only:
        baseline_ratio = 0.0
    else:
        baseline_ratio = max(0.0, min(1.0, args.baseline_ratio))

    n_baseline = int(args.episodes * baseline_ratio)
    n_incident = args.episodes - n_baseline

    # ---- Banner ----
    W = 66
    print(f"\n┌{'─'*(W-2)}┐")
    print(f"│  NetSignal — Phase 1 Data Collection{' '*(W-39)}│")
    print(f"├{'─'*(W-2)}┤")
    print(f"│  Episodes      : {args.episodes:<{W-20}}│")
    print(f"│  Baseline      : {n_baseline} ({baseline_ratio*100:.0f}%){' '*(W-22-len(str(n_baseline))-len(f'{baseline_ratio*100:.0f}'))}│")
    print(f"│  With Incident : {n_incident} ({(1-baseline_ratio)*100:.0f}%){' '*(W-22-len(str(n_incident))-len(f'{(1-baseline_ratio)*100:.0f}'))}│")
    print(f"│  Workers       : {args.workers:<{W-20}}│")
    print(f"│  Sim duration  : {sim_duration}s per episode{' '*(W-32-len(str(sim_duration)))}│")
    print(f"│  Write every   : every {write_every} step(s){' '*(W-26-len(str(write_every)))}│")
    print(f"│  Output        : {str(output_dir)[-44:]:<{W-20}}│")
    print(f"│  Seed          : {args.seed:<{W-20}}│")
    est_hrs = args.episodes * sim_duration / 3600 / max(args.workers, 1)
    print(f"│  Est. time     : ~{est_hrs:.0f}h (rough){' '*(W-28-len(f'{est_hrs:.0f}'))}│")
    print(f"└{'─'*(W-2)}┘\n")

    # ---- Build plan ----
    plan = build_episode_plan(args.episodes, baseline_ratio, args.seed, node_map)

    if args.dry_run:
        print("  DRY RUN — Episode plan (first 20 shown):\n")
        print(f"  {'EP':>4}  {'TYPE':<10}  {'INC_TIME':>8}  LANE")
        print(f"  {'─'*4}  {'─'*10}  {'─'*8}  {'─'*30}")
        for cfg in plan[:20]:
            mode = "INCIDENT" if cfg["has_incident"] else "BASELINE"
            t    = str(cfg.get("incident_time", ""))
            lane = str(cfg.get("incident_lane", ""))
            print(f"  {cfg['episode_id']:>4}  {mode:<10}  {t:>8}  {lane}")
        if len(plan) > 20:
            print(f"  ... ({len(plan) - 20} more)")
        return

    # ---- Run ----
    wall_start = time.time()
    print(f"  Starting collection — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.workers > 1:
        results = run_parallel(
            plan, output_dir,
            n_workers    = args.workers,
            base_port    = args.port,
            sim_duration = sim_duration,
            write_every  = write_every,
        )
    else:
        results = run_serial(
            plan, output_dir,
            base_port    = args.port,
            sim_duration = sim_duration,
            write_every  = write_every,
        )

    wall_time = time.time() - wall_start

    # ---- Write summary ----
    summary_path = write_summary(results, output_dir)

    # ---- Final report ----
    ok_count   = sum(1 for r in results if r.get("status") == "OK")
    fail_count = len(results) - ok_count
    ok_results = [r for r in results if r.get("status") == "OK"]

    print(f"\n┌{'─'*(W-2)}┐")
    print(f"│  Collection Complete{' '*(W-22)}│")
    print(f"├{'─'*(W-2)}┤")
    print(f"│  Successful    : {ok_count}/{len(results)}{' '*(W-22-len(str(ok_count))-len(str(len(results))))}│")
    if fail_count:
        print(f"│  Failed        : {fail_count:<{W-20}}│")

    if ok_results:
        avg_w  = np.mean([r["avg_waiting"]  for r in ok_results])
        peak_w = max(    [r["peak_waiting"] for r in ok_results])
        avg_s  = np.mean([r["avg_speed"]    for r in ok_results])
        print(f"│  Mean avg wait : {avg_w:.1f} vehicles{' '*(W-28-len(f'{avg_w:.1f}'))}│")
        print(f"│  Peak waiting  : {peak_w} vehicles{' '*(W-27-len(str(peak_w)))}│")
        print(f"│  Mean speed    : {avg_s:.2f} m/s{' '*(W-25-len(f'{avg_s:.2f}'))}│")

    h, rem = divmod(int(wall_time), 3600)
    m, s   = divmod(rem, 60)
    print(f"│  Wall time     : {h}h {m}m {s}s{' '*(W-26-len(str(h))-len(str(m))-len(str(s)))}│")
    print(f"│  Summary CSV   : {str(summary_path)[-44:]:<{W-20}}│")
    print(f"│  Episode dir   : {str(output_dir)[-44:]:<{W-20}}│")
    print(f"└{'─'*(W-2)}┘\n")

    # ---- Dataset statistics ----
    inc_eps  = [r for r in ok_results if r.get("has_incident")]
    base_eps = [r for r in ok_results if not r.get("has_incident")]

    print("  Dataset breakdown:")
    print(f"    Baseline episodes  : {len(base_eps)}")
    print(f"    Incident episodes  : {len(inc_eps)}")
    total_steps = sum(r.get("steps_completed", 0) for r in ok_results)
    total_rows  = total_steps * GRID_SIZE * GRID_SIZE
    print(f"    Total state rows   : {total_rows:,}  "
          f"({total_steps:,} steps × {GRID_SIZE*GRID_SIZE} nodes)")
    print(f"\n  Ready for Phase 2 — BiLSTM training.\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()   # required for Windows PyInstaller / spawn safety
    main()
