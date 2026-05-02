#!/usr/bin/env python3
"""
traciloop.py
============
Main TraCI control loop for the NetSignal SUMO simulation environment.

Responsibilities:
    - Launch SUMO (GUI or headless) and connect via TraCI
    - Read full network state each simulation step
    - Detect emergency vehicles and preempt signals along their route
    - Inject optional traffic incidents for labelled training data
    - Log state, metrics, and incident data to CSV
    - Print a live console dashboard
    - Save a summary plot on completion

Usage:
    python scripts/traciloop.py                         # headless
    python scripts/traciloop.py --gui                   # GUI
    python scripts/traciloop.py --gui --incident        # GUI + incident at t=600s
    python scripts/traciloop.py --incident --incident-time 300 --incident-lane e_1_2_1_3_0
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import networkx as nx

try:
    import traci # type: ignore[import-untyped]
    import traci.constants as tc # type: ignore[import-untyped]
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
OUTPUT_DIR   = _PROJECT_DIR / "output"
_NODE_MAP_FILE = _PROJECT_DIR / "config" / "grid_nodes.txt"


# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

TRACI_PORT            = 8813
STEP_LENGTH           = 1      # seconds per simulation step
SIM_DURATION          = 3600   # total simulation steps (1 hour)
CONTROL_FREQ          = 5      # signal control interval (steps)
DASHBOARD_FREQ        = 30     # console print interval (steps)
EV_PREFIX             = "EV_"  # vehicle ID prefix marking emergency vehicles
GRID_SIZE             = 6

DEFAULT_GREEN_DURATION  = 30   # seconds
DEFAULT_YELLOW_DURATION = 3    # seconds

PHASE_EW_GREEN  = 0
PHASE_EW_YELLOW = 1
PHASE_NS_GREEN  = 2
PHASE_NS_YELLOW = 3

_PHASE_LABELS = {
    PHASE_EW_GREEN:  "EW green",
    PHASE_EW_YELLOW: "EW yellow",
    PHASE_NS_GREEN:  "NS green",
    PHASE_NS_YELLOW: "NS yellow",
}

_DASHBOARD_WIDTH = 68


# ---------------------------------------------------------------------------
# Node map
# ---------------------------------------------------------------------------

def load_node_map() -> dict[str, str]:
    """Load the column/row → SUMO junction ID mapping from grid_nodes.txt."""
    node_map: dict[str, str] = {}
    if not _NODE_MAP_FILE.exists():
        return node_map
    for line in _NODE_MAP_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        node_map[key.strip()] = value.strip()
    return node_map


_NODE_MAP       = load_node_map()
HOSPITAL_NODE   = _NODE_MAP.get("HOSPITAL_NODE", "D3")
ALL_INTERSECTIONS = [
    _NODE_MAP.get(f"NODE_{col}_{row}", f"{col}/{row}")
    for row in range(GRID_SIZE)
    for col in range(GRID_SIZE)
    if _NODE_MAP.get(f"NODE_{col}_{row}")
]


# ---------------------------------------------------------------------------
# Road graph
# ---------------------------------------------------------------------------

def build_road_graph() -> nx.DiGraph:
    """
    Build a directed NetworkX graph of the 6×6 intersection grid.

    Nodes represent intersections identified by their SUMO junction ID.
    Edges represent road segments with a uniform weight of 300 m.
    Used for Dijkstra shortest-path computation during EV preemption.
    """
    graph: nx.DiGraph = nx.DiGraph()

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            node_id = f"{col}/{row}"
            graph.add_node(node_id, pos=(col * 300, row * 300))

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE - 1):
            graph.add_edge(f"{col}/{row}",   f"{col+1}/{row}", weight=300)
            graph.add_edge(f"{col+1}/{row}", f"{col}/{row}",   weight=300)

    for row in range(GRID_SIZE - 1):
        for col in range(GRID_SIZE):
            graph.add_edge(f"{col}/{row}",   f"{col}/{row+1}", weight=300)
            graph.add_edge(f"{col}/{row+1}", f"{col}/{row}",   weight=300)

    return graph


# ---------------------------------------------------------------------------
# Network state reader
# ---------------------------------------------------------------------------

def read_network_state() -> tuple[dict, dict, int]:
    """
    Query SUMO for the current state of every intersection.

    Returns:
        state: Mapping of junction ID to a dict with keys:
            lanes, waiting, queue, avg_speed, vehicle_count, current_phase.
        emergency_vehicles: Mapping of EV vehicle ID to its current lane ID.
        total_waiting: Sum of waiting vehicle counts across all intersections.
    """
    state: dict = {}
    emergency_vehicles: dict = {}
    total_waiting = 0

    for node_id in ALL_INTERSECTIONS:
        try:
            lane_ids = list(set(traci.trafficlight.getControlledLanes(node_id)))
            waiting_count  = 0
            vehicle_count  = 0
            speed_sum      = 0.0
            speed_readings = 0

            for lane_id in lane_ids:
                try:
                    waiting_count  += traci.lane.getLastStepHaltingNumber(lane_id)
                    vehicle_count  += traci.lane.getLastStepVehicleNumber(lane_id)
                    speed_sum      += traci.lane.getLastStepMeanSpeed(lane_id)
                    speed_readings += 1

                    for vehicle_id in traci.lane.getLastStepVehicleIDs(lane_id):
                        if vehicle_id.startswith(EV_PREFIX):
                            emergency_vehicles[vehicle_id] = lane_id
                except traci.exceptions.TraCIException:
                    pass

            avg_speed = speed_sum / speed_readings if speed_readings > 0 else 0.0

            try:
                current_phase = traci.trafficlight.getPhase(node_id)
            except traci.exceptions.TraCIException:
                current_phase = -1

            state[node_id] = {
                "lanes":         lane_ids,
                "waiting":       waiting_count,
                "queue":         waiting_count,
                "avg_speed":     round(avg_speed, 3),
                "vehicle_count": vehicle_count,
                "current_phase": current_phase,
            }
            total_waiting += waiting_count

        except traci.exceptions.TraCIException:
            state[node_id] = {
                "lanes":         [],
                "waiting":       0,
                "queue":         0,
                "avg_speed":     0.0,
                "vehicle_count": 0,
                "current_phase": -1,
            }

    return state, emergency_vehicles, total_waiting


# ---------------------------------------------------------------------------
# Fixed-time controller
# ---------------------------------------------------------------------------

class FixedTimeController:
    """
    Baseline fixed-time signal controller.

    All intersections cycle through EW-green → EW-yellow → NS-green → NS-yellow
    on a fixed schedule, staggered slightly per intersection to avoid
    simultaneous phase changes across the network.

    Replace ``get_actions`` with a GNN-based policy to use adaptive control.
    """

    def __init__(
        self,
        green_duration: int = DEFAULT_GREEN_DURATION,
        yellow_duration: int = DEFAULT_YELLOW_DURATION,
    ) -> None:
        self.green_dur  = green_duration
        self.yellow_dur = yellow_duration
        self.cycle_len  = 2 * (green_duration + yellow_duration)
        self._step      = 0

    def get_actions(self, state: dict) -> dict[str, int]:
        """
        Compute signal phase assignments for all intersections.

        Args:
            state: Current network state (unused by fixed-time logic,
                   kept for interface parity with adaptive controllers).

        Returns:
            Mapping of junction ID to phase index.
        """
        actions: dict[str, int] = {}
        stagger = 0

        for node_id in ALL_INTERSECTIONS:
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
# Emergency vehicle preemptor
# ---------------------------------------------------------------------------

class EVPreemptor:
    """
    Signal preemption controller for emergency vehicles.

    On detection of an EV, computes the shortest path from the EV's current
    position to the hospital node using Dijkstra's algorithm and preempts
    every intersection along that corridor with a green phase aligned to
    the direction of travel.
    """

    def __init__(self, road_graph: nx.DiGraph) -> None:
        self._graph  = road_graph
        self.active  = False
        self.path: list[str] = []
        self._ev_id: str | None = None

    def update(self, emergency_vehicles: dict) -> None:
        """
        Refresh the preemption corridor from the detected EV positions.

        Args:
            emergency_vehicles: Mapping of EV vehicle ID to current lane ID.
        """
        if not emergency_vehicles:
            if self.active:
                print("\n  EV has cleared the network — releasing preemption")
            self.active  = False
            self.path    = []
            self._ev_id  = None
            return

        ev_id, current_lane = next(iter(emergency_vehicles.items()))

        if ev_id != self._ev_id:
            self._ev_id = ev_id
            print(f"\n  EMERGENCY VEHICLE DETECTED: {ev_id}")
            print(f"  Current lane: {current_lane}")

        current_node = self._resolve_node(current_lane, ev_id)
        if current_node is None:
            return

        try:
            self.path   = nx.dijkstra_path(self._graph, source=current_node, target=HOSPITAL_NODE)
            self.active = True
            print(f"  Preemption corridor: {' -> '.join(self.path)}")
        except nx.NetworkXNoPath:
            print(f"  WARNING: No path from {current_node} to {HOSPITAL_NODE}")
            self.active = False

    def get_preempted_intersections(self) -> set[str]:
        """Return the set of junction IDs currently under preemption."""
        return set(self.path) if self.active else set()

    def get_phase_for_intersection(self, node_id: str) -> int | None:
        """
        Determine the green phase direction for a preempted intersection.

        Args:
            node_id: SUMO junction ID.

        Returns:
            Phase index (EW or NS green) aligned to EV travel direction,
            or None if the junction is not on the active corridor.
        """
        if node_id not in self.path:
            return None

        idx = self.path.index(node_id)
        if idx >= len(self.path) - 1:
            return PHASE_NS_GREEN

        rev_map = {
            v: (int(k.split("_")[1]), int(k.split("_")[2]))
            for k, v in _NODE_MAP.items()
            if k.startswith("NODE_")
        }
        col_a, row_a = rev_map.get(self.path[idx],     (0, 0))
        col_b, row_b = rev_map.get(self.path[idx + 1], (0, 0))

        return PHASE_EW_GREEN if row_a == row_b else PHASE_NS_GREEN

    def _resolve_node(self, lane_id: str, ev_id: str) -> str | None:
        """
        Derive the source junction ID from an EV's current road or lane.

        Args:
            lane_id: Lane ID reported by TraCI.
            ev_id:   Vehicle ID used for a direct road query.

        Returns:
            Junction ID string, or None if resolution fails.
        """
        try:
            road_id = traci.vehicle.getRoadID(ev_id)
            if road_id and not road_id.startswith(":") and "-" in road_id:
                return road_id.split("-")[0]
        except traci.exceptions.TraCIException:
            pass

        if lane_id and "-" in lane_id:
            edge_id = lane_id.rsplit("_", 1)[0]
            return edge_id.split("-")[0]

        return None


# ---------------------------------------------------------------------------
# Incident injector
# ---------------------------------------------------------------------------

class IncidentInjector:
    """
    Simulated traffic incident for generating labelled training data.

    Forces vehicles to a standstill on a target lane between
    ``start_time`` and ``end_time``, then releases them.
    """

    _DEFAULT_DURATION = 300

    def __init__(
        self,
        target_lane: str,
        start_time: int,
        end_time: int | None = None,
        label: str = "blockage",
    ) -> None:
        self.target_lane = target_lane
        self.start_time  = start_time
        self.end_time    = end_time if end_time is not None else start_time + self._DEFAULT_DURATION
        self.label       = label
        self.active      = False
        self._affected:  set[str] = set()

    def step(self, sim_time: int) -> bool:
        """
        Advance incident state by one simulation step.

        Args:
            sim_time: Current simulation time in seconds.

        Returns:
            True while the incident is active.
        """
        if self.start_time <= sim_time <= self.end_time:
            if not self.active:
                print(f"\n  INCIDENT at t={sim_time}s | lane: {self.target_lane}")
                self.active = True

            try:
                for vehicle_id in traci.lane.getLastStepVehicleIDs(self.target_lane):
                    if vehicle_id not in self._affected:
                        try:
                            traci.vehicle.setSpeed(vehicle_id, 0.0)
                            self._affected.add(vehicle_id)
                        except traci.exceptions.TraCIException:
                            pass
            except traci.exceptions.TraCIException:
                pass

            return True

        if self.active and sim_time > self.end_time:
            print(f"\n  Incident cleared at t={sim_time}s")
            for vehicle_id in self._affected:
                try:
                    traci.vehicle.setSpeed(vehicle_id, -1)
                except traci.exceptions.TraCIException:
                    pass
            self._affected.clear()
            self.active = False

        return False


# ---------------------------------------------------------------------------
# Data logger
# ---------------------------------------------------------------------------

class DataLogger:
    """
    Writes per-step simulation data to timestamped CSV files.

    Output files (written to ``output/``):
        state_log_<ts>.csv    Per-intersection state every timestep.
        metrics_log_<ts>.csv  Network-wide aggregate metrics every control step.
        incident_log_<ts>.csv Ground-truth incident labels.
    """

    _STATE_HEADER   = ["timestep", "intersection", "waiting", "queue",
                       "avg_speed", "vehicle_count", "current_phase",
                       "ev_active", "incident_active"]
    _METRICS_HEADER = ["timestep", "total_waiting", "total_vehicles",
                       "avg_speed_network", "ev_active", "incident_active"]
    _INCIDENT_HEADER = ["timestep", "lane", "label"]

    def __init__(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._state_path    = OUTPUT_DIR / f"state_log_{ts}.csv"
        self._metrics_path  = OUTPUT_DIR / f"metrics_log_{ts}.csv"
        self._incident_path = OUTPUT_DIR / f"incident_log_{ts}.csv"

        self._write_header(self._state_path,    self._STATE_HEADER)
        self._write_header(self._metrics_path,  self._METRICS_HEADER)
        self._write_header(self._incident_path, self._INCIDENT_HEADER)

    def log_state(self, timestep: int, state: dict, ev_active: bool, incident_active: bool) -> None:
        """Append one row per intersection to the state log."""
        with open(self._state_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            for node_id, data in state.items():
                writer.writerow([
                    timestep, node_id,
                    data["waiting"], data["queue"], data["avg_speed"],
                    data["vehicle_count"], data["current_phase"],
                    int(ev_active), int(incident_active),
                ])

    def log_metrics(self, timestep: int, total_waiting: int, state: dict,
                    ev_active: bool, incident_active: bool) -> None:
        """Append one network-wide metrics row to the metrics log."""
        total_vehicles = sum(s["vehicle_count"] for s in state.values())
        speeds = [s["avg_speed"] for s in state.values() if s["avg_speed"] > 0]
        avg_speed = round(float(np.mean(speeds)), 3) if speeds else 0.0

        with open(self._metrics_path, "a", newline="") as fh:
            csv.writer(fh).writerow([
                timestep, total_waiting, total_vehicles,
                avg_speed, int(ev_active), int(incident_active),
            ])

    def log_incident(self, timestep: int, lane: str, label: str) -> None:
        """Append one ground-truth incident row to the incident log."""
        with open(self._incident_path, "a", newline="") as fh:
            csv.writer(fh).writerow([timestep, lane, label])

    @staticmethod
    def _write_header(path: Path, header: list[str]) -> None:
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(header)


# ---------------------------------------------------------------------------
# Console dashboard
# ---------------------------------------------------------------------------

def print_dashboard(
    step: int,
    total_waiting: int,
    ev_active: bool,
    incident_active: bool,
    ev_path: list[str],
    state: dict,
) -> None:
    """
    Render a live status table to stdout.

    Displays a network summary row, the active EV corridor (if any),
    and per-intersection metrics for a representative sample of nodes.

    Args:
        step:             Current simulation timestep (seconds).
        total_waiting:    Network-wide waiting vehicle count.
        ev_active:        Whether an emergency vehicle is active.
        incident_active:  Whether a traffic incident is active.
        ev_path:          Ordered list of junction IDs on the EV corridor.
        state:            Full network state dict from ``read_network_state``.
    """
    W = _DASHBOARD_WIDTH

    ev_badge  = " EV ACTIVE " if ev_active      else " EV clear  "
    inc_badge = " INCIDENT  " if incident_active else "           "

    title = f"  NetSignal  ·  t = {step:>5}s"
    right = f"[ {ev_badge} ]  [ {inc_badge} ]  "
    gap   = W - 2 - len(title) - len(right)

    print(f"\n┌{'─' * (W - 2)}┐")
    print(f"│{title}{' ' * max(gap, 1)}{right}│")
    print(f"├{'─' * (W - 2)}┤")

    total_vehicles = sum(s["vehicle_count"] for s in state.values())
    speeds         = [s["avg_speed"] for s in state.values() if s["avg_speed"] > 0]
    net_speed      = float(np.mean(speeds)) if speeds else 0.0

    summary = (
        f"  waiting: {total_waiting:>4} veh    "
        f"on-road: {total_vehicles:>5} veh    "
        f"avg speed: {net_speed:>4.1f} m/s"
    )
    print(f"│{summary}{' ' * (W - 2 - len(summary))}│")

    if ev_active and ev_path:
        corridor = " -> ".join(ev_path)
        if len(corridor) > W - 16:
            corridor = corridor[:W - 19] + "..."
        line = f"  EV route  {corridor}"
        print(f"│{line}{' ' * (W - 2 - len(line))}│")

    print(f"├{'─' * (W - 2)}┤")

    header = f"  {'Node':<8}  {'Waiting':>7}  {'Vehicles':>8}  {'Avg Speed':>10}  {'Phase':<9}"
    print(f"│{header}{' ' * (W - 2 - len(header))}│")
    sep = f"  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*10}  {'─'*9}"
    print(f"│{sep}{' ' * (W - 2 - len(sep))}│")

    sample_nodes = [
        _NODE_MAP.get(f"NODE_{col}_{row}", f"{col}/{row}")
        for col, row in [(0, 0), (2, 2), (3, 3), (5, 5), (3, 1)]
    ]
    for node_id in sample_nodes:
        if node_id not in state:
            continue
        data      = state[node_id]
        phase_str = _PHASE_LABELS.get(data["current_phase"], f"phase {data['current_phase']}")
        row = (
            f"  {node_id:<8}  {data['waiting']:>7}  {data['vehicle_count']:>8}"
            f"  {data['avg_speed']:.1f} m/s  {phase_str:<9}"
        )
        print(f"│{row}{' ' * (W - 2 - len(row))}│")

    print(f"└{'─' * (W - 2)}┘")


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------

def save_summary_plot(
    waiting_history: list[float],
    incident_injected: bool,
    incident_time: int,
) -> None:
    """
    Save a professional summary chart to ``output/summary_plot.png``.

    The figure contains a main time-series chart with rolling average,
    EV and incident markers, and three stat cards (avg / peak / final
    waiting vehicle counts).

    Args:
        waiting_history:   Per-step network-wide waiting vehicle count.
        incident_injected: Whether an incident was injected this run.
        incident_time:     Timestep at which the incident was injected.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib import rcParams

        rcParams["font.family"]       = "DejaVu Sans"
        rcParams["axes.spines.top"]   = False
        rcParams["axes.spines.right"] = False

        C_BG      = "#FFFFFF"
        C_PANEL   = "#F7F8FA"
        C_BORDER  = "#E2E6EA"
        C_TEXT    = "#1E2A38"
        C_SUBTEXT = "#6B7A8D"
        C_LINE    = "#2563EB"
        C_FILL    = "#DBEAFE"
        C_ROLL    = "#1D4ED8"
        C_EV      = "#DC2626"
        C_INC     = "#D97706"
        C_GREEN   = "#059669"

        fig = plt.figure(figsize=(16, 7), facecolor=C_BG)
        gs  = GridSpec(
            1, 4, figure=fig,
            width_ratios=[3, 1, 1, 1],
            wspace=0.06, left=0.06, right=0.97, top=0.88, bottom=0.13,
        )
        ax_main = fig.add_subplot(gs[0, 0])
        ax_avg  = fig.add_subplot(gs[0, 1])
        ax_peak = fig.add_subplot(gs[0, 2])
        ax_fin  = fig.add_subplot(gs[0, 3])

        timesteps = list(range(len(waiting_history)))
        peak_val  = max(waiting_history)
        avg_val   = float(np.mean(waiting_history))
        final_val = waiting_history[-1]

        ax_main.set_facecolor(C_BG)
        ax_main.fill_between(timesteps, waiting_history, color=C_FILL, linewidth=0)
        ax_main.plot(timesteps, waiting_history,
                     color=C_LINE, linewidth=1.8, label="Waiting vehicles", zorder=3)

        window = 30
        if len(waiting_history) > window:
            rolling = np.convolve(waiting_history, np.ones(window) / window, mode="valid")
            ax_main.plot(
                range(window - 1, len(waiting_history)), rolling,
                color=C_ROLL, linewidth=2.2, label=f"{window}s rolling avg", zorder=4,
            )

        if 900 < len(waiting_history):
            ax_main.axvline(x=900, color=C_EV, linestyle="--", linewidth=1.4, alpha=0.85, zorder=5)
            ax_main.text(912, peak_val * 0.94, "EV dispatched",
                         color=C_EV, fontsize=8.5, fontweight="semibold", va="top")

        if incident_injected and incident_time < len(waiting_history):
            ax_main.axvline(x=incident_time, color=C_INC, linestyle="--",
                            linewidth=1.4, alpha=0.85, zorder=5)
            ax_main.text(incident_time + 12, peak_val * 0.78, "Incident",
                         color=C_INC, fontsize=8.5, fontweight="semibold", va="top")

        ax_main.grid(True, color=C_BORDER, linewidth=0.8, zorder=0)
        ax_main.spines["left"].set_color(C_BORDER)
        ax_main.spines["bottom"].set_color(C_BORDER)
        ax_main.tick_params(colors=C_SUBTEXT, labelsize=9)
        ax_main.set_xlabel("Simulation time (s)", fontsize=10, color=C_SUBTEXT, labelpad=8)
        ax_main.set_ylabel("Waiting vehicles",    fontsize=10, color=C_SUBTEXT, labelpad=8)
        ax_main.set_xlim(0, len(waiting_history))
        ax_main.set_ylim(0, peak_val * 1.18)

        legend = ax_main.legend(fontsize=9, frameon=True, loc="upper left",
                                framealpha=0.9, edgecolor=C_BORDER)
        for text in legend.get_texts():
            text.set_color(C_TEXT)

        def _draw_stat_card(ax, label: str, value: float, accent: str) -> None:
            ax.set_facecolor(C_PANEL)
            for spine in ax.spines.values():
                spine.set_color(C_BORDER)
                spine.set_linewidth(1.2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.54, f"{value:.0f}", transform=ax.transAxes,
                    fontsize=30, fontweight="bold", color=accent, ha="center", va="center")
            ax.text(0.5, 0.30, "vehicles", transform=ax.transAxes,
                    fontsize=9, color=C_SUBTEXT, ha="center", va="center")
            ax.text(0.5, 0.88, label, transform=ax.transAxes,
                    fontsize=9, fontweight="semibold", color=C_TEXT, ha="center", va="center")

        _draw_stat_card(ax_avg,  "AVG WAITING",   avg_val,   C_LINE)
        _draw_stat_card(ax_peak, "PEAK WAITING",  peak_val,  C_EV)
        _draw_stat_card(ax_fin,  "FINAL WAITING", final_val, C_GREEN)

        fig.text(0.06, 0.945, "NetSignal  —  Network Waiting Vehicles",
                 fontsize=14, fontweight="bold", color=C_TEXT, va="top")
        fig.text(0.06, 0.915, "Fixed-time baseline  ·  6×6 grid  ·  1-hour simulation",
                 fontsize=9.5, color=C_SUBTEXT, va="top")

        out_path = OUTPUT_DIR / "summary_plot.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        plt.close(fig)
        print(f"  Plot saved: {out_path}")

    except ImportError:
        print("  (matplotlib not installed — skipping plot)")
    except Exception as exc:
        print(f"  (Plot generation failed: {exc})")


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    gui: bool = False,
    inject_incident: bool = False,
    incident_time: int = 600,
    incident_lane: str = "e_2_2_2_3_0",
) -> None:
    """
    Execute the full simulation from SUMO launch to final report.

    Args:
        gui:              Launch sumo-gui when True, headless sumo when False.
        inject_incident:  Inject a simulated traffic incident.
        incident_time:    Timestep at which to inject the incident.
        incident_lane:    Lane ID to block during the incident.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    view_file   = _PROJECT_DIR / "config" / "netsignal.view.xml"
    sumo_binary = "sumo-gui" if gui else "sumo"

    sumo_cmd = [sumo_binary, "-c", str(CONFIG_FILE), "--ignore-route-errors"]
    if gui and view_file.exists():
        sumo_cmd += ["-g", str(view_file)]
    if not gui:
        sumo_cmd += ["--no-step-log", "--no-warnings"]

    W = _DASHBOARD_WIDTH
    mode_str = "GUI  (sumo-gui)" if gui else "Headless  (sumo)"
    cfg_str  = str(CONFIG_FILE)[-48:]

    print(f"\n┌{'─' * (W - 2)}┐")
    print(f"│  NetSignal Traffic Simulation{' ' * (W - 32)}│")
    print(f"├{'─' * (W - 2)}┤")
    print(f"│  Mode       : {mode_str:<{W - 17}}│")
    print(f"│  Config     : {cfg_str:<{W - 17}}│")
    print(f"│  Duration   : {SIM_DURATION}s  (1 hour){' ' * (W - 32)}│")
    print(f"│  Controller : Fixed-time baseline{' ' * (W - 36)}│")
    print(f"│  Hospital   : {HOSPITAL_NODE:<{W - 17}}│")
    print(f"│  EV departs : t = 900 s{' ' * (W - 26)}│")
    if inject_incident:
        inc_str = f"t = {incident_time}s  ·  lane {incident_lane}"
        print(f"│  Incident   : {inc_str:<{W - 17}}│")
    print(f"└{'─' * (W - 2)}┘\n")

    try:
        traci.start(sumo_cmd, port=TRACI_PORT)
        print("  TraCI connected\n")
    except traci.exceptions.FatalTraCIError as exc:
        print(f"  TraCI/SUMO launch failed: {exc}")
        return
    except FileNotFoundError:
        print(f"  Command not found: '{sumo_binary}'. Is SUMO installed and on PATH?")
        return
    except Exception as exc:
        print(f"  Failed to start SUMO: {exc}")
        return

    road_graph  = build_road_graph()
    controller  = FixedTimeController()
    preemptor   = EVPreemptor(road_graph)
    logger      = DataLogger()
    incident    = IncidentInjector(incident_lane, incident_time) if inject_incident else None

    for node_id in ALL_INTERSECTIONS:
        try:
            traci.junction.subscribeContext(
                node_id, tc.CMD_GET_VEHICLE_VARIABLE, 100,
                [tc.VAR_SPEED, tc.VAR_LANE_ID],
            )
        except Exception:
            pass

    waiting_history: list[float] = []
    wall_start = time.time()

    try:
        for step in range(SIM_DURATION):
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError as exc:
                print(f"\n  SUMO disconnected at step {step}: {exc}")
                break

            state, emergency_vehicles, total_waiting = read_network_state()
            waiting_history.append(total_waiting)

            ev_active       = bool(emergency_vehicles)
            incident_active = incident.step(step) if incident else False

            logger.log_state(step, state, ev_active, incident_active)
            if incident_active:
                logger.log_incident(step, incident_lane, "blockage")

            if step % CONTROL_FREQ == 0:
                preemptor.update(emergency_vehicles)
                actions = controller.get_actions(state)

                for node_id in preemptor.get_preempted_intersections():
                    ev_phase = preemptor.get_phase_for_intersection(node_id)
                    if ev_phase is not None:
                        actions[node_id] = ev_phase

                for node_id, phase in actions.items():
                    try:
                        current = traci.trafficlight.getPhase(node_id)
                        if current != phase and current not in (PHASE_EW_YELLOW, PHASE_NS_YELLOW):
                            traci.trafficlight.setPhase(node_id, phase)
                    except traci.exceptions.TraCIException:
                        pass

                logger.log_metrics(step, total_waiting, state, ev_active, incident_active)

                if step % DASHBOARD_FREQ == 0:
                    print_dashboard(step, total_waiting, ev_active, incident_active,
                                    preemptor.path, state)

    except KeyboardInterrupt:
        print("\n\n  Simulation interrupted.")

    finally:
        wall_time = time.time() - wall_start
        rt_ratio  = f"{SIM_DURATION / wall_time:.1f}x" if wall_time > 0.01 else "N/A"

        print(f"\n┌{'─' * (W - 2)}┐")
        print(f"│  Simulation complete{' ' * (W - 22)}│")
        print(f"├{'─' * (W - 2)}┤")
        print(f"│  Sim time     : {SIM_DURATION}s{' ' * (W - 19 - len(str(SIM_DURATION)))}│")
        print(f"│  Wall time    : {wall_time:.1f}s  ({rt_ratio} real-time){' ' * max(0, W - 30 - len(f'{wall_time:.1f}') - len(rt_ratio))}│")

        if waiting_history:
            avg_w  = float(np.mean(waiting_history))
            peak_w = int(max(waiting_history))
            print(f"│  Avg waiting  : {avg_w:.1f} vehicles{' ' * max(0, W - 28 - len(f'{avg_w:.1f}'))}│")
            print(f"│  Peak waiting : {peak_w} vehicles{' ' * max(0, W - 27 - len(str(peak_w)))}│")
        else:
            print(f"│  Avg waiting  : N/A{' ' * (W - 22)}│")
            print(f"│  Peak waiting : N/A{' ' * (W - 22)}│")

        out_str = str(OUTPUT_DIR)[-47:]
        print(f"│  Output       : {out_str:<{W - 19}}│")
        print(f"└{'─' * (W - 2)}┘\n")

        try:
            traci.close()
        except Exception:
            pass

        if waiting_history:
            save_summary_plot(waiting_history, inject_incident, incident_time)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NetSignal TraCI simulation loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python traciloop.py                     # headless\n"
            "  python traciloop.py --gui               # GUI\n"
            "  python traciloop.py --gui --incident    # GUI + incident at t=600s\n"
            "  python traciloop.py --incident --incident-time 300 --incident-lane e_1_2_1_3_0\n"
        ),
    )
    parser.add_argument("--gui", action="store_true",
                        help="Launch sumo-gui (default: headless sumo)")
    parser.add_argument("--incident", action="store_true",
                        help="Inject a simulated traffic incident")
    parser.add_argument("--incident-time", type=int, default=600, metavar="SECONDS",
                        help="Timestep at which to inject the incident (default: 600)")
    parser.add_argument("--incident-lane", type=str, default="e_2_2_2_3_0", metavar="LANE_ID",
                        help="Lane ID to block during the incident (default: e_2_2_2_3_0)")
    return parser


if __name__ == "__main__":
    args = _build_argument_parser().parse_args()
    run_simulation(
        gui=args.gui,
        inject_incident=args.incident,
        incident_time=args.incident_time,
        incident_lane=args.incident_lane,
    )
