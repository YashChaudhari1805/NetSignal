#!/usr/bin/env python3
"""
validate_data.py
================
Phase 1 — Data validation and inspection for collected SUMO episodes.

Reads all episode CSVs, checks for corruption / missing data, prints
per-column statistics, and flags any episodes that should be discarded
before BiLSTM training.

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --episode-dir output/episodes
    python scripts/validate_data.py --max-episodes 50   # inspect first N only
    python scripts/validate_data.py --plot               # save summary plots
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Optional matplotlib — degrade gracefully if absent
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = Path(__file__).parent.resolve()
_PROJECT_DIR = _SCRIPT_DIR.parent
_DEFAULT_EPISODE_DIR = _PROJECT_DIR / "output" / "episodes"
_DEFAULT_SUMMARY     = _PROJECT_DIR / "output" / "data_collection_summary.csv"

GRID_SIZE    = 6
N_NODES      = GRID_SIZE * GRID_SIZE   # 36
SIM_DURATION = 3600

# Expected columns in each file type
_STATE_COLS   = {"episode","step","node_id","waiting","queue","avg_speed",
                 "vehicle_count","current_phase","ev_nearby","preempted"}
_INCIDENT_COLS= {"episode","step","node_id","incident_active","incident_lane",
                 "incident_distance"}
_METRICS_COLS = {"episode","step","total_waiting","total_vehicles",
                 "network_avg_speed","ev_active","incident_active",
                 "incident_start","incident_end","incident_lane"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _cast_numeric(rows: list[dict], cols: list[str]) -> list[dict]:
    for row in rows:
        for col in cols:
            if col in row and row[col] != "":
                try:
                    row[col] = float(row[col])
                except ValueError:
                    row[col] = 0.0
    return rows


def _find_episode_files(episode_dir: Path) -> dict[int, dict[str, Path]]:
    """
    Scan episode_dir for ep_XXXX_*.csv files.

    Returns:
        {episode_id: {"state": Path, "incident": Path, "metrics": Path}}
    """
    episodes: dict[int, dict[str, Path]] = defaultdict(dict)
    for path in sorted(episode_dir.glob("ep_*.csv")):
        stem = path.stem          # ep_0001_state
        parts = stem.split("_")   # ['ep', '0001', 'state']
        if len(parts) < 3:
            continue
        try:
            ep_id   = int(parts[1])
            file_type = parts[2]          # state | incident | metrics
        except (ValueError, IndexError):
            continue
        episodes[ep_id][file_type] = path

    return dict(episodes)


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def check_completeness(ep_id: int, files: dict[str, Path]) -> list[str]:
    """Return list of issues found in this episode's file set."""
    issues = []

    for ftype in ("state", "incident", "metrics"):
        if ftype not in files:
            issues.append(f"Missing {ftype} file")
            continue
        path = files[ftype]
        if path.stat().st_size < 100:
            issues.append(f"{ftype} file nearly empty ({path.stat().st_size} bytes)")

    return issues


def check_state_file(
    ep_id: int,
    path:  Path,
) -> tuple[list[str], dict]:
    """
    Validate state CSV for an episode.

    Returns:
        (issues, stats)  where stats is a dict of numeric summaries.
    """
    issues = []
    stats  = {}

    try:
        rows = _read_csv(path)
    except Exception as exc:
        return [f"Cannot read state CSV: {exc}"], {}

    if not rows:
        return ["State CSV is empty"], {}

    # Column check
    missing_cols = _STATE_COLS - set(rows[0].keys())
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    numeric_cols = ["waiting","queue","avg_speed","vehicle_count",
                    "current_phase","ev_nearby","preempted"]
    rows = _cast_numeric(rows, numeric_cols)

    # Step count
    steps = sorted({int(r["step"]) for r in rows if r.get("step") != ""})
    expected_rows = SIM_DURATION * N_NODES
    actual_rows   = len(rows)

    if actual_rows < expected_rows * 0.95:
        issues.append(
            f"Short state file: {actual_rows} rows "
            f"(expected ~{expected_rows}, got {actual_rows/N_NODES:.0f} steps)"
        )

    # Node count per step
    step_node_counts = defaultdict(int)
    for r in rows:
        if r.get("step") != "":
            step_node_counts[int(float(r["step"]))] += 1

    bad_steps = [s for s, c in step_node_counts.items() if c != N_NODES]
    if bad_steps:
        issues.append(
            f"{len(bad_steps)} steps with wrong node count "
            f"(expected {N_NODES}, e.g. step {bad_steps[0]} has {step_node_counts[bad_steps[0]]})"
        )

    # Numeric sanity
    waitings    = [r["waiting"]       for r in rows if isinstance(r.get("waiting"), float)]
    speeds      = [r["avg_speed"]     for r in rows if isinstance(r.get("avg_speed"), float)]
    phases      = [r["current_phase"] for r in rows if isinstance(r.get("current_phase"), float)]

    if waitings:
        stats["mean_waiting"] = round(float(np.mean(waitings)), 2)
        stats["max_waiting"]  = int(max(waitings))
        if max(waitings) > 500:
            issues.append(f"Suspiciously high max waiting: {max(waitings):.0f}")

    if speeds:
        stats["mean_speed"] = round(float(np.mean(speeds)), 3)
        stats["min_speed"]  = round(float(min(speeds)), 3)
        if min(speeds) < 0:
            issues.append(f"Negative speed values found: {min(speeds):.3f}")

    if phases:
        unique_phases = set(int(p) for p in phases)
        stats["phases_seen"] = sorted(unique_phases)
        if not unique_phases.issubset({0, 1, 2, 3, -1}):
            issues.append(f"Unexpected phase values: {unique_phases - {0,1,2,3,-1}}")

    stats["n_rows"]  = actual_rows
    stats["n_steps"] = len(steps)

    return issues, stats


def check_incident_file(ep_id: int, path: Path) -> tuple[list[str], dict]:
    """Validate incident CSV."""
    issues = []
    stats  = {}

    try:
        rows = _read_csv(path)
    except Exception as exc:
        return [f"Cannot read incident CSV: {exc}"], {}

    if not rows:
        return ["Incident CSV is empty"], {}

    missing_cols = _INCIDENT_COLS - set(rows[0].keys())
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    rows = _cast_numeric(rows, ["incident_active", "incident_distance"])

    active_rows = [r for r in rows if r.get("incident_active") == 1.0]
    stats["incident_rows"]    = len(active_rows)
    stats["non_incident_rows"] = len(rows) - len(active_rows)
    stats["incident_fraction"] = round(len(active_rows) / len(rows), 4) if rows else 0

    # Check distances are non-negative
    dists = [r["incident_distance"] for r in rows
             if isinstance(r.get("incident_distance"), float)
             and r["incident_distance"] != 999.0]
    if dists and min(dists) < 0:
        issues.append(f"Negative incident distances: {min(dists)}")

    return issues, stats


def check_metrics_file(ep_id: int, path: Path) -> tuple[list[str], dict]:
    """Validate metrics CSV."""
    issues = []
    stats  = {}

    try:
        rows = _read_csv(path)
    except Exception as exc:
        return [f"Cannot read metrics CSV: {exc}"], {}

    if not rows:
        return ["Metrics CSV is empty"], {}

    missing_cols = _METRICS_COLS - set(rows[0].keys())
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    rows = _cast_numeric(rows, ["total_waiting","total_vehicles",
                                "network_avg_speed","ev_active","incident_active"])

    waitings = [r["total_waiting"]    for r in rows if isinstance(r.get("total_waiting"), float)]
    speeds   = [r["network_avg_speed"] for r in rows if isinstance(r.get("network_avg_speed"), float)]

    if waitings:
        stats["peak_network_waiting"] = int(max(waitings))
        stats["mean_network_waiting"] = round(float(np.mean(waitings)), 2)

    if speeds:
        stats["mean_network_speed"]   = round(float(np.mean(speeds)), 3)

    ev_steps  = sum(1 for r in rows if r.get("ev_active") == 1.0)
    inc_steps = sum(1 for r in rows if r.get("incident_active") == 1.0)
    stats["ev_active_steps"]       = ev_steps
    stats["incident_active_steps"] = inc_steps

    return issues, stats


# ---------------------------------------------------------------------------
# Dataset-level statistics
# ---------------------------------------------------------------------------

def compute_dataset_stats(
    all_stats: list[dict],
    episode_types: dict[int, str],
) -> dict:
    """Aggregate statistics across all validated episodes."""
    incident_eps  = [s for s in all_stats if episode_types.get(s["ep_id"]) == "incident"]
    baseline_eps  = [s for s in all_stats if episode_types.get(s["ep_id"]) == "baseline"]

    def _safe_mean(vals):
        return round(float(np.mean(vals)), 3) if vals else 0.0

    return {
        "total_episodes":     len(all_stats),
        "incident_episodes":  len(incident_eps),
        "baseline_episodes":  len(baseline_eps),
        "total_rows":         sum(s.get("n_rows", 0) for s in all_stats),
        "total_steps":        sum(s.get("n_steps", 0) for s in all_stats),
        "mean_waiting_incident":  _safe_mean([s.get("mean_waiting", 0) for s in incident_eps]),
        "mean_waiting_baseline":  _safe_mean([s.get("mean_waiting", 0) for s in baseline_eps]),
        "mean_speed_incident":    _safe_mean([s.get("mean_speed", 0) for s in incident_eps]),
        "mean_speed_baseline":    _safe_mean([s.get("mean_speed", 0) for s in baseline_eps]),
        "mean_incident_fraction": _safe_mean([s.get("incident_fraction", 0) for s in incident_eps]),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_waiting_distributions(
    all_episode_stats: list[dict],
    episode_types:     dict[int, str],
    output_dir:        Path,
) -> None:
    """Plot waiting vehicle distribution for incident vs baseline episodes."""
    if not HAS_MPL:
        print("  (matplotlib not available — skipping plots)")
        return

    incident_means = [s["mean_waiting"] for s in all_episode_stats
                      if episode_types.get(s["ep_id"]) == "incident"
                      and "mean_waiting" in s]
    baseline_means = [s["mean_waiting"] for s in all_episode_stats
                      if episode_types.get(s["ep_id"]) == "baseline"
                      and "mean_waiting" in s]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("NetSignal Phase 1 — Dataset Overview", fontsize=13)

    # Waiting distribution
    ax = axes[0]
    if incident_means:
        ax.hist(incident_means, bins=20, alpha=0.7, label="Incident",  color="#e05252")
    if baseline_means:
        ax.hist(baseline_means, bins=20, alpha=0.7, label="Baseline",  color="#5292e0")
    ax.set_xlabel("Mean waiting vehicles / episode")
    ax.set_ylabel("Episode count")
    ax.set_title("Waiting Distribution")
    ax.legend()

    # Incident active fraction
    ax = axes[1]
    inc_fracs = [s.get("incident_fraction", 0) for s in all_episode_stats
                 if episode_types.get(s["ep_id"]) == "incident"
                 and "incident_fraction" in s]
    if inc_fracs:
        ax.hist(inc_fracs, bins=20, color="#e0a052", alpha=0.8)
    ax.set_xlabel("Fraction of (node, step) pairs with incident_active=1")
    ax.set_ylabel("Episode count")
    ax.set_title("Incident Label Density (incident episodes only)")

    plt.tight_layout()
    plot_path = output_dir / "phase1_dataset_overview.png"
    plt.savefig(str(plot_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NetSignal Phase 1 — data validator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episode-dir",  type=str, default=str(_DEFAULT_EPISODE_DIR),
                        help=f"Path to episodes directory (default: {_DEFAULT_EPISODE_DIR})")
    parser.add_argument("--summary-csv",  type=str, default=str(_DEFAULT_SUMMARY),
                        help="Path to data_collection_summary.csv")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Inspect at most N episodes (default: all)")
    parser.add_argument("--plot",         action="store_true",
                        help="Save summary plots to the episode directory")
    parser.add_argument("--verbose",      action="store_true",
                        help="Print per-episode details even when OK")
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    if not episode_dir.exists():
        sys.exit(f"ERROR: Episode directory not found: {episode_dir}")

    # ---- Load summary CSV (optional) ----
    episode_types: dict[int, str] = {}  # ep_id -> "incident" | "baseline"
    summary_path = Path(args.summary_csv)
    if summary_path.exists():
        for row in _read_csv(summary_path):
            try:
                ep_id = int(row.get("episode", 0))
                has_inc = str(row.get("has_incident", "")).lower() in ("true","1","yes")
                episode_types[ep_id] = "incident" if has_inc else "baseline"
            except ValueError:
                pass

    # ---- Discover episodes ----
    episodes = _find_episode_files(episode_dir)
    if not episodes:
        sys.exit(f"No ep_XXXX_*.csv files found in {episode_dir}")

    ep_ids = sorted(episodes.keys())
    if args.max_episodes:
        ep_ids = ep_ids[:args.max_episodes]

    W = 68
    print(f"\n┌{'─'*(W-2)}┐")
    print(f"│  NetSignal — Phase 1 Data Validator{' '*(W-38)}│")
    print(f"├{'─'*(W-2)}┤")
    print(f"│  Episode dir   : {str(episode_dir)[-46:]:<{W-20}}│")
    print(f"│  Episodes found: {len(ep_ids):<{W-20}}│")
    print(f"└{'─'*(W-2)}┘\n")

    # ---- Validate each episode ----
    all_issues:   dict[int, list[str]] = {}
    all_ep_stats: list[dict]           = []

    print(f"  {'EP':>4}  {'TYPE':<9}  {'ROWS':>8}  {'STEPS':>6}  "
          f"{'AVG_W':>6}  {'AVG_SPD':>7}  {'INC_FRAC':>9}  STATUS")
    print(f"  {'─'*4}  {'─'*9}  {'─'*8}  {'─'*6}  "
          f"{'─'*6}  {'─'*7}  {'─'*9}  {'─'*10}")

    ok_count   = 0
    fail_count = 0

    for ep_id in ep_ids:
        files  = episodes[ep_id]
        issues = check_completeness(ep_id, files)

        state_stats   = {}
        incident_stats= {}

        if "state" in files and not any("Missing state" in i for i in issues):
            s_iss, state_stats = check_state_file(ep_id, files["state"])
            issues.extend(s_iss)

        if "incident" in files and not any("Missing incident" in i for i in issues):
            i_iss, incident_stats = check_incident_file(ep_id, files["incident"])
            issues.extend(i_iss)

        if "metrics" in files and not any("Missing metrics" in i for i in issues):
            m_iss, _ = check_metrics_file(ep_id, files["metrics"])
            issues.extend(m_iss)

        # Merge stats
        ep_stat = {"ep_id": ep_id}
        ep_stat.update(state_stats)
        ep_stat.update(incident_stats)
        all_ep_stats.append(ep_stat)

        all_issues[ep_id] = issues
        ep_type = episode_types.get(ep_id, "unknown")

        n_rows   = state_stats.get("n_rows",        "?")
        n_steps  = state_stats.get("n_steps",       "?")
        avg_w    = state_stats.get("mean_waiting",  "?")
        avg_spd  = state_stats.get("mean_speed",    "?")
        inc_frac = incident_stats.get("incident_fraction", "?")

        status = "OK" if not issues else f"WARN({len(issues)})"
        if issues:
            fail_count += 1
        else:
            ok_count += 1

        n_rows_str   = f"{n_rows:>8}" if isinstance(n_rows,   int)   else f"{'?':>8}"
        n_steps_str  = f"{n_steps:>6}" if isinstance(n_steps,  int)   else f"{'?':>6}"
        avg_w_str    = f"{avg_w:>6.1f}" if isinstance(avg_w,   float) else f"{'?':>6}"
        avg_spd_str  = f"{avg_spd:>7.3f}" if isinstance(avg_spd, float) else f"{'?':>7}"
        inc_frac_str = f"{inc_frac:>9.4f}" if isinstance(inc_frac, float) else f"{'?':>9}"

        line = (
            f"  {ep_id:>4}  {ep_type:<9}  {n_rows_str}  {n_steps_str}  "
            f"{avg_w_str}  {avg_spd_str}  {inc_frac_str}  {status}"
        )
        print(line)

        if (issues and True) or args.verbose:
            for issue in issues[:5]:
                print(f"         ⚠  {issue}")
            if len(issues) > 5:
                print(f"         ⚠  ... and {len(issues)-5} more")

    # ---- Dataset-level stats ----
    ds_stats = compute_dataset_stats(all_ep_stats, episode_types)

    print(f"\n┌{'─'*(W-2)}┐")
    print(f"│  Dataset Summary{' '*(W-19)}│")
    print(f"├{'─'*(W-2)}┤")
    print(f"│  Total episodes    : {ds_stats['total_episodes']:<{W-24}}│")
    print(f"│  Incident episodes : {ds_stats['incident_episodes']:<{W-24}}│")
    print(f"│  Baseline episodes : {ds_stats['baseline_episodes']:<{W-24}}│")
    total_rows_str = f"{ds_stats['total_rows']:,}"
    print(f"│  Total state rows  : {total_rows_str:<{W-24}}│")
    total_steps_str = f"{ds_stats['total_steps']:,}"
    print(f"│  Total steps       : {total_steps_str:<{W-24}}│")
    print(f"├{'─'*(W-2)}┤")
    print(f"│  Mean waiting (incident) : {ds_stats['mean_waiting_incident']:<{W-30}}│")
    print(f"│  Mean waiting (baseline) : {ds_stats['mean_waiting_baseline']:<{W-30}}│")
    print(f"│  Mean speed   (incident) : {ds_stats['mean_speed_incident']:<{W-30}}│")
    print(f"│  Mean speed   (baseline) : {ds_stats['mean_speed_baseline']:<{W-30}}│")
    print(f"│  Mean inc label density  : {ds_stats['mean_incident_fraction']:<{W-30}}│")
    print(f"├{'─'*(W-2)}┤")
    print(f"│  Passed validation : {ok_count}/{len(ep_ids)}{' '*(W-24-len(str(ok_count))-len(str(len(ep_ids))))}│")
    if fail_count:
        print(f"│  ⚠ Episodes with issues : {fail_count:<{W-28}}│")
    print(f"└{'─'*(W-2)}┘\n")

    # ---- Recommendations ----
    print("  Recommendations for Phase 2:")

    problem_eps = [ep for ep, iss in all_issues.items() if iss]
    if problem_eps:
        print(f"  ⚠  Exclude or re-run these {len(problem_eps)} episodes: "
              + ", ".join(str(e) for e in problem_eps[:10])
              + ("..." if len(problem_eps) > 10 else ""))
    else:
        print("  ✓  All episodes passed validation.")

    n_inc = ds_stats["incident_episodes"]
    n_bas = ds_stats["baseline_episodes"]
    if n_inc == 0:
        print("  ⚠  No incident episodes found — BiLSTM cannot learn anomaly detection.")
    elif n_bas == 0:
        print("  ⚠  No baseline episodes found — model may overfit to incident patterns.")
    else:
        ratio = n_inc / n_bas
        if ratio < 1.5:
            print(f"  ℹ  Inc/baseline ratio = {ratio:.1f} — consider more incident episodes.")
        else:
            print(f"  ✓  Inc/baseline ratio = {ratio:.1f} — class balance looks reasonable.")

    if ds_stats["total_rows"] < 500_000:
        print(f"  ⚠  Only {ds_stats['total_rows']:,} state rows — "
              f"consider collecting more episodes for robust BiLSTM training.")
    else:
        print(f"  ✓  {ds_stats['total_rows']:,} state rows available for training.")

    # ---- Plots ----
    if args.plot:
        plot_waiting_distributions(all_ep_stats, episode_types, episode_dir)

    print()


if __name__ == "__main__":
    main()
