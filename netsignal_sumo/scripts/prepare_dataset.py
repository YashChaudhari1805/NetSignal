#!/usr/bin/env python3
"""
prepare_dataset.py
==================
Phase 1 — Final step: convert raw episode CSVs into numpy arrays
ready for BiLSTM training (Phase 2).

What this script does:
  1. Reads all validated episode state + incident CSVs
  2. Normalises features (min-max per feature across dataset)
  3. Shapes data into (episodes, timesteps, nodes, features) tensors
  4. Splits into train / val / test sets (stratified by episode type)
  5. Saves .npy files + a dataset manifest JSON

Output files (in output/dataset/):
    X_train.npy   shape (N_train, T, N_nodes, F)   float32 input features
    y_train.npy   shape (N_train, T, N_nodes)       int8    incident labels
    X_val.npy
    y_val.npy
    X_test.npy
    y_test.npy
    feature_stats.json   min/max/mean/std per feature (for inference normalisation)
    manifest.json        dataset metadata

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --train-ratio 0.7 --val-ratio 0.15
    python scripts/prepare_dataset.py --exclude-episodes 3 7 12
    python scripts/prepare_dataset.py --downsample 10   # keep every 10th step
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = Path(__file__).parent.resolve()
_PROJECT_DIR = _SCRIPT_DIR.parent
_EPISODE_DIR = _PROJECT_DIR / "output" / "episodes"
_SUMMARY_CSV = _PROJECT_DIR / "output" / "data_collection_summary.csv"
_DATASET_DIR = _PROJECT_DIR / "output" / "dataset"
_NODE_MAP_FILE = _PROJECT_DIR / "config" / "grid_nodes.txt"

GRID_SIZE    = 6
N_NODES      = GRID_SIZE * GRID_SIZE   # 36
SIM_DURATION = 3600

# Features extracted from the state CSV per (step, node)
# Order must be stable — used downstream by the BiLSTM
FEATURE_NAMES = [
    "waiting",        # 0
    "queue",          # 1
    "avg_speed",      # 2
    "vehicle_count",  # 3
    "current_phase",  # 4
    "ev_nearby",      # 5
    "preempted",      # 6
]
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_node_map() -> dict[str, str]:
    node_map: dict[str, str] = {}
    if not _NODE_MAP_FILE.exists():
        return node_map
    for line in _NODE_MAP_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        node_map[k.strip()] = v.strip()
    return node_map


def _node_index(node_map: dict[str, str]) -> dict[str, int]:
    """
    Build a stable node_id → integer index mapping sorted by (col, row).
    This index is the 3rd dimension of the X tensor.
    """
    idx = {}
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            key = f"NODE_{col}_{row}"
            if key in node_map:
                idx[node_map[key]] = row * GRID_SIZE + col
    return idx


# ---------------------------------------------------------------------------
# Per-episode loader
# ---------------------------------------------------------------------------

def load_episode(
    ep_id:        int,
    state_path:   Path,
    incident_path:Path,
    node_index:   dict[str, int],
    downsample:   int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load one episode into numpy arrays.

    Args:
        ep_id:          Episode identifier (for error messages).
        state_path:     Path to ep_XXXX_state.csv
        incident_path:  Path to ep_XXXX_incident.csv
        node_index:     Stable node_id → integer index mapping.
        downsample:     Keep every N-th timestep (1 = keep all).

    Returns:
        (X, y) where:
            X shape: (T, N_nodes, N_features)  float32
            y shape: (T, N_nodes)               int8
        or None on failure.
    """
    try:
        state_rows    = _read_csv(state_path)
        incident_rows = _read_csv(incident_path)
    except Exception as exc:
        print(f"    [ep {ep_id}] Cannot read CSV: {exc}")
        return None

    if not state_rows or not incident_rows:
        print(f"    [ep {ep_id}] Empty CSV — skipping")
        return None

    # Determine number of timesteps
    steps = sorted({int(r["step"]) for r in state_rows if r.get("step") != ""})
    if not steps:
        return None

    T = len(steps)
    if downsample > 1:
        steps = steps[::downsample]
        T     = len(steps)

    step_to_idx = {s: i for i, s in enumerate(steps)}

    X = np.zeros((T, N_NODES, N_FEATURES), dtype=np.float32)
    y = np.zeros((T, N_NODES),             dtype=np.int8)

    # ---- Fill X from state CSV ----
    for row in state_rows:
        try:
            step = int(row["step"])
        except (ValueError, KeyError):
            continue
        if step not in step_to_idx:
            continue

        node_id = row.get("node_id", "")
        if node_id not in node_index:
            continue

        t   = step_to_idx[step]
        nid = node_index[node_id]

        try:
            X[t, nid, 0] = float(row.get("waiting",        0) or 0)
            X[t, nid, 1] = float(row.get("queue",          0) or 0)
            X[t, nid, 2] = float(row.get("avg_speed",      0) or 0)
            X[t, nid, 3] = float(row.get("vehicle_count",  0) or 0)
            X[t, nid, 4] = float(row.get("current_phase", -1) or -1)
            X[t, nid, 5] = float(row.get("ev_nearby",      0) or 0)
            X[t, nid, 6] = float(row.get("preempted",      0) or 0)
        except (ValueError, TypeError):
            pass

    # ---- Fill y from incident CSV ----
    for row in incident_rows:
        try:
            step = int(row["step"])
        except (ValueError, KeyError):
            continue
        if step not in step_to_idx:
            continue

        node_id = row.get("node_id", "")
        if node_id not in node_index:
            continue

        t   = step_to_idx[step]
        nid = node_index[node_id]

        try:
            y[t, nid] = int(float(row.get("incident_active", 0) or 0))
        except (ValueError, TypeError):
            pass

    return X, y


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_feature_stats(X_list: list[np.ndarray]) -> dict:
    """
    Compute per-feature min, max, mean, std across all episode arrays.

    Args:
        X_list: List of arrays with shape (T, N_nodes, N_features).

    Returns:
        Dict with keys "min", "max", "mean", "std" each a list of length N_features.
    """
    all_data = np.concatenate(
        [X.reshape(-1, N_FEATURES) for X in X_list], axis=0
    )  # (total_samples, N_features)

    return {
        "feature_names": FEATURE_NAMES,
        "min":  all_data.min(axis=0).tolist(),
        "max":  all_data.max(axis=0).tolist(),
        "mean": all_data.mean(axis=0).tolist(),
        "std":  all_data.std(axis=0).tolist(),
    }


def normalise(X: np.ndarray, stats: dict) -> np.ndarray:
    """
    Min-max normalise X to [0, 1] using precomputed stats.

    Args:
        X:     Array (..., N_features)
        stats: Dict from compute_feature_stats.

    Returns:
        Normalised array, same shape as X.
    """
    min_vals = np.array(stats["min"],  dtype=np.float32)
    max_vals = np.array(stats["max"],  dtype=np.float32)
    rng      = np.where(max_vals - min_vals > 0, max_vals - min_vals, 1.0)
    return (X - min_vals) / rng


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def stratified_split(
    episode_ids:   list[int],
    episode_types: dict[int, str],
    train_ratio:   float,
    val_ratio:     float,
    seed:          int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split episodes into train / val / test preserving incident/baseline ratio.

    Args:
        episode_ids:   All valid episode IDs.
        episode_types: {ep_id: "incident"|"baseline"} from summary CSV.
        train_ratio:   Fraction of episodes for training.
        val_ratio:     Fraction of episodes for validation.
        seed:          Random seed for reproducibility.

    Returns:
        (train_ids, val_ids, test_ids)
    """
    rng = random.Random(seed)

    incident_ids = [e for e in episode_ids if episode_types.get(e) == "incident"]
    baseline_ids = [e for e in episode_ids if episode_types.get(e) == "baseline"]
    unknown_ids  = [e for e in episode_ids if episode_types.get(e) not in ("incident", "baseline")]

    def split_group(ids):
        ids = list(ids)
        rng.shuffle(ids)
        n    = len(ids)
        n_tr = max(1, int(n * train_ratio))
        n_va = max(1, int(n * val_ratio))
        return ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]

    tr_inc, va_inc, te_inc = split_group(incident_ids)
    tr_bas, va_bas, te_bas = split_group(baseline_ids)
    tr_unk, va_unk, te_unk = split_group(unknown_ids)

    train = tr_inc + tr_bas + tr_unk
    val   = va_inc + va_bas + va_unk
    test  = te_inc + te_bas + te_unk

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NetSignal Phase 1 — prepare BiLSTM training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episode-dir",  type=str, default=str(_EPISODE_DIR))
    parser.add_argument("--summary-csv",  type=str, default=str(_SUMMARY_CSV))
    parser.add_argument("--output-dir",   type=str, default=str(_DATASET_DIR))
    parser.add_argument("--train-ratio",  type=float, default=0.70,
                        help="Fraction of episodes for training (default: 0.70)")
    parser.add_argument("--val-ratio",    type=float, default=0.15,
                        help="Fraction of episodes for validation (default: 0.15)")
    parser.add_argument("--downsample",   type=int, default=1,
                        help="Keep every N-th timestep (default: 1 = all)")
    parser.add_argument("--exclude-episodes", type=int, nargs="*", default=[],
                        help="Episode IDs to exclude (space-separated)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not episode_dir.exists():
        sys.exit(f"ERROR: Episode directory not found: {episode_dir}")

    # ---- Node index ----
    node_map   = _load_node_map()
    node_index = _node_index(node_map)
    if not node_index:
        sys.exit("ERROR: Empty node map. Run fix_flows.py first.")

    # ---- Load summary ----
    episode_types: dict[int, str] = {}
    summary_path = Path(args.summary_csv)
    if summary_path.exists():
        for row in _read_csv(summary_path):
            try:
                ep_id = int(row.get("episode", 0))
                has_inc = str(row.get("has_incident","")).lower() in ("true","1","yes")
                episode_types[ep_id] = "incident" if has_inc else "baseline"
            except ValueError:
                pass

    # ---- Discover episode files ----
    state_files    = {
        int(p.stem.split("_")[1]): p
        for p in sorted(episode_dir.glob("ep_*_state.csv"))
    }
    incident_files = {
        int(p.stem.split("_")[1]): p
        for p in sorted(episode_dir.glob("ep_*_incident.csv"))
    }

    valid_eps = sorted(
        ep for ep in state_files
        if ep in incident_files and ep not in args.exclude_episodes
    )

    if not valid_eps:
        sys.exit("ERROR: No valid episode pairs found.")

    W = 66
    print(f"\n┌{'─'*(W-2)}┐")
    print(f"│  NetSignal — Phase 1 Dataset Preparation{' '*(W-43)}│")
    print(f"├{'─'*(W-2)}┤")
    print(f"│  Episodes available: {len(valid_eps):<{W-23}}│")
    print(f"│  Excluded episodes : {len(args.exclude_episodes):<{W-23}}│")
    print(f"│  Downsample factor : {args.downsample:<{W-23}}│")
    print(f"│  Output dir        : {str(output_dir)[-42:]:<{W-23}}│")
    print(f"└{'─'*(W-2)}┘\n")

    # ---- Load all episodes ----
    print(f"  Loading {len(valid_eps)} episodes...")

    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    loaded_ids: list[int]   = []

    for ep_id in valid_eps:
        result = load_episode(
            ep_id         = ep_id,
            state_path    = state_files[ep_id],
            incident_path = incident_files[ep_id],
            node_index    = node_index,
            downsample    = args.downsample,
        )
        if result is None:
            print(f"    [ep {ep_id:04d}] skipped (load error)")
            continue

        X, y = result
        X_all.append(X)
        y_all.append(y)
        loaded_ids.append(ep_id)

        if len(loaded_ids) % 50 == 0:
            print(f"    Loaded {len(loaded_ids)}/{len(valid_eps)}...")

    print(f"  Successfully loaded {len(loaded_ids)} episodes.\n")

    if not loaded_ids:
        sys.exit("ERROR: No episodes could be loaded.")

    # ---- Compute & apply normalisation ----
    print("  Computing feature statistics...")
    feature_stats = compute_feature_stats(X_all)

    print("  Normalising features...")
    X_all_norm = [normalise(X, feature_stats) for X in X_all]

    # ---- Train / val / test split ----
    train_ids, val_ids, test_ids = stratified_split(
        loaded_ids, episode_types,
        args.train_ratio, args.val_ratio, args.seed,
    )

    def gather(ids):
        idx_map = {ep: i for i, ep in enumerate(loaded_ids)}
        Xs = [X_all_norm[idx_map[i]] for i in ids if i in idx_map]
        ys = [y_all[idx_map[i]]      for i in ids if i in idx_map]
        return Xs, ys

    X_tr, y_tr = gather(train_ids)
    X_va, y_va = gather(val_ids)
    X_te, y_te = gather(test_ids)

    def _stack_or_arr(lst):
        """Stack if all arrays have same T, else return list as object array."""
        if not lst:
            return np.array([], dtype=np.float32)
        shapes = {a.shape for a in lst}
        if len(shapes) == 1:
            return np.stack(lst, axis=0)
        # Pad to max T if shapes differ
        max_T = max(a.shape[0] for a in lst)
        padded = []
        for a in lst:
            t_pad = max_T - a.shape[0]
            if t_pad > 0:
                pad_shape = (t_pad,) + a.shape[1:]
                a = np.concatenate([a, np.zeros(pad_shape, dtype=a.dtype)], axis=0)
            padded.append(a)
        return np.stack(padded, axis=0)

    X_train = _stack_or_arr(X_tr)
    y_train = _stack_or_arr(y_tr).astype(np.int8) if len(y_tr) else np.array([], dtype=np.int8)
    X_val   = _stack_or_arr(X_va)
    y_val   = _stack_or_arr(y_va).astype(np.int8) if len(y_va) else np.array([], dtype=np.int8)
    X_test  = _stack_or_arr(X_te)
    y_test  = _stack_or_arr(y_te).astype(np.int8) if len(y_te) else np.array([], dtype=np.int8)

    # ---- Save arrays ----
    print("  Saving numpy arrays...")
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy",   X_val)
    np.save(output_dir / "y_val.npy",   y_val)
    np.save(output_dir / "X_test.npy",  X_test)
    np.save(output_dir / "y_test.npy",  y_test)

    # ---- Save feature stats ----
    feat_path = output_dir / "feature_stats.json"
    with open(feat_path, "w") as f:
        json.dump(feature_stats, f, indent=2)

    # ---- Save manifest ----
    def _inc_count(ids):
        return sum(1 for i in ids if episode_types.get(i) == "incident")
    def _bas_count(ids):
        return sum(1 for i in ids if episode_types.get(i) == "baseline")

    manifest = {
        "created":          datetime.now().isoformat(),
        "n_nodes":          N_NODES,
        "n_features":       N_FEATURES,
        "feature_names":    FEATURE_NAMES,
        "downsample_factor":args.downsample,
        "sim_duration_steps": SIM_DURATION,
        "grid_size":        GRID_SIZE,
        "node_order":       {v: k for k, v in node_index.items()},
        "splits": {
            "train": {
                "n_episodes":  len(train_ids),
                "n_incident":  _inc_count(train_ids),
                "n_baseline":  _bas_count(train_ids),
                "shape":       list(X_train.shape),
                "episode_ids": sorted(train_ids),
            },
            "val": {
                "n_episodes":  len(val_ids),
                "n_incident":  _inc_count(val_ids),
                "n_baseline":  _bas_count(val_ids),
                "shape":       list(X_val.shape),
                "episode_ids": sorted(val_ids),
            },
            "test": {
                "n_episodes":  len(test_ids),
                "n_incident":  _inc_count(test_ids),
                "n_baseline":  _bas_count(test_ids),
                "shape":       list(X_test.shape),
                "episode_ids": sorted(test_ids),
            },
        },
        "label_stats": {
            "train_positive_fraction": float(y_train.mean()) if y_train.size > 0 else 0,
            "val_positive_fraction":   float(y_val.mean())   if y_val.size   > 0 else 0,
            "test_positive_fraction":  float(y_test.mean())  if y_test.size  > 0 else 0,
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- Final report ----
    print(f"\n┌{'─'*(W-2)}┐")
    print(f"│  Dataset Ready{' '*(W-17)}│")
    print(f"├{'─'*(W-2)}┤")

    def shape_str(arr):
        return str(arr.shape) if hasattr(arr, "shape") else "?"

    print(f"│  X_train : {shape_str(X_train):<{W-14}}│")
    print(f"│  y_train : {shape_str(y_train):<{W-14}}│")
    print(f"│  X_val   : {shape_str(X_val):<{W-14}}│")
    print(f"│  y_val   : {shape_str(y_val):<{W-14}}│")
    print(f"│  X_test  : {shape_str(X_test):<{W-14}}│")
    print(f"│  y_test  : {shape_str(y_test):<{W-14}}│")
    print(f"├{'─'*(W-2)}┤")

    pos_tr = f"{manifest['label_stats']['train_positive_fraction']*100:.2f}%"
    pos_va = f"{manifest['label_stats']['val_positive_fraction']*100:.2f}%"
    pos_te = f"{manifest['label_stats']['test_positive_fraction']*100:.2f}%"
    print(f"│  Positive label fraction  train:{pos_tr}  val:{pos_va}  test:{pos_te}{' '*(W-2-35-len(pos_tr)-len(pos_va)-len(pos_te))}│")
    print(f"├{'─'*(W-2)}┤")
    print(f"│  Saved to: {str(output_dir)[-52:]:<{W-13}}│")
    print(f"└{'─'*(W-2)}┘\n")

    print("  Files written:")
    for fname in ["X_train.npy","y_train.npy","X_val.npy","y_val.npy",
                  "X_test.npy","y_test.npy","feature_stats.json","manifest.json"]:
        fpath = output_dir / fname
        size_kb = fpath.stat().st_size / 1024 if fpath.exists() else 0
        print(f"    {fname:<25} {size_kb:>8.1f} KB")

    print("\n  Phase 1 complete. Next step — Phase 2 (BiLSTM training):")
    print("  Load arrays with:")
    print(f"    X_train = np.load('{output_dir}/X_train.npy')")
    print(f"    y_train = np.load('{output_dir}/y_train.npy')")
    print(f"    # X shape: (episodes, timesteps, {N_NODES} nodes, {N_FEATURES} features)")
    print(f"    # y shape: (episodes, timesteps, {N_NODES} nodes)  -- binary incident label\n")


if __name__ == "__main__":
    main()
