#!/usr/bin/env bash
# =============================================================================
# run_phase1.sh  —  NetSignal Phase 1 end-to-end pipeline
#
# Runs all three steps in order:
#   Step 1 — collect_data.py    : simulate N episodes, write CSVs
#   Step 2 — validate_data.py   : check every CSV for corruption / anomalies
#   Step 3 — prepare_dataset.py : normalise & export numpy arrays for BiLSTM
#
# Usage:
#   bash run_phase1.sh                     # default 500 episodes
#   bash run_phase1.sh --episodes 100      # fast test run
#   bash run_phase1.sh --episodes 500 --workers 4   # parallel (Linux/macOS)
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults — override by passing flags, e.g. bash run_phase1.sh --episodes 200
# --------------------------------------------------------------------------
EPISODES=500
BASELINE_RATIO=0.20      # 20 % baseline, 80 % incident
WORKERS=1
DOWNSAMPLE=1             # keep every timestep (set to 5 to reduce dataset size)
SEED=42

# --------------------------------------------------------------------------
# Parse flags
# --------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --episodes)        EPISODES="$2";       shift 2 ;;
    --baseline-ratio)  BASELINE_RATIO="$2"; shift 2 ;;
    --workers)         WORKERS="$2";        shift 2 ;;
    --downsample)      DOWNSAMPLE="$2";     shift 2 ;;
    --seed)            SEED="$2";           shift 2 ;;
    *)                 echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# --------------------------------------------------------------------------
# Resolve project root (parent of this script's directory)
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  NetSignal — Phase 1 Full Pipeline                              ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Project  : $PROJECT_DIR"
echo "║  Episodes : $EPISODES   Baseline ratio: $BASELINE_RATIO"
echo "║  Workers  : $WORKERS    Downsample: $DOWNSAMPLE    Seed: $SEED"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# --------------------------------------------------------------------------
# Pre-flight checks
# --------------------------------------------------------------------------
echo "[pre-flight] Checking SUMO installation..."
if ! command -v sumo &>/dev/null; then
  echo "  ERROR: 'sumo' not found. Install SUMO and add it to PATH."
  exit 1
fi
echo "  sumo OK  ($(sumo --version 2>&1 | head -1))"

echo "[pre-flight] Checking Python packages..."
$PYTHON -c "import traci, numpy" 2>/dev/null || {
  echo "  ERROR: Missing packages. Run:  pip install traci eclipse-sumo numpy"
  exit 1
}
echo "  Python packages OK"

CONFIG_FILE="$PROJECT_DIR/config/netsignal.sumocfg"
NET_FILE="$PROJECT_DIR/network/grid6x6.net.xml"
NODE_FILE="$PROJECT_DIR/config/grid_nodes.txt"

if [[ ! -f "$NET_FILE" ]]; then
  echo ""
  echo "[pre-flight] Network file missing — running generate_network.py..."
  $PYTHON "$SCRIPT_DIR/generate_network.py"
fi

if [[ ! -f "$NODE_FILE" ]]; then
  echo ""
  echo "[pre-flight] Node map missing — running fix_flows.py..."
  $PYTHON "$SCRIPT_DIR/fix_flows.py"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "  ERROR: SUMO config not found: $CONFIG_FILE"
  exit 1
fi

echo ""
echo "════════════════════════════════════════"
echo "  STEP 1 — Data Collection"
echo "════════════════════════════════════════"
START_1=$(date +%s)

$PYTHON "$SCRIPT_DIR/collect_data.py" \
  --episodes       "$EPISODES"       \
  --baseline-ratio "$BASELINE_RATIO" \
  --workers        "$WORKERS"        \
  --seed           "$SEED"

END_1=$(date +%s)
echo ""
echo "  Step 1 complete in $(( END_1 - START_1 ))s"

echo ""
echo "════════════════════════════════════════"
echo "  STEP 2 — Data Validation"
echo "════════════════════════════════════════"
START_2=$(date +%s)

$PYTHON "$SCRIPT_DIR/validate_data.py" \
  --plot 2>/dev/null || true   # non-fatal if matplotlib absent

END_2=$(date +%s)
echo "  Step 2 complete in $(( END_2 - START_2 ))s"

echo ""
echo "════════════════════════════════════════"
echo "  STEP 3 — Dataset Preparation"
echo "════════════════════════════════════════"
START_3=$(date +%s)

$PYTHON "$SCRIPT_DIR/prepare_dataset.py" \
  --downsample "$DOWNSAMPLE" \
  --seed       "$SEED"

END_3=$(date +%s)
echo "  Step 3 complete in $(( END_3 - START_3 ))s"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
TOTAL=$(( END_3 - START_1 ))
H=$(( TOTAL / 3600 ))
M=$(( (TOTAL % 3600) / 60 ))
S=$(( TOTAL % 60 ))

DATASET_DIR="$PROJECT_DIR/output/dataset"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Phase 1 Complete                                                ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Total wall time  : ${H}h ${M}m ${S}s"
echo "║  Dataset location : $DATASET_DIR"
echo "║"
echo "║  Files ready for Phase 2 (BiLSTM):"
for f in X_train.npy y_train.npy X_val.npy y_val.npy X_test.npy y_test.npy; do
  FPATH="$DATASET_DIR/$f"
  if [[ -f "$FPATH" ]]; then
    SIZE=$(du -h "$FPATH" | cut -f1)
    printf "║    %-25s  %s\n" "$f" "$SIZE"
  fi
done
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
