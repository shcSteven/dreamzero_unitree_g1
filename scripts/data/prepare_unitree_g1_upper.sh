#!/bin/bash
#
# Prepare a LeRobot v2 dataset (Unitree G1 Dex3 upper body) for DreamZero training.
#
# Usage:
#   ./scripts/data/prepare_unitree_g1_upper.sh /path/to/lerobot_dataset /path/to/gear_output
#
# The input dataset must have the standard LeRobot v2 layout:
#   dataset/
#   ├── data/chunk-000/episode_*.parquet   (columns: observation.state[28], action[28], task_index)
#   ├── videos/chunk-000/{observation.images.cam_room,cam_left_wrist,cam_right_wrist}/
#   └── meta/info.json
#
# This script will:
#   1. Run convert_lerobot_to_gear.py to produce GEAR metadata (copies dataset to output path)
#   2. Fix tasks.jsonl (map task_index integers to task descriptions from the original tasks.jsonl)
#   3. Add the annotation entry to modality.json
#   4. Run the pre-training checklist
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <lerobot_dataset_path> <gear_output_path>"
    echo ""
    echo "  lerobot_dataset_path  Path to a LeRobot v2 dataset directory"
    echo "                        (must contain data/, videos/, meta/info.json)"
    echo "  gear_output_path      Where to write the GEAR-converted copy"
    exit 1
fi

INPUT_PATH="$(realpath "$1")"
OUTPUT_PATH="$2"
EMBODIMENT_TAG="unitree_g1_upper"

# ── Validate input ──────────────────────────────────────────────────────────
if [ ! -f "$INPUT_PATH/meta/info.json" ]; then
    echo "ERROR: $INPUT_PATH/meta/info.json not found. Is this a LeRobot v2 dataset?"
    exit 1
fi
if [ ! -d "$INPUT_PATH/data" ]; then
    echo "ERROR: $INPUT_PATH/data/ not found."
    exit 1
fi
if [ ! -d "$INPUT_PATH/videos" ]; then
    echo "ERROR: $INPUT_PATH/videos/ not found."
    exit 1
fi

# ── Back up original tasks.jsonl if it exists ───────────────────────────────
ORIG_TASKS=""
if [ -f "$INPUT_PATH/meta/tasks.jsonl" ]; then
    ORIG_TASKS=$(cat "$INPUT_PATH/meta/tasks.jsonl")
    echo "Found existing tasks.jsonl with $(echo "$ORIG_TASKS" | wc -l) task(s), will restore after conversion."
fi

# ── Step 1: Convert to GEAR format ──────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 1: Converting LeRobot v2 → GEAR format"
echo "  Input:  $INPUT_PATH"
echo "  Output: $OUTPUT_PATH"
echo "════════════════════════════════════════════════════════════════"

python "$REPO_ROOT/scripts/data/convert_lerobot_to_gear.py" \
    --dataset-path "$INPUT_PATH" \
    --output-path "$OUTPUT_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --state-keys '{"left_arm": [0, 7], "right_arm": [7, 14], "left_hand": [14, 21], "right_hand": [21, 28]}' \
    --action-keys '{"left_arm": [0, 7], "right_arm": [7, 14], "left_hand": [14, 21], "right_hand": [21, 28]}' \
    --relative-action-keys left_arm right_arm left_hand right_hand \
    --force

# ── Step 2: Restore tasks.jsonl ─────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 2: Fixing tasks.jsonl"
echo "════════════════════════════════════════════════════════════════"

if [ -n "$ORIG_TASKS" ]; then
    echo "$ORIG_TASKS" > "$OUTPUT_PATH/meta/tasks.jsonl"
    echo "Restored original tasks.jsonl from input dataset."
else
    echo "WARNING: No tasks.jsonl found in input dataset."
    echo "The converter created a placeholder. Edit $OUTPUT_PATH/meta/tasks.jsonl"
    echo "to add your task descriptions manually."
fi

# ── Step 3: Ensure annotation entry in modality.json ────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 3: Ensuring annotation entry in modality.json"
echo "════════════════════════════════════════════════════════════════"

python3 -c "
import json, sys
path = '$OUTPUT_PATH/meta/modality.json'
with open(path) as f:
    m = json.load(f)
if not m.get('annotation'):
    m['annotation'] = {'task': {'original_key': 'task_index'}}
    with open(path, 'w') as f:
        json.dump(m, f, indent=4)
    print('Added annotation.task entry to modality.json')
else:
    print('annotation section already present in modality.json')
"

# ── Step 4: Pre-training checklist ──────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 4: Pre-training checklist"
echo "════════════════════════════════════════════════════════════════"

PASS=0
FAIL=0

check() {
    local desc="$1"
    local result="$2"
    if [ "$result" = "true" ]; then
        echo "  [PASS] $desc"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $desc"
        FAIL=$((FAIL + 1))
    fi
}

check "meta/embodiment.json exists" \
    "$([ -f "$OUTPUT_PATH/meta/embodiment.json" ] && echo true || echo false)"

check "embodiment_tag is '$EMBODIMENT_TAG'" \
    "$(python3 -c "import json; d=json.load(open('$OUTPUT_PATH/meta/embodiment.json')); print('true' if d.get('embodiment_tag')=='$EMBODIMENT_TAG' else 'false')")"

check "meta/modality.json has state keys" \
    "$(python3 -c "import json; m=json.load(open('$OUTPUT_PATH/meta/modality.json')); print('true' if len(m.get('state',{}))>0 else 'false')")"

check "meta/modality.json has action keys" \
    "$(python3 -c "import json; m=json.load(open('$OUTPUT_PATH/meta/modality.json')); print('true' if len(m.get('action',{}))>0 else 'false')")"

check "meta/modality.json has video keys" \
    "$(python3 -c "import json; m=json.load(open('$OUTPUT_PATH/meta/modality.json')); print('true' if len(m.get('video',{}))>0 else 'false')")"

check "meta/modality.json has annotation keys" \
    "$(python3 -c "import json; m=json.load(open('$OUTPUT_PATH/meta/modality.json')); print('true' if len(m.get('annotation',{}))>0 else 'false')")"

check "meta/stats.json exists" \
    "$([ -f "$OUTPUT_PATH/meta/stats.json" ] && echo true || echo false)"

check "meta/relative_stats_dreamzero.json exists" \
    "$([ -f "$OUTPUT_PATH/meta/relative_stats_dreamzero.json" ] && echo true || echo false)"

check "meta/tasks.jsonl exists and is non-empty" \
    "$([ -s "$OUTPUT_PATH/meta/tasks.jsonl" ] && echo true || echo false)"

check "meta/episodes.jsonl exists" \
    "$([ -f "$OUTPUT_PATH/meta/episodes.jsonl" ] && echo true || echo false)"

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "  Results: $PASS passed, $FAIL failed"
echo "────────────────────────────────────────────────────────────────"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "WARNING: Some checks failed. Review the output above before training."
    exit 1
fi

echo ""
echo "Dataset is ready at: $OUTPUT_PATH"
echo ""
echo "To launch training:"
echo "  DATA_ROOT=$OUTPUT_PATH bash scripts/train/unitree_g1_upper_training.sh"
echo ""
