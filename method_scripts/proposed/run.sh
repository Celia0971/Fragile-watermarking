#!/usr/bin/env bash
# Proposed Scheme (IBLT + Group-Testing) — end-to-end run script
#
# Pipeline:
#   1. Register CA (embed)     → ca_registration.json (zero-distortion)
#   2. Attack watermarked DB   → attacked CSV
#   3. Detect tampering        → detect_result.json
#
# Usage:
#   bash run.sh --input db.csv [--attack a3] [--rho 0.01]
#               [--attack_seed 0] [--wm_seed 1000]
#               [--pk_col id] [--output_dir results/] [--config config/params.yaml]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$SCRIPT_DIR/config/params.yaml"

INPUT=""; ATTACK="a3"; RHO="0.01"; COUNT=""; ATTACK_SEED="0"; WM_SEED="1000"; PK_COL="id"
OUTPUT_DIR="$SCRIPT_DIR/results"
DETECTION_MODE=""
TRUE_S_INS=""; TRUE_S_DEL=""; TRUE_S_MOD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)          INPUT="$2";          shift 2 ;;
        --attack)         ATTACK="$2";         shift 2 ;;
        --rho)            RHO="$2";            shift 2 ;;
        --count)          COUNT="$2";          shift 2 ;;
        --attack_seed)    ATTACK_SEED="$2";    shift 2 ;;
        --wm_seed)        WM_SEED="$2";        shift 2 ;;
        --pk_col)         PK_COL="$2";         shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --config)         CONFIG="$2";         shift 2 ;;
        --detection_mode) DETECTION_MODE="$2"; shift 2 ;;
        --true_s_ins)     TRUE_S_INS="$2";     shift 2 ;;
        --true_s_del)     TRUE_S_DEL="$2";     shift 2 ;;
        --true_s_mod)     TRUE_S_MOD="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$INPUT" ]] && { echo "Error: --input required"; exit 1; }

TRIAL_DIR="$OUTPUT_DIR/trial_$(printf '%02d' "$ATTACK_SEED")"
mkdir -p "$TRIAL_DIR"

WM_CSV="$TRIAL_DIR/watermarked.csv"
if [[ -n "$COUNT" ]]; then
    ATK_CSV="$TRIAL_DIR/attacked_${ATTACK}_tc${COUNT}.csv"
else
    ATK_CSV="$TRIAL_DIR/attacked_${ATTACK}_rho${RHO}.csv"
fi
ATK_INFO="$TRIAL_DIR/attack_info.json"
CA_JSON="$TRIAL_DIR/ca_registration.json"

echo "=== Proposed (IBLT+GT) | attack=$ATTACK rho=$RHO attack_seed=$ATTACK_SEED wm_seed=$WM_SEED ==="

echo "--- Register CA (embed) ---"
python "$SCRIPT_DIR/embed.py" \
    --input      "$INPUT" \
    --output_dir "$TRIAL_DIR" \
    --config     "$CONFIG"

ATTACK_ARGS=(
    --input    "$WM_CSV"
    --output   "$ATK_CSV"
    --attack   "$ATTACK"
    --pk_col   "$PK_COL"
    --seed     "$ATTACK_SEED"
    --info_out "$ATK_INFO"
)
if [[ -n "$COUNT" ]]; then
    echo "--- Attack ($ATTACK, count=$COUNT) ---"
    ATTACK_ARGS+=(--count "$COUNT")
else
    echo "--- Attack ($ATTACK, rho=$RHO) ---"
    ATTACK_ARGS+=(--rho "$RHO")
fi
python "$BASELINES_DIR/attacks/attack_generator.py" "${ATTACK_ARGS[@]}"

echo "--- Detect ---"
DETECT_ARGS=(
    --input      "$ATK_CSV"
    --ca_record  "$CA_JSON"
    --config     "$CONFIG"
    --output_dir "$TRIAL_DIR"
)
[[ -n "$DETECTION_MODE" ]] && DETECT_ARGS+=(--detection_mode "$DETECTION_MODE")
[[ -n "$TRUE_S_INS"     ]] && DETECT_ARGS+=(--true_s_ins "$TRUE_S_INS")
[[ -n "$TRUE_S_DEL"     ]] && DETECT_ARGS+=(--true_s_del "$TRUE_S_DEL")
[[ -n "$TRUE_S_MOD"     ]] && DETECT_ARGS+=(--true_s_mod "$TRUE_S_MOD")

python "$SCRIPT_DIR/detect.py" "${DETECT_ARGS[@]}"

echo "=== Done. Results in $TRIAL_DIR ==="
