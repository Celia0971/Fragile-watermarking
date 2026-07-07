#!/usr/bin/env bash
# B4 Camara 2014 — end-to-end run script
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$SCRIPT_DIR/config/params.yaml"

INPUT=""; ATTACK="a3"; RHO="0.01"; ATTACK_SEED="0"; WM_SEED="1000"; PK_COL="id"
OUTPUT_DIR="$SCRIPT_DIR/results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)       INPUT="$2";       shift 2 ;;
        --attack)      ATTACK="$2";      shift 2 ;;
        --rho)         RHO="$2";         shift 2 ;;
        --attack_seed) ATTACK_SEED="$2"; shift 2 ;;
        --wm_seed)     WM_SEED="$2";     shift 2 ;;
        --pk_col)      PK_COL="$2";      shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --config)      CONFIG="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$INPUT" ]] && { echo "Error: --input required"; exit 1; }

TRIAL_DIR="$OUTPUT_DIR/trial_$(printf '%02d' "$ATTACK_SEED")"
mkdir -p "$TRIAL_DIR"

ATK_CSV="$TRIAL_DIR/attacked_${ATTACK}_rho${RHO}.csv"
ATK_INFO="$TRIAL_DIR/attack_info.json"
CA_JSON="$TRIAL_DIR/ca_registration.json"

echo "=== B4 Camara 2014 | attack=$ATTACK rho=$RHO attack_seed=$ATTACK_SEED wm_seed=$WM_SEED ==="

echo "--- Register watermark (embed on original) ---"
python "$SCRIPT_DIR/embed.py" \
    --input "$INPUT" --output_dir "$TRIAL_DIR" --config "$CONFIG"

echo "--- Attack ($ATTACK, rho=$RHO) on original DB ---"
python "$BASELINES_DIR/attacks/attack_generator.py" \
    --input "$INPUT" --output "$ATK_CSV" \
    --attack "$ATTACK" --rho "$RHO" --pk_col "$PK_COL" \
    --seed "$ATTACK_SEED" --info_out "$ATK_INFO"

echo "--- Detect ---"
python "$SCRIPT_DIR/detect.py" \
    --input "$ATK_CSV" --ca_record "$CA_JSON" \
    --config "$CONFIG" --output_dir "$TRIAL_DIR"

echo "=== Done. Results in $TRIAL_DIR ==="
