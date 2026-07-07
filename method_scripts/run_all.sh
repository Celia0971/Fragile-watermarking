#!/usr/bin/env bash
# ============================================================
# run_all.sh — Two-phase experiment runner.
#
# Phase 1 (Embed): For each method × trial, embed watermark
#   ONCE and store in watermarks/trial_NN/.  Skips if already done.
#
# Phase 2 (Detect): For every (method, attack, rho, trial),
#   apply attack in /tmp → run detect → delete attacked CSV.
#   Jobs run in parallel via GNU parallel (--ncpus controls degree).
#
# Watermark storage (shared across all attack/rho combos):
#   {out}/{machine}/{dataset}/{method}/watermarks/trial_NN/
#     ca_registration.json   (zero-distortion methods)
#     embed_info.json        (B1, B2)
#     watermarked.csv        (distortion-based: B1, B2, B6, B7)
#
# Detect results:
#   {out}/{machine}/{dataset}/{method}/{attack}/{rho}/trial_NN/
#     detect_result.json     (always — includes embedded metadata + GT)
#     timing.json            (always)
#     attack_info.json       (only if --debug)
#
# Usage:
#   bash run_all.sh --input datasets/processed/FCT.csv --dataset FCT \
#       --methods "proposed,b2" --machine_id machine29
#
# Options:
#   --input       Path to input CSV (required)
#   --dataset     Dataset name tag (default: input basename)
#   --machine_id  Machine label (default: $HOSTNAME)
#   --methods     Comma-separated (default: b1,b2,b3,b4,b5,b6,b7,proposed)
#   --attacks     Comma-separated a0–a8 (default: all 9)
#   --rhos        Comma-separated ρ values (default: full 15-pt R_attack)
#   --n_trials    Number of trials (default: 5; seeds: attack 0–4, wm 1000–1004)
#   --ncpus       Parallel detect jobs (default: auto = nproc - 1)
#   --output_dir  Base results dir (default: ../results)
#   --debug       Save attack_info.json per trial (flag)
#   --skip_embed  Skip Phase 1 if watermarks already exist (flag)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── R_attack grid (paper §7.6) — edit here to adjust ─────────────────────────
R_ATTACK="0.000001,0.00001,0.0001,0.001,0.005,0.01,0.025,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60"

# ── Absolute count grid — edit here to adjust ────────────────────────────────
# Count = per-operation count for each attack type.
# Min valid count per attack (enforced in job generation):
#   A0: N/A  A1/A2/A3: 1  A4: 1 (pairs)  A5/A6/A7: 1  A8: 1
COUNT_GRID="1,2,3,5,10,20,50,100,200,300,500,1000"

# ── Defaults ──────────────────────────────────────────────────────────────────
INPUT=""
DATASET=""
MACHINE_ID="${HOSTNAME:-local}"
METHODS="b1,b2,b3,b4,b5,b6,b7,proposed"
ATTACKS="a0,a1,a2,a3,a4,a5,a6,a7,a8"
RHOS="$R_ATTACK"
COUNTS="$COUNT_GRID"   # set to "" to disable count-based attacks
N_TRIALS=5
NCPUS=""           # auto
OUTPUT_DIR="$SCRIPT_DIR/../results"
DEBUG=false
SKIP_EMBED=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)       INPUT="$2";       shift 2 ;;
        --dataset)     DATASET="$2";     shift 2 ;;
        --machine_id)  MACHINE_ID="$2";  shift 2 ;;
        --methods)     METHODS="$2";     shift 2 ;;
        --attacks)     ATTACKS="$2";     shift 2 ;;
        --rhos)        RHOS="$2";        shift 2 ;;
        --counts)      COUNTS="$2";      shift 2 ;;
        --n_trials)    N_TRIALS="$2";    shift 2 ;;
        --ncpus)       NCPUS="$2";       shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --debug)       DEBUG=true;       shift   ;;
        --skip_embed)  SKIP_EMBED=true;  shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "$INPUT" ]]   && { echo "Error: --input required"; exit 1; }
[[ ! -f "$INPUT" ]] && { echo "Error: Input file not found: $INPUT"; exit 1; }
[[ -z "$DATASET" ]] && DATASET=$(basename "$INPUT" .csv)

# Auto CPU count: leave 1 core free
if [[ -z "$NCPUS" ]]; then
    NCPUS=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) - 1 ))
    [[ "$NCPUS" -lt 1 ]] && NCPUS=1
fi

# ── Method → subdirectory ─────────────────────────────────────────────────────
_method_subdir() {
    case "$1" in
        b1) echo "b1_li2004"        ;; b2) echo "b2_guo2006"       ;;
        b3) echo "b3_khan2013"      ;; b4) echo "b4_camara2014"    ;;
        b5) echo "b5_alfagi2016"    ;; b6) echo "b6_hamadou2020"   ;;
        b7) echo "b7_sun2020"       ;; proposed) echo "proposed"   ;;
        *) echo "UNKNOWN" ;;
    esac
}

# A0 uses a fixed rho of 0.00 (clean copy, no rho loop)
NO_RHO_ATTACKS="a0"

IFS=',' read -ra METHOD_LIST  <<< "$METHODS"
IFS=',' read -ra ATTACK_LIST  <<< "$ATTACKS"
IFS=',' read -ra RHO_LIST     <<< "$RHOS"
COUNT_LIST=()
[[ -n "$COUNTS" ]] && IFS=',' read -ra COUNT_LIST <<< "$COUNTS"

BASE_OUT="$OUTPUT_DIR/$MACHINE_ID/$DATASET"

# Minimum per-operation count per attack (counts below are skipped).
# For mixed attacks, this is the per-operation minimum (total = n_ops × min).
_min_count_for_attack() {
    case "$1" in
        a0) echo "999999" ;;   # A0 never uses count
        a1|a2|a3|a4) echo "1" ;;
        a5|a6|a7)    echo "1" ;;
        a8)          echo "1" ;;
        *)           echo "1" ;;
    esac
}

echo "========================================================"
echo "  run_all.sh — Two-phase experiment runner"
echo "  Input    : $INPUT  ($DATASET)"
echo "  Methods  : ${METHOD_LIST[*]}"
echo "  Attacks  : ${ATTACK_LIST[*]}"
echo "  Rhos     : ${#RHO_LIST[@]} values  (${RHO_LIST[0]} … ${RHO_LIST[${#RHO_LIST[@]}-1]})"
  if [[ ${#COUNT_LIST[@]} -gt 0 ]]; then
    _LAST_COUNT="${COUNT_LIST[${#COUNT_LIST[@]}-1]}"
    echo "  Counts   : ${#COUNT_LIST[@]} values  (${COUNT_LIST[0]} … ${_LAST_COUNT})"
  else
    echo "  Counts   : 0 values  (disabled)"
  fi
echo "  Trials   : $N_TRIALS  (wm seeds 1000–$((1000+N_TRIALS-1)))"
echo "  Parallel : $NCPUS CPU(s)"
echo "  Debug    : $DEBUG"
echo "  Output   : $BASE_OUT"
echo "========================================================"

# ── Phase 1: Embed (once per method × trial) ──────────────────────────────────
echo ""
echo "════ Phase 1: Embed ════"

_ms() { python3 -c "import time; print(int(time.time()*1000))"; }

for METHOD in "${METHOD_LIST[@]}"; do
    METHOD=$(echo "$METHOD" | tr -d '[:space:]')
    M_SUBDIR=$(_method_subdir "$METHOD")
    if [[ "$M_SUBDIR" == "UNKNOWN" ]]; then
        echo "  [SKIP] Unknown method: $METHOD"; continue
    fi
    M_DIR="$SCRIPT_DIR/$M_SUBDIR"
    CONFIG="$M_DIR/config/params.yaml"
    EMBED_PY="$M_DIR/embed.py"

    for TRIAL in $(seq 0 $((N_TRIALS - 1))); do
        WM_SEED=$((1000 + TRIAL))
        WM_DIR="$BASE_OUT/$METHOD/watermarks/trial_$(printf '%02d' $TRIAL)"

        # Skip if watermark already exists
        if $SKIP_EMBED && [[ -f "$WM_DIR/ca_registration.json" ]] || \
                          [[ -f "$WM_DIR/embed_info.json" ]]; then
            echo "  [skip] $METHOD  trial_$(printf '%02d' $TRIAL)  (watermark exists)"
            continue
        fi

        mkdir -p "$WM_DIR"
        echo "  Embedding: $METHOD | trial_$(printf '%02d' $TRIAL) | wm_seed=$WM_SEED"
        T0=$(_ms)

        case "$METHOD" in
            b1|b2)
                python3 "$EMBED_PY" \
                    --input "$INPUT" --output "$WM_DIR/watermarked.csv" \
                    --config "$CONFIG" --info_out "$WM_DIR/embed_info.json" \
                    > "$WM_DIR/embed.log" 2>&1
                ;;
            *)
                python3 "$EMBED_PY" \
                    --input "$INPUT" --output_dir "$WM_DIR" \
                    --config "$CONFIG" \
                    > "$WM_DIR/embed.log" 2>&1
                # proposed: watermarked.csv == original — delete to save space
                if [[ "$METHOD" == "proposed" ]]; then
                    rm -f "$WM_DIR/watermarked.csv"
                fi
                ;;
        esac

        T_EMBED=$(( $(_ms) - T0 ))
        # Remove embed log if not in debug mode
        $DEBUG || rm -f "$WM_DIR/embed.log"
        echo "    → done in ${T_EMBED} ms"
    done
done

# ── Phase 2: Build detect job list ────────────────────────────────────────────
echo ""
echo "════ Phase 2: Building detect job list ════"

DETECT_HELPER="$SCRIPT_DIR/_run_detect_job.sh"
JOB_ARGS=()

for METHOD in "${METHOD_LIST[@]}"; do
    METHOD=$(echo "$METHOD" | tr -d '[:space:]')
    M_SUBDIR=$(_method_subdir "$METHOD")
    [[ "$M_SUBDIR" == "UNKNOWN" ]] && continue

    M_DIR="$SCRIPT_DIR/$M_SUBDIR"
    CONFIG="$M_DIR/config/params.yaml"

    for ATTACK in "${ATTACK_LIST[@]}"; do
        ATTACK=$(echo "$ATTACK" | tr -d '[:space:]')
        MIN_COUNT=$(_min_count_for_attack "$ATTACK")

        # ── Rho-based jobs ────────────────────────────────────────────────────
        if echo "$NO_RHO_ATTACKS" | grep -qw "$ATTACK"; then
            EFFECTIVE_RHOS=("0.00")
        else
            EFFECTIVE_RHOS=("${RHO_LIST[@]}")
        fi

        for RHO in "${EFFECTIVE_RHOS[@]}"; do
            RHO=$(echo "$RHO" | tr -d '[:space:]')
            for TRIAL in $(seq 0 $((N_TRIALS - 1))); do
                ATTACK_SEED=$TRIAL
                WM_SEED=$((1000 + TRIAL))
                WM_DIR="$BASE_OUT/$METHOD/watermarks/trial_$(printf '%02d' $TRIAL)"
                OUT_DIR="$BASE_OUT/$METHOD/$ATTACK/rho${RHO}/trial_$(printf '%02d' $TRIAL)"
                [[ -f "$OUT_DIR/detect_result.json" ]] && continue
                JOB_ARGS+=("$METHOD $SCRIPT_DIR $INPUT $WM_DIR $ATTACK rho $RHO $ATTACK_SEED $WM_SEED $OUT_DIR $DEBUG $CONFIG")
            done
        done

        # ── Count-based jobs (skip A0 and counts below minimum) ───────────────
        if ! echo "$NO_RHO_ATTACKS" | grep -qw "$ATTACK" && [[ ${#COUNT_LIST[@]} -gt 0 ]]; then
            for COUNT in "${COUNT_LIST[@]}"; do
                COUNT=$(echo "$COUNT" | tr -d '[:space:]')
                [[ "$COUNT" -lt "$MIN_COUNT" ]] && continue  # skip invalid counts

                for TRIAL in $(seq 0 $((N_TRIALS - 1))); do
                    ATTACK_SEED=$TRIAL
                    WM_SEED=$((1000 + TRIAL))
                    WM_DIR="$BASE_OUT/$METHOD/watermarks/trial_$(printf '%02d' $TRIAL)"
                    OUT_DIR="$BASE_OUT/$METHOD/$ATTACK/k${COUNT}/trial_$(printf '%02d' $TRIAL)"
                    [[ -f "$OUT_DIR/detect_result.json" ]] && continue
                    JOB_ARGS+=("$METHOD $SCRIPT_DIR $INPUT $WM_DIR $ATTACK count $COUNT $ATTACK_SEED $WM_SEED $OUT_DIR $DEBUG $CONFIG")
                done
            done
        fi
    done
done

TOTAL=${#JOB_ARGS[@]}
echo "  Total detect jobs: $TOTAL  (skipped already-completed jobs)"

# ── Phase 3: Run detect jobs ──────────────────────────────────────────────────
echo ""
echo "════ Phase 3: Detect ($NCPUS parallel workers) ════"

if [[ $TOTAL -eq 0 ]]; then
    echo "  Nothing to do — all jobs already completed."
elif command -v parallel &>/dev/null; then
    export -f _method_subdir 2>/dev/null || true
    # --colsep ' ' splits each job line into {1}..{12} so detect script
    # receives 12 separate positional arguments (not one big quoted string).
    # Capture parallel exit code separately — do NOT let a non-zero exit
    # propagate through set -e and kill the outer sequential pipeline.
    # nohup parallel + tmpfile: permanent SIGHUP immunity.
    # parallel re-establishes its own SIGHUP handler overriding trap's SIG_IGN,
    # but nohup sets SIG_IGN at exec-time which parallel cannot override.
    TMPJOBS=$(mktemp /tmp/parallel_jobs_XXXXXX)
    printf '%s\n' "${JOB_ARGS[@]}" > "$TMPJOBS"
    _PAR_RC=0
    nohup parallel --jobs "$NCPUS" --colsep ' ' --halt never \
        bash "$DETECT_HELPER" \
        '{1}' '{2}' '{3}' '{4}' '{5}' '{6}' '{7}' '{8}' '{9}' '{10}' '{11}' '{12}' \
        :::: "$TMPJOBS" || _PAR_RC=$?
    rm -f "$TMPJOBS"
    [[ $_PAR_RC -ne 0 ]] && \
        echo "  WARNING: parallel exited $_PAR_RC — some jobs failed. Re-run to retry missing results." \
        || true
else
    echo "  [INFO] GNU parallel not found — running sequentially (install with: sudo apt install parallel)"
    DONE=0; FAILED=0
    _SEQ_T0=$(python3 -c "import time; print(int(time.time()))")

    for JOB in "${JOB_ARGS[@]}"; do
        DONE=$((DONE + 1))

        # ── Progress bar with ETA ─────────────────────────────────────────────
        _NOW=$(python3 -c "import time; print(int(time.time()))")
        _EL=$(( _NOW - _SEQ_T0 ))
        python3 -c "
done=$DONE; total=$TOTAL; el=$_EL
pct = done * 100 // total
filled = pct // 5
bar = '█' * filled + '░' * (20 - filled)
if done > 1 and el > 0:
    eta = int(el * (total - done) / (done - 1))
    eta_str = f'{eta//3600}h{eta%3600//60:02d}m'
    el_str  = f'{el//3600}h{el%3600//60:02d}m'
    print(f'  [{bar}] {done:>5}/{total} ({pct:3d}%) elapsed={el_str} ETA={eta_str}')
else:
    print(f'  [{bar}] {done:>5}/{total} ({pct:3d}%)')
" 2>/dev/null || echo "  [$DONE/$TOTAL]"

        bash "$DETECT_HELPER" $JOB || { FAILED=$((FAILED + 1)); echo "  FAILED: $JOB"; }
    done
    echo ""
    echo "  Completed: $((DONE-FAILED))/$DONE  Failed: $FAILED"
    [[ $FAILED -gt 0 ]] && exit 1
fi

echo ""
echo "========================================================"
echo "  All done.  Results: $BASE_OUT"
echo "========================================================"
