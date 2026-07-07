#!/usr/bin/env bash
# _run_detect_job.sh — Single (attack, rho|count, trial) detect job.
# Called per-line by GNU parallel from run_all.sh.
#
# Positional args:
#   1  method           b1..b7 | proposed
#   2  baselines_dir    absolute path to baselines/
#   3  input_csv        absolute path to original dataset CSV
#   4  wm_dir           absolute path to watermarks/trial_NN/
#   5  attack           a0..a8
#   6  mode             "rho" or "count"
#   7  value            rho value (e.g. 0.01) or count (e.g. 5)
#   8  attack_seed      integer 0..4
#   9  wm_seed          integer 1000..1004
#  10  out_dir          absolute path for detect_result.json
#  11  debug            true | false
#  12  config           absolute path to params.yaml
set -euo pipefail

METHOD="$1"; BASELINES_DIR="$2"; INPUT="$3"; WM_DIR="$4"
ATTACK="$5"; MODE="$6"; VALUE="$7"; ATTACK_SEED="$8"; WM_SEED="$9"
OUT_DIR="${10}"; DEBUG="${11}"; CONFIG="${12}"
CONFIG_NAME="${13:-}"   # optional: param analysis config tag e.g. "m10_r05_s05"

mkdir -p "$OUT_DIR"

_ms() { python3 -c "import time; print(int(time.time()*1000))"; }

# ── Method directory mapping ──────────────────────────────────────────────────
case "$METHOD" in
    b1) M_SUBDIR="b1_li2004"        ;;  b2) M_SUBDIR="b2_guo2006"      ;;
    b3) M_SUBDIR="b3_khan2013"      ;;  b4) M_SUBDIR="b4_camara2014"   ;;
    b5) M_SUBDIR="b5_alfagi2016"    ;;  b6) M_SUBDIR="b6_hamadou2020"  ;;
    b7) M_SUBDIR="b7_sun2020"       ;;  proposed) M_SUBDIR="proposed"  ;;
    *) echo "Unknown method: $METHOD"; exit 1 ;;
esac
DETECT_PY="$BASELINES_DIR/$M_SUBDIR/detect.py"

# ── Attack source: watermarked copy vs. original ──────────────────────────────
# b1,b2: distortion-based (LSB embed) — attack the watermarked CSV
# b6:    reversible distortion        — attack the watermarked CSV
# b7:    order-based (tuple reorder)  — attack the watermarked CSV
# b3,b4,b5,proposed: zero-distortion — attack the original
case "$METHOD" in
    b1|b2|b6|b7) ATK_SRC="$WM_DIR/watermarked.csv" ;;
    *)            ATK_SRC="$INPUT" ;;
esac

# ── Unique temp files (safe for parallel execution) ───────────────────────────
UNIQUE="${METHOD}_${ATTACK}_${MODE}${VALUE//./p}_s${ATTACK_SEED}_$$_${RANDOM}"
TMP_ATK="/tmp/atk_${UNIQUE}.csv"
TMP_INFO="/tmp/info_${UNIQUE}.json"

# ── Step 1: Attack (generate to /tmp, deleted after detect) ──────────────────
# MODE: "rho" (percentage), "count" (per-op absolute), "tc" (total absolute)
T0=$(_ms)
# Errors go to OUT_DIR/job.err; deleted on success so only failed jobs leave traces.
case "$MODE" in
    count)
        python3 "$BASELINES_DIR/attacks/attack_generator.py" \
            --input "$ATK_SRC" --output "$TMP_ATK" \
            --attack "$ATTACK" --count "$VALUE" --pk_col id \
            --seed "$ATTACK_SEED" --info_out "$TMP_INFO" \
            > /dev/null 2>> "$OUT_DIR/job.err"
        ;;
    tc)
        python3 "$BASELINES_DIR/attacks/attack_generator.py" \
            --input "$ATK_SRC" --output "$TMP_ATK" \
            --attack "$ATTACK" --total_count "$VALUE" --pk_col id \
            --seed "$ATTACK_SEED" --info_out "$TMP_INFO" \
            > /dev/null 2>> "$OUT_DIR/job.err"
        ;;
    *)  # rho (default)
        python3 "$BASELINES_DIR/attacks/attack_generator.py" \
            --input "$ATK_SRC" --output "$TMP_ATK" \
            --attack "$ATTACK" --rho "$VALUE" --pk_col id \
            --seed "$ATTACK_SEED" --info_out "$TMP_INFO" \
            > /dev/null 2>> "$OUT_DIR/job.err"
        ;;
esac
T_ATK=$(( $(_ms) - T0 ))

# ── Step 2: Detect ────────────────────────────────────────────────────────────
T0=$(_ms)
if [[ "$METHOD" == "b1" || "$METHOD" == "b2" ]]; then
    python3 "$DETECT_PY" \
        --input "$TMP_ATK" --embed_info "$WM_DIR/embed_info.json" \
        --config "$CONFIG" --output_dir "$OUT_DIR" \
        > /dev/null 2>> "$OUT_DIR/job.err"
else
    python3 "$DETECT_PY" \
        --input "$TMP_ATK" --ca_record "$WM_DIR/ca_registration.json" \
        --config "$CONFIG" --output_dir "$OUT_DIR" \
        > /dev/null 2>> "$OUT_DIR/job.err"
fi
T_DET=$(( $(_ms) - T0 ))

# ── Step 3: (TMP_ATK kept alive — needed for group expansion in Step 4) ──────

# ── Step 4: Group expansion → TP/TN/FP/FN/F1_all + compact detect_result.json
# Strategy:
#   proposed      : s_all_p = s_ins_p | s_del_p | s_mod_p  (operation-labelled)
#   group-level   : expand tampered group IDs → all PKs in those groups via TMP_ATK
#   stats-only    : TP/FP/FN = None, F1_all = N/A
# TMP_ATK (suspicious CSV) is available here and deleted in Step 5.
# Only scalars (TP, TN, FP, FN, F1_all) are stored — no PK lists written.
_METRICS_PY="$(mktemp /tmp/metrics_XXXXXX.py)"
cat > "$_METRICS_PY" << 'PYEOF'
import sys, json, os
sys.path.insert(0, sys.argv[1])   # BASELINES_DIR

import pandas as pd
from common.utils import load_config, hash_value, hash_to_int

METHOD      = sys.argv[2]
CONFIG      = sys.argv[3]
TMP_ATK     = sys.argv[4]
OUT_DIR     = sys.argv[5]
TMP_INFO    = sys.argv[6]
WM_SEED     = sys.argv[7]
ATK_SEED    = sys.argv[8]
ATTACK      = sys.argv[9]
MODE        = sys.argv[10]
VALUE       = sys.argv[11]
CONFIG_NAME = sys.argv[12] if len(sys.argv) > 12 else ""  # param analysis tag
WM_DIR      = sys.argv[13] if len(sys.argv) > 13 else ""  # path to watermarks/trial_NN
PK_COL      = "id"

result_path = os.path.join(OUT_DIR, "detect_result.json")

try:    result = json.load(open(result_path))
except: result = {}
try:    info = json.load(open(TMP_INFO))
except: info = {}
try:    cfg = load_config(CONFIG);  secret_key = cfg.get("secret_key", "")
except: secret_key = ""

# ── Storage info ──────────────────────────────────────────────────────────────
# Read storage_bytes from whichever artifact the embed produced:
#   proposed/B0/B3/B4/B5/B6/B7  → ca_registration.json
#   B1/B2                       → embed_info.json (alongside watermarked.csv)
# Key conventions:
#   proposed : {iblt, gt_syndromes, total_compressed}      (compressed mode)
#              also {tuple_digests, total_full} in full mode (B0)
#   B1, B7   : {order_index, total}                        (pk -> rank_W index)
# We record `total_compressed_bytes` as the unified "external metadata size",
# i.e. the rho numerator of the paper.
compressed_storage_bytes = None   # proposed: g × b/8 (Layer 2 syndromes only)
iblt_storage_bytes       = None   # proposed: IBLT Layer 1 bytes
total_compressed_bytes   = None   # unified |W(R)| in bytes
for cand in ("ca_registration.json", "embed_info.json"):
    p = os.path.join(WM_DIR, cand)
    if not os.path.exists(p):
        continue
    try:
        sb = json.load(open(p)).get("storage_bytes", {}) or {}
    except Exception:
        continue
    if not sb:
        continue
    compressed_storage_bytes = sb.get("gt_syndromes", compressed_storage_bytes)
    iblt_storage_bytes       = sb.get("iblt",        iblt_storage_bytes)
    # Unified total |W(R)|. Proposed embed writes BOTH total_compressed and
    # total_full regardless of mode, so we must pick by the active mode written
    # to detect_result.json (set by proposed/detect.py: "compressed" or "full").
    # B1/B7 store only `total` and `order_index`; the fallback handles others.
    active_mode = result.get("detection_mode")  # "compressed" | "full" | None
    if active_mode == "full" and sb.get("total_full") is not None:
        total_compressed_bytes = sb.get("total_full")
    elif active_mode == "compressed" and sb.get("total_compressed") is not None:
        total_compressed_bytes = sb.get("total_compressed")
    else:
        total_compressed_bytes = (
            sb.get("total") or sb.get("order_index")
            or sb.get("total_compressed") or sb.get("total_full")
            or ((compressed_storage_bytes or 0) + (iblt_storage_bytes or 0) or None)
        )
    break

# ── Ground truth ─────────────────────────────────────────────────────────────
s_ins_t = set(str(x) for x in (info.get("s_ins") or []))
s_del_t = set(str(x) for x in (info.get("s_del") or []))
s_mod_t = set(str(x) for x in (info.get("s_mod") or []))
s_all_t = s_ins_t | s_del_t | s_mod_t

# ── Predicted set (method-specific) ──────────────────────────────────────────
STATS_ONLY  = {"b3", "b5"}
GROUP_LEVEL = {"b1", "b2", "b4", "b6", "b7"}

n_susp  = 0
s_all_p = set()

if METHOD == "proposed":
    s_all_p = (set(str(x) for x in (result.get("s_ins") or [])) |
               set(str(x) for x in (result.get("s_del") or [])) |
               set(str(x) for x in (result.get("s_mod") or [])))

elif METHOD in GROUP_LEVEL:
    tampered_groups = result.get("tampered_groups", [])
    try:
        df_susp = pd.read_csv(TMP_ATK)
        n_susp  = len(df_susp)
    except Exception:
        df_susp = pd.DataFrame(); n_susp = 0

    if METHOD == "b2":
        s_all_p = set(str(x) for x in (result.get("tampered_tuple_pks") or []))

    elif METHOD == "b7":
        positions = result.get("tampered_tuple_positions", [])
        for pos in positions:
            if 0 <= pos < n_susp:
                s_all_p.add(str(df_susp.iloc[pos][PK_COL]))

    elif METHOD == "b1":
        num_groups = result.get("num_groups", 10)
        groups = {}
        for pos in range(n_susp):
            pk = str(df_susp.iloc[pos][PK_COL])
            gid = hash_to_int(secret_key, pk, mod=num_groups)
            groups.setdefault(gid, []).append(pk)
        for gid in tampered_groups:
            s_all_p.update(groups.get(gid, []))

    elif METHOD in ("b4", "b6"):
        num_groups = result.get("num_groups_original", 20)
        groups = {}
        for pos in range(n_susp):
            pk = str(df_susp.iloc[pos][PK_COL])
            h  = hash_value(secret_key, pk, secret_key)
            gid = int(h % num_groups)
            groups.setdefault(gid, []).append(pk)
        for gid in tampered_groups:
            s_all_p.update(groups.get(gid, []))

# ── TP / FP / FN / TN ────────────────────────────────────────────────────────
# STATS_ONLY methods (B3/B5) emit only WAR/WDR — no tuple predictions at all.
# Storing TP=0/FN=|s_all_t| would be actively misleading (implies predictions
# were made and all were wrong).  Use None to signal "not applicable".
if METHOD in STATS_ONLY:
    TP = TN = FP = FN = None
    precision_all = recall_all = f1_all = None
else:
    TP = len(s_all_t & s_all_p)
    FP = len(s_all_p - s_all_t)
    FN = len(s_all_t - s_all_p)
    n_universe = max(result.get("n_tuples_suspicious", 0),
                     result.get("n_tuples_original",   0),
                     n_susp)
    TN = max(0, n_universe - TP - FP - FN)

    # ── F1_all ────────────────────────────────────────────────────────────────
    if not s_all_t:
        # A0 (clean trial): no attack → F1 is undefined, not 1.0
        precision_all = recall_all = f1_all = None
    else:
        precision_all = round(TP / (TP + FP), 6) if (TP + FP) > 0 else 0.0
        recall_all    = round(TP / (TP + FN), 6) if (TP + FN) > 0 else 0.0
        f1_all        = round(2*TP / (2*TP + FP + FN), 6) if (2*TP + FP + FN) > 0 else 0.0

# ── Operation-specific F1 (proposed only) ────────────────────────────────────
def prf(true_set, pred_set):
    if not true_set: return None, None, None
    tp = len(true_set & pred_set); fp = len(pred_set-true_set); fn = len(true_set-pred_set)
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f = 2*p*r/(p+r) if (p+r)>0 else 0.0
    return round(p,6), round(r,6), round(f,6)

s_ins_p = set(str(x) for x in (result.get("s_ins") or []))
s_del_p = set(str(x) for x in (result.get("s_del") or []))
s_mod_p = set(str(x) for x in (result.get("s_mod") or []))
pi,ri,fi  = prf(s_ins_t, s_ins_p)
pd2,rd,fd = prf(s_del_t, s_del_p)
pm,rm,fm  = prf(s_mod_t, s_mod_p)

# 4-class confusion (for tuple-status classification): cross-classification counts
# {true op} x {pred op}; 9 integers. Only meaningful when the method natively
# outputs the three operation sets — Proposed (compressed/full) and B0 (= proposed
# full mode). The 4-class confusion matrix can be reconstructed from these 9 + the
# stored n_true_*/n_pred_* counts at aggregation time.
status_xclass = None
if METHOD == "proposed":
    status_xclass = {
        "true_ins_pred_ins": len(s_ins_t & s_ins_p),
        "true_ins_pred_del": len(s_ins_t & s_del_p),
        "true_ins_pred_mod": len(s_ins_t & s_mod_p),
        "true_del_pred_ins": len(s_del_t & s_ins_p),
        "true_del_pred_del": len(s_del_t & s_del_p),
        "true_del_pred_mod": len(s_del_t & s_mod_p),
        "true_mod_pred_ins": len(s_mod_t & s_ins_p),
        "true_mod_pred_del": len(s_mod_t & s_del_p),
        "true_mod_pred_mod": len(s_mod_t & s_mod_p),
    }

# ── FPR_tuple (proposed only) ─────────────────────────────────────────────────
if METHOD == "proposed" and s_all_t:
    n_orig = result.get("n_tuples_original", 0)
    denom_fpr = max(1, n_orig + len(s_ins_p) - len(s_all_t))
    fpr = round(len(s_all_p - s_all_t) / denom_fpr, 6)
else:
    fpr = None

# ── Compact JSON ─────────────────────────────────────────────────────────────
compact = {
    "_meta": {
        "method": METHOD, "attack": ATTACK, "mode": MODE,
        "rho":         float(VALUE) if MODE=="rho"   else None,
        "count":       int(VALUE)   if MODE=="count" else None,
        "total_count": int(VALUE)   if MODE=="tc"    else None,
        "attack_seed": int(ATK_SEED), "wm_seed": int(WM_SEED),
        "config_name": CONFIG_NAME if CONFIG_NAME else None,
    },
    "db_tampered_true": bool(s_all_t),
    "n_true_ins": len(s_ins_t), "n_true_del": len(s_del_t), "n_true_mod": len(s_mod_t),
    "db_tampered_pred": bool(result.get("db_tampered", False)),
    "n_pred_ins": len(s_ins_p), "n_pred_del": len(s_del_p), "n_pred_mod": len(s_mod_p),
    "iblt_decode_success": result.get("iblt_decode_success"),
    "detection_mode":      result.get("detection_mode"),
    # 4-class status confusion (only present when computed; absent otherwise)
    **({"status_xclass": status_xclass} if status_xclass is not None else {}),
    # Component-level detect timings, when emitted by proposed/detect.py (S4)
    **({"detect_timing_ms": result["detect_timing_ms"]} if "detect_timing_ms" in result else {}),
    "WAR": result.get("WAR"), "WDR": result.get("WDR"),
    "n_groups_tampered": result.get("n_tampered_groups"),
    # Storage (bytes): from ca_registration.json; None for baselines
    "compressed_storage_bytes": compressed_storage_bytes,  # g × b/8 (Layer 2)
    "iblt_storage_bytes":       iblt_storage_bytes,        # IBLT Layer 1
    "total_compressed_bytes":   total_compressed_bytes,    # iblt + gt_syndromes
    # Tuple-level confusion matrix
    "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    # Tuple-level metrics (operation-agnostic)
    "precision_all": precision_all, "recall_all": recall_all, "f1_all": f1_all,
    "fpr_tuple": fpr,
    # Operation-specific (proposed only; N/A for baselines)
    "precision_ins": pi, "recall_ins": ri, "f1_ins": fi,
    "precision_del": pd2, "recall_del": rd, "f1_del": fd,
    "precision_mod": pm, "recall_mod": rm, "f1_mod": fm,
}

json.dump(compact, open(result_path, "w"), indent=2)
PYEOF

python3 "$_METRICS_PY" \
    "$BASELINES_DIR" "$METHOD" "$CONFIG" "$TMP_ATK" "$OUT_DIR" "$TMP_INFO" \
    "$WM_SEED" "$ATTACK_SEED" "$ATTACK" "$MODE" "$VALUE" "$CONFIG_NAME" "$WM_DIR" \
    2>> "$OUT_DIR/job.err"
rm -f "$_METRICS_PY"

# ── Step 5: Delete temp files (TMP_ATK kept until after group expansion above) ─
rm -f "$TMP_ATK"
if [[ "$DEBUG" == "true" ]]; then
    cp "$TMP_INFO" "$OUT_DIR/attack_info.json"
fi
rm -f "$TMP_INFO"

# ── Step 6: Mark job as fully complete ───────────────────────────────────────
# .done is written ONLY after every step above succeeded.
# The skip check in run_fct_baselines.sh tests for .done — not detect_result.json —
# so a directory where detect.py wrote its output but Step 4 crashed will NOT be
# falsely skipped on the next run.
printf '{"attack_ms": %s, "detect_ms": %s, "total_ms": %s}\n' \
    "$T_ATK" "$T_DET" "$((T_ATK + T_DET))" > "$OUT_DIR/timing.json"
rm -f "$OUT_DIR/job.err"   # clean on success; kept on failure for diagnosis
touch "$OUT_DIR/.done"

echo "  ✓ $METHOD | $ATTACK | ${MODE}=${VALUE} | trial=$ATTACK_SEED"
