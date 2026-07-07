"""
B5: Alfagi et al. 2016 — Character Frequency & Text-Length Frequency Zero Watermarking (Detect)

Detection (Algorithm 3):
  1. Retrieve WRDB and EWRDB from CA
  2. Compute WRDB' from suspicious database R'
  3. WAR = (number of matching elements / total elements) * 100
     WDR = 1 - WAR/100   (or equivalently: mismatch fraction)
  4. If WDR != 0 → database tampered

Characterization (Algorithm 5 & 6):
  For char frequencies:
    Δfchari   = fchari_orig  - fchari_susp
    ΔFfchari  = Δfchari / fchari_orig * 100
    positive  → char was more frequent in original (deletions)
    negative  → char became more frequent in suspicious (insertions)

  For text-length frequencies:
    Same delta analysis over rfTxtLen values (matched by length key)
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Set

import pandas as pd

from common.utils import (
    load_csv, load_json, save_json, load_config, get_text_cols
)
from common.metrics import compute_war_wdr_float
from b5_alfagi2016.embed import (
    extract_char_frequencies, extract_txtlen_frequencies,
    compute_wrdb, ALPHABET
)


# ---------------------------------------------------------------------------
# WAR / WDR helpers
# ---------------------------------------------------------------------------

def compute_war_wdr_vectors(
    wrdb_orig: List[float],
    wrdb_susp: List[float],
    tol: float = 1e-9,
) -> Dict:
    """Compute WAR/WDR by element-wise comparison of two WRDB vectors."""
    min_len = min(len(wrdb_orig), len(wrdb_susp))
    matches = sum(
        1 for a, b in zip(wrdb_orig[:min_len], wrdb_susp[:min_len])
        if abs(a - b) <= tol
    )
    total = max(len(wrdb_orig), len(wrdb_susp))
    war = (matches / total * 100) if total > 0 else 100.0
    wdr = 1.0 - war / 100.0
    return {
        "WAR": war,
        "WDR": wdr,
        "matches": matches,
        "total": total,
        "length_mismatch": len(wrdb_orig) != len(wrdb_susp),
    }


# ---------------------------------------------------------------------------
# Characterization
# ---------------------------------------------------------------------------

def characterize_char_changes(
    rfchar_orig: Dict[str, float],
    rfchar_susp: Dict[str, float],
) -> Dict:
    """Compute per-character delta analysis."""
    deltas = {}
    for c in ALPHABET:
        orig_val = rfchar_orig.get(c, 0.0)
        susp_val = rfchar_susp.get(c, 0.0)
        delta = orig_val - susp_val
        pct = (delta / orig_val * 100) if orig_val != 0 else (0.0 if delta == 0 else float('inf'))
        if delta > 0:
            trend = "deletion"
        elif delta < 0:
            trend = "insertion"
        else:
            trend = "unchanged"
        deltas[c] = {
            "rfchar_orig": orig_val,
            "rfchar_susp": susp_val,
            "delta": delta,
            "delta_pct": pct,
            "trend": trend,
        }
    return deltas


def characterize_txtlen_changes(
    rftxtlen_orig: Dict[str, float],
    rftxtlen_susp: Dict[str, float],
) -> Dict:
    """Compute per-text-length delta analysis."""
    all_lengths = set(rftxtlen_orig.keys()) | set(rftxtlen_susp.keys())
    deltas = {}
    for l in sorted(all_lengths, key=lambda x: int(x)):
        orig_val = rftxtlen_orig.get(str(l), 0.0)
        susp_val = rftxtlen_susp.get(str(l), 0.0)
        delta = orig_val - susp_val
        pct = (delta / orig_val * 100) if orig_val != 0 else (0.0 if delta == 0 else float('inf'))
        if delta > 0:
            trend = "deletion"
        elif delta < 0:
            trend = "insertion"
        else:
            trend = "unchanged"
        deltas[str(l)] = {
            "rftxtlen_orig": orig_val,
            "rftxtlen_susp": susp_val,
            "delta": delta,
            "delta_pct": pct,
            "trend": trend,
        }
    return deltas


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------

def detect_watermark(
    df_suspicious: pd.DataFrame,
    ca_record: Dict,
    tol: float = 1e-9,
) -> Dict:
    """Detect tampering using B5 Alfagi 2016 method.

    Args:
        df_suspicious : suspicious database DataFrame
        ca_record     : CA registration dict (from embed phase)
        tol           : tolerance for floating-point frequency comparison

    Returns:
        result dict with db_tampered, WAR, WDR, characterization, etc.
    """
    pk_col    = ca_record["pk_col"]
    text_cols = ca_record["text_cols"]
    wrdb_orig = ca_record["WRDB"]
    wchar_orig = ca_record["wchar"]
    wtxtlen_orig = ca_record["wtxtlen"]

    alpha_susp = len(df_suspicious)
    alpha_orig = ca_record["alpha"]

    # Filter to text cols available in suspicious DB
    available_text_cols = [c for c in text_cols if c in df_suspicious.columns]

    # Compute watermarks from suspicious DB
    wchar_susp = extract_char_frequencies(df_suspicious, available_text_cols)
    wtxtlen_susp = extract_txtlen_frequencies(df_suspicious, available_text_cols)
    wrdb_susp = compute_wrdb(wchar_susp, wtxtlen_susp)

    # WAR / WDR on full WRDB vectors
    war_wdr = compute_war_wdr_vectors(wrdb_orig, wrdb_susp, tol)

    # WAR / WDR on char sub-watermark only
    wchar_orig_vec = wchar_orig["watermark_vector"]
    wchar_susp_vec = wchar_susp["watermark_vector"]
    war_wdr_char = compute_war_wdr_vectors(wchar_orig_vec, wchar_susp_vec, tol)

    # WAR / WDR on txtlen sub-watermark only
    # Note: if sorted_lengths differ, vectors have different lengths → size_mismatch
    wtxtlen_orig_vec = wtxtlen_orig["watermark_vector"]
    wtxtlen_susp_vec = wtxtlen_susp["watermark_vector"]
    war_wdr_txtlen = compute_war_wdr_vectors(wtxtlen_orig_vec, wtxtlen_susp_vec, tol)

    db_tampered = war_wdr["WDR"] > tol

    # Characterization
    char_deltas = characterize_char_changes(
        wchar_orig["rfchar"],
        wchar_susp["rfchar"],
    )
    txtlen_deltas = characterize_txtlen_changes(
        wtxtlen_orig["rftxtlen"],
        wtxtlen_susp["rftxtlen"],
    )

    # Determine overall change trend
    trend_counts = {"insertion": 0, "deletion": 0, "unchanged": 0}
    for info in char_deltas.values():
        trend_counts[info["trend"]] += 1
    if trend_counts["insertion"] > trend_counts["deletion"]:
        overall_trend = "net_insertion"
    elif trend_counts["deletion"] > trend_counts["insertion"]:
        overall_trend = "net_deletion"
    elif trend_counts["insertion"] > 0:
        overall_trend = "mixed_modification"
    else:
        overall_trend = "unchanged"

    return {
        "method": "B5_Alfagi2016",
        "n_tuples_suspicious": alpha_susp,
        "n_tuples_original": alpha_orig,
        "db_tampered": db_tampered,
        "WAR": war_wdr["WAR"],
        "WDR": war_wdr["WDR"],
        "war_wdr_char": war_wdr_char,
        "war_wdr_txtlen": war_wdr_txtlen,
        "wrdb_length_mismatch": war_wdr["length_mismatch"],
        "char_stats_suspicious": {
            "total_char_count": wchar_susp["total_char_count"],
            "rfchar": wchar_susp["rfchar"],
        },
        "txtlen_stats_suspicious": {
            "total_txtlen_count": wtxtlen_susp["total_txtlen_count"],
            "sorted_lengths": wtxtlen_susp["sorted_lengths"],
            "rftxtlen": wtxtlen_susp["rftxtlen"],
        },
        "characterization": {
            "char_deltas": char_deltas,
            "txtlen_deltas": txtlen_deltas,
            "overall_trend": overall_trend,
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B5 Alfagi 2016 — Detect watermark")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--ca_record",  required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)
    ca_record = load_json(args.ca_record)

    tol = float(cfg.get("tolerance", 1e-9))
    result = detect_watermark(df, ca_record, tol=tol)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "detect_result.json")
    save_json(result, out_path)

    print(f"[B5 Detect] DB tampered: {result['db_tampered']}")
    print(f"[B5 Detect] WAR={result['WAR']:.4f}%  WDR={result['WDR']:.6f}")
    print(f"[B5 Detect] Overall trend: {result['characterization']['overall_trend']}")
    print(f"[B5 Detect] Result saved → {out_path}")


if __name__ == "__main__":
    main()
