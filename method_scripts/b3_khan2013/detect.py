"""
B3: Khan & Husain 2013 — Fragile Zero Watermarking (Detect)

Detection:
  1. Retrieve CA registration → get encrypted watermark EωR and original ωR
  2. Compute ωR' from suspicious database D'
  3. Compare ωR and ωR' element-wise:
       WAR = match_count / total_count * 100
       WDR = 1 - WAR/100
  4. If WDR != 0 → database is tampered

Characterization (optional):
  For each element:
    Δfd_i = fd_i - fd_i'
    ΔFd_i = Δfd_i / fd_i * 100
  Positive ΔF → insertion trend; negative → deletion; mixed → update

Outputs:
  - WAR, WDR
  - db_tampered (bool: WDR != 0)
  - omega_R_original, omega_R_suspicious
  - per-element diff (characterization)
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional

import pandas as pd

from common.utils import (
    load_csv, load_json, save_json, load_config, get_numeric_cols
)
from common.metrics import compute_war_wdr_float
from b3_khan2013.embed import compute_watermark, encrypt_watermark


def detect_watermark(
    df_suspicious: pd.DataFrame,
    ca_record: Dict,
    tol: float = 1e-9,
) -> Dict:
    """Detect tampering using B3 Khan 2013 method.

    Args:
        df_suspicious : suspicious database DataFrame
        ca_record     : CA registration dict (from embed phase)
        tol           : floating-point tolerance for WAR comparison

    Returns:
        result dict with WAR, WDR, db_tampered, characterization, etc.
    """
    numeric_cols = ca_record["numeric_cols"]
    range_bins = ca_record["range_bins"]
    omega_R_orig = ca_record["watermark"]["omega_R"]

    # Filter to columns that exist in suspicious DB
    available_cols = [c for c in numeric_cols if c in df_suspicious.columns]

    # Compute suspicious watermark
    wm_susp = compute_watermark(df_suspicious, available_cols, range_bins)
    omega_R_susp = wm_susp["omega_R"]

    # Align lengths (suspicious DB may have different max_len in ωl)
    min_len = min(len(omega_R_orig), len(omega_R_susp))
    omega_R_orig_cmp = omega_R_orig[:min_len]
    omega_R_susp_cmp = omega_R_susp[:min_len]

    war_wdr = compute_war_wdr_float(omega_R_orig_cmp, omega_R_susp_cmp, tol=tol)
    WAR = war_wdr["WAR"] * 100.0  # as percentage
    WDR = war_wdr["WDR"] * 100.0

    db_tampered = WDR > 0.0

    # Characterization: per-element delta
    wm_orig = ca_record["watermark"]
    wm_susp_data = wm_susp

    # Digit characterization
    digit_deltas = {}
    for i in range(10):
        fd_orig = wm_orig["omega_d"][i]
        fd_susp = wm_susp_data["omega_d"][i] if i < len(wm_susp_data["omega_d"]) else 0.0
        delta = fd_orig - fd_susp
        delta_pct = (delta / fd_orig * 100.0) if fd_orig != 0 else float('inf')
        digit_deltas[str(i)] = {
            "fd_original": fd_orig,
            "fd_suspicious": fd_susp,
            "delta_fd": delta,
            "delta_pct": delta_pct,
            "trend": "insertion" if delta < -1e-9 else ("deletion" if delta > 1e-9 else "unchanged"),
        }

    # Range characterization
    range_deltas = {}
    for k in range(len(range_bins) + 1):
        fr_orig = wm_orig["omega_r"][k] if k < len(wm_orig["omega_r"]) else 0.0
        fr_susp = wm_susp_data["omega_r"][k] if k < len(wm_susp_data["omega_r"]) else 0.0
        delta = fr_orig - fr_susp
        range_deltas[str(k)] = {
            "fr_original": fr_orig,
            "fr_suspicious": fr_susp,
            "delta_fr": delta,
        }

    result = {
        "method": "B3_Khan2013",
        "n_tuples_suspicious": len(df_suspicious),
        "n_tuples_original": ca_record["watermark"]["n_tuples"],
        "db_tampered": db_tampered,
        "WAR": round(WAR, 4),
        "WDR": round(WDR, 4),
        "omega_R_original": omega_R_orig,
        "omega_R_suspicious": omega_R_susp,
        "watermark_original": wm_orig,
        "watermark_suspicious": wm_susp_data,
        "characterization": {
            "digit_deltas": digit_deltas,
            "range_deltas": range_deltas,
            "overall_trend": _infer_trend(wm_orig["n_tuples"], len(df_suspicious)),
        },
    }

    return result


def _infer_trend(n_orig: int, n_susp: int) -> str:
    if n_susp > n_orig:
        return "insertion_dominant"
    elif n_susp < n_orig:
        return "deletion_dominant"
    else:
        return "same_size_modification_or_substitution"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B3 Khan 2013 — Detect watermark")
    parser.add_argument("--input",       required=True)
    parser.add_argument("--ca_record",   required=True, help="CA registration JSON")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--output_dir",  default=".")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)
    ca_record = load_json(args.ca_record)
    tol = cfg.get("tolerance", 1e-9)

    result = detect_watermark(df, ca_record, tol=tol)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "detect_result.json")
    save_json(result, out_path)

    print(f"[B3 Detect] DB tampered: {result['db_tampered']}")
    print(f"[B3 Detect] WAR={result['WAR']:.2f}%, WDR={result['WDR']:.2f}%")
    print(f"[B3 Detect] Trend: {result['characterization']['overall_trend']}")
    print(f"[B3 Detect] Result saved → {out_path}")


if __name__ == "__main__":
    main()
