"""
B4: Camara et al. 2014 — Square Matrix Determinant+Minors Zero Watermarking (Detect)

Detection (Algorithm 4):
  1. Extract EWD and WD from CA
  2. Partition suspicious DB D' into ν' groups using same K
  3. For each group j:
       Compute Wj' = Dj' || Mj'_i (same as embed)
       If Wj' == Wj → group non-altered; else → altered
  4. Report tampered groups

Localization (Algorithm 5):
  If Wj != Wj':
    For each diagonal position i:
      If Mji != Mji' → "tampered data in column i (attribute i)"

Special handling:
  - If |ν'| != |ν|: size-change attack detected immediately
  - Floating point comparison uses tolerance (due to numerical det computation)
"""

import argparse
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from common.utils import (
    load_csv, load_json, save_json, load_config, hash_value, get_numeric_cols
)
from common.metrics import compute_localization_metrics
from b4_camara2014.embed import (
    partition_into_groups, compute_group_watermark, compute_determinant,
    compute_diagonal_minor, group_to_matrix
)


DET_TOL = 1e-6   # tolerance for floating-point determinant comparison


def watermark_vectors_equal(
    wj_ref: List[float],
    wj_susp: List[float],
    tol: float = DET_TOL,
) -> bool:
    """Compare two watermark vectors with floating-point tolerance."""
    if len(wj_ref) != len(wj_susp):
        return False
    return all(abs(a - b) <= tol for a, b in zip(wj_ref, wj_susp))


def detect_watermark(
    df_suspicious: pd.DataFrame,
    ca_record: Dict,
    tol: float = DET_TOL,
) -> Dict:
    """Detect tampering using B4 Camara 2014 method.

    Args:
        df_suspicious : suspicious database DataFrame
        ca_record     : CA registration dict (from embed phase)
        tol           : tolerance for floating-point watermark comparison

    Returns:
        result dict with db_tampered, tampered_groups, localization, etc.
    """
    pk_col      = ca_record["pk_col"]
    numeric_cols= ca_record["numeric_cols"]
    secret_key  = ca_record["method"]  # not stored directly; re-read from config
    num_groups_orig = ca_record["num_groups"]
    gamma       = ca_record["gamma"]
    group_wm_ref= ca_record["group_watermarks"]

    # We need secret_key from the caller; it is not stored in ca_record for security.
    # The caller must inject it. We'll accept it via a special field if present.
    # In the CLI we'll pass it from config.
    secret_key  = ca_record.get("_secret_key_internal", "")

    alpha_susp = len(df_suspicious)
    num_groups_susp = max(1, math.ceil(alpha_susp / gamma))

    # Detect size-change attack
    size_mismatch = (num_groups_susp != num_groups_orig)

    # Filter numeric_cols to those available in suspicious DB
    available_cols = [c for c in numeric_cols if c in df_suspicious.columns]

    # Partition suspicious DB
    groups_susp = partition_into_groups(
        df_suspicious, pk_col, secret_key, num_groups_orig
    )

    group_results = {}
    tampered_groups = []
    tampered_attributes = {}  # gid → list of attribute indices

    for gid in range(num_groups_orig):
        ref_key = str(gid)
        positions = groups_susp.get(gid, [])

        if ref_key not in group_wm_ref:
            group_results[gid] = {"status": "missing_reference", "authentic": False}
            tampered_groups.append(gid)
            continue

        ref_wj = group_wm_ref[ref_key]["watermark"]
        orig_size = group_wm_ref[ref_key].get("original_size", 1)

        if len(positions) == 0:
            if orig_size == 0:
                # Group was also empty at embed time → authentic (expected)
                group_results[gid] = {"size": 0, "status": "authentic_empty", "authentic": True}
            else:
                # Group had members but is now empty → tampered
                group_results[gid] = {
                    "size": 0, "status": "tampered_empty", "authentic": False,
                    "wj_reference": ref_wj, "wj_suspicious": None,
                }
                tampered_groups.append(gid)
            continue

        # Build group DataFrame, pad if needed (same logic as embed: use group-0 members first)
        group_df = df_suspicious.iloc[positions].copy()
        orig_size = len(group_df)

        if len(group_df) < gamma:
            n_needed = gamma - len(group_df)
            first_group_positions = groups_susp.get(0, [])
            pad_positions = first_group_positions[:n_needed]
            if len(pad_positions) < n_needed:
                pad_positions = list(range(min(n_needed, len(df_suspicious))))
            pad_df = df_suspicious.iloc[pad_positions].copy()
            group_df = pd.concat([group_df, pad_df], ignore_index=True)

        group_df = group_df.head(gamma).reset_index(drop=True)

        # Compute suspicious watermark
        g_susp_info = compute_group_watermark(
            group_df, pk_col, secret_key, available_cols, len(available_cols)
        )
        wj_susp = g_susp_info["watermark"]

        # Use reference watermark length (gamma may differ if cols changed)
        wj_ref_trimmed = ref_wj[:len(wj_susp)]
        wj_susp_trimmed = wj_susp[:len(wj_ref_trimmed)]

        is_authentic = watermark_vectors_equal(wj_ref_trimmed, wj_susp_trimmed, tol)

        # Localization: if tampered, find which diagonal minor differs
        tampered_attr_cols = []
        if not is_authentic:
            # ref_wj = [det, minor_0, ..., minor_{gamma-1}]
            # wj_susp = [det', minor_0', ...]
            for attr_idx in range(min(gamma, len(wj_ref_trimmed) - 1)):
                ref_minor = wj_ref_trimmed[attr_idx + 1]
                susp_minor = wj_susp_trimmed[attr_idx + 1] if attr_idx + 1 < len(wj_susp_trimmed) else None
                if susp_minor is None or abs(ref_minor - susp_minor) > tol:
                    tampered_attr_cols.append(attr_idx)

        group_results[gid] = {
            "original_group_size": group_wm_ref[ref_key].get("original_size", gamma),
            "suspicious_group_size": orig_size,
            "wj_reference": wj_ref_trimmed,
            "wj_suspicious": wj_susp_trimmed,
            "determinant_ref": wj_ref_trimmed[0] if wj_ref_trimmed else None,
            "determinant_susp": wj_susp_trimmed[0] if wj_susp_trimmed else None,
            "authentic": is_authentic,
            "status": "authentic" if is_authentic else "tampered",
            "tampered_attribute_indices": tampered_attr_cols,
            "tampered_attribute_names": [available_cols[i] for i in tampered_attr_cols
                                          if i < len(available_cols)],
        }

        if not is_authentic:
            tampered_groups.append(gid)
            tampered_attributes[gid] = tampered_attr_cols

    db_tampered = len(tampered_groups) > 0 or size_mismatch

    return {
        "method": "B4_Camara2014",
        "n_tuples_suspicious": alpha_susp,
        "n_tuples_original": ca_record["alpha"],
        "num_groups_original": num_groups_orig,
        "num_groups_suspicious": num_groups_susp,
        "size_mismatch": size_mismatch,
        "db_tampered": db_tampered,
        "tampered_groups": tampered_groups,
        "tampered_attribute_columns": {str(k): [numeric_cols[i] for i in v if i < len(numeric_cols)]
                                        for k, v in tampered_attributes.items()},
        "n_tampered_groups": len(tampered_groups),
        "group_results": group_results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B4 Camara 2014 — Detect watermark")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--ca_record",  required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--true_tampered_groups", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)
    ca_record = load_json(args.ca_record)
    # Inject secret key for detection (not stored in CA for security; pass via config)
    ca_record["_secret_key_internal"] = cfg["secret_key"]

    result = detect_watermark(df, ca_record)

    if args.true_tampered_groups:
        true_groups = set(int(x) for x in args.true_tampered_groups.split(",") if x.strip())
        all_groups = set(range(ca_record["num_groups"]))
        flagged = set(result["tampered_groups"])
        metrics = compute_localization_metrics(true_groups, flagged, all_groups)
        result["localization_metrics"] = metrics

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "detect_result.json")
    save_json(result, out_path)

    print(f"[B4 Detect] DB tampered: {result['db_tampered']}")
    print(f"[B4 Detect] Tampered groups: {result['tampered_groups']}")
    print(f"[B4 Detect] Size mismatch: {result['size_mismatch']}")
    print(f"[B4 Detect] Result saved → {out_path}")


if __name__ == "__main__":
    main()
