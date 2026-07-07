"""
B2: Guo et al. 2006 — Fragile Watermarking for Numeric Relational Data (Detect)

Detection:
For each group (same partitioning as embed):
  1. Sort tuples by PK ascending
  2. Recompute H0 from PKs (ignoring 2 LSBs of attribute values)
  3. For each attribute col A_j:
       H1^j = HASH(H0, col, v1_cleared, ...) [ignore 2 LSBs]
       W1^j (expected) = first v bits of H1^j
       W1^j_extracted[i] = LSB(ri.Aj)   [actual LSBs in suspicious DB]
       V1[col] = (W1_expected == W1_extracted) per tuple
  4. For each tuple ri:
       H2^i = HASH(K, pk_i, attr_vals_cleared...)
       W2^i (expected) = first γ bits of H2^i
       W2^i_extracted[j] = next_LSB(ri.Aj)
       V2[i] = (W2_expected == W2_extracted) per attribute

  A tuple is flagged tampered if V1 or V2 inconsistency is detected for it.
  A group is flagged tampered if any tuple in it is flagged.

Outputs:
  - tampered_tuples (list of PKs)
  - tampered_groups (list of group IDs)
  - db_tampered (bool)
  - per-group and per-tuple detail
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from common.utils import (
    load_csv, load_json, save_json, load_config, hash_value, get_numeric_cols
)
from common.metrics import compute_localization_metrics
from b2_guo2006.embed import clear_lsbs, get_lsb


def detect_watermark(
    df_suspicious: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    num_groups: int,
    numeric_cols: Optional[List[str]] = None,
    embed_info: Optional[Dict] = None,
) -> Dict:
    """Detect B2 watermark tampering in suspicious database."""
    if numeric_cols is None:
        numeric_cols = get_numeric_cols(df_suspicious, exclude_cols=[pk_col])

    # Partition
    groups: Dict[int, List[int]] = {g: [] for g in range(num_groups)}
    for pos in range(len(df_suspicious)):
        pk_val = df_suspicious.iloc[pos][pk_col]
        gid = hash_value(secret_key, pk_val) % num_groups
        groups[gid].append(pos)

    group_results = {}
    tampered_groups = []
    tampered_pk_set = set()

    for gid in range(num_groups):
        positions = groups[gid]
        if len(positions) == 0:
            # Check against expected non-empty group
            group_results[gid] = {"size": 0, "status": "empty"}
            continue

        # Sort by PK ascending
        grp = df_suspicious.iloc[positions].copy().reset_index(drop=True)
        try:
            grp = grp.sort_values(by=pk_col).reset_index(drop=True)
        except TypeError:
            grp = grp.sort_values(by=pk_col, key=lambda x: x.astype(str)).reset_index(drop=True)

        v = len(grp)
        gamma = len(numeric_cols)
        pk_values = [str(grp.iloc[i][pk_col]) for i in range(v)]

        # H0
        H0 = hash_value(secret_key, *pk_values)

        # --- Attribute watermark (W1) verification ---
        v1_results = {}  # col → list of bool (per tuple: LSB matches?)
        W1_expected = {}
        W1_extracted = {}
        for col in numeric_cols:
            if col not in grp.columns:
                continue
            cleared_vals = [int(clear_lsbs(grp.iloc[i][col], 2)) for i in range(v)]
            H1 = hash_value(str(H0), col, *cleared_vals)
            expected_bits = [(H1 >> i) & 1 for i in range(v)]
            extracted_bits = [get_lsb(grp.iloc[i][col], position=0) for i in range(v)]
            matches = [e == x for e, x in zip(expected_bits, extracted_bits)]
            v1_results[col] = matches
            W1_expected[col] = expected_bits
            W1_extracted[col] = extracted_bits

        # --- Tuple watermark (W2) verification ---
        # We use the cleared versions for H2 computation (as in embed)
        v2_results = {}  # i → list of bool (per attribute: next_LSB matches?)
        W2_expected = {}
        W2_extracted = {}
        for i in range(v):
            cleared_vals = [int(clear_lsbs(grp.at[i, col], 2)) for col in numeric_cols if col in grp.columns]
            H2 = hash_value(secret_key, pk_values[i], *cleared_vals)
            cols_present = [col for col in numeric_cols if col in grp.columns]
            expected_bits = [(H2 >> b) & 1 for b in range(len(cols_present))]
            extracted_bits = [get_lsb(grp.at[i, col], position=1) for col in cols_present]
            matches = [e == x for e, x in zip(expected_bits, extracted_bits)]
            v2_results[i] = matches
            W2_expected[i] = expected_bits
            W2_extracted[i] = extracted_bits

        # Determine tampered tuples: flagged if ANY V1 or V2 bit mismatch
        tampered_tuples_in_group = []
        per_tuple_status = {}
        for i in range(v):
            pk_i = grp.iloc[i][pk_col]
            v1_ok = all(v1_results[col][i] for col in v1_results)
            v2_ok = all(v2_results[i])
            is_authentic = v1_ok and v2_ok
            per_tuple_status[str(pk_i)] = {
                "v1_ok": v1_ok,
                "v2_ok": v2_ok,
                "authentic": is_authentic,
            }
            if not is_authentic:
                tampered_tuples_in_group.append(pk_i)
                tampered_pk_set.add(pk_i)

        group_tampered = len(tampered_tuples_in_group) > 0

        group_results[gid] = {
            "size": v,
            "pk_values": pk_values,
            "v1_expected": W1_expected,
            "v1_extracted": W1_extracted,
            "W2_expected": {str(k): v for k, v in W2_expected.items()},
            "W2_extracted": {str(k): v for k, v in W2_extracted.items()},
            "tampered_tuples_pks": [str(x) for x in tampered_tuples_in_group],
            "per_tuple_status": per_tuple_status,
            "authentic": not group_tampered,
            "status": "authentic" if not group_tampered else "tampered",
        }

        if group_tampered:
            tampered_groups.append(gid)

    db_tampered = len(tampered_groups) > 0

    return {
        "method": "B2_Guo2006",
        "n_tuples_suspicious": len(df_suspicious),
        "num_groups": num_groups,
        "db_tampered": db_tampered,
        "tampered_groups": tampered_groups,
        "tampered_tuple_pks": [str(x) for x in tampered_pk_set],
        "n_tampered_groups": len(tampered_groups),
        "group_results": group_results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B2 Guo 2006 — Detect watermark")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--embed_info", default=None)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--true_tampered_pks", default=None,
                        help="Comma-separated PKs of truly tampered tuples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)
    embed_info = load_json(args.embed_info) if args.embed_info else None
    numeric_cols = cfg.get("numeric_cols") or None

    result = detect_watermark(
        df,
        pk_col=cfg["pk_col"],
        secret_key=cfg["secret_key"],
        num_groups=cfg["num_groups"],
        numeric_cols=numeric_cols,
        embed_info=embed_info,
    )

    # Evaluate if ground truth provided
    if args.true_tampered_pks:
        true_pks = set(args.true_tampered_pks.split(","))
        all_pks = set(str(x) for x in df[cfg["pk_col"]].tolist())
        flagged = set(result["tampered_tuple_pks"])
        metrics = compute_localization_metrics(true_pks, flagged, all_pks)
        result["localization_metrics"] = metrics

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "detect_result.json")
    save_json(result, out_path)

    print(f"[B2 Detect] DB tampered: {result['db_tampered']}")
    print(f"[B2 Detect] Tampered groups: {result['tampered_groups']}")
    print(f"[B2 Detect] Tampered tuples: {len(result['tampered_tuple_pks'])}")
    print(f"[B2 Detect] Result saved → {out_path}")


if __name__ == "__main__":
    main()
