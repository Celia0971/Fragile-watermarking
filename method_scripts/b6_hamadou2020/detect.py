"""
B6: Hamadou et al. 2020 — Reversible Prediction-Error Expansion on LSD (Detect)

Detection algorithm:
  For each group:
    1. Partition suspicious DB using same key and CA-stored num_groups
    2. For each attribute Aj:
       a. For each EMBEDDED position (embedded_mask[col][i] = True):
          - Decode val_wm → (orig_val, extracted_bit)
          - Compare extracted_bit against stored wj_bits[col][i] from CA
          - Mismatch → this position was attacked
       b. If any mismatch in this group → group is tampered
    3. Also recover original values from non-embedded positions (value unchanged)
       and from embedded positions (decode gives original).
    4. Recompute Wj from the recovered group data, following Algorithm 4.

Result policy:
  Recomputed Wj is computed to mirror Algorithm 4, but it is not persisted in
  detect_result.json because benchmark tamper decisions still compare extracted
  bits against the CA-stored wj_bits. The group length follows the paper: each
  hash group contributes as many watermark bits as it has tuples; μ is only
  retained as the target size used to derive v=ceil(alpha/μ).
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Tuple

import pandas as pd

from common.utils import (
    load_csv, load_json, save_json, load_config, hash_value, get_numeric_cols
)
from common.metrics import compute_localization_metrics
from b6_hamadou2020.embed import (
    partition_into_groups, get_sorted_pk_hashes, decode_lsd,
    compute_group_watermark_bits
)


def detect_watermark(
    df_suspicious: pd.DataFrame,
    ca_record: Dict,
) -> Dict:
    """Detect and localise tampering using B6 Hamadou 2020 method.

    Compares extracted watermark bits against CA-stored reference bits.
    Secret key injected as '_secret_key_internal'.
    """
    pk_col       = ca_record["pk_col"]
    numeric_cols = ca_record["numeric_cols"]
    secret_key   = ca_record.get("_secret_key_internal", "")
    num_groups   = ca_record["num_groups"]
    alpha_orig   = ca_record["alpha"]
    group_info_ca = ca_record.get("group_info", {})

    alpha_susp = len(df_suspicious)
    # size_mismatch: True iff the suspicious DB has a different row count than original.
    # NOTE: Do NOT re-derive num_groups as ceil(alpha_susp/mu) — num_groups is
    # fixed by the CA record generated at embedding time.
    # Structural change is correctly captured by row-count difference.
    size_mismatch = (alpha_susp != alpha_orig)

    available_cols = [c for c in numeric_cols if c in df_suspicious.columns]

    groups_susp = partition_into_groups(df_suspicious, pk_col, secret_key, num_groups)

    tampered_groups = []
    tampered_attributes: Dict[int, List[str]] = {}
    recovered_df = df_suspicious.copy()

    for gid in range(num_groups):
        positions = groups_susp.get(gid, [])

        # Retrieve CA reference for this group
        orig_info = group_info_ca.get(str(gid), {})

        if len(positions) == 0:
            was_also_empty = orig_info.get("skipped", False) or \
                             orig_info.get("original_group_size", 1) == 0
            if was_also_empty:
                # Expected empty group; not tampered
                continue
            else:
                # Group had members during embed but is now empty → tampered
                tampered_groups.append(gid)
                continue
        if not orig_info:
            tampered_groups.append(gid)
            continue

        ref_wj_bits    = orig_info.get("wj_bits", {})
        ref_embed_mask = orig_info.get("embedded_mask", {})
        expected_group_size = orig_info.get(
            "group_size",
            orig_info.get("original_group_size", len(orig_info.get("positions_sorted", [])))
        )

        positions_sorted, pk_hashes_sorted = get_sorted_pk_hashes(
            df_suspicious, positions, pk_col, secret_key
        )
        group_df_wm = df_suspicious.iloc[positions_sorted].reset_index(drop=True)
        group_size = len(positions_sorted)
        group_size_mismatch = (group_size != expected_group_size)

        # Per-attribute: decode embedded positions and compare against CA reference
        attr_tampered: List[str] = []
        recovered_values: Dict[str, List[float]] = {col: [] for col in available_cols}

        for col in available_cols:
            col_refbits = ref_wj_bits.get(col, [])
            col_mask    = ref_embed_mask.get(col, [True] * len(col_refbits))

            has_mismatch = False
            col_recovered = []

            for i in range(group_size):
                val_wm  = float(group_df_wm.iloc[i][col])
                hi      = pk_hashes_sorted[i]
                hi_pred = hi % 10
                was_embedded = col_mask[i] if i < len(col_mask) else True

                if not was_embedded:
                    # Value never modified in this position; keep as-is
                    col_recovered.append(val_wm)
                    continue

                orig_val, extracted_bit = decode_lsd(val_wm, hi_pred)

                if extracted_bit is None:
                    # Decode failed: value changed → tampered
                    has_mismatch = True
                    col_recovered.append(val_wm)
                else:
                    ref_bit = col_refbits[i] if i < len(col_refbits) else None
                    if ref_bit is None or extracted_bit != ref_bit:
                        has_mismatch = True
                    col_recovered.append(orig_val)

            recovered_values[col] = col_recovered

            missing_reference_bits = max(0, len(col_refbits) - group_size)
            is_col_tampered = (
                has_mismatch
                or missing_reference_bits > 0
                or group_size_mismatch
            )
            if is_col_tampered:
                attr_tampered.append(col)

        # Algorithm 4 recomputes Wj from recovered group data. The result is
        # intentionally not saved or used for benchmark metrics; final decisions
        # above use extracted bits vs CA-stored wj_bits.
        recovered_group_df = group_df_wm.copy()
        for col in available_cols:
            for idx, val in enumerate(recovered_values.get(col, [])):
                if idx < len(recovered_group_df):
                    recovered_group_df.at[recovered_group_df.index[idx], col] = val

        compute_group_watermark_bits(
            recovered_group_df, pk_hashes_sorted, secret_key, available_cols
        )

        is_authentic = len(attr_tampered) == 0
        if not is_authentic:
            tampered_groups.append(gid)
            tampered_attributes[gid] = attr_tampered

        # Write recovered values back for the currently observed group members.
        for idx, pos in enumerate(positions_sorted):
            for col in available_cols:
                if idx < len(recovered_values[col]):
                    recovered_df.at[recovered_df.index[pos], col] = \
                        recovered_values[col][idx]

    db_tampered = len(tampered_groups) > 0 or size_mismatch

    return {
        "method": "B6_Hamadou2020",
        "n_tuples_suspicious": alpha_susp,
        "n_tuples_original": alpha_orig,
        "num_groups_original": num_groups,
        "num_groups_suspicious": None,  # not applicable (hash-based partitioning)
        "size_mismatch": size_mismatch,
        "db_tampered": db_tampered,
        "tampered_groups": tampered_groups,
        "n_tampered_groups": len(tampered_groups),
        "tampered_attributes_per_group": {str(k): v for k, v in tampered_attributes.items()},
        "_recovered_df": recovered_df,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B6 Hamadou 2020 — Detect watermark")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--ca_record",  required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--true_tampered_groups", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)
    ca_record = load_json(args.ca_record)
    ca_record["_secret_key_internal"] = cfg["secret_key"]

    result = detect_watermark(df, ca_record)

    os.makedirs(args.output_dir, exist_ok=True)
    recovered_df = result.pop("_recovered_df")
    recovered_path = os.path.join(args.output_dir, "recovered.csv")
    recovered_df.to_csv(recovered_path, index=False)

    if args.true_tampered_groups:
        true_groups = set(int(x) for x in args.true_tampered_groups.split(",") if x.strip())
        all_groups  = set(range(ca_record["num_groups"]))
        flagged     = set(result["tampered_groups"])
        metrics = compute_localization_metrics(true_groups, flagged, all_groups)
        result["localization_metrics"] = metrics

    out_path = os.path.join(args.output_dir, "detect_result.json")
    save_json(result, out_path)

    print(f"[B6 Detect] DB tampered: {result['db_tampered']}")
    print(f"[B6 Detect] Tampered groups: {result['tampered_groups']}")
    print(f"[B6 Detect] Size mismatch: {result['size_mismatch']}")
    print(f"[B6 Detect] Recovered DB → {recovered_path}")
    print(f"[B6 Detect] Result saved → {out_path}")


if __name__ == "__main__":
    main()
