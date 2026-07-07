"""
B1: Li et al. 2004 — Fragile Watermarking for Categorical Data (Detect)

Detection algorithm:
  1. Partition suspicious DB into groups using same key K and g
  2. Within each group, sort by tuple hash ascending
  3. Recompute group watermark W from group hash H
  4. Extract W' from actual physical ordering of pairs in the group:
       W'[pair_idx] = 0 if h_i <= h_j  (for positions 2*pair_idx, 2*pair_idx+1)
                       1 otherwise
  5. If W == W' for a group → group is authentic; else → tampered

Outputs:
  - db_level: bool (True = database is authentic)
  - group_results: per-group {authentic: bool, wm_expected: bits, wm_observed: bits}
  - tampered_groups: list of tampered group IDs

Note on insertions/deletions:
  - Inserted tuples hash to some group; they change the group's pair structure
    → group watermark comparison fails → detected
  - Deleted tuples reduce group size → group comparison may fail → detected
  - If the number of groups changes (different tuples → different group membership),
    extra detection via group-size mismatch is reported

Usage:
    python detect.py --input suspicious.csv --embed_info embed_info.json \
                     --config config/params.yaml --output_dir results/
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from common.utils import (
    load_csv, load_json, save_json, load_config, hash_value, hash_to_int
)
from common.metrics import compute_localization_metrics, summarize_trials


def detect_watermark(
    df_suspicious: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    num_groups: int,
    embed_info: Optional[Dict] = None,
) -> Dict:
    """Detect watermark in suspicious database.

    Args:
        df_suspicious : suspicious DataFrame (may be attacked)
        pk_col        : primary key column
        secret_key    : same key used during embedding
        num_groups    : same number of groups used during embedding
        embed_info    : optional dict from embed phase (for per-group comparison)

    Returns:
        result dict with:
          - db_tampered       : bool (True = tampered detected)
          - tampered_groups   : list of group IDs flagged as tampered
          - group_results     : per-group detail
          - n_tuples_suspicious : number of tuples in suspicious DB
    """
    # Step 0: Reconstruct the watermarked release order from the stored order
    # index (relational soundness). B1's watermark is encoded in tuple ordering;
    # in the relational model row order is non-semantic, so verification must not
    # trust the suspect's physical order. We re-rank surviving tuples by their
    # stored rank_W (inserted tuples, absent from the index, go last). Under an
    # order-preserving attack this is a no-op; under a benign re-ordering it
    # restores the watermarked order so the watermark is read correctly.
    if embed_info and embed_info.get("order_index_pks"):
        rank = {pk: i for i, pk in enumerate(embed_info["order_index_pks"])}
        n_idx = len(rank)
        ord_key = df_suspicious[pk_col].astype(str).map(lambda p: rank.get(p, n_idx))
        df_suspicious = (df_suspicious.assign(_wm_ord=ord_key)
                         .sort_values("_wm_ord", kind="stable")
                         .drop(columns="_wm_ord").reset_index(drop=True))

    n = len(df_suspicious)

    # Step 1: Compute tuple hashes for all rows in suspicious DB
    # h_i = H(K, ri.D, ri.P): hash includes ALL attributes so that value
    # modifications change h_i and are therefore detectable (Li et al. 2004).
    def _tuple_hash(row):
        pk_val = row[pk_col]
        other_vals = [str(row[c]) for c in row.index if c != pk_col]
        return hash_value(secret_key, pk_val, *other_vals)

    tuple_hashes = {i: _tuple_hash(df_suspicious.iloc[i]) for i in range(n)}

    # Step 2: Partition suspicious DB into groups
    groups_suspicious: Dict[int, List[int]] = {g: [] for g in range(num_groups)}
    for pos in range(n):
        pk_val = df_suspicious.iloc[pos][pk_col]
        gid = hash_to_int(secret_key, pk_val, mod=num_groups)
        groups_suspicious[gid].append(pos)

    group_results = {}
    tampered_groups = []
    authentic_groups = []

    for gid in range(num_groups):
        positions = groups_suspicious[gid]
        qk = len(positions)

        # Get expected watermark bits by recomputing from the group
        # (same computation as embed: sort by hash, compute group hash, extract bits)
        if qk < 2:
            # Cannot embed/detect watermark with < 2 tuples
            # Compare with embed_info if available
            expected_size = (
                embed_info["groups"].get(str(gid), {}).get("size", 0)
                if embed_info else qk
            )
            status = "authentic" if qk == expected_size else "tampered_size_mismatch"
            group_results[gid] = {
                "size": qk,
                "expected_size": expected_size,
                "status": status,
                "authentic": status == "authentic",
            }
            if status != "authentic":
                tampered_groups.append(gid)
            else:
                authentic_groups.append(gid)
            continue

        # Sort suspicious group positions by tuple hash ascending
        positions_sorted = sorted(positions, key=lambda p: tuple_hashes[p])
        hashes_sorted = [tuple_hashes[p] for p in positions_sorted]

        # Recompute expected watermark W from group hash
        h_group = hash_value(secret_key, *hashes_sorted)
        n_wm_bits = qk // 2
        wm_expected = [(h_group >> i) & 1 for i in range(n_wm_bits)]

        # Extract observed watermark W' from actual physical ordering
        # The suspicious DB rows appear in some physical order in df_suspicious.
        # We look at the actual positions (row indices in df_suspicious) for this group
        # and compare consecutive pairs by their hash values.
        # W'[pair_idx]: 0 if hash[physical_pos_2i] <= hash[physical_pos_2i+1], else 1
        wm_observed = []
        for pair_idx in range(n_wm_bits):
            i_phys = positions[2 * pair_idx]      # actual row position in df_suspicious
            j_phys = positions[2 * pair_idx + 1]
            h_i = tuple_hashes[i_phys]
            h_j = tuple_hashes[j_phys]
            wm_observed.append(0 if h_i <= h_j else 1)

        is_authentic = (wm_expected == wm_observed)

        group_results[gid] = {
            "size": qk,
            "hashes_sorted": hashes_sorted,
            "n_watermark_bits": n_wm_bits,
            "wm_expected": wm_expected,
            "wm_observed": wm_observed,
            "authentic": is_authentic,
            "status": "authentic" if is_authentic else "tampered",
        }

        if is_authentic:
            authentic_groups.append(gid)
        else:
            tampered_groups.append(gid)

    db_tampered = len(tampered_groups) > 0

    result = {
        "method": "B1_Li2004",
        "n_tuples_suspicious": n,
        "num_groups": num_groups,
        "db_tampered": db_tampered,
        "tampered_groups": tampered_groups,
        "authentic_groups": authentic_groups,
        "n_tampered_groups": len(tampered_groups),
        "n_authentic_groups": len(authentic_groups),
        "group_results": group_results,
    }

    return result


def evaluate_detection(
    result: Dict,
    true_tampered_groups: Optional[Set[int]] = None,
    all_group_ids: Optional[Set[int]] = None,
) -> Dict:
    """Compute localization metrics given ground truth.

    Args:
        result              : output of detect_watermark()
        true_tampered_groups: set of group IDs that were actually tampered
        all_group_ids       : set of all group IDs

    Returns:
        metrics dict
    """
    if true_tampered_groups is None:
        return result

    flagged = set(result["tampered_groups"])
    tampered = set(true_tampered_groups)
    all_ids = all_group_ids or set(range(result["num_groups"]))

    metrics = compute_localization_metrics(tampered, flagged, all_ids)
    result["localization_metrics"] = metrics
    result["db_level_correct"] = (
        (len(tampered) > 0 and result["db_tampered"]) or
        (len(tampered) == 0 and not result["db_tampered"])
    )
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B1 Li 2004 — Detect watermark")
    parser.add_argument("--input",       required=True,  help="Suspicious CSV")
    parser.add_argument("--embed_info",  default=None,   help="JSON from embed phase")
    parser.add_argument("--config",      required=True,  help="Path to params.yaml")
    parser.add_argument("--output_dir",  default=".",    help="Directory for result JSON")
    parser.add_argument("--true_tampered_groups", default=None,
                        help="Comma-separated list of truly tampered group IDs (for eval)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)
    embed_info = load_json(args.embed_info) if args.embed_info else None

    result = detect_watermark(
        df,
        pk_col=cfg["pk_col"],
        secret_key=cfg["secret_key"],
        num_groups=cfg["num_groups"],
        embed_info=embed_info,
    )

    # Optional evaluation
    if args.true_tampered_groups:
        true_groups = set(int(x) for x in args.true_tampered_groups.split(",") if x.strip())
        all_groups = set(range(cfg["num_groups"]))
        result = evaluate_detection(result, true_groups, all_groups)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "detect_result.json")
    save_json(result, out_path)

    print(f"[B1 Detect] DB tampered: {result['db_tampered']}")
    print(f"[B1 Detect] Tampered groups: {result['tampered_groups']}")
    print(f"[B1 Detect] Result saved → {out_path}")


if __name__ == "__main__":
    main()
