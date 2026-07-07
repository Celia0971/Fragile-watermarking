"""
B1: Li et al. 2004 — Fragile Watermarking for Categorical Data (Embed)

Algorithm (zero-distortion, tuple-reordering):
  1. Partition tuples into g groups: group_id = HASH(K, ri.P) mod g
  2. Within each group, sort tuples by tuple hash h_i = HASH(K, ri.P) ascending
  3. Compute group hash H = HASH(K, h1, h2, ..., h_qk) where h_i are sorted tuple hashes
  4. Extract W = first (qk // 2) bits of H  [watermark for this group]
  5. Reorder pairs within the group according to W:
       for each consecutive pair (i, i+1):
         W[i//2] == 1 → ensure h_i > h_{i+1}  (swap if needed)
         W[i//2] == 0 → ensure h_i <= h_{i+1} (swap if needed)

The watermark is embedded by the physical ordering of tuple pairs.
The original data VALUES are NOT changed (zero-distortion).

Output:
  - watermarked_df  : DataFrame with tuples reordered to encode the watermark
  - watermark_info  : dict with per-group info (group_id, watermark bits, ordering)

Usage:
    python embed.py --input db.csv --output wm_db.csv --config config/params.yaml
"""

import argparse
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from common.utils import (
    load_csv, save_csv, save_json, load_config, hash_to_int, hash_value
)


def partition_into_groups(
    df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    num_groups: int,
) -> Dict[int, List[int]]:
    """Partition DataFrame rows into groups by HASH(K, pk) mod num_groups.

    Returns:
        groups: dict mapping group_id → list of row positional indices (0-based)
    """
    groups: Dict[int, List[int]] = {g: [] for g in range(num_groups)}
    for pos in range(len(df)):
        pk_val = df.iloc[pos][pk_col]
        group_id = hash_to_int(secret_key, pk_val, mod=num_groups)
        groups[group_id].append(pos)
    return groups


def compute_tuple_hash(row: pd.Series, pk_col: str, secret_key: str) -> int:
    """Compute h_i = HASH(K, ri.D, ri.P) for a tuple.

    Paper Li et al. 2004: hash includes ALL attributes (ri.D = non-key data,
    ri.P = primary key).  This makes value modifications detectable: changing
    any attribute changes h_i, alters the group hash, and breaks the watermark.
    """
    pk_val = row[pk_col]
    # Include all attribute values in canonical string order (pk first, then others)
    other_vals = [str(row[c]) for c in row.index if c != pk_col]
    return hash_value(secret_key, pk_val, *other_vals)


def embed_watermark(
    df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    num_groups: int,
) -> Tuple[pd.DataFrame, Dict]:
    """Embed watermark by reordering tuple pairs within each group.

    Returns:
        watermarked_df : DataFrame with rows reordered to encode watermark
        info           : detailed per-group information
    """
    # Step 1: Compute tuple hashes for all rows
    tuple_hashes = [
        compute_tuple_hash(df.iloc[i], pk_col, secret_key)
        for i in range(len(df))
    ]

    # Step 2: Partition into groups
    groups = partition_into_groups(df, pk_col, secret_key, num_groups)

    # We'll build the watermarked DataFrame by collecting rows in their new order
    # Rows not in any group or in groups with < 2 tuples pass through unchanged.
    # We maintain a mapping: original position → new position in output.
    # Output = concatenation of [reordered group blocks] + [ungrouped rows].

    # For each group: sort by tuple_hash ascending, apply watermark reordering.
    group_info = {}
    reordered_positions = []  # output row order (positional indices into df)

    # Collect positions that appear in groups with >= 2 tuples
    in_group_positions = set()

    for gid in sorted(groups.keys()):
        positions = groups[gid]
        if len(positions) == 0:
            group_info[gid] = {"size": 0, "watermark_bits": [], "status": "empty"}
            continue

        # Sort positions by their tuple hash (ascending)
        positions_sorted = sorted(positions, key=lambda p: tuple_hashes[p])
        hashes_sorted = [tuple_hashes[p] for p in positions_sorted]
        qk = len(positions_sorted)

        if qk < 2:
            # Single tuple: no pairs, watermark trivially 0 bits, pass through
            group_info[gid] = {
                "size": qk,
                "sorted_positions": positions_sorted,
                "tuple_hashes": hashes_sorted,
                "watermark_bits": [],
                "status": "single_tuple",
            }
            in_group_positions.update(positions_sorted)
            reordered_positions.extend(positions_sorted)
            continue

        # Step 3: Compute group watermark
        # H = HASH(K, h1, h2, ..., h_qk)  where h_i are sorted tuple hashes
        h_group = hash_value(secret_key, *hashes_sorted)
        n_wm_bits = qk // 2  # number of bits = number of consecutive pairs

        # Extract watermark bits (LSB-first from h_group)
        wm_bits = [(h_group >> i) & 1 for i in range(n_wm_bits)]

        # Step 4 & 5: Reorder pairs based on watermark bits
        # Work on a mutable copy of sorted positions
        ordered = list(positions_sorted)
        for pair_idx in range(n_wm_bits):
            i = 2 * pair_idx
            j = i + 1
            h_i = tuple_hashes[ordered[i]]
            h_j = tuple_hashes[ordered[j]]
            bit = wm_bits[pair_idx]
            if bit == 1:
                # Want h_i > h_j  → swap if h_i <= h_j
                if h_i <= h_j:
                    ordered[i], ordered[j] = ordered[j], ordered[i]
            else:
                # Want h_i <= h_j → swap if h_i > h_j
                if h_i > h_j:
                    ordered[i], ordered[j] = ordered[j], ordered[i]

        group_info[gid] = {
            "size": qk,
            "sorted_positions": positions_sorted,
            "tuple_hashes": hashes_sorted,
            "group_hash": h_group,
            "n_watermark_bits": n_wm_bits,
            "watermark_bits": wm_bits,
            "reordered_positions": ordered,
            "status": "watermarked",
        }
        in_group_positions.update(positions_sorted)
        reordered_positions.extend(ordered)

    # Append any rows that were not placed in any group (shouldn't happen normally)
    for pos in range(len(df)):
        if pos not in in_group_positions:
            reordered_positions.append(pos)

    # Build output DataFrame in new order
    watermarked_df = df.iloc[reordered_positions].reset_index(drop=True)

    # ── Storage accounting (relational soundness) ─────────────────────────────
    # B1 encodes its watermark in the released tuple order. In the relational
    # model a relation is a set: row order carries no information and a benign
    # re-ordering must not be flagged as tampering. To remain verifiable, B1 must
    # therefore store an external order index  pk -> rank_W(pk)  that reconstructs
    # the watermarked release order (the index table of Li et al. 2004), costing
    # n*ceil(log2 n) bits. This dominates the few per-group reference bits.
    n = len(df)
    rank_bits = max(1, math.ceil(math.log2(max(2, n))))
    order_index_bytes = math.ceil(n * rank_bits / 8)
    # Order index pk -> rank_W: the released watermarked order, stored so that
    # verification can reconstruct it independently of the suspect's physical
    # row order (relational soundness; Li et al. 2004 index table).
    order_index_pks = [str(df.iloc[p][pk_col]) for p in reordered_positions]
    info = {
        "method": "B1_Li2004",
        "pk_col": pk_col,
        "num_groups": num_groups,
        "n_tuples": n,
        "groups": group_info,
        "output_order": reordered_positions,
        "order_index_pks": order_index_pks,   # pk in watermarked rank order
        "storage_bytes": {
            "order_index": order_index_bytes,   # n*ceil(log2 n)/8
            "total": order_index_bytes,
        },
    }

    return watermarked_df, info


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B1 Li 2004 — Embed watermark")
    parser.add_argument("--input",   required=True,  help="Input CSV")
    parser.add_argument("--output",  required=True,  help="Watermarked CSV output")
    parser.add_argument("--config",  required=True,  help="Path to params.yaml")
    parser.add_argument("--info_out",default=None,   help="JSON file for intermediate info")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)

    wm_df, info = embed_watermark(
        df,
        pk_col=cfg["pk_col"],
        secret_key=cfg["secret_key"],
        num_groups=cfg["num_groups"],
    )

    save_csv(wm_df, args.output)
    print(f"[B1 Embed] Watermarked {len(df)} tuples → {args.output}")

    info_path = args.info_out or args.output.replace(".csv", "_embed_info.json")
    save_json(info, info_path)
    print(f"[B1 Embed] Info saved → {info_path}")


if __name__ == "__main__":
    main()
