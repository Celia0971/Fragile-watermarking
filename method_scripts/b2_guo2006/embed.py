"""
B2: Guo et al. 2006 — Fragile Watermarking for Numeric Relational Data (Embed)

Algorithm (modifies 2 LSBs of numeric attribute values):

For each group G_j (v tuples, γ attributes):
  1. Sort tuples in group by PK ascending
  2. Compute H0 = MAC(K ∘ r1.P ∘ ... ∘ rv.P)
     [ignore 2 LSBs of each attribute value when computing H0]
  3. Attribute watermark W1^j: for each attribute column A_j (j=1..γ):
       H1^j = HASH(H0 ∘ r1.Aj ∘ ... ∘ rv.Aj)  [ignore 2 LSBs of values]
       W1^j = first v bits of H1^j  (one bit per tuple)
       LSB(ri.Aj) ← W1^j(i)  for i=1..v
  4. Tuple watermark W2^i: for each tuple ri (i=1..v):
       H2^i = HASH(K ∘ ri.A1 ∘ ... ∘ ri.Aγ)  [ignore 2 LSBs of already-LSB-modified values]
       W2^i = first γ bits of H2^i  (one bit per attribute)
       next_LSB(ri.Aj) ← W2^i(j)  for j=1..γ

"Ignore 2 LSBs" means: when computing hashes, clear the 2 LSBs of integer values,
or treat float values' integer parts with 2 LSBs cleared.

The watermarked database has its 2 LSBs of each numeric value overwritten.

Usage:
    python embed.py --input db.csv --output wm_db.csv --config config/params.yaml
"""

import argparse
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.utils import (
    load_csv, save_csv, save_json, load_config, hash_value, hash_to_int,
    get_numeric_cols
)


def clear_lsbs(val, n: int = 2):
    """Clear the n least significant bits of an integer value."""
    if isinstance(val, (int, np.integer)):
        return int(val) & ~((1 << n) - 1)
    elif isinstance(val, (float, np.floating)):
        int_part = int(val)
        cleared = int_part & ~((1 << n) - 1)
        return float(cleared) + (val - int_part)
    return val


def set_lsb(val, bit: int, position: int = 0):
    """Set bit at `position` (0=LSB, 1=next LSB) of val to `bit` (0 or 1)."""
    if isinstance(val, (int, np.integer)):
        v = int(val)
        v = (v & ~(1 << position)) | (bit << position)
        return type(val)(v)
    elif isinstance(val, (float, np.floating)):
        int_part = int(val)
        frac = val - int(val)
        int_part = (int_part & ~(1 << position)) | (bit << position)
        return float(int_part) + frac
    return val


def get_lsb(val, position: int = 0) -> int:
    """Extract bit at `position` (0=LSB, 1=next LSB) of val."""
    if isinstance(val, (float, np.floating)):
        val = int(val)
    return (int(val) >> position) & 1


def embed_group(
    group_df: pd.DataFrame,
    pk_col: str,
    numeric_cols: List[str],
    secret_key: str,
) -> Tuple[pd.DataFrame, Dict]:
    """Embed watermark into a single group.

    Returns:
        watermarked group DataFrame, group-level info dict.
    """
    group_df = group_df.copy().reset_index(drop=True)
    v = len(group_df)      # number of tuples in group
    gamma = len(numeric_cols)  # number of numeric attributes

    # Step 1: Sort by PK ascending (as integer if possible)
    try:
        group_df = group_df.sort_values(by=pk_col).reset_index(drop=True)
    except TypeError:
        group_df = group_df.sort_values(by=pk_col, key=lambda x: x.astype(str)).reset_index(drop=True)

    # Step 2: Compute H0 = HASH(K, pk1_cleared, pk2_cleared, ..., pkv_cleared)
    # The PKs may be non-numeric; just use them as strings.
    # "ignore 2 LSBs" applies to attribute values, not PKs.
    pk_values = [str(group_df.iloc[i][pk_col]) for i in range(v)]
    H0 = hash_value(secret_key, *pk_values)

    # Step 3: Attribute watermarks W1^j (one per numeric attribute column)
    # Use int(clear_lsbs(...)) for hashing to avoid float precision drift after CSV round-trip.
    # The fractional part of a numeric value does not participate in bit manipulation.
    W1 = {}  # col → list of v bits
    for col in numeric_cols:
        cleared_vals = [int(clear_lsbs(group_df.iloc[i][col], 2)) for i in range(v)]
        H1 = hash_value(str(H0), col, *cleared_vals)
        bits = [(H1 >> i) & 1 for i in range(v)]
        W1[col] = bits

    # Embed W1: set LSB(ri.Aj) ← W1^j[i]
    for col in numeric_cols:
        for i in range(v):
            old_val = group_df.at[i, col]
            new_val = set_lsb(old_val, W1[col][i], position=0)
            group_df.at[i, col] = new_val

    # Step 4: Tuple watermarks W2^i (one per tuple)
    W2 = {}  # i → list of γ bits
    for i in range(v):
        # After W1 embed: clear 2 LSBs (integer part only) for stable hash input
        cleared_vals = [int(clear_lsbs(group_df.at[i, col], 2)) for col in numeric_cols]
        H2 = hash_value(secret_key, pk_values[i], *cleared_vals)
        bits = [(H2 >> b) & 1 for b in range(gamma)]
        W2[i] = bits

    # Embed W2: set next_LSB(ri.Aj) ← W2^i[j]
    for i in range(v):
        for j, col in enumerate(numeric_cols):
            old_val = group_df.at[i, col]
            new_val = set_lsb(old_val, W2[i][j], position=1)
            group_df.at[i, col] = new_val

    info = {
        "v": v,
        "gamma": gamma,
        "H0": H0,
        "W1": {col: W1[col] for col in numeric_cols},
        "W2": {i: W2[i] for i in range(v)},
        "pk_values": pk_values,
    }
    return group_df, info


def embed_watermark(
    df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    num_groups: int,
    numeric_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Embed B2 watermark into entire database.

    Returns:
        watermarked_df, info dict
    """
    if numeric_cols is None:
        numeric_cols = get_numeric_cols(df, exclude_cols=[pk_col])

    # Partition into groups: group_id = HASH(K, pk) mod num_groups
    groups: Dict[int, List[int]] = {g: [] for g in range(num_groups)}
    for pos in range(len(df)):
        pk_val = df.iloc[pos][pk_col]
        gid = hash_value(secret_key, pk_val) % num_groups
        groups[gid].append(pos)

    watermarked_df = df.copy()
    group_info = {}

    for gid in sorted(groups.keys()):
        positions = groups[gid]
        if len(positions) == 0:
            group_info[gid] = {"size": 0, "status": "empty"}
            continue

        group_df = df.iloc[positions].copy()
        wm_group_df, g_info = embed_group(group_df, pk_col, numeric_cols, secret_key)

        # Write watermarked values back (matching by PK to be safe)
        # Map original positions to group-internal positions after sort
        for local_idx in range(len(wm_group_df)):
            pk_val = wm_group_df.iloc[local_idx][pk_col]
            # Find in original df
            orig_mask = df[pk_col] == pk_val
            orig_positions = df.index[orig_mask].tolist()
            if orig_positions:
                orig_pos = orig_positions[0]
                for col in numeric_cols:
                    watermarked_df.at[orig_pos, col] = wm_group_df.at[local_idx, col]

        g_info["status"] = "watermarked"
        g_info["group_positions"] = positions
        group_info[gid] = g_info

    info = {
        "method": "B2_Guo2006",
        "pk_col": pk_col,
        "numeric_cols": numeric_cols,
        "num_groups": num_groups,
        "n_tuples": len(df),
        "groups": group_info,
    }

    return watermarked_df.reset_index(drop=True), info


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B2 Guo 2006 — Embed watermark")
    parser.add_argument("--input",   required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--config",  required=True)
    parser.add_argument("--info_out",default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)
    numeric_cols = cfg.get("numeric_cols") or None

    wm_df, info = embed_watermark(
        df,
        pk_col=cfg["pk_col"],
        secret_key=cfg["secret_key"],
        num_groups=cfg["num_groups"],
        numeric_cols=numeric_cols,
    )

    save_csv(wm_df, args.output)
    info_path = args.info_out or args.output.replace(".csv", "_embed_info.json")
    save_json(info, info_path)
    print(f"[B2 Embed] Watermarked {len(df)} tuples → {args.output}")
    print(f"[B2 Embed] Info saved → {info_path}")


if __name__ == "__main__":
    main()
