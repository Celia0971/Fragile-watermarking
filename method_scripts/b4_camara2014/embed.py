"""
B4: Camara et al. 2014 — Distortion-Free Watermarking via Square Matrix Groups (Embed)

Algorithm (zero-distortion; numeric data):

1. Data Partitioning (Algorithm 1):
   - ν = ceil(α / γ) groups  (α=num_tuples, γ=num_numeric_attributes)
   - h_i = Hash(K || ri.P || K)  for each tuple ri
   - group_id j = h_i mod ν
   - If last group is incomplete (size < γ), fill it by repeating first tuple(s)
     of the first group (these added rows are deleted after watermark generation)

2. Group Watermark Generation (Algorithm 2):
   - Sort tuples in Gj by ascending primary key hash
   - Treat Gj as a γ×γ square matrix (rows=tuples, cols=numeric attributes)
   - Compute Dj = det(Gj)
   - Compute Mji = minor of diagonal element (i,i) for i=1..γ
   - Wj = Dj || Mj1 || Mj2 || ... || Mjγ

3. Watermark Computation and Registration (Algorithm 3):
   - WR = W1 || W2 || ... || Wν
   - EWR = Encrypt(WR || K)  (simulated via HMAC)
   - WC = EWR || owner_id || date  → registered with CA

No data values are modified (zero-distortion).

Usage:
    python embed.py --input db.csv --output_dir results/ --config config/params.yaml
"""

import argparse
import hashlib
import hmac as hmaclib
import json
import math
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.utils import (
    load_csv, save_json, load_config, hash_value, get_numeric_cols
)


# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------

def compute_determinant(matrix: np.ndarray) -> float:
    """Compute determinant of a square matrix (using numpy)."""
    return float(np.linalg.det(matrix))


def compute_diagonal_minor(matrix: np.ndarray, i: int) -> float:
    """Compute minor of diagonal element (i, i): det of matrix with row i and col i removed."""
    n = matrix.shape[0]
    rows = [r for r in range(n) if r != i]
    cols = [c for c in range(n) if c != i]
    submatrix = matrix[np.ix_(rows, cols)]
    if submatrix.size == 0:
        return 1.0  # 0x0 matrix: minor = 1 by convention
    return float(np.linalg.det(submatrix))


def group_to_matrix(group_df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
    """Convert group DataFrame to numpy matrix (rows=tuples, cols=numeric_attrs)."""
    return group_df[numeric_cols].values.astype(float)


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def partition_into_groups(
    df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    num_groups: int,
) -> Dict[int, List[int]]:
    """Partition rows into num_groups groups by Hash(K || pk || K) mod num_groups."""
    groups: Dict[int, List[int]] = {g: [] for g in range(num_groups)}
    for pos in range(len(df)):
        pk_val = df.iloc[pos][pk_col]
        h = hash_value(secret_key, pk_val, secret_key)
        gid = h % num_groups
        groups[gid].append(pos)
    return groups


# ---------------------------------------------------------------------------
# Group watermark
# ---------------------------------------------------------------------------

def compute_group_watermark(
    group_df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    numeric_cols: List[str],
    gamma: int,
) -> Dict:
    """Compute watermark for a single group.

    1. Sort by primary key hash (ascending)
    2. If group size < gamma, pad with copies of first group's first tuple
       (caller responsibility to do this before calling this function)
    3. Compute det and diagonal minors
    4. Wj = [Dj, Mj0, Mj1, ..., Mj_{gamma-1}]
    """
    group_df = group_df.copy().reset_index(drop=True)

    # Sort by primary key hash ascending
    pk_hashes = [hash_value(secret_key, group_df.iloc[i][pk_col], secret_key)
                 for i in range(len(group_df))]
    sort_order = sorted(range(len(group_df)), key=lambda i: pk_hashes[i])
    group_df = group_df.iloc[sort_order].reset_index(drop=True)

    # Build matrix (should be gamma x gamma after padding)
    matrix = group_to_matrix(group_df, numeric_cols)

    # Compute determinant
    det = compute_determinant(matrix)

    # Compute minors of diagonal elements
    minors = [compute_diagonal_minor(matrix, i) for i in range(gamma)]

    # Watermark = [det, minor_0, ..., minor_{gamma-1}]
    wj = [det] + minors

    return {
        "pk_values": [str(group_df.iloc[i][pk_col]) for i in range(len(group_df))],
        "pk_hashes": [pk_hashes[sort_order[i]] for i in range(len(group_df))],
        "matrix": matrix.tolist(),
        "determinant": det,
        "diagonal_minors": minors,
        "watermark": wj,
    }


# ---------------------------------------------------------------------------
# Full embed / register
# ---------------------------------------------------------------------------

def register_watermark(
    df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    owner_id: str,
    numeric_cols: Optional[List[str]],
    output_dir: str,
    num_groups: Optional[int] = None,
) -> Dict:
    """Compute group watermarks and create CA registration.

    Returns ca_record dict.
    """
    if numeric_cols is None:
        numeric_cols = get_numeric_cols(df, exclude_cols=[pk_col])

    alpha = len(df)
    gamma = len(numeric_cols)
    assert gamma >= 2, f"B4 requires >= 2 numeric attributes; got {gamma}"

    if num_groups is None:
        num_groups = math.ceil(alpha / gamma)
    # num_groups may be 0 if dataset is tiny; ensure at least 1
    num_groups = max(1, num_groups)

    # Partition
    groups = partition_into_groups(df, pk_col, secret_key, num_groups)

    group_watermarks = {}
    all_wj = []

    for gid in range(num_groups):
        positions = groups[gid]
        group_df = df.iloc[positions].copy()

        # Pad incomplete last group to gamma tuples using first group's first rows
        if len(group_df) < gamma:
            first_group_positions = groups.get(0, [])
            n_needed = gamma - len(group_df)
            pad_positions = first_group_positions[:n_needed]
            if len(pad_positions) < n_needed and len(df) > 0:
                # Fall back to first n_needed rows of entire df
                pad_positions = list(range(min(n_needed, len(df))))
            pad_df = df.iloc[pad_positions].copy()
            group_df = pd.concat([group_df, pad_df], ignore_index=True)

        # KNOWN LIMITATION: B4 uses square gamma×gamma matrices.
        # Hash-based partitioning can produce groups with > gamma tuples;
        # only the first gamma are watermarked — excess tuples are unprotected.
        # Coverage < 100% is an inherent property of Camara 2014 on unequal groups.
        group_df = group_df.head(gamma).reset_index(drop=True)

        g_info = compute_group_watermark(
            group_df, pk_col, secret_key, numeric_cols, gamma
        )
        g_info["original_size"] = len(positions)
        g_info["padded_size"] = len(group_df)
        group_watermarks[gid] = g_info
        all_wj.append(g_info["watermark"])

    # WR = concatenation of all group watermarks
    WR = [val for wj in all_wj for val in wj]

    # EWR = HMAC(WR || K)
    key = secret_key.encode('utf-8')
    msg = (json.dumps(WR, separators=(',', ':')) + secret_key).encode('utf-8')
    EWR = hmaclib.new(key, msg, hashlib.sha256).hexdigest()

    # ── Storage accounting (rho numerator |W(R)|) ─────────────────────────────
    # B4 registers, per group, the watermark W_j = [det, M_0, ..., M_{gamma-1}]
    # = (gamma+1) real scalars, over num_groups groups. WR is the concatenation
    # of all groups' watermarks, so |WR| = num_groups * (gamma+1). Each scalar is
    # an IEEE-754 double (64 bits) -- the precision at which det/minors are
    # compared during verification -- so the external metadata is
    #     |W(R)| = len(WR) * 64 / 8  bytes = num_groups * (gamma+1) * 8.
    # The detect runner reads storage_bytes["total"] from this file; emitting it
    # here keeps B4 on the storage axis (it was previously absent -> NaN in the
    # collected results, requiring an out-of-band analytic estimate).
    W_VAL_BITS = 64
    n_scalars = len(WR)                         # == num_groups * (gamma + 1)
    total_storage_bytes = n_scalars * W_VAL_BITS // 8
    storage_bytes = {
        "watermark_scalars": n_scalars,
        "bits_per_scalar": W_VAL_BITS,
        "watermark": total_storage_bytes,
        "total": total_storage_bytes,
    }

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ca_record = {
        "method": "B4_Camara2014",
        "owner_id": owner_id,
        "timestamp": timestamp,
        "EWR": EWR,
        "WR": WR,                     # stored for verification
        "num_groups": num_groups,
        "alpha": alpha,
        "gamma": gamma,
        "pk_col": pk_col,
        "numeric_cols": numeric_cols,
        "storage_bytes": storage_bytes,
        "group_watermarks": {str(k): v for k, v in group_watermarks.items()},
    }

    os.makedirs(output_dir, exist_ok=True)
    ca_path = os.path.join(output_dir, "ca_registration.json")
    save_json(ca_record, ca_path)
    print(f"[B4 Embed] CA registration saved → {ca_path}")
    print(f"[B4 Embed] ν={num_groups} groups, γ={gamma} attributes, α={alpha} tuples")
    print(f"[B4 Embed] storage |W(R)| = {total_storage_bytes} bytes "
          f"({n_scalars} scalars × {W_VAL_BITS} bits)")
    return ca_record


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B4 Camara 2014 — Register watermark")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config",     required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)
    numeric_cols = cfg.get("numeric_cols") or None
    num_groups_override = cfg.get("num_groups") or None

    register_watermark(
        df,
        pk_col=cfg["pk_col"],
        secret_key=cfg["secret_key"],
        owner_id=cfg.get("owner_id", "owner_001"),
        numeric_cols=numeric_cols,
        output_dir=args.output_dir,
        num_groups=num_groups_override,
    )


if __name__ == "__main__":
    main()
