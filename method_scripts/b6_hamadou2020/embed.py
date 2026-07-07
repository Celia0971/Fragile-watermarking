"""
B6: Hamadou et al. 2020 — Reversible Prediction-Error Expansion on LSD (Embed)

Algorithm (non-zero distortion, reversible; numeric data):

Partitioning:
  hi = HMAC(Ks || ti.pk || Ks) for each tuple ti
  group_id = hi mod v
  Within each group, sort tuples ascending by hi

Group watermark embedding (per group, per attribute):
  Let m be the actual number of tuples in the group.
  Hp = HMAC(Ks || h1 || h2 || ... || hm || Ks)
  For each numeric attribute Aj (j = 1..γ):
    Hj = HMAC(Hp || t1.Aj || ... || tm.Aj || Ks)
    Wj = first m bits of Hj  (MSB-first from bytes)

  For each tuple i in group, attribute j:
    hi_pred = hi % 10          (prediction: LSD of pk-hash, keeps hi_pred in [0,9])
    lsd = abs(int(val)) % 10   (last decimal digit of the integer part)
    msd = abs(int(val)) // 10
    e   = lsd - hi_pred        (prediction error)
    ew  = 2 * e + Wj[i]        (expanded error, encodes 1 bit)
    new_lsd = ew + hi_pred     (encoded LSD)

    Embeddable iff new_lsd ∈ [0, 9]:
      new_val = sign * (msd * 10 + new_lsd) + frac_part
    Non-embeddable: keep original value, mark position as skipped.

Reversibility:
  Given val_wm and hi_pred:
    new_lsd = abs(int(val_wm)) % 10       (always in [0,9] by embeddability constraint)
    ew      = new_lsd - hi_pred            (in range since both ∈ [0,9])
    bit     = ew % 2
    e       = (ew - bit) // 2
    orig_lsd = new_lsd - e - bit          (recovers original LSD)
    orig_val = sign * (msd_stored * 10 + orig_lsd) + frac_part

CA registration: stores group watermark bits, which positions were embedded, original
and modified values per group. Secret key is NOT stored (injected at detect time from config).

Usage:
    python embed.py --input db.csv --output_dir results/ --config config/params.yaml
"""

import argparse
import math
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from common.utils import (
    load_csv, save_json, load_config, hash_value, hash_to_bits, get_numeric_cols
)


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def partition_into_groups(
    df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    num_groups: int,
) -> Dict[int, List[int]]:
    """Partition rows into groups, sorted ascending by pk-hash within each group."""
    pk_hashes: List[Tuple[int, int]] = []
    for pos in range(len(df)):
        pk_val = df.iloc[pos][pk_col]
        h = hash_value(secret_key, pk_val, secret_key)
        pk_hashes.append((pos, h))

    groups: Dict[int, List[Tuple[int, int]]] = {g: [] for g in range(num_groups)}
    for pos, h in pk_hashes:
        gid = h % num_groups
        groups[gid].append((pos, h))

    sorted_groups: Dict[int, List[int]] = {}
    for gid, members in groups.items():
        members_sorted = sorted(members, key=lambda x: x[1])
        sorted_groups[gid] = [pos for pos, _ in members_sorted]

    return sorted_groups


def get_sorted_pk_hashes(
    df: pd.DataFrame,
    positions: List[int],
    pk_col: str,
    secret_key: str,
) -> Tuple[List[int], List[int]]:
    """Return (sorted_positions, sorted_pk_hashes) for given positions."""
    pairs = sorted(
        [(pos, hash_value(secret_key, df.iloc[pos][pk_col], secret_key))
         for pos in positions],
        key=lambda x: x[1]
    )
    return [p for p, _ in pairs], [h for _, h in pairs]


# ---------------------------------------------------------------------------
# LSD encoding / decoding (safe, embeddability-constrained)
# ---------------------------------------------------------------------------

def encode_lsd_safe(val: float, hi_pred: int, bit: int) -> Tuple[float, bool]:
    """Encode one watermark bit into the LSD of a numeric value (safe version).

    Only embeds if new_lsd ∈ [0, 9]; otherwise returns original value unchanged.

    Returns:
        (new_val, embedded) — embedded=True if the bit was successfully encoded.
    """
    int_part = int(val)
    frac_part = val - int_part
    sign = -1 if int_part < 0 else 1
    abs_int = abs(int_part)

    lsd = abs_int % 10
    msd = abs_int // 10
    e = lsd - hi_pred
    ew = 2 * e + bit
    new_lsd = ew + hi_pred

    if 0 <= new_lsd <= 9:
        new_val = sign * (msd * 10 + new_lsd) + frac_part
        return new_val, True
    else:
        return val, False


def decode_lsd(val_wm: float, hi_pred: int) -> Tuple[float, Optional[int]]:
    """Recover original value and embedded bit from a watermarked value.

    Uses val_wm % 10 as new_lsd (guaranteed [0,9] if embed used safe encoding).

    Returns:
        (orig_val, bit) — bit is None if decode is invalid (orig_lsd outside [0,9]).
    """
    int_part = int(val_wm)
    frac_part = val_wm - int_part
    sign = -1 if int_part < 0 else 1
    abs_int = abs(int_part)

    msd_stored = abs_int // 10
    new_lsd = abs_int % 10          # always in [0,9]
    ew = new_lsd - hi_pred          # may be in [-9, 9]
    bit = ew % 2                    # Python % → always 0 or 1
    e = (ew - bit) // 2             # prediction error (floor div)
    orig_lsd = new_lsd - e - bit    # reconstruct original LSD

    if not (0 <= orig_lsd <= 9):
        return val_wm, None         # decode invalid: this value was not embedded

    orig_abs = msd_stored * 10 + orig_lsd
    orig_val = sign * orig_abs + frac_part
    return orig_val, bit


# ---------------------------------------------------------------------------
# Group watermark
# ---------------------------------------------------------------------------

def compute_group_watermark_bits(
    group_df: pd.DataFrame,
    pk_hashes_sorted: List[int],
    secret_key: str,
    numeric_cols: List[str],
) -> Dict[str, List[int]]:
    """Compute per-attribute watermark bit vectors Wj for a group.

    Hp = HMAC(Ks || h1 || ... || hm || Ks)
    Hj = HMAC(Hp || t1.Aj || ... || tm.Aj || Ks)
    Wj = first m bits of Hj, where m is the actual group cardinality.
    """
    group_size = len(group_df)
    hp_int = hash_value(secret_key, *pk_hashes_sorted, secret_key)
    hp_str = str(hp_int)

    wj_bits = {}
    for col in numeric_cols:
        # Always use float string to ensure consistent representation across int/float cols
        col_vals = [str(float(group_df.iloc[i][col])) for i in range(group_size)]
        bits = hash_to_bits(hp_str, *col_vals, secret_key, n_bits=group_size)
        wj_bits[col] = bits

    return wj_bits


def embed_group(
    df: pd.DataFrame,
    positions_sorted: List[int],
    pk_hashes_sorted: List[int],
    secret_key: str,
    numeric_cols: List[str],
) -> Tuple[Dict[str, List[float]], Dict]:
    """Embed watermark bits into one group.

    Returns:
        modified_values : {col: [new_val_per_position]}
        info            : detailed embedding record
    """
    group_df = df.iloc[positions_sorted].reset_index(drop=True)
    group_size = len(group_df)

    wj_bits = compute_group_watermark_bits(
        group_df, pk_hashes_sorted, secret_key, numeric_cols
    )

    modified_values:  Dict[str, List[float]] = {col: [] for col in numeric_cols}
    original_values:  Dict[str, List[float]] = {col: [] for col in numeric_cols}
    embedded_mask:    Dict[str, List[bool]]  = {col: [] for col in numeric_cols}

    for col in numeric_cols:
        bits = wj_bits[col]
        for i in range(group_size):
            val = float(group_df.iloc[i][col])
            hi  = pk_hashes_sorted[i]
            hi_pred = hi % 10
            bit = bits[i]
            new_val, embedded = encode_lsd_safe(val, hi_pred, bit)
            original_values[col].append(val)
            modified_values[col].append(new_val)
            embedded_mask[col].append(embedded)

    info = {
        "positions_sorted": positions_sorted,
        "pk_hashes_sorted": pk_hashes_sorted,
        "group_size": group_size,
        "wj_bits": wj_bits,
        "original_values": original_values,
        "modified_values": modified_values,
        "embedded_mask": embedded_mask,
    }
    return modified_values, info


# ---------------------------------------------------------------------------
# Full embed
# ---------------------------------------------------------------------------

def embed_watermark(
    df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    owner_id: str,
    numeric_cols: Optional[List[str]],
    output_dir: str,
    num_groups: Optional[int] = None,
    mu: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Embed B6 watermark into df.

    Returns (watermarked_df, ca_record).
    """
    if numeric_cols is None:
        numeric_cols = get_numeric_cols(df, exclude_cols=[pk_col])
    assert len(numeric_cols) >= 1, "B6 requires at least 1 numeric attribute"

    alpha = len(df)
    if mu is None:
        mu = len(numeric_cols)
    target_mu = max(1, mu)

    if num_groups is None:
        num_groups = max(1, math.ceil(alpha / target_mu))

    groups = partition_into_groups(df, pk_col, secret_key, num_groups)

    wm_df = df.copy()
    group_info = {}

    for gid in range(num_groups):
        positions = groups[gid]
        if len(positions) == 0:
            group_info[str(gid)] = {"positions": [], "skipped": True}
            continue

        positions_sorted, pk_hashes_sorted = get_sorted_pk_hashes(
            df, positions, pk_col, secret_key
        )

        modified_values, info = embed_group(
            df, positions_sorted, pk_hashes_sorted,
            secret_key, numeric_cols
        )

        # Write modified values back for every tuple that belongs to this group.
        actual_positions_set = set(positions)
        for idx, pos in enumerate(positions_sorted):
            if pos in actual_positions_set:
                for col in numeric_cols:
                    wm_df.at[wm_df.index[pos], col] = modified_values[col][idx]

        info["original_group_size"] = len(positions)
        group_info[str(gid)] = info

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ca_record = {
        "method": "B6_Hamadou2020",
        "owner_id": owner_id,
        "timestamp": timestamp,
        "alpha": alpha,
        "pk_col": pk_col,
        "numeric_cols": numeric_cols,
        "num_groups": num_groups,
        # The paper uses μ to derive v=ceil(alpha/μ). The watermark length for
        # each group is not forced to μ; it is that group's actual tuple count.
        "mu": target_mu,
        "group_size_mode": "actual_group_cardinality",
        "group_info": group_info,
    }

    os.makedirs(output_dir, exist_ok=True)
    ca_path = os.path.join(output_dir, "ca_registration.json")
    wm_path = os.path.join(output_dir, "watermarked.csv")
    save_json(ca_record, ca_path)
    wm_df.to_csv(wm_path, index=False)
    print(f"[B6 Embed] CA registration saved → {ca_path}")
    print(f"[B6 Embed] Watermarked DB saved  → {wm_path}")
    print(f"[B6 Embed] ν={num_groups} groups, target μ={target_mu}, γ={len(numeric_cols)}, α={alpha}")
    return wm_df, ca_record


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B6 Hamadou 2020 — Embed watermark")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config",     required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)
    numeric_cols  = cfg.get("numeric_cols") or None
    num_groups    = cfg.get("num_groups")   or None
    mu            = cfg.get("mu")           or None

    embed_watermark(
        df,
        pk_col=cfg["pk_col"],
        secret_key=cfg["secret_key"],
        owner_id=cfg.get("owner_id", "owner_001"),
        numeric_cols=numeric_cols,
        output_dir=args.output_dir,
        num_groups=num_groups,
        mu=mu,
    )


if __name__ == "__main__":
    main()
