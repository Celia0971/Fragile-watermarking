"""
B3: Khan & Husain 2013 — Fragile Zero Watermarking (Embed / Register)

Algorithm (zero-distortion; watermark derived from DB statistics):

Three sub-watermarks computed from ALL numeric attributes:

1. ωd (digit sub-watermark):
   - For every decimal digit d in every value across all numeric attributes:
     count digit_frequency[d] for d ∈ {0,...,9}
   - rfd_i = (digit_frequency[i] / total_digit_count) * 100   for i=0..9
   - ωd = [rfd_0, ..., rfd_9, total_digit_count]

2. ωl (length sub-watermark):
   - For every numeric value, compute len(str(abs(int(value))))
   - rfl_j = (length_frequency[j] / total_length_count) * 100  for j=1..max_len
   - ωl = [rfl_1, ..., rfl_max_len, total_length_count]

3. ωr (range sub-watermark):
   - For every numeric value, classify into range bins
   - rfr_k = (range_frequency[k] / total_range_count) * 100   for k=0..num_bins-1
   - ωr = [rfr_0, ..., rfr_{K-1}, total_range_count]

ωR = ωd ∥ ωl ∥ ωr   (concatenated list of floats)
EωR = Encrypt(ωR, SK)  → simulated by HMAC-SHA256 of serialized ωR with SK
WC = EωR ∥ owner_id ∥ timestamp  → saved as "CA registration" JSON

The original database is NOT modified (zero-distortion).

Usage:
    python embed.py --input db.csv --output_dir results/ --config config/params.yaml
"""

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.utils import (
    load_csv, save_json, load_config, get_numeric_cols
)


def extract_digits(value: float) -> List[int]:
    """Extract all decimal digits from the integer representation of a value."""
    int_val = abs(int(value))
    if int_val == 0:
        return [0]
    digits = []
    while int_val > 0:
        digits.append(int_val % 10)
        int_val //= 10
    return digits


def value_to_length(value: float) -> int:
    """Return the number of digits in the integer representation of abs(value)."""
    int_val = abs(int(value))
    if int_val == 0:
        return 1
    n = 0
    while int_val > 0:
        n += 1
        int_val //= 10
    return n


def classify_range(value: float, range_bins: List[List[int]]) -> int:
    """Return index of the first range bin that value falls into, or len(bins) for 'other'."""
    for k, (lo, hi) in enumerate(range_bins):
        if lo <= abs(int(value)) < hi:
            return k
    return len(range_bins)  # 'other' bin


def compute_watermark(
    df: pd.DataFrame,
    numeric_cols: List[str],
    range_bins: List[List[int]],
) -> Dict:
    """Compute the three sub-watermarks from the database.

    Returns:
        dict with keys: omega_d, omega_l, omega_r, omega_R (full watermark list),
        and detailed frequency statistics.
    """
    # ── ωd: digit sub-watermark ──────────────────────────────────────────────
    digit_freq = [0] * 10
    total_digit_count = 0

    for col in numeric_cols:
        for val in df[col].dropna():
            for d in extract_digits(float(val)):
                digit_freq[d] += 1
                total_digit_count += 1

    if total_digit_count > 0:
        rfd = [digit_freq[i] / total_digit_count * 100.0 for i in range(10)]
    else:
        rfd = [0.0] * 10

    omega_d = rfd + [float(total_digit_count)]

    # ── ωl: length sub-watermark ─────────────────────────────────────────────
    length_freq: Dict[int, int] = {}
    total_length_count = 0

    for col in numeric_cols:
        for val in df[col].dropna():
            ln = value_to_length(float(val))
            length_freq[ln] = length_freq.get(ln, 0) + 1
            total_length_count += 1

    max_len = max(length_freq.keys()) if length_freq else 1
    rfl = []
    for j in range(1, max_len + 1):
        cnt = length_freq.get(j, 0)
        rfl.append(cnt / total_length_count * 100.0 if total_length_count > 0 else 0.0)

    omega_l = rfl + [float(total_length_count), float(max_len)]

    # ── ωr: range sub-watermark ───────────────────────────────────────────────
    num_bins = len(range_bins) + 1  # +1 for 'other'
    range_freq = [0] * num_bins
    total_range_count = 0

    for col in numeric_cols:
        for val in df[col].dropna():
            k = classify_range(float(val), range_bins)
            range_freq[k] += 1
            total_range_count += 1

    if total_range_count > 0:
        rfr = [range_freq[k] / total_range_count * 100.0 for k in range(num_bins)]
    else:
        rfr = [0.0] * num_bins

    omega_r = rfr + [float(total_range_count)]

    # ── ωR: full watermark ────────────────────────────────────────────────────
    omega_R = omega_d + omega_l + omega_r

    return {
        "omega_d": omega_d,
        "omega_l": omega_l,
        "omega_r": omega_r,
        "omega_R": omega_R,
        "digit_freq": digit_freq,
        "total_digit_count": total_digit_count,
        "length_freq": {str(k): v for k, v in length_freq.items()},
        "total_length_count": total_length_count,
        "max_len": max_len,
        "range_freq": range_freq,
        "total_range_count": total_range_count,
        "numeric_cols": numeric_cols,
        "n_tuples": len(df),
    }


def encrypt_watermark(omega_R: List[float], secret_key: str) -> str:
    """Simulate watermark encryption: HMAC-SHA256(serialized omega_R, SK).

    Returns hex string (simulating EωR).
    """
    key = secret_key.encode('utf-8')
    msg = json.dumps(omega_R, separators=(',', ':')).encode('utf-8')
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def register_watermark(
    df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    owner_id: str,
    numeric_cols: Optional[List[str]],
    range_bins: List[List[int]],
    output_dir: str,
) -> Dict:
    """Compute watermark and create CA registration artifact.

    Returns:
        ca_record : dict representing the watermark certificate WC
    """
    if numeric_cols is None:
        numeric_cols = get_numeric_cols(df, exclude_cols=[pk_col])

    wm_data = compute_watermark(df, numeric_cols, range_bins)
    omega_R = wm_data["omega_R"]
    EwR = encrypt_watermark(omega_R, secret_key)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    ca_record = {
        "method": "B3_Khan2013",
        "owner_id": owner_id,
        "timestamp": timestamp,
        "EwR": EwR,       # encrypted watermark (for integrity check)
        "watermark": wm_data,  # stored for detection (simulating CA retrieval)
        "pk_col": pk_col,
        "numeric_cols": numeric_cols,
        "range_bins": range_bins,
    }

    os.makedirs(output_dir, exist_ok=True)
    ca_path = os.path.join(output_dir, "ca_registration.json")
    save_json(ca_record, ca_path)
    print(f"[B3 Embed] CA registration saved → {ca_path}")
    return ca_record


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B3 Khan 2013 — Compute & register watermark")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config",     required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)
    numeric_cols = cfg.get("numeric_cols") or None
    range_bins = cfg.get("range_bins", [[0,100],[100,1000],[1000,10000],[10000,100000],[100000,1000000]])

    ca_record = register_watermark(
        df,
        pk_col=cfg["pk_col"],
        secret_key=cfg["secret_key"],
        owner_id=cfg.get("owner_id", "owner_001"),
        numeric_cols=numeric_cols,
        range_bins=range_bins,
        output_dir=args.output_dir,
    )
    wm = ca_record["watermark"]
    print(f"[B3 Embed] n_tuples={wm['n_tuples']}, |ωR|={len(wm['omega_R'])}")
    print(f"[B3 Embed] EωR (first 16 chars): {ca_record['EwR'][:16]}...")


if __name__ == "__main__":
    main()
