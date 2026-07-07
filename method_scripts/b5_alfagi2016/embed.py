"""
B5: Alfagi et al. 2016 — Character Frequency & Text-Length Frequency Zero Watermarking (Embed)

Algorithm (zero-distortion; textual data):

Watermark generation (Algorithm 1 & 2):
  1. Character frequency watermark (Wchar):
     - For all textual attributes, collect all lowercase alphabetic characters
     - rfchari = freq(char_i) / total_chars * 100   for i in a-z
     - Wchar = [rfchar_a, ..., rfchar_z, total_char_count]

  2. Text-length frequency watermark (Wtxtlen):
     - For all textual attributes, collect lengths of every string value
     - rfTxtLeni = freq(len_i) / total_values * 100   for each distinct length
     - Wtxtlen = list of (length, rfTxtLen) pairs, sorted by length;
       also stores total_txtlen_count and the full sorted lengths list

  3. Combined watermark:
     WRDB = Wchar || Wtxtlen (serialised)
     EWRDB = HMAC-SHA256(WRDB || secret_key)
     Wcer = EWRDB || owner_id || timestamp  → registered with CA (JSON)

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

import pandas as pd

from common.utils import (
    load_csv, save_json, load_config, get_text_cols
)


# ---------------------------------------------------------------------------
# Frequency extraction
# ---------------------------------------------------------------------------

ALPHABET = list('abcdefghijklmnopqrstuvwxyz')


def extract_char_frequencies(df: pd.DataFrame, text_cols: List[str]) -> Dict:
    """Compute per-character relative frequency watermark (Wchar).

    Returns dict with:
      char_freq: {char: count}
      total_char_count: int
      rfchar: {char: rfchari}   rfchari = freq / total * 100
    """
    char_freq = {c: 0 for c in ALPHABET}
    total = 0

    for col in text_cols:
        for val in df[col].dropna().astype(str):
            for ch in val.lower():
                if ch.isalpha() and ch in char_freq:
                    char_freq[ch] += 1
                    total += 1

    rfchar = {}
    for c in ALPHABET:
        rfchar[c] = (char_freq[c] / total * 100) if total > 0 else 0.0

    return {
        "char_freq": char_freq,
        "total_char_count": total,
        "rfchar": rfchar,
        # Flat list in a-z order for serialisation / comparison
        "watermark_vector": [rfchar[c] for c in ALPHABET] + [float(total)],
    }


def extract_txtlen_frequencies(df: pd.DataFrame, text_cols: List[str]) -> Dict:
    """Compute text-length relative frequency watermark (Wtxtlen).

    Returns dict with:
      txtlen_freq: {length: count}
      total_txtlen_count: int
      rftxtlen: {length: rfTxtLeni}
      sorted_lengths: sorted list of distinct lengths
      watermark_vector: [rfTxtLen for each sorted length] + [total_count]
    """
    txtlen_freq: Dict[int, int] = {}
    total = 0

    for col in text_cols:
        for val in df[col].dropna().astype(str):
            length = len(val)
            txtlen_freq[length] = txtlen_freq.get(length, 0) + 1
            total += 1

    rftxtlen = {}
    for length, cnt in txtlen_freq.items():
        rftxtlen[length] = (cnt / total * 100) if total > 0 else 0.0

    sorted_lengths = sorted(txtlen_freq.keys())
    watermark_vector = [rftxtlen[l] for l in sorted_lengths] + [float(total)]

    return {
        "txtlen_freq": {str(k): v for k, v in txtlen_freq.items()},
        "total_txtlen_count": total,
        "rftxtlen": {str(k): v for k, v in rftxtlen.items()},
        "sorted_lengths": sorted_lengths,
        "watermark_vector": watermark_vector,
    }


def compute_wrdb(wchar_info: Dict, wtxtlen_info: Dict) -> List[float]:
    """Concatenate Wchar and Wtxtlen vectors into WRDB."""
    return wchar_info["watermark_vector"] + wtxtlen_info["watermark_vector"]


def encrypt_wrdb(wrdb: List[float], secret_key: str) -> str:
    """HMAC-SHA256 of WRDB||secret_key → hex digest (EWRDB)."""
    key = secret_key.encode('utf-8')
    msg = (json.dumps(wrdb, separators=(',', ':')) + secret_key).encode('utf-8')
    return hmaclib.new(key, msg, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_watermark(
    df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    owner_id: str,
    text_cols: Optional[List[str]],
    output_dir: str,
) -> Dict:
    """Compute and register the B5 watermark.

    Returns ca_record dict saved to output_dir/ca_registration.json.
    """
    if text_cols is None:
        text_cols = get_text_cols(df, exclude_cols=[pk_col])

    assert len(text_cols) >= 1, f"B5 requires at least 1 textual attribute; got {len(text_cols)}"

    alpha = len(df)

    wchar_info = extract_char_frequencies(df, text_cols)
    wtxtlen_info = extract_txtlen_frequencies(df, text_cols)

    wrdb = compute_wrdb(wchar_info, wtxtlen_info)
    ewrdb = encrypt_wrdb(wrdb, secret_key)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    ca_record = {
        "method": "B5_Alfagi2016",
        "owner_id": owner_id,
        "timestamp": timestamp,
        "EWRDB": ewrdb,
        "WRDB": wrdb,
        "alpha": alpha,
        "pk_col": pk_col,
        "text_cols": text_cols,
        # Sub-watermarks stored for detailed comparison at detect time
        "wchar": wchar_info,
        "wtxtlen": wtxtlen_info,
    }

    os.makedirs(output_dir, exist_ok=True)
    ca_path = os.path.join(output_dir, "ca_registration.json")
    save_json(ca_record, ca_path)
    print(f"[B5 Embed] CA registration saved → {ca_path}")
    print(f"[B5 Embed] Text cols: {text_cols}, α={alpha} tuples")
    print(f"[B5 Embed] Total chars: {wchar_info['total_char_count']}, "
          f"Total txtlen samples: {wtxtlen_info['total_txtlen_count']}")
    return ca_record


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B5 Alfagi 2016 — Register watermark")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config",     required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)
    text_cols = cfg.get("text_cols") or None
    if isinstance(text_cols, list) and len(text_cols) == 0:
        text_cols = None

    register_watermark(
        df,
        pk_col=cfg["pk_col"],
        secret_key=cfg["secret_key"],
        owner_id=cfg.get("owner_id", "owner_001"),
        text_cols=text_cols,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
