"""
Common utilities for fragile watermarking baselines.

Provides:
- CSV I/O
- Deterministic hashing (HMAC-SHA256)
- Seed management (base_seed + trial_index)
- JSON intermediate result storage
"""

import hashlib
import hmac
import json
import os
import struct
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def load_csv(path: str, pk_col: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV database file into a DataFrame.

    Args:
        path: Path to the CSV file.
        pk_col: If provided, set this column as the index (but keep it in data).

    Returns:
        DataFrame with all columns; no index manipulation by default.
    """
    df = pd.read_csv(path)
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hash_value(key: Union[str, bytes], *args) -> int:
    """Compute HMAC-SHA256 of (key, *args) and return as unsigned integer.

    All args are converted to bytes via str → UTF-8 encoding.
    Returns a 256-bit integer (non-negative).

    Args:
        key: Secret key (str or bytes).
        *args: Values to hash (will be str()-converted and concatenated with '||').
    """
    if isinstance(key, str):
        key = key.encode('utf-8')
    msg = '||'.join(str(a) for a in args).encode('utf-8')
    h = hmac.new(key, msg, hashlib.sha256).digest()
    return int.from_bytes(h, byteorder='big')


def hash_to_bits(key: Union[str, bytes], *args, n_bits: int = 32) -> List[int]:
    """Hash args and return the first n_bits bits as a list of 0/1 integers."""
    h = hash_value(key, *args)
    bits = []
    for i in range(n_bits - 1, -1, -1):
        bits.append((h >> i) & 1)
    # Return in MSB-first order, length n_bits
    # Actually use LSB-first by taking low bits
    result = []
    for i in range(n_bits):
        result.append((h >> i) & 1)
    return result


def hash_to_int(key: Union[str, bytes], *args, mod: Optional[int] = None) -> int:
    """Hash args and return integer, optionally modulo some value."""
    h = hash_value(key, *args)
    if mod is not None:
        return h % mod
    return h


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def get_trial_seed(base_seed: int, trial_index: int) -> int:
    """Return reproducible seed for trial: base_seed + trial_index.

    This gives different random behavior per trial while remaining reproducible.
    """
    return base_seed + trial_index


def make_rng(base_seed: int, trial_index: int) -> np.random.Generator:
    """Create a numpy default_rng for the given trial."""
    seed = get_trial_seed(base_seed, trial_index)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# JSON intermediate results
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str) -> None:
    """Save arbitrary data to a JSON file (pretty-printed)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def _json_default(obj):
    """JSON serialization fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Numeric column utilities
# ---------------------------------------------------------------------------

def get_numeric_cols(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """Return list of numeric column names, optionally excluding some columns."""
    exclude = set(exclude_cols or [])
    return [c for c in df.select_dtypes(include='number').columns if c not in exclude]


def get_text_cols(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """Return list of object/string column names."""
    exclude = set(exclude_cols or [])
    return [c for c in df.select_dtypes(include='object').columns if c not in exclude]


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def make_output_dir(base_dir: str, method: str, trial: int) -> str:
    """Create and return output directory for a method/trial run."""
    out = os.path.join(base_dir, method, f"trial_{trial:02d}")
    os.makedirs(out, exist_ok=True)
    return out
