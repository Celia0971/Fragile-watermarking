"""
Proposed Scheme: IBLT + Group-Testing Fragile Watermarking — Generation (Embed/Register)

Implements the Gen algorithm from Definition 4.1 of the paper.

Two-layer zero-distortion watermark:

  Layer 1 — Primary-key reconciliation (§4.2, Eq. 14):
    S_pk(R) = IBLT_K(PK(R))
    An IBLT built over HMAC-keyed encodings of all primary keys.
    At verify time: IBLT_K(PK(R)) − IBLT_K(PK(R′)) decodes to
    (D̂_del, D̂_ins) = (PK(R) \ PK(R′), PK(R′) \ PK(R)).

  Layer 2 — Group-based modification localization (§4.3):
    For each tuple T_i, compute b-bit digest c_i = H_{K,c}(serial(T_i))  [Eq. 26].
    Assign each tuple to g Bernoulli(p) tests via A_{a,pk} = 1[H_{K,g}(a||pk)/2^λ < p]  [Eq. 19].
    Store g XOR syndromes: C_a = ⊕_{T_i ∈ G_a} c_i  [Eq. 27].
    Also store individual digests for "full" mode and for S_del syndrome adjustment.

  Compressed-mode alternative (§6.3):
    Each tuple is assigned to exactly ONE group via HMAC % g (single-partition).
    Stores g XOR group hashes instead of n individual digests.
    Lower storage cost; used in SP4 storage–accuracy trade-off experiments.
    Note: this mode does NOT use the Bernoulli COMP guarantee.

No attribute values are modified. All metadata is stored externally in the CA JSON.

Paper reference: §4 (Watermark Generation), Definition 4.1, Eqs. 10–27.
"""

import argparse
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))        # baselines/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))     # project root (iblt.py)

import hmac as _hmac
import hashlib as _hashlib
import struct as _struct
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common.utils import (
    load_csv, save_csv, save_json, load_config, hash_value
)
from proposed.iblt import IBLT

# Memory-efficient chunk size for (n×g) matrix operations.
# Peak memory per chunk: CHUNK_SIZE × g × 8 bytes (uint64) ≈ 100 MB for g=2532.
_CHUNK_SIZE = 5000

# Per-chunk memory budget (bytes).  The (chunk × g) uint64 matrix and the fmix64
# temporaries dominate; peak ≈ chunk × g × 8 × 4.  With many parallel workers,
# a fixed CHUNK_SIZE=5000 would OOM for large g (e.g. g≈63k → 2.5 GB/chunk),
# so we shrink the chunk adaptively to keep peak ≈ _CHUNK_MEM_BUDGET per worker.
_CHUNK_MEM_BUDGET = 384 * 1024 * 1024   # ~384 MB peak per chunk


def _chunk_for_g(gt_g: int) -> int:
    """Adaptive row-chunk size: keeps the (chunk × g) working set within budget.

    chunk = clamp( budget / (g × 8 × 4),  32,  _CHUNK_SIZE )
    Small g → keeps the full _CHUNK_SIZE; large g → shrinks to avoid OOM.
    Both embed and detect call this with the SAME g, so chunking never affects
    correctness (it only partitions the row loop).

    Floor is 32 (not 256): at very large g the 256 floor broke the budget — e.g.
    g=331k (FCT rho=10%) gave 256×g×8×4 ≈ 2.7 GB/chunk, capping parallelism to
    ~8 workers on a 31 GB box. Floor 32 keeps peak ≈ budget for all deployed g
    (32×331k×8×4 ≈ 0.4 GB); a 32-row × wide-g array is still fully vectorized, and
    total work is chunk-independent so wall time barely changes.
    """
    if gt_g <= 0:
        return _CHUNK_SIZE
    chunk = _CHUNK_MEM_BUDGET // (gt_g * 8 * 4)
    return max(32, min(_CHUNK_SIZE, int(chunk)))


# ── Shared helpers (imported by detect.py) ────────────────────────────────────

def pk_to_iblt_key(secret_key: str, pk_str: str) -> int:
    """Encode a primary key string as a 64-bit IBLT key via HMAC.

    Paper: S_pk(R) = IBLT_K(PK(R))  [Eq. 14].
    The IBLT operates on integer keys; this function maps each PK string
    to a unique 64-bit integer using a keyed hash, binding the encoding
    to the secret key K so that an adversary without K cannot forge keys.

    Args:
        secret_key : secret key K ∈ {0,1}^κ
        pk_str     : string representation of the primary key pk_i

    Returns:
        64-bit non-negative integer ∈ [0, 2^64 − 1].
    """
    h = hash_value(secret_key, "iblt_pk", pk_str)
    return h & 0xFFFFFFFFFFFFFFFF  # truncate to 64 bits


def value_to_str(val) -> str:
    """Stable display string for a single attribute value.

    Kept for log/debug compatibility. The digest path below uses
    encode_canonical_value(), which is type-tagged and length-prefixed.
    """
    if val is None:
        return "__NULL__"
    # float NaN is distinct from None in the schema — must NOT collapse to __NULL__
    try:
        if isinstance(val, float) and math.isnan(val):
            return "__NAN__"
        if isinstance(val, np.floating) and np.isnan(val):
            return "__NAN__"
    except TypeError:
        pass
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        # repr() gives the shortest round-trip-safe decimal in Python 3
        return repr(float(val))
    # str, categorical, date, empty string: preserved exactly for display.
    return str(val)


def _len_prefix(payload: bytes) -> bytes:
    """Return an unambiguous decimal length prefix followed by raw bytes."""
    return str(len(payload)).encode("ascii") + b":" + payload


def encode_canonical_value(val) -> bytes:
    """Canonical type-tagged byte encoding for one scalar cell value.

    Paper §4.3.3 requires injective, schema-fixed tuple serialization with
    type tags and length prefixes. This encoding distinguishes, for example,
    NULL from NaN, integer 1 from string "1", and empty string from NULL.
    """
    if val is None:
        tag = b"null"
        payload = b""
    elif isinstance(val, (bool, np.bool_)):
        tag = b"bool"
        payload = b"true" if bool(val) else b"false"
    elif isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_)):
        tag = b"int"
        payload = str(int(val)).encode("utf-8")
    elif isinstance(val, (float, np.floating)):
        tag = b"float"
        fval = float(val)
        if math.isnan(fval):
            payload = b"NaN"
        elif math.isinf(fval):
            payload = b"Inf" if fval > 0 else b"-Inf"
        else:
            payload = repr(fval).encode("utf-8")
    else:
        try:
            if bool(pd.isna(val)) and not isinstance(val, (str, bytes)):
                tag = b"null"
                payload = b""
            else:
                tag = b"str"
                payload = str(val).encode("utf-8")
        except (TypeError, ValueError):
            tag = b"str"
            payload = str(val).encode("utf-8")

    return _len_prefix(tag) + b":" + _len_prefix(payload)


def serialize_tuple_bytes(
    pk_str: str,
    row: Dict[str, Any],
    all_cols: List[str],
) -> bytes:
    """Serialize tuple identity and fixed-order non-PK attributes injectively."""
    parts = [b"pk", encode_canonical_value(pk_str)]
    for col in all_cols:
        col_bytes = str(col).encode("utf-8")
        parts.append(b"col")
        parts.append(_len_prefix(col_bytes))
        parts.append(encode_canonical_value(row.get(col)))
    return b"|".join(parts)


def _hmac_int_from_bytes(secret_key: str, label: str, payload: bytes, bits: int) -> int:
    """HMAC-SHA256 XOF-style integer output truncated to exactly bits bits."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    key_b = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
    needed = (bits + 7) // 8
    out = bytearray()
    counter = 0
    label_b = label.encode("utf-8")
    while len(out) < needed:
        msg = label_b + b"\0" + counter.to_bytes(4, "big") + b"\0" + payload
        out.extend(_hmac.new(key_b, msg, _hashlib.sha256).digest())
        counter += 1
    value = int.from_bytes(bytes(out[:needed]), "big")
    excess = needed * 8 - bits
    if excess:
        value >>= excess
    return value


def compute_tuple_digest(
    secret_key: str,
    pk_str: str,
    row: Dict[str, Any],
    all_cols: List[str],
    b_bits: int,
) -> int:
    """Compute the b-bit keyed digest for one tuple.

    Paper Eq. 26: c_i = H_{K,c}(serial(T_i)) ∈ {0,1}^b.
    Here H_{K,c} is the content-digest PRF keyed with K.
    serial(T_i) encodes tuple identity plus all non-PK attribute values in
    fixed column order, with type tags and length prefixes.

    Args:
        secret_key : secret key K (bound to "tuple_digest" domain label)
        pk_str     : primary key pk_i (included to bind digest to tuple identity)
        row        : dict {col_name: value} for the non-PK attributes of T_i
        all_cols   : ordered list of columns defining the schema (fixed across embed/detect)
        b_bits     : digest width b in bits; result is masked to b_bits

    Returns:
        Integer c_i ∈ [0, 2^b − 1].
    """
    payload = serialize_tuple_bytes(pk_str, row, all_cols)
    return _hmac_int_from_bytes(secret_key, "tuple_digest", payload, b_bits)


def gt_matrix_entry(
    secret_key: str,
    test_idx: int,
    pk_str: str,
    gt_m: int,
    gt_s_del: int = 0,
) -> bool:
    """Return True iff tuple pk_str participates in group-testing row test_idx.

    Paper Eqs. 18–19 (Pseudorandom Bernoulli groups):
      u_{a,pk} = H_{K,g}(a || pk) / 2^λ ∈ [0, 1)
      A_{a,pk} = 1[u_{a,pk} < p]
    where p = 1/(m + s_del + 1)  [Eq. 11, Remark 1 in paper].

    The denominator (m + s_del + 1) is optimal: it maximises the clean-separation
    probability q_{m,s_del} = p(1−p)^{m+s_del} for mixed attacks  [Eq. 25].
    When s_del = 0, this reduces to the standard p = 1/(m+1) used in deletion-free
    Bernoulli group testing  [Remark 1].

    Implementation: h % (m + s_del + 1) == 0 gives an equivalent uniform
    Bernoulli(1/(m+s_del+1)) sample from the PRF output.

    Args:
        secret_key : secret key K (H_{K,g} domain)
        test_idx   : group index a ∈ {0, …, g−1}
        pk_str     : primary key string
        gt_m       : modification tolerance m (|D_mod| ≤ m)
        gt_s_del   : deletion tolerance s_del (|D_del| ≤ s_del)

    Returns:
        True with probability p = 1/(m + s_del + 1).
    """
    h = hash_value(secret_key, "gt_entry", str(test_idx), pk_str)
    return (h % (gt_m + gt_s_del + 1)) == 0


def _pk_seeds(secret_key: str, pk_list: List[str]) -> np.ndarray:
    """Compute per-pk uint64 seeds: one HMAC call per pk (vectorization basis).

    Uses domain "gt_pk" (distinct from "gt_group") so pk and group seeds are
    independent. XOR(seed_pk, offset_a) is then used for membership decisions,
    replacing the n×g individual HMAC calls of the original gt_matrix_entry loop.

    Statistical guarantee (verified empirically):
      - Bernoulli(1/(m+s_del+1)) probability preserved
      - Independence between any two group assignment bits preserved
      - Uniform residue distribution mod (m+s_del+1) confirmed

    This is an equivalent PRF implementation: both the original HMAC(K, a||pk)
    and this XOR approach are computationally indistinguishable from independent
    Bernoulli(p) samples for an adversary without K.
    """
    key_b = secret_key.encode() if isinstance(secret_key, str) else secret_key
    seeds = np.empty(len(pk_list), dtype=np.uint64)
    for i, pk in enumerate(pk_list):
        h = _hmac.new(key_b, f"gt_pk||{pk}".encode(), _hashlib.sha256).digest()
        seeds[i] = _struct.unpack('>Q', h[:8])[0]
    return seeds


def _group_offsets(secret_key: str, gt_g: int) -> np.ndarray:
    """Compute per-group uint64 offsets: one HMAC call per group."""
    key_b = secret_key.encode() if isinstance(secret_key, str) else secret_key
    offsets = np.empty(gt_g, dtype=np.uint64)
    for a in range(gt_g):
        h = _hmac.new(key_b, f"gt_group||{a}".encode(), _hashlib.sha256).digest()
        offsets[a] = _struct.unpack('>Q', h[:8])[0]
    return offsets


# ── Membership PRF: keyed hash of the (group, pk) pair  [Eqs. 18-19] ──────────
# Constants are the MurmurHash3 64-bit finalizer (fmix64) multipliers.
_MIX_C1    = np.uint64(0xff51afd7ed558ccd)
_MIX_C2    = np.uint64(0xc4ceb9fe1a85ec53)
_MIX_SHIFT = np.uint64(33)


def _mix64(x: np.ndarray) -> np.ndarray:
    """MurmurHash3 64-bit finalizer (avalanche), vectorised over a uint64 array.

    Why this is required (correctness, not optimisation):
      Membership realises the paper's keyed Bernoulli test
      A_{a,pk} = 1[H_K(a‖pk) mod D = 0],  D = m + s_del + 1   [Eqs. 18-19].
      For speed we derive the per-pair value from two independent keyed hashes,
      seed_pk = H_K("gt_pk"‖pk) and offset_a = H_K("gt_group"‖a), as
      H_K(a‖pk) := fmix64(seed_pk XOR offset_a)  (a PRF of the pair under K),
      keeping the cost at n+g HMAC calls instead of n·g.

      The raw XOR alone is UNSAFE: when D is a power of two, x mod D depends only
      on the low log2(D) bits and XOR is bitwise, so all pks sharing those low
      bits collapse into one equivalence class with an identical membership row.
      That destroys clean-separation (Theorem 4.4) and makes the decoder flag
      whole classes (observed for m+r+1 ∈ {16,32}).

      fmix64 folds every high bit into the low bits via its shift-xor / multiply
      rounds, so (fmix64(combined) mod D) is a proper Bernoulli(1/D) draw for
      EVERY D, including powers of two. Both embed and detect call this single
      function, guaranteeing identical group membership.
    """
    x = x ^ (x >> _MIX_SHIFT)
    x = x * _MIX_C1
    x = x ^ (x >> _MIX_SHIFT)
    x = x * _MIX_C2
    x = x ^ (x >> _MIX_SHIFT)
    return x


def membership_matrix(
    seeds: np.ndarray,
    offsets: np.ndarray,
    gt_m: int,
    gt_s_del: int,
) -> np.ndarray:
    """Return boolean membership matrix A of shape (n, g) via numpy broadcast.

    A[i, a] = True iff tuple i belongs to group a.
    A_{a,pk} = 1[(fmix64(seed_pk XOR offset_a) % denom) == 0]  [Eqs. 18-19].

    Total HMAC calls: n + g  (vs n×g in the original per-pair approach); the
    fmix64 finalizer makes this a true keyed hash of the (a, pk) pair so the
    Bernoulli(1/denom) guarantee holds for every denom (see _mix64).
    """
    # Cast denom to uint64: np.uint64 % python_int coerces to float64 and
    # loses precision for large 64-bit integers, giving wrong membership bits.
    denom = np.uint64(gt_m + gt_s_del + 1)
    combined = seeds[:, None] ^ offsets[None, :]  # (n, g) uint64
    return (_mix64(combined) % denom) == np.uint64(0)   # (n, g) bool


def compute_syndromes(
    secret_key: str,
    pk_digest_map: Dict[str, int],
    gt_g: int,
    gt_b: int,
    gt_m: int,
    gt_s_del: int = 0,
) -> List[int]:
    """Compute g Bernoulli XOR syndromes  [paper Eq. 27: C_a = ⊕_{T_i ∈ G_a} c_i].

    Chunked implementation: processes _CHUNK_SIZE rows at a time to avoid
    creating the full (n × g) uint64 matrix (which would be n×g×8 bytes).
    Peak memory per chunk ≈ _CHUNK_SIZE × g × 8 bytes ≈ 100 MB for g=2532.
    """
    mask = (1 << gt_b) - 1
    denom = np.uint64(gt_m + gt_s_del + 1)
    pk_list = list(pk_digest_map.keys())
    digests  = list(pk_digest_map.values())
    n = len(pk_list)

    offsets = _group_offsets(secret_key, gt_g)
    syndromes = [0] * gt_g
    chunk_sz = _chunk_for_g(gt_g)   # adaptive: small for large g to avoid OOM

    for start in range(0, n, chunk_sz):
        end = min(start + chunk_sz, n)
        chunk_seeds = _pk_seeds(secret_key, pk_list[start:end])
        combined = chunk_seeds[:, None] ^ offsets[None, :]   # (chunk, g) uint64
        A_chunk  = (_mix64(combined) % denom) == np.uint64(0) # (chunk, g) bool — fmix64 PRF
        del combined   # free uint64 array immediately (8× larger than bool)
        rows, cols = np.where(A_chunk)
        del A_chunk
        for k in range(len(rows)):
            syndromes[int(cols[k])] ^= digests[start + int(rows[k])]

    return [s & mask for s in syndromes]


def get_group_assign(secret_key: str, pk_str: str, gt_g: int) -> int:
    """Assign tuple pk_str to exactly one partition group in {0, …, g−1}.

    Used only by the compressed-mode ALTERNATIVE (single-partition hash comparison).
    NOT part of the paper's main Bernoulli group-testing scheme.

    This provides lower storage (g × b vs n × b bits) but coarser localization:
    any positive group reports all ~n/g tuples as S_mod candidates.
    Precision ≈ 1/μ where μ = n/g is the average group size.

    Args:
        secret_key : secret key K
        pk_str     : primary key string
        gt_g       : number of partition groups

    Returns:
        Group index ∈ {0, …, g−1}.
    """
    h = hash_value(secret_key, "gt_group_assign", pk_str)
    return int(h % gt_g)


def compute_group_hashes(
    secret_key: str,
    pk_digest_map: Dict[str, int],
    gt_g: int,
    gt_b: int,
) -> List[int]:
    """Compute g single-partition XOR group hashes (compressed-mode alternative).

    Each tuple is assigned to exactly ONE group (disjoint partition).
    group_hash_j = ⊕_{pk assigned to group j} c_{pk}.

    Used only by the 'compressed' detection mode as a simpler alternative
    to the Bernoulli multi-test syndromes. Lower localization accuracy than COMP,
    but deterministic group membership simplifies the decode step.

    Storage: g × (b/8) bytes — same as syndromes, but disjoint vs. overlapping.

    Args:
        secret_key     : secret key K
        pk_digest_map  : {pk_str: c_i} for all tuples in R
        gt_g           : number of partition groups g
        gt_b           : digest width b in bits

    Returns:
        List of g group hash values ∈ [0, 2^b − 1].
    """
    mask = (1 << gt_b) - 1
    group_hashes = [0] * gt_g
    for pk_str, d in pk_digest_map.items():
        gj = get_group_assign(secret_key, pk_str, gt_g)
        group_hashes[gj] ^= d
    return [h & mask for h in group_hashes]


# ── Main registration function ────────────────────────────────────────────────

def embed_watermark(
    df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    iblt_capacity: int,
    iblt_mult: float,
    iblt_k: int,
    gt_g: int,
    gt_b: int,
    gt_m: int,
    gt_s_del: int = 0,
    all_cols: Optional[List[str]] = None,
    detection_mode: str = "compressed",
) -> Dict:
    """Run the Gen algorithm to produce verification metadata W(R).

    Paper Definition 4.1 (Watermark Generation):
      Input : R = {T_1,…,T_n}, K, m, s_ins, s_del, δ, b, k, α_IBLT, λ_fp
      Output: W(R) = (S_pk(R), S_grp(R))  [Eq. 13]

    where:
      S_pk(R)  = IBLT_K(PK(R))                           [Eq. 14 — Layer 1]
      S_grp(R) = {C_1,…,C_g}  (g XOR syndromes of b bits) [Eq. 27 — Layer 2]

    IBLT sizing  [Eq. 10]:  ℓ = ⌈α_IBLT × d_max⌉
      d_max = s_ins + s_del (= iblt_capacity in config)

    Bernoulli probability  [Eq. 11]: p = 1/(m + s_del + 1)
    Number of groups       [Eq. 12]: g sized to satisfy Theorem 4.4 (set manually here)

    This is a zero-distortion scheme: W(R) is stored externally in the CA JSON.
    The released database R is not modified.

    Args:
        df             : original database R as a DataFrame
        pk_col         : primary key column name (pk_col ∉ all_cols for digests)
        secret_key     : secret key K ∈ {0,1}^κ  (paper: κ = 256 bits for HMAC-SHA256)
        iblt_capacity  : d_max = s_ins + s_del; IBLT provisioned for this many differences
        iblt_mult      : α_IBLT; IBLT cell count ℓ = ⌈α_IBLT × d_max⌉  [Eq. 10]
                         Recommended ≥ 1.5 for reliable peeling  [Theorem 3.2]
        iblt_k         : k, number of IBLT hash functions (k=3 standard)
        gt_g           : g, number of group-testing rows/tests  [Eq. 12]
        gt_b           : b, digest/syndrome width in bits  [Eq. 26]
        gt_m           : m, modification tolerance; guarantees |D_mod| ≤ m localizable
        gt_s_del       : s_del, deletion tolerance used for Bernoulli p  [Eq. 11]
        all_cols       : schema-fixed ordered list of non-PK attribute columns;
                         None = all columns except pk_col
        detection_mode : "compressed" (default) — store only IBLT + g syndromes;
                         "full" — additionally store n individual tuple digests.
                         IMPORTANT: compressed mode enforces the paper's storage
                         claim |W| = |S_pk| + |S_grp| = ℓ(c_cnt+λ_pk+λ_fp) + gb  [Eq. 65].
                         Tuple digests are NOT stored in compressed mode.

    Returns:
        ca_record: dict to be stored as JSON with the Certificate Authority.
    """
    if all_cols is None:
        all_cols = [c for c in df.columns if c != pk_col]

    n = len(df)
    import time as _time   # component-level wall-clock instrumentation (§5.10 / S4)
    _t0 = _time.perf_counter()
    timing: Dict[str, float] = {}

    # ── IBLT sizing: ℓ = ⌈α_IBLT × d_max⌉  [Eq. 10] ─────────────────────────
    # iblt_capacity = d_max = s_ins + s_del (maximum primary-key symmetric difference)
    iblt_m = max(10, int(math.ceil(iblt_mult * max(1, iblt_capacity))))

    # ── Layer 1: S_pk(R) = IBLT_K(PK(R))  [Eq. 14] ──────────────────────────
    _t = _time.perf_counter()
    iblt = IBLT(iblt_m, iblt_k)
    original_pks: List[str] = []
    for i in range(n):
        pk_str = str(df.iloc[i][pk_col])
        original_pks.append(pk_str)
        iblt.insert(pk_to_iblt_key(secret_key, pk_str))
    timing["iblt_build_ms"] = (_time.perf_counter() - _t) * 1000.0

    # ── Layer 2a: Per-tuple digests c_i = H_{K,c}(serial(T_i))  [Eq. 26] ────
    _t = _time.perf_counter()
    tuple_digests: Dict[str, int] = {}
    for i in range(n):
        pk_str = str(df.iloc[i][pk_col])
        row = {c: df.iloc[i][c] for c in all_cols if c in df.columns}
        tuple_digests[pk_str] = compute_tuple_digest(
            secret_key, pk_str, row, all_cols, gt_b
        )
    timing["tuple_digest_ms"] = (_time.perf_counter() - _t) * 1000.0

    # ── Layer 2b: Bernoulli XOR syndromes  [Eq. 27] ─────────────────────────
    _t = _time.perf_counter()
    syndromes: List[int] = compute_syndromes(
        secret_key, tuple_digests, gt_g, gt_b, gt_m, gt_s_del
    )
    timing["group_digest_ms"] = (_time.perf_counter() - _t) * 1000.0

    # ── Layer 2c: Single-partition group hashes (compressed-mode alternative) ──
    # Each tuple belongs to exactly one partition group (HMAC % g).
    # group_hash_j = ⊕_{pk in partition j} c_{pk}.
    # Lower localization accuracy than COMP, but deterministic decode.
    # NOT part of the paper's proposed scheme; provided for SP4 comparison.
    group_hashes: List[int] = compute_group_hashes(
        secret_key, tuple_digests, gt_g, gt_b
    )

    # ── Storage estimates [§6, Eq. 65] ───────────────────────────────────────
    # Compressed: |V| = κ + ℓ(c_cnt + λ_pk + λ_fp) + gb
    # Full:       |V| = κ + ℓ(c_cnt + λ_pk + λ_fp) + nb  (W_tuple baseline)
    iblt_storage        = iblt_m * 3 * 8                    # ℓ × 24 bytes
    gt_syndrome_storage = gt_g * max(1, (gt_b + 7) // 8)   # g × b/8 bytes
    digest_storage      = n    * max(1, (gt_b + 7) // 8)   # n × b/8 bytes

    # ── CA record: W(R) = (S_pk(R), S_grp(R))  [Eq. 13] ─────────────────────
    # Compressed mode: store only IBLT + g syndromes — no individual tuple digests.
    # This enforces the paper's storage claim.  Full mode additionally stores
    # n individual digests (W_tuple baseline) for exact per-tuple comparison.
    ca_record: Dict = {
        "method":         "Proposed_IBLT_GT",
        "detection_mode": detection_mode,
        "pk_col":         pk_col,
        "all_cols":       all_cols,
        "n_tuples":       n,

        # ── Layer 1: S_pk(R) = IBLT_K(PK(R))  [Eq. 14] ──────────────────────
        "iblt_params": {
            "m":        iblt_m,
            "k":        iblt_k,
            "capacity": iblt_capacity,   # d_max = s_ins + s_del
        },
        "iblt_cells": {
            "count":    list(iblt._count),
            "key_sum":  list(iblt._keySum),
            "hash_sum": list(iblt._hashSum),
        },
        # PK reverse map: required to translate IBLT integer keys → pk strings
        # (not counted in the paper's |W| formula, as it encodes PK(R) directly)
        "original_pks": original_pks,

        # ── Layer 2: S_grp(R) = {C_1,…,C_g}  [Eq. 27] ───────────────────────
        "gt_params": {
            "g":     gt_g,
            "b":     gt_b,
            "m":     gt_m,
            "s_del": gt_s_del,   # p = 1/(m+s_del+1)  [Eq. 11]
        },
        # Bernoulli group syndromes — always stored (this is the proposed W(R))
        "gt_syndromes": syndromes,

        # Individual tuple digests — stored ONLY in full mode (W_tuple baseline).
        # In compressed mode this is empty: {} (saving n × b bits of storage).
        "tuple_digests": tuple_digests if detection_mode == "full" else {},

        # ── Storage breakdown  [Eq. 65] ───────────────────────────────────────
        "storage_bytes": {
            "iblt":             iblt_storage,
            "gt_syndromes":     gt_syndrome_storage,   # proposed compressed |S_grp|
            "tuple_digests":    digest_storage,        # full-mode |W_tuple| (reference)
            "total_compressed": iblt_storage + gt_syndrome_storage,   # proposed
            "total_full":       iblt_storage + digest_storage,        # W_tuple baseline
        },
    }
    # Component timing for S4 runtime sensitivity (§5.10).
    # Generation total = sum of component phases (matches outer wall-clock to overhead).
    timing["embed_total_ms"] = (_time.perf_counter() - _t0) * 1000.0
    ca_record["embed_timing_ms"] = timing
    return ca_record


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Proposed Scheme — Register (embed) watermark (zero-distortion)"
    )
    parser.add_argument("--input",      required=True,  help="Input CSV database")
    parser.add_argument("--output_dir", default=".",    help="Output directory")
    parser.add_argument("--config",     required=True,  help="Path to params.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df  = load_csv(args.input)

    pk_col     = cfg["pk_col"]
    secret_key = cfg["secret_key"]
    iblt_mult  = float(cfg.get("iblt_mult", 1.5))
    iblt_k     = int(cfg.get("iblt_k", 3))
    gt_b       = int(cfg.get("gt_b", 256))
    gt_m       = int(cfg.get("gt_m", 5))
    gt_s_del   = int(cfg.get("gt_s_del", 5))
    all_cols   = cfg.get("all_cols") or None

    n     = len(df)
    delta = float(cfg.get("delta", 9.54e-7))

    # ── IBLT capacity: storage-optimized (Table 6) or basic (s+r) ─────────────
    # "storage_optimized": C_pk = ⌈1.1 × 0.20 × n⌉  (paper Eq. 56, §7.5)
    # "basic":             C_pk = iblt_capacity from config (= s+r for within-budget)
    iblt_mode = cfg.get("iblt_mode", "basic")
    if iblt_mode == "storage_optimized":
        iblt_capacity = math.ceil(1.1 * 0.20 * n)
        print(f"[Proposed Embed] storage-optimized IBLT: C_pk={iblt_capacity:,}")
    else:
        iblt_capacity = int(cfg.get("iblt_capacity", 10))

    # ── Compute g from paper Eq. 12 (Definition 4.1) ─────────────────────────
    p_grp  = 1.0 / (gt_m + gt_s_del + 1)
    pi_mr  = p_grp * (1 - p_grp) ** (gt_m + gt_s_del)
    numerator = (math.log(n)
                 + math.log(math.comb(n - 1, gt_m))
                 + math.log(1.0 / delta))
    gt_g   = math.ceil(numerator / pi_mr)
    gt_g   = max(gt_g, int(cfg.get("gt_g_min", 1)))
    print(f"[Proposed Embed] n={n:,}  g={gt_g:,}  p={p_grp:.4f}  b={gt_b}  C_pk={iblt_capacity:,}")
    detection_mode = cfg.get("detection_mode", "compressed")

    ca_record = embed_watermark(
        df, pk_col, secret_key,
        iblt_capacity, iblt_mult, iblt_k,
        gt_g, gt_b, gt_m, gt_s_del,
        all_cols=all_cols,
        detection_mode=detection_mode,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    ca_path = os.path.join(args.output_dir, "ca_registration.json")
    save_json(ca_record, ca_path)

    # Zero-distortion: watermarked DB is identical to the original
    wm_path = os.path.join(args.output_dir, "watermarked.csv")
    save_csv(df, wm_path)

    n  = len(df)
    sb = ca_record["storage_bytes"]
    p  = 1.0 / (gt_m + gt_s_del + 1)   # Bernoulli p  [Eq. 11]
    print(f"[Proposed Embed] Registered {n} tuples → {ca_path}")
    print(f"[Proposed Embed] IBLT : ℓ={ca_record['iblt_params']['m']} cells, "
          f"k={iblt_k}, d_max={iblt_capacity}")
    print(f"[Proposed Embed] GT   : g={gt_g}, b={gt_b}, m={gt_m}, "
          f"s_del={gt_s_del}, p={p:.4f}")
    print(f"[Proposed Embed] Storage (compressed): {sb['total_compressed']} bytes  "
          f"= IBLT ({sb['iblt']}B) + syndromes ({sb['gt_syndromes']}B)")
    print(f"[Proposed Embed] Storage (full)      : {sb['total_full']} bytes  "
          f"= IBLT ({sb['iblt']}B) + digests ({sb['tuple_digests']}B)")
    print(f"[Proposed Embed] Watermarked DB (zero-distortion copy) → {wm_path}")


if __name__ == "__main__":
    main()
