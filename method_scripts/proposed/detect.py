"""
Proposed Scheme: IBLT + Group-Testing Fragile Watermarking — Verification (Detect)

Implements the Verify algorithm from Definition 5.1 of the paper.

Two-layer detection pipeline:

  Layer 1 — Primary-key reconciliation  [§5.1, Eqs. 28–32]:
    1. Build IBLT_K(PK(R′)) from the suspicious database.
    2. ΔS_pk = S_pk(R) − S_pk(R′)  [Eq. 29]
    3. Peeling decode → D̂_del = PK(R) \ PK(R′),  D̂_ins = PK(R′) \ PK(R)  [Eqs. 30–31]
    4. Matched key set: PK_mat = PK(R) ∩ PK(R′) = PK(R′) \ D̂_ins  [Eq. 32]

  Layer 2 — Group-based modification localisation  [§5.2, Eqs. 33–38]:
    Only clean groups (D_a^del = ∅) are used; contaminated groups are discarded  [§5.2.1].

    Full mode (detection_mode = "full"):
      Compare stored digest c_i vs recomputed c_i′ = H_{K,c}(serial(T′(pk))) for
      every pk ∈ PK_mat. Reports D̂_mod = {pk : c_i ≠ c_i′}.
      Zero false positives; zero false negatives (under PRF assumption).
      Storage: n × b bits (tuple-wise baseline W_tuple from §3.5).

    Compressed mode (detection_mode = "compressed")  [§5.2.3]:
      Uses the g stored Bernoulli XOR syndromes (S_grp(R)) instead of n digests.
      Steps:
        a. Discard groups containing recovered deleted primary keys.
        b. Recompute clean-group syndromes C_a′ over PK_mat from suspicious digests.
        c. delta_a = C_a ⊕ C_a′ on clean groups; test a is positive iff delta_a ≠ 0.
        d. Elimination decoder  [Eq. 38]: flag pk when no clean negative group certifies it.
      Guarantee (Theorem 5.2): if |D_mod| ≤ m, this recovers D_mod exactly
      (except with probability δ + mg × 2^{−b})  [Theorem 5.3].
      Storage: g × b bits (compressed proposal — main paper contribution).

  Output: D̂_ins, D̂_del, D̂_mod as sets of primary-key strings.

Paper reference: §5 (Watermark Verification), Definition 5.1, Eqs. 28–42.
"""

import argparse
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))        # baselines/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))     # project root (iblt.py)

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from common.utils import (
    load_csv, load_json, save_json, load_config, hash_value
)
from common.metrics import compute_localization_metrics
from proposed.iblt import IBLT
from proposed.embed import (
    pk_to_iblt_key,
    compute_tuple_digest,
    gt_matrix_entry,
    get_group_assign,
    _pk_seeds,
    _group_offsets,
    _CHUNK_SIZE,        # chunk size for memory-efficient (n×g) matrix ops
    _chunk_for_g,       # adaptive chunk size (large g → smaller chunk to avoid OOM)
    membership_matrix,  # boolean (n×g) membership matrix for group-testing decode
    _mix64,             # fmix64 finalizer — must match embed-side membership PRF
)


# ── IBLT restore ──────────────────────────────────────────────────────────────

def restore_iblt(cells: Dict, m: int, k: int) -> IBLT:
    """Reconstruct the stored IBLT S_pk(R) from the CA record cell arrays.

    Paper §5.1: The verifier has stored S_pk(R) = IBLT_K(PK(R))  [Eq. 14].
    Since the IBLT is serialised to JSON as three integer arrays (count, keySum,
    hashSum), this function restores the IBLT object by directly loading those arrays.

    Args:
        cells : dict with keys "count", "key_sum", "hash_sum" (from CA record)
        m     : number of IBLT cells ℓ
        k     : number of hash functions k

    Returns:
        Restored IBLT object representing S_pk(R).
    """
    iblt = IBLT(m, k)
    iblt._count   = list(cells["count"])
    iblt._keySum  = list(cells["key_sum"])
    iblt._hashSum = list(cells["hash_sum"])
    return iblt


# ── COMP decoder  [§5.2.3, Eq. 38] ──────────────────────────────────────────

def comp_decode(
    n_tests: int,
    deltas: List[int],
    stable_pks: List[str],
    secret_key: str,
    gt_m: int,
    gt_s_del: int = 0,
    clean_groups: Optional[set] = None,
    _offsets: Optional[np.ndarray] = None,
) -> Set[str]:
    """Elimination decoder for the Bernoulli group-testing layer.

    Paper §5.2.3 (Elimination Decoder on Clean Groups), Eqs. 37–38:

      B(pk) = {a : A_{a,pk}=1  and  D_a^del=∅}            [Eq. 37]
      D̂_mod = {pk ∈ PK_mat : B(pk)=∅  or  y_a=1 ∀a∈B(pk)}  [Eq. 38]

    Intuition:
      - y_a = 1[delta_a ≠ 0]: group a is abnormal (positive test)  [Eq. 36].
      - B(pk): the set of clean covering groups for pk — groups that contain pk
        and contain no deleted tuple (so their stored digest is still valid).
      - A clean covering group with y_a = 0 CERTIFIES pk as intact:
        if pk were modified, its digest change would make delta_a ≠ 0.
      - pk is declared modified iff NO clean covering group certifies it as intact,
        i.e., B(pk) = ∅ (uncovered) or every group in B(pk) is positive.

    Correctness guarantee (Theorem 5.2):
      Under Theorem 4.4 (clean-separating property) and |D_mod| ≤ m:
      - Every modified pk: all its covering groups contain the modification
        → all are positive → no group certifies it → correctly flagged.
      - Every unmodified pk: by Theorem 4.4, ∃ a clean covering group G_a
        with y_a = 0 → it certifies pk as intact → correctly excluded.
      Hence D̂_mod = D_mod exactly, except with probability mg × 2^{−b}
      (XOR cancellation, Theorem 5.3  [Eq. 41]).

    Note: this is the ELIMINATION rule (exclude if certified intact), NOT the
    classical COMP "include if in all positive tests" rule from Du-Hwang.
    The two are equivalent in the ideal OR model but differ under XOR outcomes.

    Args:
        n_tests     : total number of tests g
        deltas      : delta_a = C_a ⊕ C_a′ for a=0..g−1  [Eq. 36]
        stable_pks  : PK_mat = PK(R) ∩ PK(R′) = PK(R) \ D̂_del  [Eq. 32]
        secret_key  : secret key K
        gt_m        : modification tolerance m
        gt_s_del    : deletion tolerance s_del (Bernoulli p = 1/(m+s_del+1))
        clean_groups: set of group indices with D_a^del=∅  [§5.2.1, Eq. 34].
                      None = all groups are treated as clean (deletion-free case).

    Returns:
        D̂_mod: set of PK strings declared value-modified.
    """
    valid_list = sorted(clean_groups) if clean_groups is not None else list(range(n_tests))
    valid_arr  = np.array(valid_list, dtype=np.int64)

    if not stable_pks or len(valid_list) == 0:
        return set(stable_pks)  # no clean groups → flag everything

    stable_list   = list(stable_pks)
    n_stable      = len(stable_list)
    offsets_all   = _offsets if _offsets is not None else _group_offsets(secret_key, n_tests)
    offsets_clean = offsets_all[valid_arr]   # (|clean|,) uint64

    # Which clean groups are positive (delta ≠ 0)?
    positive = np.array([deltas[int(a)] != 0 for a in valid_arr], dtype=bool)

    # Chunked: process _CHUNK_SIZE stable PKs at a time to avoid (n × |clean|)
    # uint64 matrix that would be n × g × 8 bytes (up to ~12 GB for FCT).
    denom = np.uint64(gt_m + gt_s_del + 1)
    row_has_covering   = np.zeros(n_stable, dtype=bool)
    row_has_clean_cert = np.zeros(n_stable, dtype=bool)
    chunk_sz = _chunk_for_g(len(offsets_clean))   # adaptive on #clean groups

    for start in range(0, n_stable, chunk_sz):
        end = min(start + chunk_sz, n_stable)
        chunk_seeds = _pk_seeds(secret_key, stable_list[start:end])
        combined    = chunk_seeds[:, None] ^ offsets_clean[None, :]   # (chunk, |clean|)
        A_chunk     = (_mix64(combined) % denom) == np.uint64(0)      # fmix64 PRF
        del combined
        row_has_covering[start:end]   = A_chunk.any(axis=1)
        row_has_clean_cert[start:end] = (A_chunk & ~positive[None, :]).any(axis=1)
        del A_chunk

    # Elimination rule [Eq. 38]:
    flagged = (~row_has_covering) | (~row_has_clean_cert)

    return {stable_list[i] for i in np.where(flagged)[0]}


# ── Main detection function ───────────────────────────────────────────────────

def compute_theorem_2_7_counts(
    ca_record: Dict,
    secret_key: str,
    s_del_true: Set[str],
    s_mod_true: Set[str],
) -> Dict[str, int]:
    """Compute the five internal counts of Theorem 2.7 for one realized attack.

    Used by the S2 sensitivity experiment (diagnostic-only — these counts are
    not available to a deployed verifier). All work is in-memory; no large
    intermediate structures are persisted.

      N_g_clean: groups with no deleted tuple
      N_g_pure : groups with no deleted AND no modified tuple
      N_t_cov  : surviving tuples covered by ≥1 deletion-clean group
      N_t_cert : intact surviving tuples covered by ≥1 pure intact group
      N_U      : (n - q_del) - N_t_cov

    Returns a dict with the 5 integers and (n, g, p, q_del, q_mod, q_int).
    """
    gt_params = ca_record["gt_params"]
    gt_g  = gt_params["g"]
    gt_m  = gt_params["m"]
    gt_s_del = gt_params.get("s_del", 0)
    n = ca_record["n_tuples"]
    pks_all = list(ca_record["original_pks"])
    pks_del = set(s_del_true)
    pks_mod = set(s_mod_true)
    pks_int = [pk for pk in pks_all if pk not in pks_del and pk not in pks_mod]
    pks_surv = [pk for pk in pks_all if pk not in pks_del]

    offsets = _group_offsets(secret_key, gt_g)
    denom_b = np.uint64(gt_m + gt_s_del + 1)

    # Membership of deleted PKs across all groups → flags per group
    def _membership(pks):
        if not pks:
            return np.zeros((0, gt_g), dtype=bool)
        seeds = _pk_seeds(secret_key, list(pks))
        combined = seeds[:, None] ^ offsets[None, :]
        return (_mix64(combined) % denom_b) == np.uint64(0)

    A_del = _membership(list(pks_del))           # (|del|, g)
    A_mod = _membership(list(pks_mod))           # (|mod|, g)
    group_has_del = A_del.any(axis=0) if A_del.size else np.zeros(gt_g, dtype=bool)
    group_has_mod = A_mod.any(axis=0) if A_mod.size else np.zeros(gt_g, dtype=bool)
    n_g_clean = int((~group_has_del).sum())
    n_g_pure  = int((~group_has_del & ~group_has_mod).sum())

    # For surviving tuples: covered by ≥1 deletion-clean group?
    n_t_cov = 0
    if pks_surv:
        CHUNK = _chunk_for_g(gt_g)   # adaptive: keeps (chunk × g) peak ≈ budget
        for start in range(0, len(pks_surv), CHUNK):
            chunk = pks_surv[start:start+CHUNK]
            seeds = _pk_seeds(secret_key, chunk)
            combined = seeds[:, None] ^ offsets[None, :]
            A_surv = (_mix64(combined) % denom_b) == np.uint64(0)  # (chunk, g)
            # covered by a deletion-clean group
            cov = (A_surv & (~group_has_del[None, :])).any(axis=1)
            n_t_cov += int(cov.sum())

    # For intact surviving tuples: covered by ≥1 pure intact group?
    n_t_cert = 0
    if pks_int:
        CHUNK = _chunk_for_g(gt_g)   # adaptive: keeps (chunk × g) peak ≈ budget
        for start in range(0, len(pks_int), CHUNK):
            chunk = pks_int[start:start+CHUNK]
            seeds = _pk_seeds(secret_key, chunk)
            combined = seeds[:, None] ^ offsets[None, :]
            A_int = (_mix64(combined) % denom_b) == np.uint64(0)
            pure_mask = (~group_has_del & ~group_has_mod)
            cert = (A_int & pure_mask[None, :]).any(axis=1)
            n_t_cert += int(cert.sum())

    n_q_del = len(pks_del)
    n_q_mod = len(pks_mod)
    n_u = (n - n_q_del) - n_t_cov

    return {
        "n": n, "g": gt_g, "p": 1.0/(gt_m + gt_s_del + 1),
        "q_del": n_q_del, "q_mod": n_q_mod,
        "N_g_clean": n_g_clean,
        "N_g_pure":  n_g_pure,
        "N_t_cov":   n_t_cov,
        "N_t_cert":  n_t_cert,
        "N_U":       int(n_u),
    }


def detect_watermark(
    df_suspicious: pd.DataFrame,
    ca_record: Dict,
    secret_key: str,
    detection_mode: str = "full",
) -> Dict:
    """Run the Verify algorithm against a suspect database R′.

    Paper Definition 5.1:
      Input : R′, W(R) = (S_pk(R), S_grp(R)), K, Π
      Output: 'authentic'  if no tampering detected
              (D̂_ins, D̂_del, D̂_mod)  otherwise

    Args:
        df_suspicious  : suspect database R′ as a DataFrame
        ca_record      : CA registration dict from embed_watermark()
        secret_key     : secret key K (same used during registration)
        detection_mode : "full"       → direct per-tuple digest comparison [§3.5]
                         "compressed" → Bernoulli COMP decoder  [§5.2.3]

    Returns:
        Result dict with db_tampered, s_ins, s_del, s_mod, iblt_decode_success.
    """
    import time as _time   # component-level instrumentation for S4 runtime
    _t0 = _time.perf_counter()
    timing: Dict[str, float] = {}

    pk_col   = ca_record["pk_col"]
    all_cols = ca_record["all_cols"]
    # Always read detection_mode from ca_record so embed and detect are consistent.
    # The function parameter is kept for backward compatibility but ca_record takes precedence.
    detection_mode = ca_record.get("detection_mode", detection_mode)

    iblt_params = ca_record["iblt_params"]
    gt_params   = ca_record["gt_params"]
    gt_g        = gt_params["g"]
    gt_b        = gt_params["b"]
    gt_m        = gt_params["m"]
    gt_s_del    = gt_params.get("s_del", 0)   # paper Eq. 11: p = 1/(m+s_del+1)
    mask        = (1 << gt_b) - 1

    # ── Layer 1: Primary-key reconciliation  [§5.1] ───────────────────────────
    _t = _time.perf_counter()

    # Restore S_pk(R) = IBLT_K(PK(R)) from CA  [Eq. 14]
    orig_iblt = restore_iblt(
        ca_record["iblt_cells"], iblt_params["m"], iblt_params["k"]
    )

    # Build S_pk(R′) = IBLT_K(PK(R′))  [Eq. 28]
    susp_iblt = IBLT(iblt_params["m"], iblt_params["k"])
    susp_pk_list: List[str] = []
    for i in range(len(df_suspicious)):
        pk_str = str(df_suspicious.iloc[i][pk_col])
        susp_pk_list.append(pk_str)
        susp_iblt.insert(pk_to_iblt_key(secret_key, pk_str))

    # ΔS_pk = S_pk(R) − S_pk(R′)  [Eq. 29]
    # By Eq. 9: IBLT_K(X) − IBLT_K(Y) = IBLT_K(X △ Y)
    # So ΔS_pk encodes PK(R) △ PK(R′)
    diff = orig_iblt.subtract(susp_iblt)

    timing["iblt_diff_ms"] = (_time.perf_counter() - _t) * 1000.0
    _t = _time.perf_counter()

    # Peeling decode → (D̂_del, D̂_ins)  [Eqs. 30–31]
    deleted_keys, inserted_keys, iblt_success = diff.decode()
    timing["iblt_peel_ms"] = (_time.perf_counter() - _t) * 1000.0

    # Map IBLT integer keys back to PK strings
    # Original: HMAC key → pk string (from CA's original_pks)
    key_to_orig_pk: Dict[int, str] = {
        pk_to_iblt_key(secret_key, pk_str): pk_str
        for pk_str in ca_record["original_pks"]
    }
    # Suspicious: HMAC key → pk string (from R′)
    key_to_susp_pk: Dict[int, str] = {
        pk_to_iblt_key(secret_key, pk_str): pk_str
        for pk_str in susp_pk_list
    }

    # D̂_del = PK(R) \ PK(R′)  [Eq. 30]
    s_del: Set[str] = {key_to_orig_pk[k] for k in deleted_keys  if k in key_to_orig_pk}
    # D̂_ins = PK(R′) \ PK(R)  [Eq. 31]
    s_ins: Set[str] = {key_to_susp_pk[k] for k in inserted_keys if k in key_to_susp_pk}

    if not iblt_success:
        # Layer 2 is defined only after successful primary-key reconciliation.
        # Keep top-level prediction sets empty so downstream metric code can run
        # and score the trial as a localization failure, while preserving partial
        # peeled keys under diagnostic fields.
        return {
            "method":              "Proposed_IBLT_GT",
            "detection_mode":      detection_mode,
            "status":              "iblt_decode_failed",
            "localization_valid":  False,
            "failure_reason":      "iblt_decode_failed",
            "n_tuples_original":   ca_record["n_tuples"],
            "n_tuples_suspicious": len(df_suspicious),
            "iblt_decode_success": False,
            "db_tampered":         True,
            "s_ins":               [],
            "s_del":               [],
            "s_mod":               [],
            "n_ins":               0,
            "n_del":               0,
            "n_mod":               0,
            "stable_set_size":     0,
            "s_ins_partial":       sorted(s_ins),
            "s_del_partial":       sorted(s_del),
            "n_ins_partial":       len(s_ins),
            "n_del_partial":       len(s_del),
        }

    # ── Layer 2 setup  [§5.2] ─────────────────────────────────────────────────

    # PK_mat = PK(R) ∩ PK(R′) = PK(R) \ D̂_del  [Eq. 32]
    # Layer 2 operates only on these matched primary keys.
    # Inserted primary keys are excluded (they have no original digest).
    orig_pk_set = set(ca_record["original_pks"])
    stable_pks  = orig_pk_set - s_del   # PK_mat

    # Stored digests c_i = H_{K,c}(serial(T_i))  [Eq. 26] (from CA)
    stored_digests_raw = ca_record.get("tuple_digests", {})
    stored_digests: Dict[str, int] = {k: int(v) for k, v in stored_digests_raw.items()}

    # Guard against schema changes between embed and detect
    available_cols = [c for c in all_cols if c in df_suspicious.columns]

    # Index R′ by PK string for O(1) row lookup
    susp_pk_to_idx: Dict[str, int] = {
        str(df_suspicious.iloc[i][pk_col]): i
        for i in range(len(df_suspicious))
    }

    # Recompute c_i′ = H_{K,c}(serial(T′(pk))) for all pk ∈ PK_mat  [Eq. 35]
    _t = _time.perf_counter()
    susp_digest_map: Dict[str, int] = {}
    for pk_str in stable_pks:
        idx = susp_pk_to_idx.get(pk_str)
        if idx is None:
            continue
        row = {c: df_suspicious.iloc[idx][c] for c in available_cols}
        susp_digest_map[pk_str] = compute_tuple_digest(
            secret_key, pk_str, row, available_cols, gt_b
        )
    timing["group_recompute_ms"] = (_time.perf_counter() - _t) * 1000.0
    _t = _time.perf_counter()

    s_mod: Set[str] = set()

    # ── Detection mode: full (direct digest comparison)  [§3.5 baseline] ──────
    # For each pk ∈ PK_mat: if c_i ≠ c_i′ then pk ∈ D̂_mod.
    # This is equivalent to storing W_tuple(R) = {H_K(serial(T(pk))) : pk ∈ PK(R)}
    # from §3.5. Zero false positives and zero false negatives under PRF assumption.
    # Storage cost: n × b bits (NOT the proposed compressed approach).
    if detection_mode == "full":
        for pk_str in stable_pks:
            orig_d = stored_digests.get(pk_str)
            susp_d = susp_digest_map.get(pk_str)
            if orig_d is None or susp_d is None:
                continue
            if orig_d != susp_d:
                s_mod.add(pk_str)

    # ── Detection mode: compressed (Bernoulli group-testing)  [§5.2, §5.2.3] ──
    # Uses only the g stored syndromes S_grp(R) = {C_1,…,C_g}.
    # No individual tuple digests are needed — enforcing the paper's storage claim.
    # Storage cost: g × b bits  [Eq. 65: |V| = κ + ℓ(c_cnt+λ_pk+λ_fp) + gb].
    #
    # Algorithm (paper §5.2.1–§5.2.3):
    #   Step a. Identify contaminated groups: G_a is contaminated if any
    #           pk ∈ D̂_del has A_{a,pk}=1  [Eq. 34, §5.2.1].
    #           Discard contaminated groups entirely — do NOT adjust syndromes.
    #           This is the paper's approach; it does not require tuple digests.
    #   Step b. For each clean group a: recompute C_a′ from matched tuples  [Eq. 35].
    #   Step c. delta_a = C_a ⊕ C_a′; y_a = 1[delta_a ≠ 0]  [Eq. 36].
    #   Step d. Elimination decoder on clean covering groups  [Eqs. 37–38].
    #
    # Guarantee (Theorem 5.2): D̂_mod = D_mod exactly when |D_mod| ≤ m,
    # except with probability δ + mg × 2^{−b}  [Theorem 5.3].
    else:  # detection_mode == "compressed"
        stored_syndromes_raw = ca_record.get("gt_syndromes", [])
        stored_syndromes = [int(s) for s in stored_syndromes_raw]

        if not stored_syndromes:
            raise ValueError(
                "Compressed mode requires 'gt_syndromes' in CA record. "
                "Re-embed with detection_mode='compressed'."
            )

        # Pre-compute group offsets once (g HMAC calls, shared by steps a–d)
        offsets = _group_offsets(secret_key, gt_g)

        # Step a: Identify contaminated groups  [§5.2.1, Eq. 34].
        # Vectorized: build membership rows for all deleted PKs at once.
        contaminated: set = set()
        if s_del:
            del_list  = list(s_del)
            del_seeds = _pk_seeds(secret_key, del_list)
            A_del = membership_matrix(del_seeds, offsets, gt_m, gt_s_del)
            contaminated = set(int(a) for a in np.where(A_del.any(axis=0))[0])
        clean_groups = [a for a in range(gt_g) if a not in contaminated]
        clean_set    = set(clean_groups)

        # Step b: Recompute C_a′ = ⊕_{pk ∈ PK_mat : A_{a,pk}=1} c_pk′  [Eq. 35].
        # Chunked: process _CHUNK_SIZE rows at a time to avoid (n×g) uint64 matrix.
        susp_syndromes = np.zeros(gt_g, dtype=np.uint64)
        if susp_digest_map and clean_groups:
            susp_pks     = list(susp_digest_map.keys())
            susp_digests = np.fromiter(
                (int(d) for d in susp_digest_map.values()),
                dtype=np.uint64, count=len(susp_digest_map))
            n_susp       = len(susp_pks)
            clean_arr    = np.array(sorted(clean_groups), dtype=np.int64)
            n_clean      = len(clean_arr)
            offsets_clean = offsets[clean_arr]  # (n_clean,) uint64
            denom_b      = np.uint64(gt_m + gt_s_del + 1)
            chunk_sz     = _chunk_for_g(n_clean)   # adaptive on #clean groups

            for start in range(0, n_susp, chunk_sz):
                end = min(start + chunk_sz, n_susp)
                chunk_seeds = _pk_seeds(secret_key, susp_pks[start:end])
                combined    = chunk_seeds[:, None] ^ offsets_clean[None, :]
                A_chunk     = (_mix64(combined) % denom_b) == np.uint64(0)  # fmix64 PRF
                del combined
                rows, cols  = np.where(A_chunk)
                del A_chunk
                # Grouped XOR-reduction in C [Eq. 35]: for every membership (row, col),
                # susp_syndromes[clean_arr[col]] ^= digest[row]. Replaces the former
                # per-membership Python loop (~n·g·p iterations) that dominated runtime
                # at large g — e.g. ~1.2e9 iters for FCT at rho=10% (g=331k), the cause
                # of multi-hour detects in the sensitivity high-rho sweep.
                np.bitwise_xor.at(
                    susp_syndromes, clean_arr[cols], susp_digests[start + rows])

        # Step c: delta_a = C_a ⊕ C_a′ for clean groups; 0 for contaminated  [Eq. 36].
        # Contaminated groups get delta=0 and are effectively ignored by the decoder.
        deltas = [0] * gt_g
        for a in clean_groups:
            deltas[a] = (stored_syndromes[a] ^ int(susp_syndromes[a])) & mask

        # Step d: Elimination decoder on clean covering groups B(pk)  [Eqs. 37–38].
        # Pass pre-computed offsets to avoid recomputing g HMAC calls inside.
        s_mod = comp_decode(
            gt_g, deltas, list(stable_pks), secret_key, gt_m, gt_s_del,
            clean_groups=set(clean_groups),
            _offsets=offsets,
        )

    timing["decoder_ms"] = (_time.perf_counter() - _t) * 1000.0
    timing["detect_total_ms"] = (_time.perf_counter() - _t0) * 1000.0

    # ── Combined tampering verdict  [Definition 5.1] ──────────────────────────
    # db_tampered = True iff any of D̂_ins, D̂_del, D̂_mod is non-empty
    db_tampered = bool(s_del or s_ins or s_mod)

    result: Dict = {
        "detect_timing_ms": timing,
        "method":              "Proposed_IBLT_GT",
        "detection_mode":      detection_mode,
        "status":              "authentic" if not db_tampered else "tampered",
        "localization_valid":  True,
        "failure_reason":      None,
        # n = |R| (original);  |R′| = n_tuples_suspicious
        "n_tuples_original":   ca_record["n_tuples"],
        "n_tuples_suspicious": len(df_suspicious),
        # iblt_decode_success: False → D̂_del/D̂_ins may be incomplete  [Theorem 3.2]
        "iblt_decode_success": iblt_success,
        "db_tampered":         db_tampered,
        # Operation-aware output: separate estimates for ins/del/mod  [Eq. 3.4]
        "s_ins":               sorted(s_ins),   # D̂_ins  [Eq. 31]
        "s_del":               sorted(s_del),   # D̂_del  [Eq. 30]
        "s_mod":               sorted(s_mod),   # D̂_mod  [Eq. 38]
        "n_ins":               len(s_ins),
        "n_del":               len(s_del),
        "n_mod":               len(s_mod),
        "stable_set_size":     len(stable_pks),  # |PK_mat|  [Eq. 32]
    }
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Proposed Scheme — Detect watermark tampering"
    )
    parser.add_argument("--input",          required=True, help="Suspicious CSV")
    parser.add_argument("--ca_record",      required=True, help="CA registration JSON")
    parser.add_argument("--config",         required=True, help="Path to params.yaml")
    parser.add_argument("--output_dir",     default=".",   help="Directory for result JSON")
    parser.add_argument("--detection_mode", default=None,
                        help="Override detection_mode: full|compressed")
    # Ground-truth for per-operation localization metrics (optional)
    parser.add_argument("--true_s_ins",  default=None,
                        help="Comma-separated PKs of truly inserted tuples (D_ins)")
    parser.add_argument("--true_s_del",  default=None,
                        help="Comma-separated PKs of truly deleted tuples (D_del)")
    parser.add_argument("--true_s_mod",  default=None,
                        help="Comma-separated PKs of truly modified tuples (D_mod)")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    df         = load_csv(args.input)
    ca_record  = load_json(args.ca_record)

    secret_key     = cfg["secret_key"]
    detection_mode = args.detection_mode or cfg.get("detection_mode", "full")

    result = detect_watermark(df, ca_record, secret_key, detection_mode)

    # ── Optional ground-truth evaluation ──────────────────────────────────────
    all_pks  = set(str(df.iloc[i][ca_record["pk_col"]]) for i in range(len(df)))
    orig_pks = set(ca_record["original_pks"])
    universe = all_pks | orig_pks   # PK(R) ∪ PK(R′)

    if args.true_s_ins or args.true_s_del or args.true_s_mod:
        true_s_ins = set(args.true_s_ins.split(",")) if args.true_s_ins else set()
        true_s_del = set(args.true_s_del.split(",")) if args.true_s_del else set()
        true_s_mod = set(args.true_s_mod.split(",")) if args.true_s_mod else set()
        true_tampered = true_s_ins | true_s_del | true_s_mod  # D_all  [§3.3 Eq. 4 union]

        pred_tampered = set(result["s_ins"]) | set(result["s_del"]) | set(result["s_mod"])

        # Precision_x, Recall_x, F1_x for x ∈ {all, ins, del, mod}  [§7.7]
        result["localization_metrics"] = {
            "combined": compute_localization_metrics(true_tampered, pred_tampered, universe),
            "s_ins":    compute_localization_metrics(true_s_ins, set(result["s_ins"]),  universe),
            "s_del":    compute_localization_metrics(true_s_del, set(result["s_del"]),  universe),
            "s_mod":    compute_localization_metrics(true_s_mod, set(result["s_mod"]),  universe),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "detect_result.json")
    save_json(result, out_path)

    print(f"[Proposed Detect] DB tampered      : {result['db_tampered']}")
    print(f"[Proposed Detect] IBLT decode OK   : {result['iblt_decode_success']}")
    print(f"[Proposed Detect] D̂_ins ({result['n_ins']:3d})   : "
          f"{result['s_ins'][:5]}{'...' if result['n_ins'] > 5 else ''}")
    print(f"[Proposed Detect] D̂_del ({result['n_del']:3d})   : "
          f"{result['s_del'][:5]}{'...' if result['n_del'] > 5 else ''}")
    print(f"[Proposed Detect] D̂_mod ({result['n_mod']:3d})   : "
          f"{result['s_mod'][:5]}{'...' if result['n_mod'] > 5 else ''}")
    print(f"[Proposed Detect] Result saved → {out_path}")


if __name__ == "__main__":
    main()
