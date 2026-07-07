"""
B7: Sun et al. 2020 — Two-Level Group Eigenvalue Fragile Watermarking (Embed)

Algorithm overview (zero-distortion; mixed data: numeric + textual):

Data Grouping (two-level):
  Level 1 — Teams:
    Sort all tuples by primary key ascending.
    Take γ tuples at a time to form a "team".  t = ceil(α / γ) teams total.
    (Last team padded with first-team tuples if incomplete.)

  Level 2 — Groups:
    n = argmin{i ∈ N | i*16 - i! + 1 < 0}  →  n = 5  (fixed by the formula)
    Take n teams at a time to form a "group". G = ceil(t / n) groups total.

Eigenvalue extraction per team (generalised from material-gene-DB to arbitrary CSV):
  Wd : digit frequency (digits 0–9) from numeric column values in the team
  Wt : char frequency (a–z) + distinct text-length frequency from text column values
  Wr  = Wd || Wt    (concatenated as a list of counts)
  EWr = SHA-256(Wr) → 256-bit hash hex string  (team eigenvalue)

Watermark generation:
  W_team  = MSB (bit 7 of byte 0) of EWr              → 1-bit team watermark
  WB_group = sum of W_team for all teams in the group  → integer in {0,…,n}
  WD_team  = W_team bit  →  0 = ascending tuple sort, 1 = descending tuple sort

Watermark embedding (zero-distortion — order only):
  Group-level (encode WB into team ordering):
    Start from identity permutation π = [0, 1, …, n-1].
    Apply unrank(n, WB, π)  [Myrvold-Ruskey 2001] to obtain target permutation.
    Reorder teams within the group according to π.

  Tuple-level (encode WD into tuple ordering within each team):
    Compute h_i = HMAC(Ks || pk_i || Ks) for each tuple i in the team.
    If WD = 0: sort tuples ascending by h_i.
    If WD = 1: sort tuples descending by h_i.

CA registration: store eigenvalues, WB/WD, team/group structure (no value modification).

Usage:
    python embed.py --input db.csv --output_dir results/ --config config/params.yaml
"""

import argparse
import hashlib
import json
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from common.utils import (
    load_csv, save_json, load_config, hash_value, get_numeric_cols, get_text_cols
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHABET = list('abcdefghijklmnopqrstuvwxyz')


def compute_n_teams_per_group() -> int:
    """Find n = argmin{i ∈ N | i*16 - i! + 1 < 0}.

    For i=5: 5*16 - 5! + 1 = 80 - 120 + 1 = -39 < 0  → n = 5.
    """
    factorial = 1
    for i in range(1, 20):
        factorial *= i
        if i * 16 - factorial + 1 < 0:
            return i
    return 5  # should never reach here


N_TEAMS_PER_GROUP = compute_n_teams_per_group()  # = 5


# ---------------------------------------------------------------------------
# Myrvold-Ruskey rank / unrank (linear-time permutation coding)
# ---------------------------------------------------------------------------

def _unrank_inplace(n: int, r: int, perm: List[int]) -> None:
    """Myrvold-Ruskey unrank: given rank r, produce permutation in perm[0..n-1].

    Modifies perm in-place.  perm should start as [0, 1, …, len(perm)-1].
    The rank r must be in [0, n!-1].
    """
    if n > 0:
        perm[n - 1], perm[r % n] = perm[r % n], perm[n - 1]
        _unrank_inplace(n - 1, r // n, perm)


def unrank_permutation(n: int, r: int) -> List[int]:
    """Return the permutation of [0..n-1] with Myrvold-Ruskey rank r."""
    perm = list(range(n))
    _unrank_inplace(n, r, perm)
    return perm


def rank_permutation(perm: List[int]) -> int:
    """Compute Myrvold-Ruskey rank of a permutation.

    perm must be a list containing exactly the values [0..n-1] in some order.
    Returns rank r ∈ [0, n!-1].
    """
    n = len(perm)
    perm = list(perm)
    inv = [0] * n
    for i in range(n):
        inv[perm[i]] = i

    r = 0
    factor = 1
    for i in range(1, n):
        # After (n-i) calls to rank step, accumulate
        pass

    # Use the iterative interpretation of the Myrvold-Ruskey rank algorithm:
    # rank(n, π, π⁻¹):
    #   if n == 1: return 0
    #   s = π[n-1]
    #   swap(π[n-1], π[π⁻¹[n-1]])
    #   swap(π⁻¹[s], π⁻¹[n-1])
    #   return s + n * rank(n-1, π, π⁻¹)
    # This gives r = s_{n-1} + n*(s_{n-2} + (n-1)*(...))
    # where s_i = π[i] before swapping.
    # We accumulate from the innermost outward.

    # Recompute using iterative approach
    perm2 = list(perm)
    inv2 = list(inv)
    r = 0
    for step in range(n, 1, -1):
        s = perm2[step - 1]
        # swap perm2[step-1] and perm2[inv2[step-1]]
        j = inv2[step - 1]
        perm2[step - 1], perm2[j] = perm2[j], perm2[step - 1]
        inv2[s], inv2[step - 1] = inv2[step - 1], inv2[s]
        r += s
        if step > 1:
            r_so_far = r
            r = r_so_far  # accumulated in factorial number system below

    # Actually build the rank correctly using the Myrvold-Ruskey recurrence:
    perm3 = list(perm)
    inv3 = [0] * n
    for i in range(n):
        inv3[perm3[i]] = i

    r = 0
    multiplier = 1
    # rank = s_{n-1} + n * (s_{n-2} + (n-1) * (... + 2 * s_1))
    # We compute from inner to outer:
    values = []
    p = list(perm3)
    inv_p = list(inv3)
    for step in range(n, 1, -1):
        s = p[step - 1]
        j = inv_p[step - 1]
        p[step - 1], p[j] = p[j], p[step - 1]
        inv_p[s], inv_p[step - 1] = inv_p[step - 1], inv_p[s]
        values.append((s, step))

    # Build rank: values = [(s_{n-1}, n), (s_{n-2}, n-1), ...]
    # rank = s_{n-1} + n * (s_{n-2} + (n-1) * (...))
    r = 0
    for s, step in reversed(values):
        r = s + step * r
    return r


# ---------------------------------------------------------------------------
# Eigenvalue extraction
# ---------------------------------------------------------------------------

def extract_team_eigenvalue(
    team_df: pd.DataFrame,
    numeric_cols: List[str],
    text_cols: List[str],
) -> Tuple[List[int], str]:
    """Compute team eigenvalue Wr and its SHA-256 hash EWr.

    Wr = Wd || Wt  (digit freq 0-9 from numeric, char freq a-z + len freq from text)

    Returns:
        wr  : list of counts (raw eigenvalue)
        ewr : SHA-256 hex digest
    """
    # Wd: digit 0-9 frequency from numeric column values
    digit_freq = [0] * 10
    for col in numeric_cols:
        for val in team_df[col].dropna().astype(str):
            for ch in val:
                if ch.isdigit():
                    digit_freq[int(ch)] += 1

    # Wt: char a-z frequency + length frequency from text column values
    char_freq = [0] * 26
    len_freq: Dict[int, int] = {}
    for col in text_cols:
        for val in team_df[col].dropna().astype(str):
            for ch in val.lower():
                if 'a' <= ch <= 'z':
                    char_freq[ord(ch) - ord('a')] += 1
            l = len(val)
            len_freq[l] = len_freq.get(l, 0) + 1

    sorted_lengths = sorted(len_freq.keys())
    len_counts = [len_freq[l] for l in sorted_lengths]

    wr = digit_freq + char_freq + len_counts
    ewr = hashlib.sha256(json.dumps(wr, separators=(',', ':')).encode()).hexdigest()
    return wr, ewr


def get_team_watermark_bit(ewr: str) -> int:
    """Extract MSB (bit 7) of first byte of EWr."""
    return (int(ewr[:2], 16) >> 7) & 1


# ---------------------------------------------------------------------------
# Tuple hash for ordering
# ---------------------------------------------------------------------------

def compute_tuple_hash(row: pd.Series, pk_col: str, secret_key: str) -> int:
    """Compute HMAC(Ks || pk || Ks) for a tuple (used to define sort order)."""
    pk_val = row[pk_col]
    return hash_value(secret_key, pk_val, secret_key)


# ---------------------------------------------------------------------------
# Two-level grouping (on sorted original DB)
# ---------------------------------------------------------------------------

def form_teams(df: pd.DataFrame, pk_col: str, gamma: int) -> List[List[int]]:
    """Sort df by pk ascending, then take γ tuples per team.

    Returns list of teams, each team = list of row positions in df.
    Last team padded from first team if needed (for eigenvalue computation).
    """
    pk_series = df[pk_col].reset_index(drop=True)
    sorted_positions = list(pk_series.sort_values().index)

    alpha = len(sorted_positions)
    n_teams = math.ceil(alpha / gamma)

    teams = []
    for t in range(n_teams):
        start = t * gamma
        end = min(start + gamma, alpha)
        team_positions = sorted_positions[start:end]
        # Pad last team if needed (padding only for eigenvalue; not emitted in output)
        if len(team_positions) < gamma:
            n_needed = gamma - len(team_positions)
            pad = sorted_positions[:n_needed]
            team_positions = team_positions + pad
        teams.append(team_positions)

    return teams


def form_groups(teams: List[List[int]], n: int) -> List[List[int]]:
    """Group consecutive teams into groups of n teams each.

    Returns list of groups, each group = list of team indices.
    """
    t = len(teams)
    n_groups = math.ceil(t / n)
    groups = []
    for g in range(n_groups):
        start = g * n
        end = min(start + n, t)
        team_indices = list(range(start, end))
        groups.append(team_indices)
    return groups


# ---------------------------------------------------------------------------
# Core embed
# ---------------------------------------------------------------------------

def embed_watermark(
    df: pd.DataFrame,
    pk_col: str,
    secret_key: str,
    owner_id: str,
    numeric_cols: Optional[List[str]],
    text_cols: Optional[List[str]],
    gamma: Optional[int],
    output_dir: str,
) -> Tuple[pd.DataFrame, Dict]:
    """Embed B7 watermark by reordering teams and tuples.

    Returns (watermarked_df, ca_record).
    """
    import time

    if numeric_cols is None:
        numeric_cols = get_numeric_cols(df, exclude_cols=[pk_col])
    if text_cols is None:
        text_cols = get_text_cols(df, exclude_cols=[pk_col])
    if gamma is None:
        gamma = max(1, len(numeric_cols) + len(text_cols))

    alpha = len(df)
    df_indexed = df.reset_index(drop=True)

    # Pre-compute PK-sorted positions so we know which are real vs padding per team
    pk_series = df_indexed[pk_col].reset_index(drop=True)
    pk_sorted_positions = list(pk_series.sort_values().index)

    # Step 1: Form teams from PK-sorted df (last team may be padded for eigenvalue)
    teams = form_teams(df_indexed, pk_col, gamma)
    t_count = len(teams)

    # Real positions per team: only the non-padded rows
    real_positions_per_team: List[Set[int]] = []
    for t_idx in range(t_count):
        start = t_idx * gamma
        end = min(start + gamma, alpha)
        real_positions_per_team.append(set(pk_sorted_positions[start:end]))

    # Step 2: Form groups of n=N_TEAMS_PER_GROUP teams
    n = N_TEAMS_PER_GROUP
    groups_team_indices = form_groups(teams, n)

    # Step 3: Compute eigenvalues and watermark bits per team
    team_info = []
    for tid, positions in enumerate(teams):
        team_df = df_indexed.iloc[positions].reset_index(drop=True)
        wr, ewr = extract_team_eigenvalue(team_df, numeric_cols, text_cols)
        w_bit = get_team_watermark_bit(ewr)
        real_size = len(real_positions_per_team[tid])
        team_info.append({
            "team_id": tid,
            "positions": positions,
            "real_size": real_size,
            "wr": wr,
            "ewr": ewr,
            "w_bit": w_bit,
        })

    # Step 4: Embed group watermark (reorder teams within each group)
    group_info = []
    reordered_teams_global: List[int] = []  # team IDs in final output order

    for gid, team_idx_list in enumerate(groups_team_indices):
        actual_n = len(team_idx_list)
        w_bits = [team_info[tid]["w_bit"] for tid in team_idx_list]
        WB = sum(w_bits)  # group watermark value ∈ {0,...,actual_n}

        # Clamp WB to valid range [0, actual_n! - 1]
        # For n=5: 5! = 120; WB ∈ {0,1,2,3,4,5} which is always < 120, so safe
        target_perm = unrank_permutation(actual_n, WB)
        # target_perm[i] = which team (from local index) goes to position i
        reordered_local = [team_idx_list[target_perm[i]] for i in range(actual_n)]

        group_info.append({
            "group_id": gid,
            "original_team_ids": team_idx_list,
            "w_bits": w_bits,
            "WB": WB,
            "target_perm": target_perm,
            "reordered_team_ids": reordered_local,
        })
        reordered_teams_global.extend(reordered_local)

    # Step 5: Embed tuple watermark (reorder tuples within each team)
    # Only emit real (non-padding) positions in the output.
    output_rows: List[int] = []  # row positions in df_indexed, in final output order

    for tid in reordered_teams_global:
        info = team_info[tid]
        positions = info["positions"]  # includes padding for eigenvalue computation
        w_bit = info["w_bit"]
        real_set = real_positions_per_team[tid]  # only the non-padded rows

        # Compute pk-hash for each tuple in team (including padding, for consistent sort)
        pos_hash_pairs = [
            (pos, compute_tuple_hash(df_indexed.iloc[pos], pk_col, secret_key))
            for pos in positions
        ]

        # WD = w_bit: 0 → ascending hash order, 1 → descending hash order
        reverse_sort = (w_bit == 1)
        sorted_pairs = sorted(pos_hash_pairs, key=lambda x: x[1], reverse=reverse_sort)
        sorted_positions = [p for p, _ in sorted_pairs]

        # Only emit real positions (preserve relative order from hash sort)
        real_sorted_positions = [p for p in sorted_positions if p in real_set]

        info["WD"] = w_bit
        info["tuple_sort_order"] = real_sorted_positions  # only real positions stored

        output_rows.extend(real_sorted_positions)

    # Build watermarked DataFrame
    wm_df = df_indexed.iloc[output_rows].reset_index(drop=True)

    # ── Storage accounting (relational soundness) ─────────────────────────────
    # B7 embeds its watermark in the order of teams within a group and of tuples
    # within a team. In the relational model row order carries no information and
    # a benign re-ordering must not be flagged; to remain verifiable B7 must store
    # an external order index  pk -> rank_W(pk)  reconstructing the released order,
    # costing n*ceil(log2 n) bits (this dominates the per-team reference bits).
    rank_bits = max(1, math.ceil(math.log2(max(2, alpha))))
    order_index_bytes = math.ceil(alpha * rank_bits / 8)

    # CA registration record
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ca_record = {
        "method": "B7_Sun2020",
        "owner_id": owner_id,
        "timestamp": timestamp,
        "alpha": alpha,
        "pk_col": pk_col,
        "numeric_cols": numeric_cols,
        "text_cols": text_cols,
        "gamma": gamma,
        "n_teams_per_group": n,
        "n_teams_total": t_count,
        "n_groups": len(groups_team_indices),
        "team_info": team_info,
        "group_info": group_info,
        "reordered_team_sequence": reordered_teams_global,
        "output_row_order": output_rows,
        # Order index pk -> rank_W: released watermarked order, stored so
        # verification reconstructs it independently of the suspect's physical
        # row order (relational soundness; B7 embeds in team/tuple ordering).
        "order_index_pks": [str(df_indexed.iloc[p][pk_col]) for p in output_rows],
        "storage_bytes": {
            "order_index": order_index_bytes,   # n*ceil(log2 n)/8
            "total": order_index_bytes,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    ca_path = os.path.join(output_dir, "ca_registration.json")
    wm_path = os.path.join(output_dir, "watermarked.csv")
    save_json(ca_record, ca_path)
    wm_df.to_csv(wm_path, index=False)

    print(f"[B7 Embed] CA registration saved → {ca_path}")
    print(f"[B7 Embed] Watermarked DB saved  → {wm_path}")
    print(f"[B7 Embed] α={alpha}, γ={gamma}, t={t_count} teams, "
          f"n={n}, G={len(groups_team_indices)} groups")
    print(f"[B7 Embed] Numeric cols: {numeric_cols}")
    print(f"[B7 Embed] Text cols: {text_cols}")

    return wm_df, ca_record


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B7 Sun 2020 — Embed watermark")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config",     required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_csv(args.input)

    numeric_cols = cfg.get("numeric_cols") or None
    text_cols    = cfg.get("text_cols")    or None
    gamma        = cfg.get("gamma")        or None

    if isinstance(numeric_cols, list) and len(numeric_cols) == 0:
        numeric_cols = None
    if isinstance(text_cols, list) and len(text_cols) == 0:
        text_cols = None

    embed_watermark(
        df,
        pk_col=cfg["pk_col"],
        secret_key=cfg["secret_key"],
        owner_id=cfg.get("owner_id", "owner_001"),
        numeric_cols=numeric_cols,
        text_cols=text_cols,
        gamma=gamma,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
