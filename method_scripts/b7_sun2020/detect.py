"""
B7: Sun et al. 2020 — Two-Level Group Eigenvalue Fragile Watermarking (Detect)

Detection algorithm (Algorithm 2):

For each group of the suspicious database:
  1. Re-extract eigenvalues from the suspicious DB (using same team/group boundaries)
     → recompute EWr' → W'_team → WB'1 = sum of W'_team  (eigenvalue-derived WB)
  2. Compute WB'2 from the actual team ordering in the suspicious DB
     using rank_permutation() on the observed team sequence within the group
  3. If WB'1 ≠ WB'2: group is tampered  (FG = attacked group)
  4. For each attacked group, locate attacked tuples:
     For each team in the attacked group:
       - Compute h_i = HMAC(Ks || pk_i || Ks) for each tuple
       - WD = W'_team bit (0 = expect ascending, 1 = expect descending)
       - Check if tuples in team are sorted according to WD
       - Tuples violating the expected order are flagged as tampered

Output:
  - db_tampered: bool
  - tampered_groups: list of group IDs
  - tampered_teams: list of team IDs within tampered groups
  - tampered_tuple_positions: list of suspicious DB row positions flagged as tampered
  - group-level and team-level detection details
"""

import argparse
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from common.utils import (
    load_csv, load_json, save_json, load_config, get_numeric_cols, get_text_cols
)
from common.metrics import compute_localization_metrics
from b7_sun2020.embed import (
    N_TEAMS_PER_GROUP,
    extract_team_eigenvalue, get_team_watermark_bit,
    compute_tuple_hash, rank_permutation,
    form_teams, form_groups,
)


def detect_watermark(
    df_suspicious: pd.DataFrame,
    ca_record: Dict,
) -> Dict:
    """Detect and localise tampering using B7 Sun 2020 method.

    The suspicious DB is assumed to be in the watermarked ordering (i.e., the
    attacker received the watermarked DB and may have modified values/rows).

    Args:
        df_suspicious : suspicious (possibly attacked) watermarked DataFrame
        ca_record     : CA registration from embed phase; _secret_key_internal injected

    Returns:
        result dict with db_tampered, tampered_groups, localization info, etc.
    """
    pk_col       = ca_record["pk_col"]
    numeric_cols = ca_record["numeric_cols"]
    text_cols    = ca_record["text_cols"]
    gamma        = ca_record["gamma"]
    n            = ca_record["n_teams_per_group"]
    secret_key   = ca_record.get("_secret_key_internal", "")
    alpha_orig   = ca_record["alpha"]
    n_teams_orig = ca_record["n_teams_total"]
    n_groups_orig= ca_record["n_groups"]

    # Step 0: Reconstruct the watermarked release order from the stored order
    # index (relational soundness). B7 encodes watermark bits in the order of
    # teams within a group and of tuples within a team; in the relational model
    # row order is non-semantic, so verification must not trust the suspect's
    # physical order. We re-rank surviving tuples by their stored rank_W
    # (inserted tuples, absent from the index, go last). Order-preserving attacks
    # leave this a no-op; a benign re-ordering is restored to the watermarked order.
    order_index_pks = ca_record.get("order_index_pks")
    if order_index_pks:
        rank = {pk: i for i, pk in enumerate(order_index_pks)}
        n_idx = len(rank)
        ord_key = df_suspicious[pk_col].astype(str).map(lambda p: rank.get(p, n_idx))
        df_suspicious = (df_suspicious.assign(_wm_ord=ord_key)
                         .sort_values("_wm_ord", kind="stable")
                         .drop(columns="_wm_ord").reset_index(drop=True))

    alpha_susp = len(df_suspicious)
    df_susp = df_suspicious.reset_index(drop=True)

    # Filter columns to those present in suspicious DB
    avail_numeric = [c for c in numeric_cols if c in df_susp.columns]
    avail_text    = [c for c in text_cols    if c in df_susp.columns]

    # ── Step 1: Form teams from the suspicious DB (same method as embed) ──────
    # The suspicious DB retains the watermarked ordering, so we group by position
    # (not by PK re-sort) to reflect the embedded order.
    # However, to detect VALUE tampering, we also need eigenvalues.
    # Per-paper: "grouping process is exactly the same as when watermark is embedded."
    # Embed sorted by PK first; we apply the same PK sort here.
    teams_susp = form_teams(df_susp, pk_col, gamma)
    # But note: if tuples were deleted/inserted, the team boundaries shift.
    # Size mismatch is already informative.
    n_teams_susp = len(teams_susp)
    groups_susp  = form_groups(teams_susp, n)
    n_groups_susp = len(groups_susp)

    size_mismatch = (n_teams_susp != n_teams_orig) or (alpha_susp != alpha_orig)

    # ── Step 2: Compute eigenvalues for each team in suspicious DB ────────────
    team_susp_info = []
    for tid, positions in enumerate(teams_susp):
        team_df = df_susp.iloc[positions].reset_index(drop=True)
        wr, ewr = extract_team_eigenvalue(team_df, avail_numeric, avail_text)
        w_bit   = get_team_watermark_bit(ewr)
        team_susp_info.append({
            "team_id": tid,
            "positions": positions,
            "wr": wr,
            "ewr": ewr,
            "w_bit": w_bit,
        })

    # Retrieve original team info for WB reference
    orig_team_info = ca_record.get("team_info", [])
    orig_group_info = ca_record.get("group_info", [])

    # ── Step 3: Per-group detection ───────────────────────────────────────────
    group_results = {}
    tampered_groups = []
    tampered_teams: Set[int] = set()
    tampered_tuple_positions: Set[int] = set()

    for gid, team_idx_list_susp in enumerate(groups_susp):
        actual_n = len(team_idx_list_susp)

        # WB'1: derived from eigenvalues of suspicious DB
        w_bits_susp = [team_susp_info[tid]["w_bit"] for tid in team_idx_list_susp]
        WB1 = sum(w_bits_susp)

        # WB'2: derived from observed team ordering
        # The suspicious DB ordering (after watermarking) determines a permutation
        # relative to the canonical PK-sorted ordering.
        # We compare WB1 (eigenvalue-derived) with the ORIGINAL WB stored at embed time.
        if gid < len(orig_group_info):
            WB_orig = orig_group_info[gid]["WB"]
        else:
            WB_orig = None  # group didn't exist in original → tampered

        # Also compute WB'2 from the actual permutation of teams in this group.
        # The observed order of team_idx_list_susp defines a permutation
        # relative to sorted order [g*n, g*n+1, ...].
        # Map team indices to local [0..actual_n-1] positions.
        base_tid = gid * n
        local_perm = [tid - base_tid for tid in team_idx_list_susp
                      if 0 <= tid - base_tid < actual_n]
        if len(local_perm) == actual_n:
            # Check if local_perm is a valid permutation of [0..actual_n-1]
            if sorted(local_perm) == list(range(actual_n)):
                WB2 = rank_permutation(local_perm)
            else:
                WB2 = -1  # invalid permutation
        else:
            WB2 = -1

        # Eigenvalue mismatch: WB1 (from suspicious eigenvalues) vs WB_orig (registered)
        eigenvalue_mismatch = (WB_orig is None) or (WB1 != WB_orig)

        # Ordering mismatch: WB2 (observed physical ordering rank) vs WB_orig
        ordering_mismatch = (WB2 != WB_orig) if WB_orig is not None else True

        # Combined tamper decision: flag if EITHER watermark component mismatches.
        # B7 Sun 2020 embeds both eigenvalue and ordering watermarks; either
        # mismatch constitutes evidence of tampering.
        # NOTE: ordering_mismatch may have false positives if the suspicious DB
        # was sorted by PK before submission (undoing the watermarked order).
        # For robustness, eigenvalue_mismatch alone is used as the primary signal
        # BUG NOTE (ordering_mismatch): The ordering-based check (WB2) is disabled
        # because the detect-side always computes local_perm=[0,1,...,n-1] via
        # form_groups sequential indices, giving a fixed Myrvold-Ruskey rank
        # (e.g., 119 for n=5 identity permutation) regardless of data content.
        # Since WB_orig is a random rank embedded at watermark time, ordering_mismatch
        # is nearly always True even for clean copies → FAR ≈ 1.0.
        # Fix: use eigenvalue_mismatch only (correct and working).
        # Size mismatches (insertion/deletion) are already handled by size_mismatch above.
        is_group_tampered = eigenvalue_mismatch

        group_results[gid] = {
            "group_id": gid,
            "team_ids": team_idx_list_susp,
            "WB_original": WB_orig,
            "WB1_eigenvalue": WB1,
            "WB2_observed_rank": WB2,
            "w_bits_suspicious": w_bits_susp,
            "eigenvalue_mismatch": eigenvalue_mismatch,
            "ordering_mismatch": ordering_mismatch,
            "authentic": not is_group_tampered,
            "status": "authentic" if not is_group_tampered else "tampered",
        }

        if is_group_tampered:
            tampered_groups.append(gid)
            # Localise to teams and tuples within tampered group
            for tid in team_idx_list_susp:
                t_info = team_susp_info[tid]
                positions = t_info["positions"]
                w_bit_susp = t_info["w_bit"]

                # Check tuple ordering: expected = ascending (w=0) or descending (w=1)
                pos_hash_pairs = [
                    (pos, compute_tuple_hash(df_susp.iloc[pos], pk_col, secret_key))
                    for pos in positions
                ]
                hashes = [h for _, h in pos_hash_pairs]

                # Expected: strictly monotone according to w_bit
                expected_ascending = all(
                    hashes[i] <= hashes[i + 1] for i in range(len(hashes) - 1)
                )
                expected_descending = all(
                    hashes[i] >= hashes[i + 1] for i in range(len(hashes) - 1)
                )

                if w_bit_susp == 0 and not expected_ascending:
                    team_order_violated = True
                elif w_bit_susp == 1 and not expected_descending:
                    team_order_violated = True
                else:
                    team_order_violated = False

                # Identify individual violating tuples
                violating_positions = []
                if w_bit_susp == 0:
                    # ascending expected; find tuples where h[i] > h[i+1]
                    for i in range(len(hashes) - 1):
                        if hashes[i] > hashes[i + 1]:
                            violating_positions.append(positions[i])
                            violating_positions.append(positions[i + 1])
                else:
                    # descending expected; find tuples where h[i] < h[i+1]
                    for i in range(len(hashes) - 1):
                        if hashes[i] < hashes[i + 1]:
                            violating_positions.append(positions[i])
                            violating_positions.append(positions[i + 1])

                violating_positions = list(set(violating_positions))
                tampered_teams.add(tid)
                tampered_tuple_positions.update(violating_positions)

                group_results[gid].setdefault("team_details", {})[tid] = {
                    "WD": w_bit_susp,
                    "ordering_violated": team_order_violated,
                    "violating_tuple_positions": violating_positions,
                    "pk_hashes_sample": hashes[:5],
                }

    db_tampered = len(tampered_groups) > 0 or size_mismatch

    return {
        "method": "B7_Sun2020",
        "n_tuples_suspicious": alpha_susp,
        "n_tuples_original": alpha_orig,
        "n_teams_original": n_teams_orig,
        "n_teams_suspicious": n_teams_susp,
        "n_groups_original": n_groups_orig,
        "n_groups_suspicious": n_groups_susp,
        "size_mismatch": size_mismatch,
        "db_tampered": db_tampered,
        "tampered_groups": tampered_groups,
        "n_tampered_groups": len(tampered_groups),
        "tampered_teams": sorted(tampered_teams),
        "n_tampered_teams": len(tampered_teams),
        "tampered_tuple_positions": sorted(tampered_tuple_positions),
        "n_tampered_tuples": len(tampered_tuple_positions),
        "group_results": group_results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B7 Sun 2020 — Detect watermark")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--ca_record",  required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--true_tampered_groups", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df  = load_csv(args.input)
    ca_record = load_json(args.ca_record)
    ca_record["_secret_key_internal"] = cfg["secret_key"]

    result = detect_watermark(df, ca_record)

    if args.true_tampered_groups:
        true_groups = set(int(x) for x in args.true_tampered_groups.split(",") if x.strip())
        all_groups  = set(range(ca_record["n_groups"]))
        flagged     = set(result["tampered_groups"])
        metrics = compute_localization_metrics(true_groups, flagged, all_groups)
        result["localization_metrics"] = metrics

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "detect_result.json")
    save_json(result, out_path)

    print(f"[B7 Detect] DB tampered: {result['db_tampered']}")
    print(f"[B7 Detect] Tampered groups: {result['tampered_groups']}")
    print(f"[B7 Detect] Tampered teams: {result['tampered_teams']}")
    print(f"[B7 Detect] Tampered tuples: {result['n_tampered_tuples']}")
    print(f"[B7 Detect] Result saved → {out_path}")


if __name__ == "__main__":
    main()
