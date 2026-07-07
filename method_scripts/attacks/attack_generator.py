"""
Attack generator for fragile watermarking experiments.

Attack taxonomy (paper §7.6, Table 8, A0–A8):

  A0  Clean          No operation; used for FAR measurement.
  A1  Ins            Tuple insertion: max{1, ⌊ρ·n⌋} fresh tuples with
                     column-wise empirical sampling for non-key values.
  A2  Del            Tuple deletion: max{1, ⌊ρ·n⌋} tuples removed uniformly.
  A3  Mod            Value modification: max{1, ⌊ρ·n⌋} tuples; one non-key
                     attribute per tuple changed using column empirical sampling
                     (works for numerical, categorical, and textual attributes).
  A4  Sub            Attribute substitution: max{1, ⌊ρ·n/2⌋} disjoint pairs;
                     ONE shared editable attribute per pair is swapped.
  A5  DelIns         Deletion-insertion: max{1, ⌊ρ·n/2⌋} each; net zero change.
  A6  DelMod         Deletion-modification: max{1, ⌊ρ·n/2⌋} each.
  A7  InsMod         Insertion-modification: max{1, ⌊ρ·n/2⌋} each.
  A8  InsDelMod      Balanced three-operation: max{1, ⌊ρ·n/3⌋} each.

Cardinality rule (§7.6):
  A1–A3: max{1, ⌊ρn⌋}
  A4–A7: each component receives max{1, ⌊ρn/2⌋}  (A4: n_pairs = max{1,⌊ρn/2⌋})
  A8:    each component receives max{1, ⌊ρn/3⌋}

Each attack returns (attacked_df, attack_info) where attack_info records
ground-truth tampered primary keys for metric computation.

Usage:
    from attacks.attack_generator import AttackGenerator
    ag = AttackGenerator(rng=np.random.default_rng(seed))
    attacked_df, info = ag.a1_ins(df, rho=0.01, pk_col='id')
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class AttackGenerator:
    """Stateless attack generator. Pass an rng for reproducibility."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _non_pk_cols(self, df: pd.DataFrame, pk_col: Optional[str]) -> List[str]:
        """Return all non-primary-key columns."""
        excl = {pk_col} if pk_col else set()
        return [c for c in df.columns if c not in excl]

    def _count(self, n: int, rho: float, divisor: int = 1) -> int:
        """Cardinality rule: max{1, floor(rho * n / divisor)}."""
        return max(1, int(math.floor(rho * n / divisor)))

    def _resolve_k(self, n: int, rho: Optional[float], count: Optional[int] = None,
                   divisor: int = 1) -> int:
        """Return per-operation k from either explicit count or rho×n formula."""
        if count is not None:
            return max(1, int(count))
        if rho is not None:
            return self._count(n, rho, divisor)
        raise ValueError("Either rho or count must be provided")

    def _distribute_total(self, total: int, n_ops: int) -> List[int]:
        """Distribute `total` across `n_ops` operations as evenly as possible.

        Total count = sum of all per-operation counts.
        Remainder (total % n_ops) is assigned randomly to `remainder` operations
        using self.rng for reproducibility given a fixed attack seed.

        Example: total=10, n_ops=3 → base=3, remainder=1
          One randomly chosen op gets 4; others get 3 → e.g. [3, 4, 3].
        """
        total = max(n_ops, int(total))   # ensure at least 1 per op
        base  = total // n_ops
        rem   = total % n_ops
        counts = [base] * n_ops
        if rem > 0:
            for idx in self.rng.choice(n_ops, size=rem, replace=False).tolist():
                counts[idx] += 1
        return counts

    def _sample_new_tuples(
        self,
        df: pd.DataFrame,
        k: int,
        pk_col: Optional[str],
    ) -> pd.DataFrame:
        """Sample k new rows with column-wise independent empirical sampling.

        Each non-key column is sampled independently from its observed values.
        Categorical and textual values are sampled from observed frequencies;
        numerical values are sampled from observed values (paper §7.6).
        """
        new_rows: Dict[str, np.ndarray] = {}
        for col in df.columns:
            if col == pk_col:
                continue
            new_rows[col] = self.rng.choice(df[col].values, size=k, replace=True)
        result = pd.DataFrame(new_rows)
        non_pk = self._non_pk_cols(df, pk_col)
        return result[non_pk]

    def _sample_replacement(
        self,
        col_values: np.ndarray,
        current_val,
        max_attempts: int = 20,   # kept for signature compatibility (unused)
    ):
        """Sample a replacement value different from current_val.

        Frequency-weighted (empirical) sampling restricted to the values that
        differ from current_val. This preserves the column's empirical
        distribution among the alternatives while GUARANTEEING a real change
        whenever the column has more than one distinct value. (The previous
        attempt-capped loop produced no-op edits on skewed categorical columns
        — e.g. ~0.9^20≈12% of the time on a 90%-dominant value — which silently
        capped every method's recall; see attack no-op analysis.)
        """
        mask = col_values != current_val
        if not mask.any():
            return current_val  # single distinct value — no change possible
        return self.rng.choice(col_values[mask])

    # ── A0: Clean ────────────────────────────────────────────────────────────

    def a0_clean(self, df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
        """No-op. Database returned byte-identical for FAR measurement (§7.7)."""
        return df.copy(), {
            "attack": "A0_Clean",
            "s_ins": [], "s_del": [], "s_mod": [],
        }

    # ── A1: Insertion ─────────────────────────────────────────────────────────

    def a1_ins(
        self,
        df: pd.DataFrame,
        rho: Optional[float] = None,
        pk_col: Optional[str] = None,
        count: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Insert max{1, ⌊ρn⌋} fresh tuples, or exactly `count` tuples (§7.6 A1).

        New PKs start at max(PK(R)) + 1. Non-key values sampled column-wise
        from cleaned empirical distributions.
        """
        df = df.copy()
        n = len(df)
        k = self._resolve_k(n, rho, count)

        new_rows = self._sample_new_tuples(df, k, pk_col)
        if pk_col and pk_col in df.columns:
            max_pk = int(df[pk_col].max()) if pd.api.types.is_numeric_dtype(df[pk_col]) else 0
            new_rows.insert(0, pk_col, list(range(max_pk + 1, max_pk + 1 + k)))

        attacked = pd.concat([df, new_rows], ignore_index=True)
        s_ins = new_rows[pk_col].tolist() if pk_col and pk_col in new_rows.columns else []

        return attacked, {
            "attack": "A1_Ins", "rho": rho, "k": k,
            "s_ins": [str(x) for x in s_ins], "s_del": [], "s_mod": [],
            "original_n": n, "new_n": len(attacked),
        }

    # ── A2: Deletion ──────────────────────────────────────────────────────────

    def a2_del(
        self,
        df: pd.DataFrame,
        rho: Optional[float] = None,
        pk_col: Optional[str] = None,
        count: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Delete max{1, ⌊ρn⌋} tuples, or exactly `count` tuples (§7.6 A2)."""
        df = df.copy()
        n = len(df)
        k = min(self._resolve_k(n, rho, count), n - 1)

        del_pos = sorted(self.rng.choice(n, size=k, replace=False).tolist())
        del_idx = [df.index[i] for i in del_pos]
        s_del = (df.loc[del_idx, pk_col].tolist()
                 if pk_col and pk_col in df.columns else del_pos)

        attacked = df.drop(index=del_idx).reset_index(drop=True)

        return attacked, {
            "attack": "A2_Del", "rho": rho, "k": k,
            "s_ins": [], "s_del": [str(x) for x in s_del], "s_mod": [],
            "original_n": n, "new_n": len(attacked),
        }

    # ── A3: Value modification ────────────────────────────────────────────────

    def a3_mod(
        self,
        df: pd.DataFrame,
        rho: Optional[float] = None,
        pk_col: Optional[str] = None,
        editable_cols: Optional[List[str]] = None,
        count: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Modify max{1, ⌊ρn⌋} tuples, or exactly `count` tuples (§7.6 A3)."""
        df = df.copy()
        n = len(df)
        k = min(self._resolve_k(n, rho, count), n)

        cols = editable_cols or self._non_pk_cols(df, pk_col)
        if not cols:
            return df, {"attack": "A3_Mod", "rho": rho, "k": 0,
                        "s_ins": [], "s_del": [], "s_mod": []}

        col_values = {c: df[c].values for c in cols}
        target_pos = sorted(self.rng.choice(n, size=k, replace=False).tolist())
        s_mod = []
        edits = []

        for pos in target_pos:
            idx = df.index[pos]
            pk = df.at[idx, pk_col] if pk_col and pk_col in df.columns else pos

            col = str(self.rng.choice(cols))
            current_val = df.at[idx, col]
            new_val = self._sample_replacement(col_values[col], current_val)
            df.at[idx, col] = new_val

            s_mod.append(str(pk))
            edits.append({"pk": str(pk), "col": col,
                          "old": str(current_val), "new": str(new_val)})

        return df, {
            "attack": "A3_Mod", "rho": rho, "k": k,
            "s_ins": [], "s_del": [], "s_mod": s_mod,
            "edits": edits,
            "original_n": n, "new_n": n,
        }

    # ── A4: Attribute substitution ────────────────────────────────────────────

    def a4_sub(
        self,
        df: pd.DataFrame,
        rho: Optional[float] = None,
        pk_col: Optional[str] = None,
        editable_cols: Optional[List[str]] = None,
        count: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Swap one shared attribute between pairs; count=n_pairs or rho-based (§7.6 A4)."""
        df = df.copy()
        n = len(df)
        n_pairs = min(self._resolve_k(n, rho, count, divisor=2), n // 2)

        cols = editable_cols or self._non_pk_cols(df, pk_col)
        if not cols or n_pairs == 0:
            return df, {"attack": "A4_Sub", "rho": rho, "n_pairs": 0,
                        "s_ins": [], "s_del": [], "s_mod": []}

        available = list(range(n))
        self.rng.shuffle(available)
        s_mod = []
        swap_log = []

        pair_idx = 0
        used = set()
        max_attempts = n_pairs * 10

        attempt = 0
        while len(swap_log) < n_pairs and attempt < max_attempts:
            attempt += 1
            if len(available) < 2:
                break
            i, j = int(self.rng.choice(n)), int(self.rng.choice(n))
            if i == j or i in used or j in used:
                continue

            idx_i, idx_j = df.index[i], df.index[j]
            col = str(self.rng.choice(cols))
            val_i, val_j = df.at[idx_i, col], df.at[idx_j, col]

            if val_i == val_j:
                continue  # no actual modification — resample

            df.at[idx_i, col] = val_j
            df.at[idx_j, col] = val_i
            used.update([i, j])

            pk_i = (str(df.at[idx_i, pk_col]) if pk_col and pk_col in df.columns
                    else str(i))
            pk_j = (str(df.at[idx_j, pk_col]) if pk_col and pk_col in df.columns
                    else str(j))
            s_mod.extend([pk_i, pk_j])
            swap_log.append({"pos": (i, j), "col": col,
                             "pk_i": pk_i, "pk_j": pk_j})

        return df, {
            "attack": "A4_Sub", "rho": rho, "n_pairs": len(swap_log),
            "s_ins": [], "s_del": [], "s_mod": s_mod,
            "swaps": swap_log,
            "original_n": n, "new_n": n,
        }

    # ── Internal fixed-count primitives (for mixed attacks) ───────────────────

    def _del_k(self, df: pd.DataFrame, k: int, pk_col: Optional[str]) -> Tuple[pd.DataFrame, Dict]:
        """Delete exactly k tuples uniformly without replacement."""
        df = df.copy()
        n = len(df)
        k = min(k, n - 1)
        del_pos = sorted(self.rng.choice(n, size=k, replace=False).tolist())
        del_idx = [df.index[i] for i in del_pos]
        s_del = (df.loc[del_idx, pk_col].tolist()
                 if pk_col and pk_col in df.columns else del_pos)
        attacked = df.drop(index=del_idx).reset_index(drop=True)
        return attacked, {"k": k, "s_del": [str(x) for x in s_del]}

    def _ins_k(self, df: pd.DataFrame, k: int, pk_col: Optional[str]) -> Tuple[pd.DataFrame, Dict]:
        """Insert exactly k fresh tuples with column-wise empirical sampling."""
        df = df.copy()
        new_rows = self._sample_new_tuples(df, k, pk_col)
        if pk_col and pk_col in df.columns:
            max_pk = int(df[pk_col].max()) if pd.api.types.is_numeric_dtype(df[pk_col]) else 0
            new_rows.insert(0, pk_col, list(range(max_pk + 1, max_pk + 1 + k)))
        attacked = pd.concat([df, new_rows], ignore_index=True)
        s_ins = new_rows[pk_col].tolist() if pk_col and pk_col in new_rows.columns else []
        return attacked, {"k": k, "s_ins": [str(x) for x in s_ins]}

    def _mod_k(self, df: pd.DataFrame, k: int, pk_col: Optional[str],
               editable_cols: Optional[List[str]]) -> Tuple[pd.DataFrame, Dict]:
        """Modify exactly k tuples; one non-key attribute per tuple."""
        df = df.copy()
        n = len(df)
        k = min(k, n)
        cols = editable_cols or self._non_pk_cols(df, pk_col)
        if not cols or k == 0:
            return df, {"k": 0, "s_mod": []}
        col_values = {c: df[c].values for c in cols}
        target_pos = sorted(self.rng.choice(n, size=k, replace=False).tolist())
        s_mod = []
        for pos in target_pos:
            idx = df.index[pos]
            pk = df.at[idx, pk_col] if pk_col and pk_col in df.columns else pos
            col = str(self.rng.choice(cols))
            current_val = df.at[idx, col]
            df.at[idx, col] = self._sample_replacement(col_values[col], current_val)
            s_mod.append(str(pk))
        return df, {"k": k, "s_mod": s_mod}

    # ── A5: Deletion-insertion (zero net change) ───────────────────────────────

    def a5_del_ins(
        self,
        df: pd.DataFrame,
        rho: Optional[float] = None,
        pk_col: Optional[str] = None,
        count: Optional[int] = None,
        total_count: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Delete + insert. count=per-op OR total_count=total (distributed 50/50)."""
        n = len(df)
        if total_count is not None:
            k_del, k_ins = self._distribute_total(total_count, 2)
        else:
            k = self._resolve_k(n, rho, count, divisor=2)
            k_del = k_ins = k
        df_del, i_del = self._del_k(df, k_del, pk_col)
        df_ins, i_ins = self._ins_k(df_del, k_ins, pk_col)
        return df_ins, {
            "attack": "A5_DelIns", "rho": rho, "total_count": total_count,
            "s_ins": i_ins["s_ins"], "s_del": i_del["s_del"], "s_mod": [],
        }

    # ── A6: Deletion-modification ─────────────────────────────────────────────

    def a6_del_mod(
        self,
        df: pd.DataFrame,
        rho: Optional[float] = None,
        pk_col: Optional[str] = None,
        editable_cols: Optional[List[str]] = None,
        count: Optional[int] = None,
        total_count: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Delete + modify. count=per-op OR total_count=total (distributed 50/50)."""
        n = len(df)
        if total_count is not None:
            k_del, k_mod = self._distribute_total(total_count, 2)
        else:
            k = self._resolve_k(n, rho, count, divisor=2)
            k_del = k_mod = k
        df_del, i_del = self._del_k(df, k_del, pk_col)
        df_mod, i_mod = self._mod_k(df_del, k_mod, pk_col, editable_cols)
        return df_mod, {
            "attack": "A6_DelMod", "rho": rho, "total_count": total_count,
            "s_ins": [], "s_del": i_del["s_del"], "s_mod": i_mod["s_mod"],
        }

    # ── A7: Insertion-modification ────────────────────────────────────────────

    def a7_ins_mod(
        self,
        df: pd.DataFrame,
        rho: Optional[float] = None,
        pk_col: Optional[str] = None,
        editable_cols: Optional[List[str]] = None,
        count: Optional[int] = None,
        total_count: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Modify + insert. count=per-op OR total_count=total (distributed 50/50)."""
        n = len(df)
        if total_count is not None:
            k_mod, k_ins = self._distribute_total(total_count, 2)
        else:
            k = self._resolve_k(n, rho, count, divisor=2)
            k_mod = k_ins = k
        df_mod, i_mod = self._mod_k(df, k_mod, pk_col, editable_cols)
        df_ins, i_ins = self._ins_k(df_mod, k_ins, pk_col)
        return df_ins, {
            "attack": "A7_InsMod", "rho": rho, "total_count": total_count,
            "s_ins": i_ins["s_ins"], "s_del": [], "s_mod": i_mod["s_mod"],
        }

    # ── A8: Insertion-deletion-modification ───────────────────────────────────

    def a8_ins_del_mod(
        self,
        df: pd.DataFrame,
        rho: Optional[float] = None,
        pk_col: Optional[str] = None,
        editable_cols: Optional[List[str]] = None,
        count: Optional[int] = None,
        total_count: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Del+Mod+Ins. count=per-op OR total_count=total (distributed as evenly as possible)."""
        n = len(df)
        if total_count is not None:
            k_del, k_mod, k_ins = self._distribute_total(total_count, 3)
        else:
            k = self._resolve_k(n, rho, count, divisor=3)
            k_del = k_mod = k_ins = k
        df_del, i_del = self._del_k(df, k_del, pk_col)
        df_mod, i_mod = self._mod_k(df_del, k_mod, pk_col, editable_cols)
        df_ins, i_ins = self._ins_k(df_mod, k_ins, pk_col)
        return df_ins, {
            "attack": "A8_InsDelMod", "rho": rho, "total_count": total_count,
            "s_ins": i_ins["s_ins"], "s_del": i_del["s_del"], "s_mod": i_mod["s_mod"],
        }

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def apply(
        self,
        attack_name: str,
        df: pd.DataFrame,
        **kwargs,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Dispatch to named attack (a0–a8, case-insensitive).

        kwargs are forwarded to the specific attack method.
        Attacks that require rho accept it as kwarg 'rho'.
        """
        dispatch = {
            'a0': self.a0_clean,
            'a1': self.a1_ins,
            'a2': self.a2_del,
            'a3': self.a3_mod,
            'a4': self.a4_sub,
            'a5': self.a5_del_ins,
            'a6': self.a6_del_mod,
            'a7': self.a7_ins_mod,
            'a8': self.a8_ins_del_mod,
        }
        key = attack_name.lower().strip()
        if key not in dispatch:
            raise ValueError(
                f"Unknown attack '{attack_name}'. Valid: {sorted(dispatch.keys())}"
            )
        return dispatch[key](df, **kwargs)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from common.utils import load_csv, save_csv, save_json

    parser = argparse.ArgumentParser(
        description="Apply a single attack (A0–A8) to a CSV database"
    )
    parser.add_argument("--input",    required=True,  help="Input CSV path")
    parser.add_argument("--output",   required=True,  help="Output attacked CSV path")
    parser.add_argument("--attack",   required=True,
                        help="Attack ID: a0 (clean) | a1 (Ins) | a2 (Del) | "
                             "a3 (Mod) | a4 (Sub) | a5 (DelIns) | "
                             "a6 (DelMod) | a7 (InsMod) | a8 (InsDelMod)")
    parser.add_argument("--rho",   type=float, default=None,
                        help="Attack strength ρ ∈ R_attack (percentage-based)")
    parser.add_argument("--count", type=int,   default=None,
                        help="Absolute attack count (per-operation; overrides --rho)")
    parser.add_argument("--total_count", type=int, default=None,
                        help="Total count across ALL sub-operations (overrides --count and --rho)."
                             " For mixed attacks, distributed as evenly as possible using attack seed.")
    parser.add_argument("--pk_col",   default="id",   help="Primary key column name")
    parser.add_argument("--seed",     type=int,  default=0,
                        help="Attack random seed (paper: 0–9 for 10 trials)")
    parser.add_argument("--info_out", default=None,   help="Path to save attack_info JSON")
    args = parser.parse_args()

    tc = getattr(args, 'total_count', None)
    if args.rho is None and args.count is None and tc is None and args.attack.lower().strip() != 'a0':
        args.rho = 0.01  # default fallback

    rng = np.random.default_rng(args.seed)
    ag  = AttackGenerator(rng=rng)
    df  = load_csv(args.input)

    # Attacks that need "total_count" distributed across sub-operations:
    MIXED_ATTACKS = {'a5', 'a6', 'a7', 'a8'}
    # Single-op attacks (a1-a4): total_count == per-op count (use --count alias)
    SINGLE_ATTACKS = {'a1', 'a2', 'a3', 'a4'}

    no_attack = {'a0'}
    attack_key = args.attack.lower().strip()
    kwargs: Dict = {}
    if attack_key not in no_attack:
        if tc is not None:
            if attack_key in MIXED_ATTACKS:
                kwargs['total_count'] = tc   # distribute among sub-ops
            else:
                kwargs['count'] = tc         # single-op: total = per-op
        elif args.count is not None:
            kwargs['count'] = args.count
        elif args.rho is not None:
            kwargs['rho'] = args.rho
    if args.pk_col:
        kwargs['pk_col'] = args.pk_col

    attacked_df, info = ag.apply(args.attack, df, **kwargs)
    save_csv(attacked_df, args.output)

    n_in, n_out = len(df), len(attacked_df)
    print(f"[{args.attack.upper()}] n={n_in} → {n_out} | "
          f"ins={len(info.get('s_ins',[]))} "
          f"del={len(info.get('s_del',[]))} "
          f"mod={len(info.get('s_mod',[]))}")

    if args.info_out:
        save_json(info, args.info_out)
        print(f"[attack_info] → {args.info_out}")
