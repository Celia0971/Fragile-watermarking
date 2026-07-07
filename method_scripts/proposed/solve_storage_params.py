#!/usr/bin/env python3
"""
solve_storage_params.py — Storage-budgeted (m, s, r) solver for the proposed scheme.

Solves, per dataset, the optimization problem (storage_optimization.tex Eq. 5):

    max t   s.t.  m = r = t,  metadata_bits(m, s, r) <= 8 * rho * B_R

where
    metadata_bits = kappa + ceil(alpha_IBLT*(s+r)) * (c_cnt + lam_pk + lam_chk) + g*b
    g = ceil( (ln n + ln C(n-1, m) + ln(1/delta)) / pi_{m,r} )
    pi_{m,r} = p(1-p)^{m+r},  p = 1/(m+r+1)

F(t) is strictly increasing, so t* is found by linear scan; the leftover
budget is then spent on s (insertion-only tolerance, IBLT cells only).

Outputs a summary table and one params YAML per dataset (format consumed by
baselines/proposed/embed.py).

Usage:
    python3 solve_storage_params.py                       # rho=1%, delta=2^-10
    python3 solve_storage_params.py --rho 0.05            # 5% budget
    python3 solve_storage_params.py --delta-exp 20        # delta = 2^-20
    python3 solve_storage_params.py --no-s-bonus          # keep s = 0
    python3 solve_storage_params.py --outdir baselines/proposed/config/budget
"""

import argparse
import math
import os

# ── Fixed accounting parameters (storage_optimization.tex Table 1) ──────────
KAPPA    = 128   # secret-key length (bits)
ALPHA    = 1.5   # IBLT load factor (k=3 peeling threshold c3≈1.222 + slack)
LAM_CHK  = 32    # hashSum fingerprint width (bits)
B_DIGEST = 64    # group-digest width b (bits)
IBLT_K   = 3

# ── Dataset table: name -> dict ─────────────────────────────────────────────
# csv: path used to (re)measure B_R; n: tuple count;
# c_cnt = ceil(log2(n+1))+1 ; lam_pk = ceil(log2(pk_max)) (integer ids 1..n).
DATASETS = {
    "FCT":    dict(csv="datasets/processed/FCT.csv",    n=581012, c_cnt=21, lam_pk=20),
    "AGNews": dict(csv="datasets/processed/AGNews.csv", n=120000, c_cnt=18, lam_pk=17),
    "Adult":  dict(csv="datasets/processed/Adult.csv",  n=48842,  c_cnt=17, lam_pk=16),
    "Bank":   dict(csv="datasets/processed/Bank.csv",   n=41188,  c_cnt=17, lam_pk=16),
}

YAML_TEMPLATE = """\
# ── Proposed scheme: storage-budgeted config for {name} ──
# Solved by solve_storage_params.py: rho={rho}, delta={delta:.3e}
# t* = {t} (balanced tolerance: q_mod<=m, q_del<=r, q_ins+q_del<=s+r)
# g = {g}  |  metadata = {meta_bytes:,} B = {rate:.3f}% of {B:,} B
pk_col: "id"
secret_key: "Proposed_IBLT_GT_SecretKey_v1"
all_cols: null
iblt_mode: "basic"
iblt_capacity: {cap}          # C_pk = s + r
iblt_mult: {alpha}
iblt_k: {k}
delta: {delta:.6e}
gt_b: {b}
gt_m: {m}
gt_s_del: {r}
detection_mode: "compressed"
"""


def g_groups(n: int, m: int, r: int, delta: float) -> int:
    """Group count g from Eq. 12 (Definition 4.1)."""
    p = 1.0 / (m + r + 1)
    pi = p * (1.0 - p) ** (m + r)
    num = math.log(n) + math.log(math.comb(n - 1, m)) + math.log(1.0 / delta)
    return math.ceil(num / pi)


def metadata_bits(n, m, s, r, c_cnt, lam_pk, delta):
    cells = math.ceil(ALPHA * (s + r))
    iblt = cells * (c_cnt + lam_pk + LAM_CHK)
    synd = g_groups(n, m, r, delta) * B_DIGEST
    return KAPPA + iblt + synd


def solve(n, B, c_cnt, lam_pk, rho, delta, mode="balanced", r_fixed=0,
          s_bonus=True):
    """Return dict with optimal (m, s, r), g, and the metadata breakdown.

    mode="balanced": m = r = t  (hard guarantee for any mixed split of t).
    mode="modheavy": m = t, r = r_fixed (small buffer; mixed mod+del beyond
                     r relies on the soft capacity), s+r >= t via s.
    mode="aligned" : m = t, r = r_fixed, and s+r = t_soft(m,r) so that the
                     A1/A2 hard cliff coincides with the A3 soft capacity
                     (all pure attacks degrade at the same scale).
    """
    budget_bits = 8.0 * rho * B

    def soft_cap(t, r):
        g = g_groups(n, t, r, delta)
        p = 1.0 / (t + r + 1)
        return math.log(g * p / math.log(n)) / (-math.log(1.0 - p))

    def F(t):
        if mode == "balanced":
            return metadata_bits(n, t, 0, t, c_cnt, lam_pk, delta)
        if mode == "aligned":
            r = min(r_fixed, t)
            sr = max(int(soft_cap(t, r)), t)   # s+r target
            return metadata_bits(n, t, sr - r, r, c_cnt, lam_pk, delta)
        # modheavy: IBLT must hold s+r >= t, so charge cells for t
        return metadata_bits(n, t, max(t - r_fixed, 0), r_fixed,
                             c_cnt, lam_pk, delta)

    t = 0
    while F(t + 1) <= budget_bits:
        t += 1
    if t == 0:
        raise ValueError("budget too small for t=1")

    r = t if mode == "balanced" else min(r_fixed, t)
    if mode == "aligned":
        s = max(int(soft_cap(t, r)), t) - r
    else:
        s = 0 if mode == "balanced" else t - r
    if mode != "aligned" and s_bonus:
        used = metadata_bits(n, t, s, r, c_cnt, lam_pk, delta)
        cell_cost = ALPHA * (c_cnt + lam_pk + LAM_CHK)
        s += int((budget_bits - used) // cell_cost)
        while metadata_bits(n, t, s, r, c_cnt, lam_pk, delta) > budget_bits:
            s -= 1

    g = g_groups(n, t, r, delta)
    cells = math.ceil(ALPHA * (s + r))
    iblt_bits = cells * (c_cnt + lam_pk + LAM_CHK)
    total = KAPPA + iblt_bits + g * B_DIGEST
    # soft capacity: E[FP]~1 at  g*p*(1-p)^t_soft = ln n   (witness analysis;
    # applies to q_mod alone and to q_mod+q_del jointly in mixed attacks)
    p = 1.0 / (t + r + 1)
    t_soft = math.log(g * p / math.log(n)) / (-math.log(1.0 - p))
    return dict(m=t, s=s, r=r, g=g, cells=cells, t_soft=int(t_soft),
                kappa_B=KAPPA / 8, iblt_B=iblt_bits / 8,
                synd_B=g * B_DIGEST / 8, total_B=total / 8,
                rate=total / (8 * B))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rho", type=float, default=0.01,
                    help="compression-rate budget (default 0.01 = 1%%)")
    ap.add_argument("--delta-exp", type=int, default=10,
                    help="delta = 2^-EXP (default 10)")
    ap.add_argument("--delta", type=float, default=None,
                    help="explicit delta value (overrides --delta-exp; e.g. 1e-6)")
    ap.add_argument("--no-s-bonus", action="store_true",
                    help="do not spend leftover budget on s")
    ap.add_argument("--mode", choices=["balanced", "modheavy", "aligned"],
                    default="balanced",
                    help="balanced: m=r=t* | modheavy: m=t*, r=--r-fixed | "
                         "aligned: m=t*, r=--r-fixed, s+r=t_soft")
    ap.add_argument("--r-fixed", type=int, default=0,
                    help="r buffer for modheavy mode (default 0)")
    ap.add_argument("--outdir", default=None,
                    help="directory for generated params YAMLs "
                         "(default baselines/proposed/config/budget_<mode>)")
    args = ap.parse_args()

    delta = args.delta if args.delta is not None else 2.0 ** (-args.delta_exp)
    if args.outdir is None:
        args.outdir = f"baselines/proposed/config/budget_{args.mode}"
    os.makedirs(args.outdir, exist_ok=True)

    hdr = (f"{'dataset':8} {'B_R(B)':>12} {'n':>8} | "
           f"{'m':>4} {'s':>5} {'r':>4} {'g':>8} {'t_soft':>6} | "
           f"{'IBLT(B)':>9} {'synd(B)':>10} {'total(B)':>10} {'rate':>7}")
    print(f"mode={args.mode}  rho={args.rho}  delta=2^-{args.delta_exp}  "
          f"b={B_DIGEST}  alpha={ALPHA}  lam_chk={LAM_CHK}  kappa={KAPPA}"
          + (f"  r_fixed={args.r_fixed}" if args.mode == "modheavy" else ""))
    print(hdr)
    print("-" * len(hdr))

    for name, d in DATASETS.items():
        B = (os.path.getsize(d["csv"]) if os.path.exists(d["csv"])
             else d.get("bytes"))
        if B is None:
            print(f"{name:8} SKIPPED: {d['csv']} not found and no fallback size")
            continue
        sol = solve(d["n"], B, d["c_cnt"], d["lam_pk"],
                    args.rho, delta, mode=args.mode, r_fixed=args.r_fixed,
                    s_bonus=not args.no_s_bonus)
        print(f"{name:8} {B:>12,} {d['n']:>8,} | "
              f"{sol['m']:>4} {sol['s']:>5} {sol['r']:>4} {sol['g']:>8,} "
              f"{sol['t_soft']:>6} | "
              f"{sol['iblt_B']:>9,.0f} {sol['synd_B']:>10,.0f} "
              f"{sol['total_B']:>10,.0f} {sol['rate']*100:>6.3f}%")

        path = os.path.join(args.outdir, f"params_{name}.yaml")
        with open(path, "w") as f:
            f.write(YAML_TEMPLATE.format(
                name=name, rho=args.rho,
                t=sol["m"], g=sol["g"], meta_bytes=int(sol["total_B"]),
                rate=sol["rate"] * 100, B=B, cap=sol["s"] + sol["r"],
                alpha=ALPHA, k=IBLT_K, delta=delta, b=B_DIGEST,
                m=sol["m"], r=sol["r"]))

    print(f"\nYAML configs written to {args.outdir}/")


if __name__ == "__main__":
    main()
