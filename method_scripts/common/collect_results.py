#!/usr/bin/env python3
"""
collect_results.py — Collect and aggregate experiment results.

Expected directory structure (written by run_all.sh):
  {results_dir}/{machine_id}/{dataset}/{method}/
    watermarks/trial_NN/          ← skipped by walker
    {attack}/{rho}/trial_NN/
      detect_result.json          ← always present; contains _meta and _gt blocks
      timing.json                 ← always present
      attack_info.json            ← only in --debug runs

detect_result.json contains:
  _meta: {method, attack, rho, attack_seed, wm_seed}
  _gt:   {s_ins, s_del, s_mod, n_true_ins, n_true_del, n_true_mod, db_tampered_true}
  + method-specific detection fields

Usage:
  python collect_results.py --results_dir results/ --output_dir results/

Outputs:
  all_results_raw.csv        — one row per (machine, dataset, method, attack, rho, trial)
  all_results_agg.csv        — mean ± std per (dataset, method, attack, rho)
  summaries/{dataset}_{method}_{attack}.csv  — per-attack table: rows=rho, cols=mean±std metrics
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

TRIAL_RE = __import__('re').compile(r'^trial_(\d+)$')


def _load(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ── Ground-truth extraction ───────────────────────────────────────────────────

def _gt_from_detect(detect: Dict) -> Dict[str, Any]:
    """Read ground truth from compact detect_result.json (top-level scalars).

    New compact format (written by _run_detect_job.sh Step 4) stores ground
    truth as top-level scalar fields (db_tampered_true, n_true_ins, …).
    No '_gt' sub-block exists in the new format; we read directly.
    """
    n_ins = int(detect.get("n_true_ins", 0) or 0)
    n_del = int(detect.get("n_true_del", 0) or 0)
    n_mod = int(detect.get("n_true_mod", 0) or 0)
    db_t  = bool(detect.get("db_tampered_true", (n_ins + n_del + n_mod) > 0))
    return {
        "db_tampered_true": db_t,
        "n_true_ins": n_ins,
        "n_true_del": n_del,
        "n_true_mod": n_mod,
        # PK sets not available in compact format; callers that need them
        # must use attack_info.json (debug runs only).
        "true_ins_pks": set(),
        "true_del_pks": set(),
        "true_mod_pks": set(),
    }


def _gt_from_attack_info(info: Optional[Dict]) -> Dict[str, Any]:
    """New unified format: top-level s_ins / s_del / s_mod lists."""
    gt = {"db_tampered_true": False,
          "n_true_ins": 0, "n_true_del": 0, "n_true_mod": 0,
          "true_ins_pks": set(), "true_del_pks": set(), "true_mod_pks": set()}
    if info is None:
        return gt
    s_ins = info.get("s_ins", []) or []
    s_del = info.get("s_del", []) or []
    s_mod = info.get("s_mod", []) or []
    gt["true_ins_pks"] = set(str(x) for x in s_ins)
    gt["true_del_pks"] = set(str(x) for x in s_del)
    gt["true_mod_pks"] = set(str(x) for x in s_mod)
    gt["n_true_ins"] = len(gt["true_ins_pks"])
    gt["n_true_del"] = len(gt["true_del_pks"])
    gt["n_true_mod"] = len(gt["true_mod_pks"])
    gt["db_tampered_true"] = bool(s_ins or s_del or s_mod)
    return gt


# ── Localization metrics for Proposed ────────────────────────────────────────

def _proposed_localization(detect: Dict, gt: Dict) -> Dict[str, Optional[float]]:
    """Fallback F1_all computation from raw detect output (pre-compact format).

    Only called when f1_all is absent from detect_result.json — i.e., results
    produced before Step 4 was rewritten.  The new compact format pre-computes
    f1_all in Step 4 and stores it as a scalar, so this path is not triggered
    for any newly generated results.

    NOTE: detect.get("s_ins") etc. are lists in the OLD format (detect.py full
    output), but NOT present in the new compact format.  If this fallback is
    somehow called on new-format results it will silently return None/None/None,
    which is safe (the caller checks for None and skips the row in aggregation).
    """
    s_ins_raw = detect.get("s_ins") or []
    s_del_raw = detect.get("s_del") or []
    s_mod_raw = detect.get("s_mod") or []
    # New compact format: s_ins/del/mod are absent → lists are empty → return None
    if not s_ins_raw and not s_del_raw and not s_mod_raw:
        return {"precision": None, "recall": None, "f1": None, "fpr_tuple": None}
    true_all = gt["true_ins_pks"] | gt["true_del_pks"] | gt["true_mod_pks"]
    if not true_all:
        return {"precision": None, "recall": None, "f1": None, "fpr_tuple": None}
    pred_all = (set(str(x) for x in s_ins_raw) |
                set(str(x) for x in s_del_raw) |
                set(str(x) for x in s_mod_raw))
    tp = len(true_all & pred_all)
    fp = len(pred_all - true_all)
    fn = len(true_all - pred_all)
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return {"precision": prec, "recall": recall, "f1": f1, "fpr_tuple": None}


# ── Row extraction ────────────────────────────────────────────────────────────

def _extract_row(
    trial_dir: Path,
    machine_id: str, dataset: str, method: str,
    attack: str, rho: float, trial: int,
) -> Optional[Dict[str, Any]]:
    detect = _load(trial_dir / "detect_result.json")
    if detect is None:
        return None

    timing = _load(trial_dir / "timing.json") or {}
    atk    = _load(trial_dir / "attack_info.json")  # may be None in production

    # Ground truth: prefer compact top-level fields (new format written by Step 4),
    # then attack_info.json (debug runs), then zeros as last resort.
    # New format has no "_gt" sub-block; _gt_from_detect() reads top-level fields.
    if "db_tampered_true" in detect or "n_true_ins" in detect:
        gt = _gt_from_detect(detect)
    elif atk is not None:
        gt = _gt_from_attack_info(atk)
    else:
        gt = {"db_tampered_true": False, "n_true_ins": 0, "n_true_del": 0,
              "n_true_mod": 0, "true_ins_pks": set(),
              "true_del_pks": set(), "true_mod_pks": set()}

    # Metadata from _meta block (if available)
    meta = detect.get("_meta", {})
    wm_seed     = meta.get("wm_seed",     1000 + trial)
    attack_seed = meta.get("attack_seed", trial)
    mode        = meta.get("mode", "rho")         # "rho", "count", or "tc"
    count       = meta.get("count", None)         # per-op absolute count
    # total_count: from _meta if present, else infer from directory name (tc{N})
    total_count = meta.get("total_count", None)
    if total_count is None and mode == "tc":
        # parent of trial_dir is the tc{N} directory
        try:
            tc_dirname = trial_dir.parent.name   # e.g. "tc10"
            if tc_dirname.startswith("tc"):
                total_count = int(tc_dirname[2:])
        except Exception:
            pass

    meta = detect.get("_meta", {})
    config_name = meta.get("config_name")   # param analysis tag; None for normal runs

    row: Dict[str, Any] = {
        "machine_id":  machine_id,
        "dataset":     dataset,
        "method":      method,
        "config_name": config_name,
        "attack":      attack,
        "mode":        mode,
        "rho":         rho,
        "count":       count,
        "total_count": total_count,
        "trial":       trial,
        "wm_seed":     wm_seed,
        "attack_seed": attack_seed,
        # Ground truth (compact format: counts only)
        "db_tampered_true": detect.get("db_tampered_true", gt["db_tampered_true"]),
        "n_true_ins":       detect.get("n_true_ins",       gt["n_true_ins"]),
        "n_true_del":       detect.get("n_true_del",       gt["n_true_del"]),
        "n_true_mod":       detect.get("n_true_mod",       gt["n_true_mod"]),
        # Detection
        "db_tampered_pred":      detect.get("db_tampered_pred", detect.get("db_tampered")),
        "n_groups_tampered_pred": detect.get("n_groups_tampered", detect.get("n_tampered_groups")),
        "WAR":                   detect.get("WAR"),
        "WDR":                   detect.get("WDR"),
        "n_pred_ins":            detect.get("n_pred_ins", detect.get("n_ins")),
        "n_pred_del":            detect.get("n_pred_del", detect.get("n_del")),
        "n_pred_mod":            detect.get("n_pred_mod", detect.get("n_mod")),
        "iblt_decode_success":   detect.get("iblt_decode_success"),
        "status":                detect.get("status"),
        "localization_valid":    detect.get("localization_valid"),
        "failure_reason":        detect.get("failure_reason"),
        # Localization metrics (pre-computed in compact format, or compute from _gt)
        "precision_ins": detect.get("precision_ins"),
        "recall_ins":    detect.get("recall_ins"),
        "f1_ins":        detect.get("f1_ins"),
        "precision_del": detect.get("precision_del"),
        "recall_del":    detect.get("recall_del"),
        "f1_del":        detect.get("f1_del"),
        "precision_mod": detect.get("precision_mod"),
        "recall_mod":    detect.get("recall_mod"),
        "f1_mod":        detect.get("f1_mod"),
        "precision_all": detect.get("precision_all"),
        "recall_all":    detect.get("recall_all"),
        "f1_all":        detect.get("f1_all"),
        "fpr_tuple":     detect.get("fpr_tuple"),
        # Tuple-level confusion matrix (stored in compact format from Step 4).
        # None for B3/B5 (stats-only) and for old results without this field.
        "TP": detect.get("TP"),
        "TN": detect.get("TN"),
        "FP": detect.get("FP"),
        "FN": detect.get("FN"),
        # Storage (bytes): from ca_registration.json via Step 4.
        # Only populated for proposed method; None for baselines (no CA storage_bytes).
        "compressed_storage_bytes": detect.get("compressed_storage_bytes"),  # g×b/8
        "iblt_storage_bytes":       detect.get("iblt_storage_bytes"),
        "total_compressed_bytes":   detect.get("total_compressed_bytes"),
        # Runtime (ms) from timing.json (written by _run_detect_job.sh)
        "attack_ms": timing.get("attack_ms"),
        "detect_ms": timing.get("detect_ms"),
        "total_ms":  timing.get("total_ms"),
    }
    # 4-class tuple-status confusion (only Proposed/B0; absent for B1..B7).
    # 9 integers: rows = true op, cols = pred op; intact rows/cols derived from
    # n_true_*/n_pred_* at aggregation/plot time.
    xc = detect.get("status_xclass") or {}
    for k in ("true_ins_pred_ins","true_ins_pred_del","true_ins_pred_mod",
              "true_del_pred_ins","true_del_pred_del","true_del_pred_mod",
              "true_mod_pred_ins","true_mod_pred_del","true_mod_pred_mod"):
        row[f"xc_{k}"] = xc.get(k)

    pred = row["db_tampered_pred"]
    db_t = row["db_tampered_true"]
    row["detection_correct"] = (bool(pred) == bool(db_t)) if pred is not None else None

    # B1–B7 do not natively output operation-labelled primary-key prediction
    # sets (D_hat_ins, D_hat_del, D_hat_mod).  Older compact result files may
    # contain fpr_tuple=0 because the wrapper compared empty s_* sets; coerce
    # those baseline values to N/A so aggregate tables do not imply zero false
    # positives for tuple-level localization that the baseline never attempted.
    if method != "proposed":
        row["fpr_tuple"] = None

    # Statistics-only baselines (B3 Khan, B5 Alfagi) emit only WAR/WDR aggregate
    # signals — they produce NO tuple-level primary-key predictions at all.
    # Older detect_result.json files stored precision/recall/f1 = 0.0 for them,
    # which is misleading (it implies "attempted localization and got everything
    # wrong" rather than "no localization capability").  Force these to N/A so
    # downstream tables exclude them from tuple-level comparisons.
    STATS_ONLY_METHODS = {"b3", "b5"}
    if method in STATS_ONLY_METHODS:
        row["precision_all"] = None
        row["recall_all"]    = None
        row["f1_all"]        = None
        row["precision_ins"] = row["recall_ins"] = row["f1_ins"] = None
        row["precision_del"] = row["recall_del"] = row["f1_del"] = None
        row["precision_mod"] = row["recall_mod"] = row["f1_mod"] = None
        row["TP"] = row["TN"] = row["FP"] = row["FN"] = None

    # Fallback: if pre-computed metrics are missing (old format), compute now
    if row["f1_all"] is None and method == "proposed":
        loc = _proposed_localization(detect, gt)
        row["precision_all"] = loc.get("precision")
        row["recall_all"]    = loc.get("recall")
        row["f1_all"]        = loc.get("f1")
        row["fpr_tuple"]     = loc.get("fpr_tuple")

    return row


# ── Directory walker ──────────────────────────────────────────────────────────

def walk_results(
    results_dir: Path,
    machine_ids: Optional[List[str]] = None,
    flat: bool = False,
) -> List[Dict[str, Any]]:
    """Walk results into per-trial rows.

    Default layout:  results/{machine}/{dataset}/{method}/{attack}/{rho}/trial_N/
    Flat layout      (flat=True, no machine level):
                     results/{dataset}/{method}/{attack}/{rho}/trial_N/
    In flat mode the machine_id field is set to the results_dir basename.
    """
    rows = []

    if flat:
        machine_dirs = [results_dir]          # results_dir itself holds datasets
    else:
        machine_dirs = [d for d in sorted(results_dir.iterdir()) if d.is_dir()]

    for machine_dir in machine_dirs:
        mid = results_dir.name if flat else machine_dir.name
        if machine_ids and mid not in machine_ids:
            continue

        for dataset_dir in sorted(machine_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            for method_dir in sorted(dataset_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name

                for atk_dir in sorted(method_dir.iterdir()):
                    if not atk_dir.is_dir() or atk_dir.name == "watermarks":
                        continue
                    attack = atk_dir.name  # e.g. "a3"

                    for rho_dir in sorted(atk_dir.iterdir()):
                        if not rho_dir.is_dir():
                            continue
                        dname = rho_dir.name
                        # rho-based:   "rho0.01"  → rho=float, count=None
                        # count-based: "k5"       → rho=None,  count=int
                        # legacy:      "0.01"     → rho=float, count=None
                        if dname.startswith("rho"):
                            try:
                                rho = float(dname[3:])
                            except ValueError:
                                continue
                        elif dname.startswith("k"):
                            try:
                                int(dname[1:])   # validate it's actually a number
                            except ValueError:
                                continue
                            rho = None           # per-op count directory
                        elif dname.startswith("tc"):
                            try:
                                int(dname[2:])   # validate total-count number
                            except ValueError:
                                continue
                            rho = None           # total-count directory
                        else:
                            try:
                                rho = float(dname)   # legacy bare float
                            except ValueError:
                                continue

                        for trial_dir in sorted(rho_dir.iterdir()):
                            if not trial_dir.is_dir():
                                continue
                            tm = TRIAL_RE.match(trial_dir.name)
                            if not tm:
                                continue
                            trial = int(tm.group(1))

                            row = _extract_row(
                                trial_dir, mid, dataset, method,
                                attack, rho, trial,
                            )
                            if row is not None:
                                rows.append(row)
    return rows


# ── Aggregation ───────────────────────────────────────────────────────────────

_METRIC_COLS = [
    "detection_correct", "precision", "recall", "f1", "fpr_tuple",
    "precision_ins", "recall_ins", "f1_ins",
    "precision_del", "recall_del", "f1_del",
    "precision_mod", "recall_mod", "f1_mod",
    "precision_all", "recall_all", "f1_all",
    "WAR", "WDR", "n_pred_ins", "n_pred_del", "n_pred_mod",
    "attack_ms", "detect_ms", "total_ms",
]
# NOTE: rho is NaN for count/tc-mode rows (and vice versa) — group with
# dropna=False so absolute-count experiments are not silently dropped.
_GROUP_COLS = ["dataset", "method", "attack", "mode", "rho", "count", "total_count"]


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per (dataset, method, attack, strength) + DR and FAR."""
    agg_rows = []
    group_cols = [c for c in _GROUP_COLS if c in df.columns]
    for keys, grp in df.groupby(group_cols, dropna=False):
        row: Dict[str, Any] = dict(zip(group_cols, keys))
        row["n_trials"]   = len(grp)
        row["n_machines"] = int(grp["machine_id"].nunique())

        tampered = grp[grp["db_tampered_true"] == True]
        clean    = grp[grp["db_tampered_true"] == False]
        row["DR"]  = float(tampered["db_tampered_pred"].mean()) if len(tampered) > 0 else None
        row["FAR"] = float(clean["db_tampered_pred"].mean())    if len(clean)    > 0 else None

        for col in _METRIC_COLS:
            if col not in grp.columns:
                row[f"{col}_mean"] = None
                row[f"{col}_std"]  = None
                continue
            vals = grp[col].dropna().astype(float)
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else None
            row[f"{col}_std"]  = float(vals.std())  if len(vals) > 1 else None

        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


# ── Per-attack summary tables ─────────────────────────────────────────────────

def summarize_by_attack(df_raw: pd.DataFrame, output_dir: Path) -> None:
    """For each (dataset, method, attack): one CSV with rho as rows.

    Columns: rho, n_trials, DR, FAR, F1_mean, F1_std, precision_mean,
             recall_mean, detect_ms_mean, detect_ms_std
    Also embeds dataset/method/attack as header metadata.
    """
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    key_cols = ["dataset", "method", "attack"]
    metric_pairs = [
        ("db_tampered_pred", "DR/FAR"),
        ("f1",        "F1"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("detect_ms", "detect_ms"),
    ]

    for (dataset, method, attack), grp in df_raw.groupby(key_cols):
        rows = []
        for rho, rho_grp in grp.groupby("rho"):
            r: Dict[str, Any] = {
                "rho": rho,
                "n_trials": len(rho_grp),
                "dataset": dataset,
                "method": method,
                "attack": attack,
            }
            # DR (tampered trials only) and FAR (clean trials only)
            tampered = rho_grp[rho_grp["db_tampered_true"] == True]
            clean    = rho_grp[rho_grp["db_tampered_true"] == False]
            r["DR"]  = round(float(tampered["db_tampered_pred"].mean()), 4) if len(tampered) > 0 else None
            r["FAR"] = round(float(clean["db_tampered_pred"].mean()), 4)    if len(clean)    > 0 else None

            for col, label in [("f1","F1"), ("precision","Precision"),
                                ("recall","Recall"), ("detect_ms","detect_ms")]:
                if col in rho_grp.columns:
                    vals = rho_grp[col].dropna().astype(float)
                    r[f"{label}_mean"] = round(float(vals.mean()), 4) if len(vals) > 0 else None
                    r[f"{label}_std"]  = round(float(vals.std()),  4) if len(vals) > 1 else None

            rows.append(r)

        if not rows:
            continue

        summary_df = pd.DataFrame(rows)
        fname = f"{dataset}_{method}_{attack}.csv"
        summary_df.to_csv(summary_dir / fname, index=False)

    n = sum(1 for _ in summary_dir.glob("*.csv"))
    print(f"  Per-attack summaries → {summary_dir}/  ({n} files)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Collect and aggregate fragile watermarking results"
    )
    p.add_argument("--results_dir", required=True,
                   help="Base results dir (contains machine_id sub-dirs)")
    p.add_argument("--output_dir", default=None,
                   help="Where to write CSVs (default: same as --results_dir)")
    p.add_argument("--machine_ids", default=None,
                   help="Comma-separated machine IDs to include (default: all)")
    p.add_argument("--raw_file",    default="all_results_raw.csv")
    p.add_argument("--agg_file",    default="all_results_agg.csv")
    p.add_argument("--no_summary",  action="store_true",
                   help="Skip per-attack summary CSV generation")
    p.add_argument("--flat", action="store_true",
                   help="No machine-name level: results_dir holds {dataset}/ "
                        "directly (layout results/<batch>/{dataset}/proposed/...)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    machine_ids = [m.strip() for m in args.machine_ids.split(",")] \
                  if args.machine_ids else None

    print(f"Scanning: {results_dir}")
    if machine_ids:
        print(f"Machines: {machine_ids}")

    rows = walk_results(results_dir, machine_ids, flat=args.flat)
    print(f"Completed trials found: {len(rows)}")

    if not rows:
        print("No results found. Exiting.")
        return

    df_raw = pd.DataFrame(rows)
    # Reorder: identity cols first
    id_cols = ["machine_id", "dataset", "method", "attack", "rho", "trial",
               "wm_seed", "attack_seed"]
    other = [c for c in df_raw.columns if c not in id_cols]
    df_raw = df_raw[id_cols + other]

    raw_path = output_dir / args.raw_file
    df_raw.to_csv(raw_path, index=False)
    print(f"Raw CSV   → {raw_path}  ({len(df_raw)} rows × {len(df_raw.columns)} cols)")

    df_agg = aggregate_results(df_raw)
    agg_path = output_dir / args.agg_file
    df_agg.to_csv(agg_path, index=False)
    print(f"Agg CSV   → {agg_path}  ({len(df_agg)} rows)")

    if getattr(args, 'summary', False):
        summarize_by_attack(df_raw, output_dir)
    # summaries/ not generated by default (redundant with all_results_agg.csv)


if __name__ == "__main__":
    main()
