"""
Evaluation metrics for fragile watermarking baselines.

Metrics implemented:
- DR   (Detection Rate)     : fraction of tampered databases correctly detected
- FAR  (False Alarm Rate)   : fraction of clean databases falsely flagged
- Precision                 : TP / (TP + FP)  at tuple/group level
- Recall                    : TP / (TP + FN)  at tuple/group level
- F1                        : harmonic mean of Precision and Recall
- FPR_tuple                 : FP / (FP + TN)  at tuple/group level
- WAR  (Watermark Agreement Rate) : used by B3 Khan 2013 and B5 Alfagi 2016
- WDR  (Watermark Disagreement Rate) = 1 - WAR
- Jaccard similarity

All functions accept simple Python scalars or lists/arrays.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np


# ---------------------------------------------------------------------------
# Database-level detection metrics (DR / FAR)
# Used across multiple trials: pass lists of boolean outcomes.
# ---------------------------------------------------------------------------

def compute_dr(detected_list: List[bool]) -> float:
    """Detection Rate: fraction of tampered databases correctly detected.

    Args:
        detected_list: list of bool, one per trial, True = tampering detected.

    Returns:
        DR in [0, 1].
    """
    if len(detected_list) == 0:
        return 0.0
    return float(np.mean(detected_list))


def compute_far(false_alarm_list: List[bool]) -> float:
    """False Alarm Rate: fraction of clean databases falsely flagged.

    Args:
        false_alarm_list: list of bool, one per trial on CLEAN database,
                          True = falsely flagged as tampered.

    Returns:
        FAR in [0, 1].
    """
    if len(false_alarm_list) == 0:
        return 0.0
    return float(np.mean(false_alarm_list))


# ---------------------------------------------------------------------------
# Tuple/group-level localization metrics
# ---------------------------------------------------------------------------

def compute_localization_metrics(
    tampered_ids: Union[Set, List],
    flagged_ids: Union[Set, List],
    total_ids: Union[Set, List],
) -> Dict[str, float]:
    """Compute Precision, Recall, F1, FPR at tuple/group level.

    Empty-set convention (paper §7.7):
      - Both D_x = ∅ and D̂_x = ∅ → perfect clean outcome → P = R = F1 = 1.
      - Exactly one is empty (missed detection or false alarm) → P or R = 0, F1 = 0.

    Args:
        tampered_ids : set of IDs that were actually tampered.
        flagged_ids  : set of IDs flagged by the detector.
        total_ids    : set of all IDs (tampered + clean).

    Returns:
        Dict with keys: precision, recall, f1, fpr_tuple, tp, fp, fn, tn.
    """
    tampered = set(tampered_ids)
    flagged  = set(flagged_ids)
    total    = set(total_ids)
    clean    = total - tampered

    # Perfect clean outcome: verifier correctly outputs nothing on a clean input
    if len(tampered) == 0 and len(flagged) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "fpr_tuple": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "tn": len(clean)}

    tp = len(tampered & flagged)
    fp = len(flagged - tampered)
    fn = len(tampered - flagged)
    tn = len(clean - flagged)

    # One set empty: precision or recall is undefined → set to 0, F1 = 0 (paper §7.7)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr_tuple = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision":  round(precision, 6),
        "recall":     round(recall, 6),
        "f1":         round(f1, 6),
        "fpr_tuple":  round(fpr_tuple, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def aggregate_localization_metrics(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Average localization metrics over multiple trials.

    Returns dict of {metric_name: {"mean": ..., "std": ...}}.
    """
    keys = ["precision", "recall", "f1", "fpr_tuple"]
    result = {}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        result[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return result


# ---------------------------------------------------------------------------
# WAR / WDR  (B3 Khan 2013, B5 Alfagi 2016)
# ---------------------------------------------------------------------------

def compute_war_wdr(
    wm_original: List,
    wm_detected: List,
) -> Dict[str, float]:
    """Watermark Agreement Rate and Watermark Disagreement Rate.

    Compares two watermark vectors element-wise.

    Args:
        wm_original : reference watermark (list of floats or ints).
        wm_detected : detected watermark (list of floats or ints).

    Returns:
        Dict with 'WAR' and 'WDR' (both in [0, 1]).
    """
    assert len(wm_original) == len(wm_detected), "Watermark length mismatch"
    n = len(wm_original)
    if n == 0:
        return {"WAR": 1.0, "WDR": 0.0}

    matches = sum(1 for a, b in zip(wm_original, wm_detected) if a == b)
    war = matches / n
    wdr = 1.0 - war
    return {"WAR": round(war, 6), "WDR": round(wdr, 6)}


def compute_war_wdr_float(
    wm_original: List[float],
    wm_detected: List[float],
    tol: float = 1e-9,
) -> Dict[str, float]:
    """WAR/WDR with floating-point tolerance for frequency comparisons."""
    assert len(wm_original) == len(wm_detected)
    n = len(wm_original)
    if n == 0:
        return {"WAR": 1.0, "WDR": 0.0}
    matches = sum(1 for a, b in zip(wm_original, wm_detected) if abs(a - b) <= tol)
    war = matches / n
    return {"WAR": round(war, 6), "WDR": round(1.0 - war, 6)}


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def jaccard(set_a: Union[Set, List], set_b: Union[Set, List]) -> float:
    """Jaccard similarity between two sets."""
    a, b = set(set_a), set(set_b)
    if len(a | b) == 0:
        return 1.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarize_trials(
    dr_list: Optional[List[bool]] = None,
    far_list: Optional[List[bool]] = None,
    localization_list: Optional[List[Dict]] = None,
    war_list: Optional[List[float]] = None,
    wdr_list: Optional[List[float]] = None,
) -> Dict:
    """Aggregate all metrics across trials into mean ± std dict."""
    summary = {}

    if dr_list is not None and len(dr_list) > 0:
        vals = [float(v) for v in dr_list]
        summary["DR"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    if far_list is not None and len(far_list) > 0:
        vals = [float(v) for v in far_list]
        summary["FAR"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    if localization_list is not None and len(localization_list) > 0:
        summary["localization"] = aggregate_localization_metrics(localization_list)

    if war_list is not None and len(war_list) > 0:
        summary["WAR"] = {"mean": float(np.mean(war_list)), "std": float(np.std(war_list))}

    if wdr_list is not None and len(wdr_list) > 0:
        summary["WDR"] = {"mean": float(np.mean(wdr_list)), "std": float(np.std(wdr_list))}

    return summary
