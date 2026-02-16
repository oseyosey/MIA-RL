#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve


def _load_scores(path: str) -> List[float]:
    scores: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            obj = json.loads(ln)
            scores.append(float(obj["score"]))
    return scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate MIA scores – outputs AUROC and curves")
    p.add_argument("--members", required=True, help="JSONL with per-sample scores for training members")
    p.add_argument("--nonmembers", required=True, help="JSONL with per-sample scores for non-members")
    p.add_argument("--output", required=True, help="Path to write metrics JSON")
    p.add_argument("--fpr-list", type=str, default=None, help="Comma-separated list of FPR thresholds (e.g., 0.001,0.01) for TPR@FPR computation")
    p.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap resamples for AUROC CI (e.g., 1000). 0 disables bootstrapping.")
    p.add_argument("--higher-is-member", action="store_true", help="If set, higher scores indicate membership (default: lower scores indicate membership)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_scores = _load_scores(args.members)
    out_scores = _load_scores(args.nonmembers)

    # Filter out NaN and inf values, replacing them with 0.0
    in_scores = [0.0 if not np.isfinite(s) else float(s) for s in in_scores]
    out_scores = [0.0 if not np.isfinite(s) else float(s) for s in out_scores]

    # In MIA, often lower loss => more likely member. Convert to a decision score where higher => member.
    # Our attacks already tend to output lower-is-member; flip sign so that higher = member.
    # However, some metrics (like similarity scores) have higher-is-member semantics.
    if args.higher_is_member:
        y_scores = np.array(in_scores + out_scores)
    else:
        y_scores = np.array(in_scores + out_scores) * -1.0
    y_true = np.array([1] * len(in_scores) + [0] * len(out_scores))

    # Final check: ensure no NaN or inf values before computing AUROC
    if np.any(~np.isfinite(y_scores)):
        print(f"Warning: Found {np.sum(~np.isfinite(y_scores))} non-finite scores, replacing with 0.0")
        y_scores = np.nan_to_num(y_scores, nan=0.0, posinf=0.0, neginf=0.0)

    auroc = float(roc_auc_score(y_true, y_scores))
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = float(auc(recall, precision))

    # TPR@FPR thresholds
    tpr_at_fpr = {}
    if args.fpr_list:
        try:
            fpr_targets = [float(x) for x in args.fpr_list.split(",") if x.strip() != ""]
        except Exception:
            fpr_targets = []
        for ub in fpr_targets:
            # Find the last index where FPR <= ub; if none, tpr_at_fpr = 0.0
            idx = np.where(fpr <= ub)[0]
            tpr_at_fpr[str(ub)] = float(tpr[idx[-1]]) if len(idx) > 0 else 0.0

    # Bootstrap AUROC CI (simple percentile CI)
    auroc_ci = None
    if args.bootstrap and args.bootstrap > 0:
        rng = np.random.default_rng(seed=0)
        n = len(y_true)
        samples = []
        for _ in range(args.bootstrap):
            idx = rng.integers(0, n, size=n)
            try:
                samples.append(roc_auc_score(y_true[idx], y_scores[idx]))
            except Exception:
                continue
        if samples:
            lo, hi = np.percentile(samples, [2.5, 97.5]).tolist()
            auroc_ci = {"low": float(lo), "high": float(hi), "level": 0.95}

    out = {
        "num_members": len(in_scores),
        "num_nonmembers": len(out_scores),
        "auroc": auroc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
    }
    if tpr_at_fpr:
        out["tpr_at_fpr"] = tpr_at_fpr
    if auroc_ci is not None:
        out["auroc_ci95"] = auroc_ci
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()



