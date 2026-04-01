# ------------------------------------------------------------
# NTU Capstone Project: Group Cohesion Estimation using LLM and Deep Learning
# Maryam Ali Aljanobi
# April 2026
#
# File: Step7_ML_AVS_buckets.py
#
# This code is set to experimnt with post-hoc bucket evaluation for cohesion for the ML model:
# - Reads ML best predictions.csv (clip_id + predicted_cohesion)
# - Reads labels.csv (clip_id + y_true)
# - Converts BOTH y_true and y_pred into low/mid/high buckets
# - Computes bucket metrics 
# ------------------------------------------------------------


import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

BUCKET_ORDER = ["low", "mid", "high"]

def bucketize_score(x: float) -> str:
    # your updated rule
    # low = 1-2, mid = 3-5, high = 6-7
    if x <= 2:
        return "low"
    elif x <= 5:
        return "mid"
    else:
        return "high"

def clamp_1_7(x):
    try:
        v = float(x)
    except Exception:
        v = 4.0
    return min(7.0, max(1.0, v))

def ordinal_tolerance_accuracy(y_true, y_pred, tol=1):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) <= tol))

def bucket_neighbor_tolerance(y_true_b, y_pred_b):
    idx = {b: i for i, b in enumerate(BUCKET_ORDER)}
    ok = 0
    for t, p in zip(y_true_b, y_pred_b):
        if abs(idx[t] - idx[p]) <= 1:
            ok += 1
    return float(ok / len(y_true_b)) if len(y_true_b) > 0 else float("nan")

def evaluate_one_csv(csv_path, true_col="y_true", pred_col="y_pred"):
    df = pd.read_csv(csv_path)

    if true_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"{csv_path} must contain columns {true_col} and {pred_col}")

    y_true = df[true_col].apply(clamp_1_7).to_numpy()
    y_pred = df[pred_col].apply(clamp_1_7).to_numpy()

    y_true_b = [bucketize_score(x) for x in y_true]
    y_pred_b = [bucketize_score(x) for x in y_pred]

    acc = float(accuracy_score(y_true_b, y_pred_b))
    macro_f1 = float(f1_score(y_true_b, y_pred_b, average="macro"))
    weighted_f1 = float(f1_score(y_true_b, y_pred_b, average="weighted"))

    prec, rec, f1s, support = precision_recall_fscore_support(
        y_true_b, y_pred_b, labels=BUCKET_ORDER, zero_division=0
    )

    per_class = {}
    for i, b in enumerate(BUCKET_ORDER):
        per_class[b] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1s[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true_b, y_pred_b, labels=BUCKET_ORDER)

    return {
        "n_samples": int(len(df)),
        "bucket_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "neighbor_tolerance_bucket_accuracy": bucket_neighbor_tolerance(y_true_b, y_pred_b),
        "tolerance_acc_abs_error_le_1": ordinal_tolerance_accuracy(y_true, y_pred, tol=1),
        "tolerance_acc_abs_error_le_2": ordinal_tolerance_accuracy(y_true, y_pred, tol=2),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": BUCKET_ORDER,
            "counts": cm.tolist(),
        },
    }

def mean_std(values):
    arr = np.array(values, dtype=float)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="Root results dir containing seed_* folders")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--true_col", default="y_true")
    ap.add_argument("--pred_col", default="y_pred")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    split_file = f"predictions_{args.split}.csv"

    seed_dirs = []
    for name in sorted(os.listdir(args.root_dir)):
        full = os.path.join(args.root_dir, name)
        if os.path.isdir(full) and name.startswith("seed_"):
            seed_dirs.append(full)

    if not seed_dirs:
        raise RuntimeError(f"No seed_* folders found in {args.root_dir}")

    all_results = {}
    metric_lists = {
        "bucket_accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
        "neighbor_tolerance_bucket_accuracy": [],
        "tolerance_acc_abs_error_le_1": [],
        "tolerance_acc_abs_error_le_2": [],
    }

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        csv_path = os.path.join(seed_dir, split_file)

        if not os.path.exists(csv_path):
            print(f"[warn] missing {csv_path}, skipping")
            continue

        res = evaluate_one_csv(csv_path, true_col=args.true_col, pred_col=args.pred_col)
        all_results[seed_name] = res

        for k in metric_lists:
            metric_lists[k].append(res[k])

        print(f"\n{seed_name} ({args.split})")
        print(f"  Bucket accuracy: {res['bucket_accuracy']:.4f}")
        print(f"  Macro-F1: {res['macro_f1']:.4f}")
        print(f"  Weighted-F1: {res['weighted_f1']:.4f}")
        print(f"  Neighbor-tolerance bucket accuracy: {res['neighbor_tolerance_bucket_accuracy']:.4f}")
        print(f"  Tolerance acc (|err|<=1): {res['tolerance_acc_abs_error_le_1']:.4f}")
        print(f"  Tolerance acc (|err|<=2): {res['tolerance_acc_abs_error_le_2']:.4f}")

    summary = {k: mean_std(v) for k, v in metric_lists.items()}

    out = {
        "bucket_rule": {"low": "1-2", "mid": "3-5", "high": "6-7"},
        "split_evaluated": args.split,
        "per_seed": all_results,
        "summary_mean_std": summary,
    }

    out_json = args.out_json
    if out_json is None:
        out_json = os.path.join(args.root_dir, f"bucket_eval_{args.split}_mean_std.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== Summary across seeds ===")
    for k, stats in summary.items():
        print(f"{k}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\nSaved:", out_json)

if __name__ == "__main__":
    main()