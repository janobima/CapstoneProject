# ------------------------------------------------------------
# NTU Capstone Project: Group Cohesion Estimation using LLM and Deep Learning
# Maryam Ali Aljanobi
# April 2026
#
# File: Step6.6_LLM_GPU_learning-buckets
#
# This code is set to experimnt with post-hoc bucket evaluation for cohesion:
# - Reads predictions.csv (clip_id + predicted_cohesion)
# - Reads labels.csv (clip_id + y_true)
# - Converts BOTH y_true and y_pred into low/mid/high buckets
# - Computes bucket metrics 
# - Saves JSON metrics + several plots
#
# Example:
# python Step6.6_LLM_GPU_learning-buckets.py \
#   --pred_csv $/gce_project/results/llm_runs/qwen3b_retrieval_k16/predictions.csv \
#   --labels_csv $/gce_project/results/llm_eval_labels.csv \
#   --outdir $/gce_project/results/llm_runs/qwen3b_retrieval_k16/bucket_eval \
#   --clip_col clip_id \
#   --pred_col predicted_cohesion \
#   --true_col y_true

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# --------------------------
# Bucketing rule 
# --------------------------
def bucketize_score(x: float) -> str:
    """
    Default rule:
      1-2 -> low
      4-5 -> mid
      6-7 -> high
    """
    if x <= 2:
        return "low"
    elif x <= 5:
        return "mid"
    else:
        return "high"

BUCKET_ORDER = ["low", "mid", "high"]

def clamp_1_7(x) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        v = 4
    return max(1, min(7, v))

def safe_make_dir(d: str):
    os.makedirs(d, exist_ok=True)

# --------------------------
# Plot helpers 
# --------------------------
def plot_confusion_matrix(cm, labels, title, outpath, normalize=False):
    """
    cm: confusion matrix (counts & normalized)
    """
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True bucket")
    ax.set_xlabel("Predicted bucket")

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if normalize:
                text = f"{val:.2f}"
            else:
                text = f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_bucket_distribution(y_true_b, y_pred_b, outpath):
    true_counts = pd.Series(y_true_b).value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)
    pred_counts = pd.Series(y_pred_b).value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)

    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()

    x = np.arange(len(BUCKET_ORDER))
    width = 0.35

    ax.bar(x - width/2, true_counts.values, width, label="True")
    ax.bar(x + width/2, pred_counts.values, width, label="Pred")

    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_ORDER)
    ax.set_ylabel("Count")
    ax.set_title("Bucket distribution (true vs predicted)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_scatter_true_vs_pred(y_true, y_pred, outpath):
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.set_xlabel("True cohesion (1–7)")
    ax.set_ylabel("Predicted cohesion (1–7)")
    ax.set_title("True vs predicted cohesion (scatter)")
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0.5, 7.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_error_histogram(y_true, y_pred, outpath):
    err = (np.asarray(y_pred) - np.asarray(y_true)).astype(float)
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.hist(err, bins=np.arange(-6.5, 7.5, 1), rwidth=0.9)
    ax.set_xlabel("Prediction error (pred - true)")
    ax.set_ylabel("Count")
    ax.set_title("Error histogram")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

# --------------------------
# Metric helpers
# --------------------------
def ordinal_tolerance_accuracy(y_true, y_pred, tol: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) <= tol))

def bucket_neighbor_tolerance(y_true_b, y_pred_b) -> float:
    """
    Neighbor tolerance on buckets:
      correct = exact match OR adjacent class (low<->mid, mid<->high).
      low<->high is NOT adjacent.
    """
    idx = {b: i for i, b in enumerate(BUCKET_ORDER)}
    ok = 0
    for t, p in zip(y_true_b, y_pred_b):
        if t not in idx or p not in idx:
            continue
        if abs(idx[t] - idx[p]) <= 1:
            ok += 1
    return float(ok / len(y_true_b)) if len(y_true_b) > 0 else float("nan")

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="Path to predictions.csv from your LLM run")
    ap.add_argument("--labels_csv", required=True, help="Path to llm_eval_labels.csv (clip_id, y_true, ...)")
    ap.add_argument("--outdir", required=True, help="Output directory for bucket eval + plots")

    ap.add_argument("--clip_col", default="clip_id", help="clip id column name (pred + label files)")
    ap.add_argument("--pred_col", default="predicted_cohesion", help="prediction column name in pred_csv")
    ap.add_argument("--true_col", default="y_true", help="ground truth column name in labels_csv")

    ap.add_argument("--split_col", default="split", help="optional split column in labels_csv")
    ap.add_argument("--only_split", default="", help="if set, filter labels by split name (e.g., test)")

    args = ap.parse_args()
    safe_make_dir(args.outdir)

    # Load
    pred_df = pd.read_csv(args.pred_csv)
    lab_df = pd.read_csv(args.labels_csv)

    # Optional split filtering
    if args.only_split:
        if args.split_col in lab_df.columns:
            lab_df = lab_df[lab_df[args.split_col].astype(str).str.lower() == args.only_split.lower()].copy()

    # Keep needed columns
    if args.clip_col not in pred_df.columns:
        raise ValueError(f"pred_csv missing clip id column: {args.clip_col}")
    if args.pred_col not in pred_df.columns:
        raise ValueError(f"pred_csv missing prediction column: {args.pred_col}")

    if args.clip_col not in lab_df.columns:
        raise ValueError(f"labels_csv missing clip id column: {args.clip_col}")
    if args.true_col not in lab_df.columns:
        raise ValueError(f"labels_csv missing ground truth column: {args.true_col}")

    pred_df = pred_df[[args.clip_col, args.pred_col]].copy()
    lab_df = lab_df[[args.clip_col, args.true_col]].copy()

    # Merge (inner join: only matched IDs)
    df = pd.merge(lab_df, pred_df, on=args.clip_col, how="inner")
    if df.empty:
        raise RuntimeError("No matched clip_ids between labels and predictions. Check clip_id formatting.")

    # Clean + clamp
    df["y_true"] = df[args.true_col].apply(clamp_1_7).astype(int)
    df["y_pred"] = df[args.pred_col].apply(clamp_1_7).astype(int)

    # Buckets
    df["y_true_bucket"] = df["y_true"].apply(bucketize_score)
    df["y_pred_bucket"] = df["y_pred"].apply(bucketize_score)

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    y_true_b = df["y_true_bucket"].tolist()
    y_pred_b = df["y_pred_bucket"].tolist()

    # Bucket metrics
    acc = float(accuracy_score(y_true_b, y_pred_b))
    f1_macro = float(f1_score(y_true_b, y_pred_b, average="macro"))
    f1_weighted = float(f1_score(y_true_b, y_pred_b, average="weighted"))

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

    cm_counts = confusion_matrix(y_true_b, y_pred_b, labels=BUCKET_ORDER)
    cm_norm = cm_counts.astype(float) / np.maximum(cm_counts.sum(axis=1, keepdims=True), 1.0)

    # Ordinal tolerance on 1–7
    tol1 = ordinal_tolerance_accuracy(y_true, y_pred, tol=1)
    tol2 = ordinal_tolerance_accuracy(y_true, y_pred, tol=2)

    # Bucket neighbor tolerance
    bucket_adj = bucket_neighbor_tolerance(y_true_b, y_pred_b)

    # Save metrics JSON
    out_metrics = {
        "n_matched": int(len(df)),
        "bucket_rule": {"low": "1-2", "mid": "4-5", "high": "6-7"},
        "bucket_metrics": {
            "accuracy": acc,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "neighbor_tolerance_accuracy": bucket_adj,
            "per_class": per_class,
        },
        "ordinal_metrics_1to7": {
            "tolerance_acc_abs_error_le_1": tol1,
            "tolerance_acc_abs_error_le_2": tol2,
            "mean_abs_error": float(np.mean(np.abs(y_true - y_pred))),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        },
        "confusion_matrix": {
            "labels": BUCKET_ORDER,
            "counts": cm_counts.tolist(),
            "normalized_by_true_row": cm_norm.tolist(),
        },
    }

    metrics_path = os.path.join(args.outdir, "bucket_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    # Save merged table too (helpful for debugging/report)
    merged_path = os.path.join(args.outdir, "merged_bucket_eval.csv")
    df[[args.clip_col, "y_true", "y_pred", "y_true_bucket", "y_pred_bucket"]].to_csv(merged_path, index=False)

    # Plots
    plot_confusion_matrix(
        cm_counts, BUCKET_ORDER,
        title="Confusion matrix (bucket counts)",
        outpath=os.path.join(args.outdir, "confusion_matrix_counts.png"),
        normalize=False
    )
    plot_confusion_matrix(
        cm_norm, BUCKET_ORDER,
        title="Confusion matrix (row-normalized)",
        outpath=os.path.join(args.outdir, "confusion_matrix_normalized.png"),
        normalize=True
    )
    plot_bucket_distribution(y_true_b, y_pred_b, os.path.join(args.outdir, "bucket_distribution.png"))
    plot_scatter_true_vs_pred(y_true, y_pred, os.path.join(args.outdir, "scatter_true_vs_pred.png"))
    plot_error_histogram(y_true, y_pred, os.path.join(args.outdir, "error_histogram.png"))

    print("Saved:", metrics_path)
    print("Saved:", merged_path)
    print("Saved plots into:", args.outdir)
    print("\nBucket accuracy:", acc)
    print("Macro-F1:", f1_macro, "| Weighted-F1:", f1_weighted)
    print("Neighbor-tolerance bucket accuracy:", bucket_adj)
    print("Tolerance acc (|err|<=1):", tol1, "| (|err|<=2):", tol2)


if __name__ == "__main__":
    main()