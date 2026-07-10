"""
src/training/evaluate.py
Evaluation helpers for the HAN spam detector.

Responsibilities:
  - Load a trained checkpoint and run full metrics on any (X, y) split
  - Produce confusion matrix, ROC curve, precision-recall curve
  - Print a comparison table vs paper-reported numbers (Table 3 / Table 4)
  - Save all outputs to outputs/<run_name>/

CLI usage (same-dataset):
    python -m src.training.evaluate \\
        --weights checkpoints/fold_01.weights.h5 \\
        --data    data/processed \\
        --split   test \\
        --run     baseline_enron

Cross-dataset usage:
    python -m src.training.evaluate \\
        --weights  checkpoints/fold_01.weights.h5 \\
        --data     data/processed \\
        --split    test \\
        --run      cross_enron_to_sa \\
        --x-file   data/processed/X_spamassassin_test.npy \\
        --y-file   data/processed/y_spamassassin_test.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)

from src.models.han import load_han_model
from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger("evaluate")

# Paper-reported numbers from Table 3 (same-dataset, Enron EN)
# Used only for the comparison print — not for training decisions.
PAPER_RESULTS = {
    "enron": {
        "accuracy":  0.958,
        "precision": 0.981,
        "recall":    0.937,
        "f1":        0.958,
        "auc":       0.989,
    },
    "spamassassin": {
        "accuracy":  0.955,
        "precision": 0.893,
        "recall":    0.978,
        "f1":        0.933,
        "auc":       0.987,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    batch_size: int | None = None,
) -> dict:
    """
    Run the model on (X, y) and return a dict of all metrics.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, auc,
        y_true, y_pred (binary), y_prob (continuous)
    """
    bs = batch_size or CFG.training.batch_size
    log.info(f"Predicting on {len(X)} samples (batch_size={bs}) ...")
    y_prob = model.predict(X, batch_size=bs, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    # sklearn metrics
    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "auc":       float(roc_auc),
        # raw arrays for plotting
        "y_true":    y,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "fpr":       fpr,
        "tpr":       tpr,
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham (0)", "Spam (1)"])
    ax.set_yticklabels(["Ham (0)", "Spam (1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{labels[i][j]}\n{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved confusion matrix → {out_path}")


def plot_roc_curve(fpr, tpr, roc_auc: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — HAN Spam Detector")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved ROC curve → {out_path}")


def plot_precision_recall(y_true, y_prob, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="purple", lw=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved P-R curve → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Paper comparison table
# ──────────────────────────────────────────────────────────────────────────────
def print_comparison_table(metrics: dict, dataset_key: str | None = None) -> None:
    """Print a side-by-side comparison with paper-reported values."""
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
    paper = PAPER_RESULTS.get(dataset_key or "", {})

    header = f"{'Metric':<12} {'Ours':>10} {'Paper':>10} {'Δ':>8}"
    print("\n" + "=" * 45)
    print("  MODEL vs PAPER COMPARISON")
    if dataset_key:
        print(f"  Dataset: {dataset_key}")
    print("=" * 45)
    print(header)
    print("-" * 45)
    for k in metric_keys:
        our_val   = metrics.get(k, float("nan"))
        paper_val = paper.get(k, float("nan"))
        delta     = our_val - paper_val if paper_val == paper_val else float("nan")
        delta_str = f"{delta:+.4f}" if delta == delta else "  n/a"
        paper_str = f"{paper_val:.4f}" if paper_val == paper_val else "  n/a"
        print(f"  {k:<10} {our_val:>10.4f} {paper_str:>10} {delta_str:>8}")
    print("=" * 45 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# High-level run function
# ──────────────────────────────────────────────────────────────────────────────
def run_evaluation(
    weights_path: str,
    data_dir: str,
    run_name: str = "eval",
    x_file: str | None = None,
    y_file: str | None = None,
    split: str = "test",
    threshold: float = 0.5,
    dataset_key: str | None = None,
    output_dir: str = "outputs",
) -> dict:
    """
    End-to-end evaluation pipeline.

    Parameters
    ----------
    weights_path : path to .weights.h5 checkpoint
    data_dir     : directory containing manifest.json, embedding_matrix.npy, X_*.npy, y_*.npy
    run_name     : subdirectory name under output_dir for saving plots/metrics
    x_file       : override X array path (for cross-dataset eval)
    y_file       : override y array path (for cross-dataset eval)
    split        : "test" or "train" — used to build default X/y paths
    threshold    : decision threshold (default 0.5)
    dataset_key  : one of "enron" | "spamassassin" — for paper comparison
    output_dir   : root output directory

    Returns
    -------
    dict of scalar metrics (accuracy, precision, recall, f1, auc)
    """
    data_dir = Path(data_dir)
    out_dir  = Path(output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load manifest and embedding matrix ───────────────────────────────────
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)

    embedding_matrix = np.load(data_dir / "embedding_matrix.npy")
    vocab_size = embedding_matrix.shape[0]
    embed_dim  = embedding_matrix.shape[1]
    max_sentences = manifest["max_sentences"]
    max_words     = manifest["max_words"]

    log.info(
        f"manifest: vocab_size={vocab_size}, embed_dim={embed_dim}, "
        f"max_sentences={max_sentences}, max_words={max_words}"
    )

    # ── Load evaluation data ─────────────────────────────────────────────────
    X_path = x_file or str(data_dir / f"X_{split}.npy")
    y_path = y_file or str(data_dir / f"y_{split}.npy")
    log.info(f"Loading X from {X_path}")
    log.info(f"Loading y from {y_path}")
    X = np.load(X_path)
    y = np.load(y_path)
    log.info(f"Eval set: {X.shape}, spam={y.sum()}, ham={(y==0).sum()}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_han_model(
        weights_path=weights_path,
        max_sentences=max_sentences,
        max_words=max_words,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        embedding_matrix=embedding_matrix,
    )

    # ── Run evaluation ────────────────────────────────────────────────────────
    metrics = evaluate_model(model, X, y, threshold=threshold)

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(
        metrics["y_true"], metrics["y_pred"],
        target_names=["Ham", "Spam"],
    )
    log.info(f"\n{report}")
    (out_dir / "classification_report.txt").write_text(report)

    # ── Save scalar metrics ───────────────────────────────────────────────────
    scalar_metrics = {k: v for k, v in metrics.items()
                      if k not in ("y_true", "y_pred", "y_prob", "fpr", "tpr")}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(scalar_metrics, f, indent=2)
    log.info(f"Saved metrics.json → {out_dir / 'metrics.json'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        metrics["y_true"], metrics["y_pred"],
        out_dir / "confusion_matrix.png",
    )
    plot_roc_curve(
        metrics["fpr"], metrics["tpr"], metrics["auc"],
        out_dir / "roc_curve.png",
    )
    plot_precision_recall(
        metrics["y_true"], metrics["y_prob"],
        out_dir / "precision_recall.png",
    )

    # ── Print comparison ──────────────────────────────────────────────────────
    print_comparison_table(scalar_metrics, dataset_key=dataset_key)

    log.info(f"All outputs saved to: {out_dir}")
    return scalar_metrics


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained HAN checkpoint.")
    p.add_argument("--weights",   required=True,  help="Path to .weights.h5 checkpoint")
    p.add_argument("--data",      required=True,  help="Path to data/processed directory")
    p.add_argument("--run",       default="eval", help="Name for output subdirectory")
    p.add_argument("--split",     default="test", choices=["train", "test"],
                   help="Which split to evaluate (default: test)")
    p.add_argument("--x-file",    default=None,   help="Override X path (cross-dataset)")
    p.add_argument("--y-file",    default=None,   help="Override y path (cross-dataset)")
    p.add_argument("--threshold", default=0.5,    type=float,
                   help="Decision threshold (default: 0.5)")
    p.add_argument("--dataset",   default=None,
                   choices=["enron", "spamassassin"],
                   help="Dataset key for paper comparison table")
    p.add_argument("--output-dir", default="outputs",
                   help="Root output directory (default: outputs)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(
        weights_path=args.weights,
        data_dir=args.data,
        run_name=args.run,
        x_file=args.x_file,
        y_file=args.y_file,
        split=args.split,
        threshold=args.threshold,
        dataset_key=args.dataset,
        output_dir=args.output_dir,
    )
