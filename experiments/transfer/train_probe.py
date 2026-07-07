"""Train simple layerwise linear probes from activations to faithfulness labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from common import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="Scored trace JSONL")
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--label_field", type=str, default="faithfulness_score")
    parser.add_argument("--ridge", type=float, default=1e-3)
    return parser.parse_args()


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average ranks for ties.
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = ranks[positives].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Average Precision (AUC of the precision-recall curve).
    Equivalent to sklearn.metrics.average_precision_score for binary labels."""
    y_true = y_true.astype(int)
    n = len(y_true)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0 or n_pos == n:
        return None
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_sorted)
    rank = np.arange(1, n + 1, dtype=float)
    precision_at_k = tp_cumsum / rank
    return float((precision_at_k * y_sorted).sum() / n_pos)


def fit_ridge_probe(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    xtx = x_aug.T @ x_aug
    penalty = np.eye(xtx.shape[0], dtype=x.dtype) * ridge
    penalty[-1, -1] = 0.0
    return np.linalg.solve(xtx + penalty, x_aug.T @ y)


def main() -> None:
    args = parse_args()
    activations = torch.load(args.activations, weights_only=False)
    task_ids = list(activations["task_ids"])
    if "labels" in activations:
        labels_tensor = activations["labels"].float().cpu().numpy()
        if len(labels_tensor) != len(task_ids):
            raise ValueError("Activation labels length does not match task_ids")
        labels_by_id = {
            task_id: float(labels_tensor[i])
            for i, task_id in enumerate(task_ids)
        }
    else:
        labels_by_id = {
            record["task_id"]: float(record[args.label_field])
            for record in read_jsonl(args.labels)
            if args.label_field in record
            and not record.get("degenerate_reason")
            and record.get("degenerate_score") != 1
        }

    valid_idx = [i for i, task_id in enumerate(task_ids) if task_id in labels_by_id]
    if not valid_idx:
        raise ValueError("No overlapping task_ids between activations and labels")

    y = np.array([labels_by_id[task_ids[i]] for i in valid_idx], dtype=np.float32)
    rows = []
    layer_keys = sorted(
        [key for key in activations if key.startswith("layer_")],
        key=lambda key: int(key.split("_")[1]),
    )

    for layer_key in layer_keys:
        layer = int(layer_key.split("_")[1])
        x = activations[layer_key][valid_idx].float().cpu().numpy()
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
        weights = fit_ridge_probe(x, y, args.ridge)
        scores = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1) @ weights
        auc = roc_auc(y.astype(int), scores)
        rows.append({
            "layer": layer,
            "n": len(valid_idx),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "auc": auc,
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "n", "n_positive", "n_negative", "auc"])
        writer.writeheader()
        writer.writerows(rows)

    best = max((row for row in rows if row["auc"] is not None), key=lambda row: row["auc"], default=None)
    summary = {
        "activations": str(args.activations),
        "labels": str(args.labels),
        "label_field": args.label_field,
        "n_overlap": len(valid_idx),
        "best_layer": best["layer"] if best else None,
        "best_auc": best["auc"] if best else None,
        "layers": rows,
    }
    out_json = args.output_json or args.output_csv.with_suffix(".json")
    write_json(out_json, summary)
    print(f"Wrote probe AUCs to {args.output_csv}")
    print(json.dumps({"best_layer": summary["best_layer"], "best_auc": summary["best_auc"]}, indent=2))


if __name__ == "__main__":
    main()
