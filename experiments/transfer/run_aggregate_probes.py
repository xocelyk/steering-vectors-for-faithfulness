"""Train ridge probes on aggregate scenarios (gpqa_all, stanford_all).

For each (model, split_type, scenario) where scenario aggregates multiple
(dataset, cue) cells, pool the per-cell activations restricted to that
split's train/test rows, fit a ridge probe per layer on the pooled train
rows (per-feature normalization computed on train), and save artifacts
into the same probe directory used for per-cell probes:

  experiments/transfer/probes/<split_type>/<model_slug>__<scenario>.json
  experiments/transfer/probes/<split_type>/weights/<model_slug>__<scenario>.pt

Per-cell probes already in that directory are not touched.

Usage:
  uv run python experiments/transfer/run_aggregate_probes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from build_contrastive_vectors import (
    MODEL_REGISTRY,
    SCENARIOS,
    load_cell,
    pool_cells,
)
from common import write_json
from train_probe import fit_ridge_probe, pr_auc, roc_auc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=sorted(MODEL_REGISTRY))
    p.add_argument("--split_types", nargs="+", default=["meek", "giovanni"], choices=["meek", "giovanni"])
    p.add_argument("--scenarios", nargs="+", default=["gpqa_all", "stanford_all"])
    p.add_argument("--activations_dir", type=Path, default=Path("experiments/transfer/activations"))
    p.add_argument("--splits_dir", type=Path, default=Path("experiments/transfer/splits"))
    p.add_argument("--output_dir", type=Path, default=Path("experiments/transfer/probes"))
    p.add_argument("--ridge", type=float, default=1e-3)
    return p.parse_args()


def fit_and_score(x_train: np.ndarray, y_train: np.ndarray,
                  x_test: np.ndarray, y_test: np.ndarray,
                  ridge: float) -> dict:
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_n = (x_train - mu) / sigma
    x_test_n = (x_test - mu) / sigma
    weights = fit_ridge_probe(x_train_n, y_train.astype(np.float32), ridge)
    train_scores = np.concatenate(
        [x_train_n, np.ones((x_train_n.shape[0], 1), dtype=x_train_n.dtype)], axis=1
    ) @ weights
    test_scores = np.concatenate(
        [x_test_n, np.ones((x_test_n.shape[0], 1), dtype=x_test_n.dtype)], axis=1
    ) @ weights
    return {
        "weights": weights.astype(np.float32),
        "feature_mean": mu.squeeze(0).astype(np.float32),
        "feature_std": sigma.squeeze(0).astype(np.float32),
        "train_scores": train_scores.astype(np.float32),
        "test_scores": test_scores.astype(np.float32),
        "train_roc_auc": roc_auc(y_train.astype(int), train_scores),
        "test_roc_auc":  roc_auc(y_test.astype(int),  test_scores),
        "train_pr_auc":  pr_auc(y_train.astype(int),  train_scores),
        "test_pr_auc":   pr_auc(y_test.astype(int),   test_scores),
    }


def _row_metrics(r: dict) -> dict:
    return {
        "layer": r["layer"],
        "train_roc_auc": r["train_roc_auc"],
        "test_roc_auc":  r["test_roc_auc"],
        "train_pr_auc":  r["train_pr_auc"],
        "test_pr_auc":   r["test_pr_auc"],
    }


def main() -> None:
    args = parse_args()

    # Validate scenarios up-front: only aggregate (multi-cell) ones make sense here.
    aggregates: list[str] = []
    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            print(f"WARNING: unknown scenario {scenario}; skipping")
            continue
        if len(SCENARIOS[scenario]) <= 1:
            print(f"WARNING: {scenario} has only 1 cell; per-cell probes already cover it. Skipping.")
            continue
        aggregates.append(scenario)
    if not aggregates:
        raise SystemExit("No aggregate scenarios to run.")

    failed: list[tuple[str, str, str, str]] = []

    for split_type in args.split_types:
        out_dir = args.output_dir / split_type
        out_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = out_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        aggregate_summary: list[dict] = []

        for model_slug in args.models:
            if model_slug not in MODEL_REGISTRY:
                print(f"WARNING: unknown model_slug {model_slug}; skipping")
                continue

            for scenario in aggregates:
                cells_spec = SCENARIOS[scenario]
                print(f"\n=== [{split_type}] {model_slug} {scenario}  cells={cells_spec} ===")

                try:
                    cells = [
                        load_cell(args.activations_dir, args.splits_dir,
                                  model_slug, d, c, split_type)
                        for d, c in cells_spec
                    ]
                except FileNotFoundError as e:
                    print(f"  SKIP: {e}")
                    failed.append((split_type, model_slug, scenario, str(e)))
                    continue

                empty = [(c.dataset, c.cue) for c in cells if not c.train_ids or not c.test_ids]
                if empty:
                    print(f"  SKIP: empty splits for {empty}")
                    failed.append((split_type, model_slug, scenario, f"empty splits: {empty}"))
                    continue

                pooled = pool_cells(cells)

                # Pooled task_ids in the same row order as pool_cells concatenates them.
                pooled_task_ids: list[str] = []
                for c in cells:
                    pooled_task_ids.extend(c.task_ids)

                train_idx = np.flatnonzero(pooled.is_train)
                test_idx = np.flatnonzero(pooled.is_test)
                y_train = pooled.labels[train_idx]
                y_test = pooled.labels[test_idx]
                n_pos_train = int((y_train == 1).sum())
                n_pos_test = int((y_test == 1).sum())
                if (
                    n_pos_train in (0, len(y_train))
                    or n_pos_test in (0, len(y_test))
                ):
                    print(f"  SKIP degenerate labels: train_pos={n_pos_train}/{len(y_train)} "
                          f"test_pos={n_pos_test}/{len(y_test)}")
                    failed.append((split_type, model_slug, scenario, "degenerate labels"))
                    continue

                cell_rows: list[dict] = []
                weights_per_layer: dict[int, np.ndarray] = {}
                mu_per_layer: dict[int, np.ndarray] = {}
                sigma_per_layer: dict[int, np.ndarray] = {}
                scores_per_layer: dict[int, dict[str, np.ndarray]] = {}

                for layer in pooled.layers:
                    x = pooled.layer_data[layer]
                    fit = fit_and_score(x[train_idx], y_train, x[test_idx], y_test, args.ridge)
                    weights_per_layer[layer] = fit["weights"]
                    mu_per_layer[layer] = fit["feature_mean"]
                    sigma_per_layer[layer] = fit["feature_std"]
                    scores_per_layer[layer] = {
                        "train_scores": fit["train_scores"],
                        "test_scores":  fit["test_scores"],
                    }
                    cell_rows.append({
                        "split_type": split_type,
                        "model_slug": model_slug,
                        "scenario": scenario,
                        "layer": layer,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "n_pos_train": n_pos_train,
                        "n_neg_train": len(y_train) - n_pos_train,
                        "n_pos_test": n_pos_test,
                        "n_neg_test": len(y_test) - n_pos_test,
                        "train_roc_auc": fit["train_roc_auc"],
                        "test_roc_auc":  fit["test_roc_auc"],
                        "train_pr_auc":  fit["train_pr_auc"],
                        "test_pr_auc":   fit["test_pr_auc"],
                    })

                best_by_roc = max(
                    (r for r in cell_rows if r["test_roc_auc"] is not None),
                    key=lambda r: r["test_roc_auc"], default=None,
                )
                best_by_pr = max(
                    (r for r in cell_rows if r["test_pr_auc"] is not None),
                    key=lambda r: r["test_pr_auc"], default=None,
                )
                if best_by_roc is None:
                    print(f"  FAIL: no valid layers for {scenario}")
                    failed.append((split_type, model_slug, scenario, "no valid layers"))
                    continue

                pr_extra = ""
                if best_by_pr is not None:
                    pr_extra = (f" | best_by_pr layer={best_by_pr['layer']} "
                                f"test_pr_auc={best_by_pr['test_pr_auc']:.3f}")
                print(f"  best_by_roc layer={best_by_roc['layer']} "
                      f"test_roc_auc={best_by_roc['test_roc_auc']:.3f} "
                      f"(train_roc_auc={best_by_roc['train_roc_auc']:.3f}){pr_extra}")

                cell_summary = {
                    "split_type": split_type,
                    "model_slug": model_slug,
                    "scenario": scenario,
                    "cells": [{"dataset": c.dataset, "cue": c.cue} for c in cells],
                    "best_by_roc": _row_metrics(best_by_roc),
                    "best_by_pr":  _row_metrics(best_by_pr) if best_by_pr is not None else None,
                    "best_layer": best_by_roc["layer"],
                    "n_train": best_by_roc["n_train"],
                    "n_test": best_by_roc["n_test"],
                    "layers": cell_rows,
                }

                cell_summary_path = out_dir / f"{model_slug}__{scenario}.json"
                write_json(cell_summary_path, cell_summary)
                aggregate_summary.append({k: v for k, v in cell_summary.items() if k != "layers"})

                weights_path = weights_dir / f"{model_slug}__{scenario}.pt"
                train_task_ids = [pooled_task_ids[i] for i in train_idx]
                test_task_ids = [pooled_task_ids[i] for i in test_idx]
                torch.save({
                    "split_type": split_type,
                    "model_slug": model_slug,
                    "scenario": scenario,
                    "cells": [{"dataset": c.dataset, "cue": c.cue} for c in cells],
                    "ridge": args.ridge,
                    "train_task_ids": train_task_ids,
                    "test_task_ids":  test_task_ids,
                    "y_train": y_train.astype(np.float32),
                    "y_test":  y_test.astype(np.float32),
                    "layers": list(pooled.layers),
                    "weights":      {l: torch.from_numpy(w) for l, w in weights_per_layer.items()},
                    "feature_mean": {l: torch.from_numpy(m) for l, m in mu_per_layer.items()},
                    "feature_std":  {l: torch.from_numpy(s) for l, s in sigma_per_layer.items()},
                    "train_scores": {l: torch.from_numpy(scores_per_layer[l]["train_scores"]) for l in scores_per_layer},
                    "test_scores":  {l: torch.from_numpy(scores_per_layer[l]["test_scores"])  for l in scores_per_layer},
                }, weights_path)

        if aggregate_summary:
            summary_path = out_dir / "aggregate_best.json"
            write_json(summary_path, {"scenarios": aggregate_summary})
            print(f"\n[{split_type}] Wrote aggregate summary to {summary_path}")

    if failed:
        print(f"\nFAILED: {len(failed)}")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
