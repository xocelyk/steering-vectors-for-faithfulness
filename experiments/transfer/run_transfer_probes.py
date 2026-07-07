"""Sweep linear probes across every (model, dataset, cue) activation cell.

For each cell with a saved `mean_all_response.pt`:
  1. Split rows into train/test using
     `experiments/phase4/splits/{dataset}_{train,test}_task_ids.txt`.
  2. For every saved layer, fit a ridge linear probe on train rows (with
     per-feature normalization computed on train), score train and test rows,
     compute ROC-AUC for each.
  3. Pick best layer per cell by test AUC.

Outputs (under experiments/transfer/probes/<split_type>/):
  - all_layers.csv: one row per (model, dataset, cue, layer)
  - <model>__<dataset>__<cue>.json: per-cell layer sweep
  - best_per_cell.json: flat best-layer aggregate
  - weights/<model>__<dataset>__<cue>.pt: fitted probe weights + saved scores

Usage:
  uv run python experiments/transfer/run_transfer_probes.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from common import write_json
from train_probe import fit_ridge_probe, roc_auc, pr_auc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations_dir", type=Path, default=Path("experiments/transfer/activations"))
    p.add_argument("--splits_dir", type=Path, default=Path("experiments/transfer/splits"),
                   help="Per-scenario splits dir built by build_scenario_splits.py")
    p.add_argument("--output_dir", type=Path, default=Path("experiments/transfer/probes"))
    p.add_argument("--split_types", nargs="+", default=["meek", "giovanni"], choices=["meek", "giovanni"],
                   help="Which faithfulness population(s) to run probes on.")
    p.add_argument("--ridge", type=float, default=1e-3, help="Ridge regularization strength")
    p.add_argument("--models", nargs="+", default=None,
                   help="Restrict to these model slugs (default: all subdirectories of activations_dir)")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--cues", nargs="+", default=None)
    p.add_argument("--min_pool_size", type=int, default=20,
                   help="Skip a (scenario, split_type) if Giovanni/Meek train+test < this many rows.")
    return p.parse_args()


def load_scenario_split(splits_dir: Path, model: str, dataset: str, cue: str,
                        split_type: str, kind: str) -> set[str]:
    path = splits_dir / model / dataset / cue / f"{split_type}_{kind}_task_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def discover_cells(activations_dir: Path,
                   models: list[str] | None,
                   datasets: list[str] | None,
                   cues: list[str] | None) -> list[tuple[str, str, str, Path]]:
    cells: list[tuple[str, str, str, Path]] = []
    for model_dir in sorted(activations_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if models and model_dir.name not in models:
            continue
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            if datasets and dataset_dir.name not in datasets:
                continue
            for cue_dir in sorted(dataset_dir.iterdir()):
                if not cue_dir.is_dir():
                    continue
                if cues and cue_dir.name not in cues:
                    continue
                pt = cue_dir / "mean_all_response.pt"
                if pt.exists():
                    cells.append((model_dir.name, dataset_dir.name, cue_dir.name, pt))
    return cells


def fit_and_score(x_train: np.ndarray, y_train: np.ndarray,
                  x_test: np.ndarray, y_test: np.ndarray,
                  ridge: float) -> dict:
    """Fit ridge probe; return weights + per-row scores + both AUC metrics."""
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_n = (x_train - mu) / sigma
    x_test_n = (x_test - mu) / sigma
    weights = fit_ridge_probe(x_train_n, y_train.astype(np.float32), ridge)
    train_scores = np.concatenate([x_train_n, np.ones((x_train_n.shape[0], 1), dtype=x_train_n.dtype)], axis=1) @ weights
    test_scores = np.concatenate([x_test_n, np.ones((x_test_n.shape[0], 1), dtype=x_test_n.dtype)], axis=1) @ weights
    return {
        "weights": weights,            # (hidden+1,) — last entry is bias
        "feature_mean": mu.squeeze(0), # (hidden,)
        "feature_std": sigma.squeeze(0),
        "train_scores": train_scores,
        "test_scores": test_scores,
        "train_roc_auc": roc_auc(y_train.astype(int), train_scores),
        "test_roc_auc":  roc_auc(y_test.astype(int),  test_scores),
        "train_pr_auc":  pr_auc(y_train.astype(int),  train_scores),
        "test_pr_auc":   pr_auc(y_test.astype(int),   test_scores),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cells = discover_cells(args.activations_dir, args.models, args.datasets, args.cues)
    if not cells:
        raise SystemExit(f"No activation cells found under {args.activations_dir}")
    print(f"Found {len(cells)} cells; split_types={args.split_types}")

    for split_type in args.split_types:
        out_dir = args.output_dir / split_type
        out_dir.mkdir(parents=True, exist_ok=True)
        all_rows: list[dict] = []
        cell_summaries: list[dict] = []

        for model_slug, dataset, cue, pt_path in cells:
            print(f"\n=== [{split_type}] {model_slug}/{dataset}/{cue} ===")
            try:
                train_set = load_scenario_split(args.splits_dir, model_slug, dataset, cue, split_type, "train")
                test_set = load_scenario_split(args.splits_dir, model_slug, dataset, cue, split_type, "test")
            except FileNotFoundError as e:
                print(f"  SKIP missing split: {e}")
                continue

            data = torch.load(pt_path, weights_only=False, map_location="cpu")
            task_ids = list(data["task_ids"])
            labels = data["labels"].float().cpu().numpy()
            layer_keys = sorted(
                [k for k in data.keys() if k.startswith("layer_")],
                key=lambda k: int(k.split("_")[1]),
            )

            train_idx = [i for i, t in enumerate(task_ids) if t in train_set]
            test_idx = [i for i, t in enumerate(task_ids) if t in test_set]

            if len(train_idx) + len(test_idx) < args.min_pool_size:
                print(f"  SKIP small pool: train_n={len(train_idx)} test_n={len(test_idx)} (< {args.min_pool_size})")
                continue
            if not train_idx or not test_idx:
                print(f"  SKIP empty side: train_n={len(train_idx)} test_n={len(test_idx)}")
                continue

            y_train = labels[train_idx]
            y_test = labels[test_idx]
            n_pos_train = int((y_train == 1).sum())
            n_pos_test = int((y_test == 1).sum())
            if n_pos_train == 0 or n_pos_train == len(y_train) or n_pos_test == 0 or n_pos_test == len(y_test):
                print(f"  SKIP degenerate labels: train_pos={n_pos_train}/{len(y_train)} test_pos={n_pos_test}/{len(y_test)}")
                continue

            cell_rows: list[dict] = []
            weights_per_layer: dict[int, np.ndarray] = {}
            mu_per_layer: dict[int, np.ndarray] = {}
            sigma_per_layer: dict[int, np.ndarray] = {}
            scores_per_layer: dict[int, dict[str, np.ndarray]] = {}
            for layer_key in layer_keys:
                layer = int(layer_key.split("_")[1])
                x = data[layer_key].float().cpu().numpy()
                x_train = x[train_idx]
                x_test = x[test_idx]
                fit = fit_and_score(x_train, y_train, x_test, y_test, args.ridge)
                weights_per_layer[layer] = fit["weights"].astype(np.float32)
                mu_per_layer[layer] = fit["feature_mean"].astype(np.float32)
                sigma_per_layer[layer] = fit["feature_std"].astype(np.float32)
                scores_per_layer[layer] = {
                    "train_scores": fit["train_scores"].astype(np.float32),
                    "test_scores":  fit["test_scores"].astype(np.float32),
                }
                cell_rows.append({
                    "split_type": split_type,
                    "model_slug": model_slug,
                    "dataset": dataset,
                    "cue": cue,
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

            def _row_metrics(r: dict) -> dict:
                return {
                    "layer": r["layer"],
                    "train_roc_auc": r["train_roc_auc"],
                    "test_roc_auc":  r["test_roc_auc"],
                    "train_pr_auc":  r["train_pr_auc"],
                    "test_pr_auc":   r["test_pr_auc"],
                }

            best_by_roc = max((r for r in cell_rows if r["test_roc_auc"] is not None),
                              key=lambda r: r["test_roc_auc"], default=None)
            best_by_pr = max((r for r in cell_rows if r["test_pr_auc"] is not None),
                             key=lambda r: r["test_pr_auc"], default=None)
            best = best_by_roc  # canonical pick used for printing + weights save
            if best_by_roc:
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
                    "dataset": dataset,
                    "cue": cue,
                    "best_by_roc": _row_metrics(best_by_roc),
                    "best_by_pr":  _row_metrics(best_by_pr) if best_by_pr is not None else None,
                    "best_layer": best_by_roc["layer"],
                    "n_train": best_by_roc["n_train"],
                    "n_test": best_by_roc["n_test"],
                    "layers": cell_rows,
                }
                cell_summary_path = out_dir / f"{model_slug}__{dataset}__{cue}.json"
                write_json(cell_summary_path, cell_summary)
                cell_summaries.append({k: v for k, v in cell_summary.items() if k != "layers"})

                # Save fitted probe weights + scores for cheap post-hoc metric add-ons.
                weights_dir = out_dir / "weights"
                weights_dir.mkdir(parents=True, exist_ok=True)
                weights_path = weights_dir / f"{model_slug}__{dataset}__{cue}.pt"
                torch.save({
                    "split_type": split_type,
                    "model_slug": model_slug,
                    "dataset": dataset,
                    "cue": cue,
                    "ridge": args.ridge,
                    "train_task_ids": [task_ids[i] for i in train_idx],
                    "test_task_ids":  [task_ids[i] for i in test_idx],
                    "y_train": y_train.astype(np.float32),
                    "y_test":  y_test.astype(np.float32),
                    "layers": [int(k.split("_")[1]) for k in layer_keys],
                    "weights":     {l: torch.from_numpy(w) for l, w in weights_per_layer.items()},
                    "feature_mean":{l: torch.from_numpy(m) for l, m in mu_per_layer.items()},
                    "feature_std": {l: torch.from_numpy(s) for l, s in sigma_per_layer.items()},
                    "train_scores":{l: torch.from_numpy(scores_per_layer[l]["train_scores"]) for l in scores_per_layer},
                    "test_scores": {l: torch.from_numpy(scores_per_layer[l]["test_scores"])  for l in scores_per_layer},
                }, weights_path)

            all_rows.extend(cell_rows)

        csv_path = out_dir / "all_layers.csv"
        with open(csv_path, "w", newline="") as f:
            if all_rows:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
        print(f"\n[{split_type}] Wrote {len(all_rows)} rows to {csv_path}")

        summary_path = out_dir / "best_per_cell.json"
        write_json(summary_path, {"cells": cell_summaries})
        print(f"[{split_type}] Wrote per-cell summary to {summary_path}")

        if cell_summaries:
            def _key_test_roc(r: dict) -> float:
                return -(r.get("best_by_roc") or {}).get("test_roc_auc", float("-inf"))
            sorted_cells = sorted(cell_summaries, key=_key_test_roc)
            print(f"\n[{split_type}] Best layer per cell (sorted by test ROC-AUC):")
            print(f"{'model':<18}{'dataset':<8}{'cue':<10}"
                  f"{'roc_layer':>10}{'roc':>8}{'pr_layer':>10}{'pr':>8}")
            for r in sorted_cells:
                roc_block = r.get("best_by_roc") or {}
                pr_block = r.get("best_by_pr") or {}
                roc_layer = roc_block.get("layer")
                roc_val = roc_block.get("test_roc_auc")
                pr_layer = pr_block.get("layer")
                pr_val = pr_block.get("test_pr_auc")
                roc_layer_s = f"{roc_layer:>10}" if roc_layer is not None else f"{'-':>10}"
                roc_s = f"{roc_val:>8.3f}" if roc_val is not None else f"{'-':>8}"
                pr_layer_s = f"{pr_layer:>10}" if pr_layer is not None else f"{'-':>10}"
                pr_s = f"{pr_val:>8.3f}" if pr_val is not None else f"{'-':>8}"
                print(f"{r['model_slug']:<18}{r['dataset']:<8}{r['cue']:<10}"
                      f"{roc_layer_s}{roc_s}{pr_layer_s}{pr_s}")


if __name__ == "__main__":
    main()
