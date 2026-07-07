"""Build difference-of-means contrastive steering vectors per scenario.

For each (model, split_type, scenario), this script:
  1. Resolves the constituent (dataset, cue) cells the scenario draws from.
  2. Determines the "best layer":
     - Single-cell scenario: reads `best_by_roc.layer` from the existing
       per-cell probe JSON in `experiments/transfer/probes/<split>/`.
     - Aggregate scenario (multi-cell): pools D+/D- across constituent cells
       and fits a ridge probe per layer on the pooled train rows; picks
       argmax(test_roc_auc).
  3. Computes the DoM vector at that layer:
       v = mean(act[D+ ∩ train]) - mean(act[D- ∩ train])
     using only train task_ids.

Outputs to `experiments/transfer/vectors/contrastive/<split_type>/<model_slug>/<scenario>.pt`:
  vector              (hidden_size,)  raw DoM vector (float32)
  vector_normalized   (hidden_size,)  unit-normalized
  layer               int             selected layer
  hidden_size         int
  scenario            str             e.g. "gpqa_all"
  cells               list[(d,c)]     constituent (dataset, cue) cells
  split_type          "meek" | "giovanni"
  model               HF id           e.g. "google/gemma-3-4b-it"
  model_slug          str
  method              "dom_train_only"
  n_d_plus_train      int
  n_d_minus_train     int
  source_pt_paths     list[str]
  best_layer_source   "probe_json" | "pooled_probe_fit"
  layer_test_roc_auc  float           ROC-AUC on the pooled test rows at the chosen layer
  layer_test_pr_auc   float           PR-AUC ditto
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from train_probe import fit_ridge_probe, roc_auc, pr_auc


from steering_vectors_for_faithfulness.config import MODEL_REGISTRY

# Scenario name -> list of (dataset, cue) cells the vector aggregates over.
SCENARIOS: dict[str, list[tuple[str, str]]] = {
    # Hold dataset = gpqa, vary cue
    "gpqa_grader":   [("gpqa", "grader")],
    "gpqa_insider":  [("gpqa", "insider")],
    "gpqa_stanford": [("gpqa", "stanford")],
    "gpqa_xml":      [("gpqa", "xml")],
    "gpqa_all":      [("gpqa", c) for c in ("stanford", "xml", "grader", "insider")],
    # Hold cue = stanford, vary dataset (gpqa_stanford = stanford_gpqa, already covered above)
    "stanford_bbh":  [("bbh", "stanford")],
    "stanford_mmlu": [("mmlu", "stanford")],
    "stanford_all":  [(d, "stanford") for d in ("bbh", "gpqa", "mmlu")],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=sorted(MODEL_REGISTRY))
    p.add_argument("--split_types", nargs="+", default=["meek", "giovanni"], choices=["meek", "giovanni"])
    p.add_argument("--scenarios", nargs="+", default=sorted(SCENARIOS))
    p.add_argument("--activations_dir", type=Path, default=Path("experiments/transfer/activations"))
    p.add_argument("--splits_dir", type=Path, default=Path("experiments/transfer/splits"))
    p.add_argument("--probes_dir", type=Path, default=Path("experiments/transfer/probes"))
    p.add_argument("--output_dir", type=Path, default=Path("experiments/transfer/vectors/contrastive"))
    p.add_argument("--ridge", type=float, default=1e-3)
    return p.parse_args()


@dataclass
class CellData:
    model_slug: str
    dataset: str
    cue: str
    task_ids: list[str]                 # ordered as saved in the .pt
    labels: np.ndarray                  # 1-d float labels per row
    layers: list[int]
    layer_data: dict[int, np.ndarray]   # layer -> (n_rows, hidden) float32
    train_ids: set[str]
    test_ids: set[str]
    pt_path: Path


def read_id_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_cell(activations_dir: Path, splits_dir: Path,
              model_slug: str, dataset: str, cue: str, split_type: str) -> CellData:
    pt_path = activations_dir / model_slug / dataset / cue / "mean_all_response.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"No activation cell at {pt_path}")
    data = torch.load(pt_path, weights_only=False, map_location="cpu")
    layers = list(data["layers"])
    layer_data = {int(l): data[f"layer_{l}"].float().numpy() for l in layers}
    return CellData(
        model_slug=model_slug,
        dataset=dataset,
        cue=cue,
        task_ids=list(data["task_ids"]),
        labels=data["labels"].float().numpy(),
        layers=[int(l) for l in layers],
        layer_data=layer_data,
        train_ids=read_id_list(splits_dir / model_slug / dataset / cue / f"{split_type}_train_task_ids.txt"),
        test_ids=read_id_list(splits_dir / model_slug / dataset / cue / f"{split_type}_test_task_ids.txt"),
        pt_path=pt_path,
    )


@dataclass
class PooledCell:
    layers: list[int]
    layer_data: dict[int, np.ndarray]   # layer -> (n_rows_total, hidden)
    labels: np.ndarray                  # (n_rows_total,)
    is_train: np.ndarray                # bool mask
    is_test: np.ndarray                 # bool mask


def pool_cells(cells: list[CellData]) -> PooledCell:
    """Concatenate rows from a list of cells along axis 0."""
    assert cells, "no cells to pool"
    layers = cells[0].layers
    for c in cells[1:]:
        if c.layers != layers:
            raise ValueError(f"layer list mismatch when pooling: {c.layers} vs {layers}")

    chunks = {l: [] for l in layers}
    label_chunks = []
    train_mask_chunks = []
    test_mask_chunks = []
    for c in cells:
        for l in layers:
            chunks[l].append(c.layer_data[l])
        label_chunks.append(c.labels)
        train_mask_chunks.append(np.array([t in c.train_ids for t in c.task_ids], dtype=bool))
        test_mask_chunks.append(np.array([t in c.test_ids for t in c.task_ids], dtype=bool))

    layer_data = {l: np.concatenate(chunks[l], axis=0) for l in layers}
    labels = np.concatenate(label_chunks, axis=0)
    is_train = np.concatenate(train_mask_chunks, axis=0)
    is_test = np.concatenate(test_mask_chunks, axis=0)
    return PooledCell(layers=layers, layer_data=layer_data, labels=labels, is_train=is_train, is_test=is_test)


def fit_layer_aucs(pooled: PooledCell, ridge: float) -> dict[int, dict]:
    """Return layer -> {train_roc, test_roc, train_pr, test_pr} on pooled data."""
    out: dict[int, dict] = {}
    train_idx = np.flatnonzero(pooled.is_train)
    test_idx = np.flatnonzero(pooled.is_test)
    y_train = pooled.labels[train_idx]
    y_test = pooled.labels[test_idx]
    if y_train.size == 0 or y_test.size == 0:
        raise ValueError("pooled data has empty train or test split")
    if int((y_train == 1).sum()) in (0, y_train.size) or int((y_test == 1).sum()) in (0, y_test.size):
        raise ValueError("pooled data has degenerate train or test labels")
    for layer in pooled.layers:
        x = pooled.layer_data[layer]
        x_train = x[train_idx]
        x_test = x[test_idx]
        mu = x_train.mean(axis=0, keepdims=True)
        sigma = x_train.std(axis=0, keepdims=True) + 1e-6
        x_train_n = (x_train - mu) / sigma
        x_test_n = (x_test - mu) / sigma
        weights = fit_ridge_probe(x_train_n, y_train.astype(np.float32), ridge)
        train_scores = np.concatenate([x_train_n, np.ones((x_train_n.shape[0], 1), dtype=x_train_n.dtype)], axis=1) @ weights
        test_scores = np.concatenate([x_test_n, np.ones((x_test_n.shape[0], 1), dtype=x_test_n.dtype)], axis=1) @ weights
        out[layer] = {
            "train_roc": roc_auc(y_train.astype(int), train_scores),
            "test_roc":  roc_auc(y_test.astype(int),  test_scores),
            "train_pr":  pr_auc(y_train.astype(int),  train_scores),
            "test_pr":   pr_auc(y_test.astype(int),   test_scores),
        }
    return out


def best_layer_from_probe_json(probes_dir: Path, split_type: str,
                               model_slug: str, dataset: str, cue: str) -> tuple[int, dict]:
    path = probes_dir / split_type / f"{model_slug}__{dataset}__{cue}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing probe JSON: {path}")
    summary = json.loads(path.read_text())
    block = summary.get("best_by_roc")
    if block is None:
        raise ValueError(f"{path} has no best_by_roc")
    return int(block["layer"]), {
        "test_roc_auc": block.get("test_roc_auc"),
        "test_pr_auc":  block.get("test_pr_auc"),
        "source": "probe_json",
    }


def dom_vector(pooled: PooledCell, layer: int) -> tuple[np.ndarray, int, int]:
    """Compute D+ - D- mean activation difference using TRAIN rows only."""
    y = pooled.labels
    is_train = pooled.is_train
    plus_mask = is_train & (y == 1)
    minus_mask = is_train & (y == 0)
    n_plus = int(plus_mask.sum())
    n_minus = int(minus_mask.sum())
    if n_plus == 0 or n_minus == 0:
        raise ValueError(f"DoM impossible: train n_d_plus={n_plus} n_d_minus={n_minus}")
    x = pooled.layer_data[layer]
    mu_plus = x[plus_mask].mean(axis=0)
    mu_minus = x[minus_mask].mean(axis=0)
    return (mu_plus - mu_minus).astype(np.float32), n_plus, n_minus


def main() -> None:
    args = parse_args()

    bad: list[tuple[str, str, str, str]] = []
    summaries: list[dict] = []

    for model_slug in args.models:
        if model_slug not in MODEL_REGISTRY:
            print(f"WARNING: unknown model_slug {model_slug}; skipping")
            continue
        model_name = MODEL_REGISTRY[model_slug]

        for split_type in args.split_types:
            for scenario in args.scenarios:
                if scenario not in SCENARIOS:
                    print(f"WARNING: unknown scenario {scenario}; skipping")
                    continue
                cells_spec = SCENARIOS[scenario]
                print(f"\n=== {model_slug} [{split_type}] {scenario}  cells={cells_spec} ===")

                try:
                    cells = [load_cell(args.activations_dir, args.splits_dir,
                                       model_slug, d, c, split_type)
                             for d, c in cells_spec]
                except FileNotFoundError as e:
                    print(f"  SKIP {scenario}: {e}")
                    bad.append((model_slug, split_type, scenario, str(e)))
                    continue

                # Skip if any constituent cell has empty pool on this split type
                empty = [(c.dataset, c.cue) for c in cells if not c.train_ids or not c.test_ids]
                if empty:
                    print(f"  SKIP {scenario}: empty splits for {empty}")
                    bad.append((model_slug, split_type, scenario, f"empty splits: {empty}"))
                    continue

                pooled = pool_cells(cells)

                # Pick layer
                try:
                    if len(cells) == 1:
                        layer, layer_info = best_layer_from_probe_json(
                            args.probes_dir, split_type,
                            model_slug, cells[0].dataset, cells[0].cue,
                        )
                    else:
                        layer_aucs = fit_layer_aucs(pooled, args.ridge)
                        layer = max(
                            (l for l, m in layer_aucs.items() if m["test_roc"] is not None),
                            key=lambda l: layer_aucs[l]["test_roc"],
                        )
                        layer_info = {
                            "test_roc_auc": layer_aucs[layer]["test_roc"],
                            "test_pr_auc":  layer_aucs[layer]["test_pr"],
                            "source": "pooled_probe_fit",
                        }
                except Exception as e:
                    print(f"  FAIL {scenario}: layer selection: {type(e).__name__}: {e}")
                    bad.append((model_slug, split_type, scenario, f"layer selection: {e}"))
                    continue

                # Build the vector
                try:
                    vec, n_plus, n_minus = dom_vector(pooled, layer)
                except ValueError as e:
                    print(f"  FAIL {scenario}: {e}")
                    bad.append((model_slug, split_type, scenario, str(e)))
                    continue

                norm = float(np.linalg.norm(vec))
                vec_unit = vec / max(norm, 1e-12)

                out_dir = args.output_dir / split_type / model_slug
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{scenario}.pt"
                payload = {
                    "vector":            torch.from_numpy(vec.astype(np.float32)),
                    "vector_normalized": torch.from_numpy(vec_unit.astype(np.float32)),
                    "vector_norm":       norm,
                    "layer":             layer,
                    "hidden_size":       int(vec.shape[0]),
                    "scenario":          scenario,
                    "cells":             [{"dataset": c.dataset, "cue": c.cue} for c in cells],
                    "split_type":        split_type,
                    "model":             model_name,
                    "model_slug":        model_slug,
                    "method":            "dom_train_only",
                    "n_d_plus_train":    n_plus,
                    "n_d_minus_train":   n_minus,
                    "source_pt_paths":   [str(c.pt_path) for c in cells],
                    "best_layer_source": layer_info["source"],
                    "layer_test_roc_auc": layer_info.get("test_roc_auc"),
                    "layer_test_pr_auc":  layer_info.get("test_pr_auc"),
                }
                torch.save(payload, out_path)

                summary = {k: v for k, v in payload.items()
                           if k not in ("vector", "vector_normalized")}
                summary["out_path"] = str(out_path)
                summaries.append(summary)
                print(f"  saved layer={layer} norm={norm:.3f} D+={n_plus} D-={n_minus}  "
                      f"roc@layer={layer_info.get('test_roc_auc'):.3f}  -> {out_path.relative_to(args.output_dir.parent.parent)}"
                      if layer_info.get("test_roc_auc") is not None
                      else f"  saved layer={layer} norm={norm:.3f} D+={n_plus} D-={n_minus}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_dir / "all_summaries.json"
    with open(aggregate_path, "w") as f:
        json.dump({"vectors": summaries, "failed": bad}, f, indent=2)
    print(f"\nWrote {len(summaries)} vectors. Aggregate summary: {aggregate_path}")
    if bad:
        print(f"Failed: {len(bad)} scenarios")
        for b in bad:
            print(f"  - {b}")


if __name__ == "__main__":
    main()
