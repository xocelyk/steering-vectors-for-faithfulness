"""Build per-scenario D+/D- contrastive pools for steering-vector creation.

For each (split_type, model, dataset, cue) where split_type in {meek, giovanni}:
  - Load the activation cell to recover (task_ids, labels).
  - Intersect with that scenario's TRAIN split (test is held out for evaluation).
  - Partition by label: D+ = faithfulness_score == 1, D- = faithfulness_score == 0.

Outputs under experiments/transfer/contrastive_pools/<split_type>/<model>/<dataset>/<cue>/:
  d_plus.txt       (one task_id per line, faithful examples in train)
  d_minus.txt      (one task_id per line, unfaithful examples in train)
  summary.json     (counts + paths)

Plus an aggregate experiments/transfer/contrastive_pools/all_summaries.json with
counts per (split_type, model, dataset, cue).

Cheap: only reads activation .pt headers + split text files; no GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations_dir", type=Path, default=Path("experiments/transfer/activations"))
    p.add_argument("--splits_dir", type=Path, default=Path("experiments/transfer/splits"))
    p.add_argument("--output_dir", type=Path, default=Path("experiments/transfer/contrastive_pools"))
    p.add_argument("--split_types", nargs="+", default=["meek", "giovanni"], choices=["meek", "giovanni"])
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--cues", nargs="+", default=None)
    return p.parse_args()


def load_train_ids(splits_dir: Path, model: str, dataset: str, cue: str, split_type: str) -> set[str]:
    path = splits_dir / model / dataset / cue / f"{split_type}_train_task_ids.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def discover_cells(activations_dir: Path,
                   models: list[str] | None,
                   datasets: list[str] | None,
                   cues: list[str] | None) -> list[tuple[str, str, str, Path]]:
    cells: list[tuple[str, str, str, Path]] = []
    for model_dir in sorted(activations_dir.iterdir()):
        if not model_dir.is_dir() or (models and model_dir.name not in models):
            continue
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir() or (datasets and dataset_dir.name not in datasets):
                continue
            for cue_dir in sorted(dataset_dir.iterdir()):
                if not cue_dir.is_dir() or (cues and cue_dir.name not in cues):
                    continue
                pt = cue_dir / "mean_all_response.pt"
                if pt.exists():
                    cells.append((model_dir.name, dataset_dir.name, cue_dir.name, pt))
    return cells


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cells = discover_cells(args.activations_dir, args.models, args.datasets, args.cues)
    if not cells:
        raise SystemExit(f"No activation cells found under {args.activations_dir}")

    aggregates: list[dict] = []

    for model_slug, dataset, cue, pt_path in cells:
        data = torch.load(pt_path, weights_only=False, map_location="cpu")
        task_ids = list(data["task_ids"])
        labels = data["labels"].float().cpu().numpy()

        for split_type in args.split_types:
            train_ids = load_train_ids(args.splits_dir, model_slug, dataset, cue, split_type)
            if not train_ids:
                aggregates.append({
                    "split_type": split_type, "model": model_slug, "dataset": dataset, "cue": cue,
                    "status": "missing_split",
                })
                print(f"  [{split_type}] {model_slug}/{dataset}/{cue}: missing split")
                continue

            d_plus, d_minus = [], []
            for tid, label in zip(task_ids, labels):
                if tid not in train_ids:
                    continue
                if int(label) == 1:
                    d_plus.append(tid)
                elif int(label) == 0:
                    d_minus.append(tid)

            out = args.output_dir / split_type / model_slug / dataset / cue
            out.mkdir(parents=True, exist_ok=True)
            (out / "d_plus.txt").write_text("\n".join(d_plus) + ("\n" if d_plus else ""))
            (out / "d_minus.txt").write_text("\n".join(d_minus) + ("\n" if d_minus else ""))

            summary = {
                "split_type": split_type,
                "model": model_slug,
                "dataset": dataset,
                "cue": cue,
                "n_train": len(d_plus) + len(d_minus),
                "n_d_plus": len(d_plus),
                "n_d_minus": len(d_minus),
                "d_plus_path": str(out / "d_plus.txt"),
                "d_minus_path": str(out / "d_minus.txt"),
                "activations_path": str(pt_path),
                "status": "ok",
            }
            write_json(out / "summary.json", summary)
            aggregates.append(summary)
            print(f"  [{split_type}] {model_slug}/{dataset}/{cue}: D+={len(d_plus)} D-={len(d_minus)}")

    write_json(args.output_dir / "all_summaries.json", {"pools": aggregates})

    # Print compact totals per split_type
    print("\n=== totals per split_type ===")
    for split_type in args.split_types:
        rows = [a for a in aggregates if a.get("split_type") == split_type and a.get("status") == "ok"]
        n_plus = sum(r["n_d_plus"] for r in rows)
        n_minus = sum(r["n_d_minus"] for r in rows)
        n_unbalanced = sum(1 for r in rows if r["n_d_plus"] == 0 or r["n_d_minus"] == 0)
        print(f"  {split_type}: {len(rows)} cells, total D+={n_plus} D-={n_minus}, "
              f"unbalanced cells (one side empty): {n_unbalanced}")

    # Flag scenarios with very small or very imbalanced pools
    print("\n=== weakest pools (< 10 in either side) ===")
    weak = [a for a in aggregates
            if a.get("status") == "ok" and (a["n_d_plus"] < 10 or a["n_d_minus"] < 10)]
    for w in sorted(weak, key=lambda r: min(r["n_d_plus"], r["n_d_minus"])):
        print(f"  [{w['split_type']}] {w['model']}/{w['dataset']}/{w['cue']}: "
              f"D+={w['n_d_plus']} D-={w['n_d_minus']}")


if __name__ == "__main__":
    main()
