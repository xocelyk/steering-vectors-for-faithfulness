"""
Extract steering vectors from D+/D- activations via difference-of-means.

For each layer, computes: normalize(mean(D+) - mean(D-))

Optionally computes affine alphas (Belrose method): projects D+ and D-
activations onto the steering direction to find data-driven steering strengths.

Usage:
    uv run python experiments/phase3/extract_vectors.py \
        --activations_dir experiments/phase3/results/activations

    # With affine alpha computation
    uv run python experiments/phase3/extract_vectors.py \
        --activations_dir experiments/phase3/results/activations \
        --compute_affine_alphas
"""

import argparse
import json
import os
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-layer steering vectors from D+/D- activations"
    )
    parser.add_argument(
        "--activations_dir", type=str, required=True,
        help="Directory containing d_plus_activations.pt and d_minus_activations.pt",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: results/vectors/)",
    )
    parser.add_argument(
        "--compute_affine_alphas", action="store_true",
        help="Compute data-driven alphas by projecting D+/D- onto the steering direction "
             "(Belrose affine method). Stored in the output .pt file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "results" / "vectors"
    os.makedirs(output_dir, exist_ok=True)

    d_plus = torch.load(activations_dir / "d_plus_activations.pt", weights_only=False)
    d_minus = torch.load(activations_dir / "d_minus_activations.pt", weights_only=False)

    layer_keys = sorted(
        [k for k in d_plus.keys() if k.startswith("layer_")],
        key=lambda k: int(k.split("_")[1]),
    )

    print(f"Loaded activations: {len(d_plus['task_ids'])} D+, {len(d_minus['task_ids'])} D- examples")
    print(f"Layers: {len(layer_keys)}")

    steering_vectors = {}
    norms = {}
    affine_alphas = {}

    for key in layer_keys:
        layer_idx = int(key.split("_")[1])
        plus_acts = d_plus[key].float()
        minus_acts = d_minus[key].float()
        plus_mean = plus_acts.mean(dim=0)
        minus_mean = minus_acts.mean(dim=0)

        raw_vector = plus_mean - minus_mean
        norm = raw_vector.norm().item()
        normalized = raw_vector / raw_vector.norm()

        steering_vectors[layer_idx] = normalized
        norms[layer_idx] = norm

        if args.compute_affine_alphas:
            plus_projs = (plus_acts @ normalized.unsqueeze(-1)).squeeze(-1)
            minus_projs = (minus_acts @ normalized.unsqueeze(-1)).squeeze(-1)
            alpha_plus = plus_projs.mean().item()
            alpha_minus = minus_projs.mean().item()
            affine_alphas[layer_idx] = {
                "alpha_plus": alpha_plus,
                "alpha_minus": alpha_minus,
                "alpha_range": alpha_plus - alpha_minus,
            }

    save_data = {
        "vectors": steering_vectors,
        "norms": {k: v for k, v in norms.items()},
        "metadata": {
            "method": "difference_of_means",
            "n_d_plus": len(d_plus["task_ids"]),
            "n_d_minus": len(d_minus["task_ids"]),
            "token_strategy": d_plus.get("token_strategy", "unknown"),
            "model": d_plus.get("model", "unknown"),
            "layers": [int(k.split("_")[1]) for k in layer_keys],
        },
    }

    if args.compute_affine_alphas:
        save_data["affine_alphas"] = affine_alphas

    out_path = output_dir / "steering_vectors.pt"
    torch.save(save_data, out_path)

    print(f"\nSaved {len(steering_vectors)} steering vectors to {out_path}")
    print(f"\nPer-layer norms (pre-normalization):")
    for layer_idx in sorted(norms.keys()):
        print(f"  Layer {layer_idx:2d}: {norms[layer_idx]:.4f}")

    if args.compute_affine_alphas:
        print(f"\nAffine alphas (Belrose method):")
        print(f"  {'Layer':>5s}  {'alpha_plus':>11s}  {'alpha_minus':>12s}  {'alpha_range':>12s}")
        for layer_idx in sorted(affine_alphas.keys()):
            aa = affine_alphas[layer_idx]
            print(f"  {layer_idx:5d}  {aa['alpha_plus']:11.4f}  {aa['alpha_minus']:12.4f}  {aa['alpha_range']:12.4f}")


if __name__ == "__main__":
    main()
