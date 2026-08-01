"""Train optimization-based steering vectors for one (model, dst_mode) pair.

Loads the model once, then iterates over the 8 scenarios for that model/dst_mode
and trains a vector per scenario via steering_opt_batched.optimize_completion.

Layer per scenario is read from the meek probe JSON (best_by_roc.layer):
  - single-cell scenario `<scenario>`: probes/meek/<model>__<dataset>__<cue>.json
  - aggregate scenario (gpqa_all, stanford_all): probes/meek/<model>__<scenario>.json

Outputs:
  experiments/transfer/vectors/optimization/<dst_mode>/<model_slug>/<scenario>.pt
  experiments/transfer/vectors/optimization/<dst_mode>/<model_slug>/<scenario>.json

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=0 uv run python experiments/transfer/train_optimization_vectors.py \\
      --model_slug gemma-3-4b-it --dst_mode specific
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import steering_opt_batched as steering_opt  # noqa: E402

from build_contrastive_vectors import MODEL_REGISTRY, SCENARIOS  # noqa: E402
from common import write_json  # noqa: E402

from steering_vectors_for_faithfulness.config import configure_hf_cache
configure_hf_cache()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
load_dotenv(PROJECT_ROOT / ".env")
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY") or os.getenv("HFTOKEN")
if _hf_token:
    os.environ.setdefault("HF_TOKEN", _hf_token)
    os.environ.setdefault("HF_KEY", _hf_token)
    os.environ.setdefault("HFTOKEN", _hf_token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_slug", required=True, choices=sorted(MODEL_REGISTRY))
    p.add_argument("--dst_mode", required=True, choices=["specific", "generic"])
    p.add_argument("--scenarios", nargs="+", default=sorted(SCENARIOS))
    p.add_argument("--examples_dir", type=Path,
                   default=Path("experiments/transfer/optimization_examples"))
    p.add_argument("--probes_dir", type=Path,
                   default=Path("experiments/transfer/probes/meek"))
    p.add_argument("--output_dir", type=Path,
                   default=Path("experiments/transfer/vectors/optimization"))
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--max_iters", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--starting_norm", type=float, default=1.0)
    p.add_argument("--max_norm", type=float, default=5.0)
    p.add_argument("--target_loss", type=float, default=None)
    p.add_argument("--normalize_token_length", action="store_true")
    p.add_argument("--device", default="cuda:0",
                   help="Primary device. Used for the steering vector and (in single mode) "
                        "the whole model.")
    p.add_argument("--multi_gpu_mode", choices=["single", "sharded"], default="single",
                   help="single: one GPU holds the entire model. "
                        "sharded: device_map=auto distributes the model across all visible "
                        "GPUs (pipeline-style); each GPU holds only a slice. "
                        "Use sharded when one GPU cannot fit (model + activations).")
    p.add_argument("--model_dtype", choices=["float16", "bfloat16", "float32"],
                   default="bfloat16",
                   help="Model load dtype. bfloat16 avoids fp16 overflow at deeper layers.")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Load model in 4-bit (NF4 via bitsandbytes). Drastically "
                        "shrinks weight memory (e.g. 24 GB bf16 -> ~6 GB), leaving "
                        "headroom for activation storage on 12B+ models.")
    p.add_argument("--load_in_8bit", action="store_true",
                   help="Load model in 8-bit (bitsandbytes). Halves weight memory.")
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def parse_token_spec(raw):
    if raw is None:
        return None
    if raw == "prompt":
        return "prompt"
    if isinstance(raw, int):
        return raw
    if isinstance(raw, list) and len(raw) == 2:
        start = int(raw[0]) if raw[0] is not None else None
        stop = int(raw[1]) if raw[1] is not None else None
        return slice(start, stop)
    raise ValueError(f"Invalid token spec: {raw}")


def load_datapoints(path: Path) -> list[steering_opt.TrainingDatapoint]:
    out: list[steering_opt.TrainingDatapoint] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt", "")
            if not prompt:
                raise ValueError(f"{path}:{line_num} missing prompt")
            src = rec.get("src_completions", [])
            dst = rec.get("dst_completions", [])
            if not src and not dst:
                raise ValueError(f"{path}:{line_num} need at least one of src/dst_completions")
            out.append(steering_opt.TrainingDatapoint(
                prompt=prompt,
                src_completions=src,
                dst_completions=dst,
                token=parse_token_spec(rec.get("token")),
                is_negative=rec.get("is_negative", False),
            ))
    return out


def resolve_layer(probes_dir: Path, model_slug: str, scenario: str) -> tuple[int, dict]:
    """Look up the probe-best ROC layer for this scenario.

    For aggregate scenarios (gpqa_all, stanford_all) read <model>__<scenario>.json.
    For single-cell scenarios (e.g. gpqa_grader, stanford_bbh) read
    <model>__<dataset>__<cue>.json based on SCENARIOS.
    """
    cells = SCENARIOS.get(scenario)
    if cells is None:
        raise SystemExit(f"Unknown scenario {scenario}")
    if len(cells) == 1:
        dataset, cue = cells[0]
        path = probes_dir / f"{model_slug}__{dataset}__{cue}.json"
    else:
        path = probes_dir / f"{model_slug}__{scenario}.json"
    if not path.exists():
        raise FileNotFoundError(f"Probe JSON not found: {path}")
    blob = json.loads(path.read_text())
    block = blob.get("best_by_roc")
    if block is None:
        raise ValueError(f"{path} has no best_by_roc")
    return int(block["layer"]), {
        "probe_path": str(path),
        "probe_test_roc_auc": block.get("test_roc_auc"),
        "probe_test_pr_auc":  block.get("test_pr_auc"),
    }


def get_num_hidden_layers(model) -> int:
    base = model.module if hasattr(model, "module") else model
    config = base.config
    text_config = getattr(config, "text_config", None) or config
    n = getattr(text_config, "num_hidden_layers", None)
    if n is None:
        raise AttributeError("Could not infer num_hidden_layers")
    return int(n)


def load_model(model_name: str, device: str, model_dtype: str,
               multi_gpu_mode: str = "single",
               load_in_4bit: bool = False, load_in_8bit: bool = False):
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} mode={multi_gpu_mode} "
          f"device={device} dtype={model_dtype} 4bit={load_in_4bit} 8bit={load_in_8bit}...",
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=_hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[model_dtype]
    quant_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    if multi_gpu_mode == "sharded":
        # Distribute layers across all visible GPUs (pipeline-style).
        device_map = "auto"
    else:
        device_map = {"": device}

    kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
        token=_hf_token,
    )
    if quant_config is None:
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if (tokenizer.pad_token is not None and not (load_in_4bit or load_in_8bit)
            and multi_gpu_mode != "sharded"):
        # resize_token_embeddings is unsafe with quantized weights AND with
        # device_map="auto" (which can leave embedding/lm_head on different
        # GPUs and break the resize step).
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded.", flush=True)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    model_name = MODEL_REGISTRY[args.model_slug]

    out_dir = args.output_dir / args.dst_mode / args.model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-resolve the work list so we can skip-existing without holding the model.
    jobs: list[tuple[str, Path, Path, int, dict]] = []
    skipped = 0
    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            print(f"WARNING: unknown scenario {scenario}; skipping")
            continue
        examples_path = args.examples_dir / args.dst_mode / args.model_slug / f"{scenario}.jsonl"
        if not examples_path.exists():
            print(f"SKIP {scenario}: examples not found at {examples_path}")
            continue
        out_pt = out_dir / f"{scenario}.pt"
        if args.skip_existing and out_pt.exists():
            print(f"SKIP {scenario}: vector exists at {out_pt}")
            skipped += 1
            continue
        try:
            layer, probe_info = resolve_layer(args.probes_dir, args.model_slug, scenario)
        except (FileNotFoundError, ValueError) as e:
            print(f"SKIP {scenario}: {e}")
            continue
        jobs.append((scenario, examples_path, out_pt, layer, probe_info))

    if not jobs:
        print(f"Nothing to train. {skipped} already-completed vectors found.")
        return

    print(f"Will train {len(jobs)} vectors for {args.model_slug} [{args.dst_mode}]: "
          f"{[j[0] for j in jobs]}")

    model, tokenizer = load_model(
        model_name, args.device, args.model_dtype,
        multi_gpu_mode=args.multi_gpu_mode,
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
    )
    n_layers = get_num_hidden_layers(model)

    failed: list[tuple[str, str]] = []
    for scenario, examples_path, out_pt, layer, probe_info in jobs:
        layer = min(layer, n_layers - 1)
        print(f"\n[{time.strftime('%H:%M:%S')}] === {args.model_slug} [{args.dst_mode}] {scenario} layer={layer} ===", flush=True)
        try:
            datapoints = load_datapoints(examples_path)
            n_src = sum(len(d.src_completions) for d in datapoints)
            n_dst = sum(len(d.dst_completions) for d in datapoints)
            print(f"  loaded {len(datapoints)} datapoints  dst={n_dst} src={n_src}", flush=True)

            t0 = time.time()
            result = steering_opt.optimize_completion(
                model, datapoints, layer,
                tokenizer=tokenizer,
                lr=args.lr,
                max_iters=args.max_iters,
                use_transformer_lens=False,
                do_target_loss_avg=False,
                return_loss=True,
                target_loss=args.target_loss,
                target_loss_target_iters=5,
                debug=False,
                batch_size=args.batch_size,
                temperature=args.temperature,
                starting_norm=args.starting_norm,
                max_norm=args.max_norm,
                normalize_token_length=args.normalize_token_length,
                device=args.device,
            )
            vector = result[0]
            loss_info = result[1] if len(result) > 1 else {}
            elapsed = time.time() - t0

            raw_norm = vector.norm().item()
            normalized = (vector / vector.norm()).detach().cpu().float()

            payload = {
                "vectors": {layer: normalized},
                "norms": {layer: raw_norm},
                "metadata": {
                    "method": "optimization",
                    "layer": layer,
                    "model": model_name,
                    "model_slug": args.model_slug,
                    "scenario": scenario,
                    "dst_mode": args.dst_mode,
                    "examples_path": str(examples_path),
                    "n_datapoints": len(datapoints),
                    "n_src_completions": n_src,
                    "n_dst_completions": n_dst,
                    "lr": args.lr,
                    "max_iters": args.max_iters,
                    "temperature": args.temperature,
                    "starting_norm": args.starting_norm,
                    "max_norm": args.max_norm,
                    "target_loss": args.target_loss,
                    "normalize_token_length": args.normalize_token_length,
                    "batch_size": args.batch_size,
                    "elapsed_seconds": round(elapsed, 2),
                    "final_loss": loss_info.get("loss") if isinstance(loss_info, dict) else None,
                    "n_iters": loss_info.get("iters") if isinstance(loss_info, dict) else None,
                    "final_vector_norm": raw_norm,
                    **probe_info,
                },
            }
            tmp = out_pt.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            tmp.replace(out_pt)
            write_json(out_pt.with_suffix(".json"), payload["metadata"])
            print(f"  saved vector layer={layer} norm={raw_norm:.3f} "
                  f"final_loss={payload['metadata']['final_loss']} "
                  f"elapsed={elapsed:.1f}s -> {out_pt}", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL {scenario}: {type(exc).__name__}: {exc}", flush=True)
            failed.append((scenario, f"{type(exc).__name__}: {exc}"))
        finally:
            # Reclaim allocator cache between scenarios so a per-scenario
            # spike does not push the next scenario into OOM territory.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for s, msg in failed:
            print(f"  - {s}: {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
