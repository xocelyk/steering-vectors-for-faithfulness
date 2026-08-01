"""Central configuration for the steering-vectors experiments.

Single source of truth for values that used to be copy-pasted across the
and transfer scripts: the repo layout, the location of the external
``measuring_cot_monitorability`` dependency, default model ids, and Hugging Face
token/cache setup.

Import this instead of hardcoding paths::

    from steering_vectors_for_faithfulness import config

    config.configure_hf_cache()          # before importing torch/vllm
    config.ensure_monitorability_importable()
    from measuring_cot_monitorability.scorers import cue_aware_adaptive_scorer
"""

from __future__ import annotations

import os
from pathlib import Path

# config.py lives at src/steering_vectors_for_faithfulness/config.py, so the
# repo root is three parents up. Editable install (`pip install -e .`) keeps
# __file__ pointing at the source tree, so this stays correct.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


_load_dotenv()


# --- External dependency: measuring_cot_monitorability ----------------------
# Vendored as a git submodule under third_party/. Set MONITORABILITY_ROOT to
# point at a checkout elsewhere.
MONITORABILITY_ROOT = Path(
    os.environ.get(
        "MONITORABILITY_ROOT",
        REPO_ROOT / "third_party" / "measuring_cot_monitorability",
    )
)
MONITORABILITY_SRC = MONITORABILITY_ROOT / "src"
MONITORABILITY_DATA = MONITORABILITY_ROOT / "data"
CORE_DATASETS_PATH = MONITORABILITY_DATA / "core_datasets_with_factors"


def ensure_monitorability_importable() -> None:
    """Put measuring_cot_monitorability on sys.path if it isn't pip-installed."""
    import sys

    src = str(MONITORABILITY_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


# --- Model ids --------------------------------------------------------------
# The subject model whose chain-of-thought we steer and evaluate.
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Transfer models, keyed by the short slug used on the CLI and in output paths.
MODEL_REGISTRY = {
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
}
CANONICAL_MODELS = list(MODEL_REGISTRY.values())
CANONICAL_DATASETS = ["bbh", "gpqa", "mmlu"]
CANONICAL_CUES = {
    "stanford": "stanford_professor_recommends",
    "xml": "xml_metadata_success_rate",
    "grader": "grader_hack_validation",
    "unethical": "insider_information",
}


def resolve_model(slug_or_id: str) -> str:
    """Map a short slug (e.g. ``gemma-3-4b-it``) to its HF id; pass ids through."""
    return MODEL_REGISTRY.get(slug_or_id, slug_or_id)


# --- Hugging Face environment ----------------------------------------------
def hf_token() -> str | None:
    """Return the HF token from any of the names used in this project."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HF_KEY")
        or os.environ.get("HFTOKEN")
    )


def ensure_hf_token_aliases() -> None:
    """Expose the HF token under every name HF libraries might read."""
    token = hf_token()
    if token:
        for name in ("HF_TOKEN", "HF_KEY", "HFTOKEN", "HUGGING_FACE_HUB_TOKEN"):
            os.environ.setdefault(name, token)


def configure_hf_cache() -> Path:
    """Point the HF cache env vars at ``HF_HOME`` (default ~/.cache/huggingface).

    Call before importing torch/transformers/vllm. Respects any values already
    set in the environment; only fills in the gaps. Returns the cache root.
    """
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("HF_XET_CACHE_DIR", str(cache / "xet"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "hub"))
    return cache
