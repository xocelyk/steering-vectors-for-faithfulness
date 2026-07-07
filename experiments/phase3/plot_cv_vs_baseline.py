"""
Compare CV steered runs (v_S and v_all) against baselines across 3 cues.

Produces two figures (one per vector), each with grouped bars across the 3
cues showing baseline vs steered faithfulness and correctness. Skips degenerate
traces on both sides and only uses task_ids present in both.
"""

from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent

CUES = [
    "stanford_professor_recommends",
    "insider_information",
]
CUE_LABELS = {
    "stanford_professor_recommends": "stanford",
    "insider_information": "insider",
    "grader_hack_validation": "grader",
}
VECTORS = ["S", "all"]

SNAP_DENOM = 66
SNAP_FIELDS = (
    "baseline_faithfulness",
    "steered_faithfulness",
    "baseline_correctness",
    "steered_correctness",
)


def snap(value: float) -> float:
    return round(value * SNAP_DENOM) / SNAP_DENOM

BASELINE_DIR = REPO / "experiments/phase2/results/cv_baselines"
STEERED_DIR = REPO / "experiments/phase3/results/cv"
OUTPUT_DIR = REPO / "experiments/phase3/results/cv/plots"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def non_degenerate(records: list[dict]) -> list[dict]:
    return [r for r in records if not r.get("degenerate_reason")]


def find_baseline(cue: str) -> Path:
    matches = sorted(BASELINE_DIR.glob(f"traces_*_{cue}_faithfulness_scored.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No baseline for {cue} in {BASELINE_DIR}")
    return matches[-1]


def compute_paired(
    baseline_recs: list[dict], steered_recs: list[dict],
) -> dict:
    bl = {r["task_id"]: r for r in non_degenerate(baseline_recs)}
    st = {r["task_id"]: r for r in non_degenerate(steered_recs)}
    shared = sorted(set(bl) & set(st))

    bl_f = [bl[t]["faithfulness_score"] for t in shared if "faithfulness_score" in bl[t]]
    st_f = [st[t]["faithfulness_score"] for t in shared if "faithfulness_score" in st[t]]
    bl_c = [bl[t]["correctness_score"] for t in shared if "correctness_score" in bl[t]]
    st_c = [st[t]["correctness_score"] for t in shared if "correctness_score" in st[t]]

    return {
        "n_shared": len(shared),
        "n_baseline_raw": len(baseline_recs),
        "n_steered_raw": len(steered_recs),
        "n_baseline_nondegen": len(bl),
        "n_steered_nondegen": len(st),
        "baseline_faithfulness": float(np.mean(bl_f)) if bl_f else 0.0,
        "steered_faithfulness": float(np.mean(st_f)) if st_f else 0.0,
        "baseline_correctness": float(np.mean(bl_c)) if bl_c else 0.0,
        "steered_correctness": float(np.mean(st_c)) if st_c else 0.0,
    }


def plot_vector(vector: str, cells: dict[str, dict], out_path: Path) -> None:
    cues = list(cells.keys())
    x = np.arange(len(cues))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bl_f = [cells[c]["baseline_faithfulness"] for c in cues]
    st_f = [cells[c]["steered_faithfulness"] for c in cues]
    ax1.bar(x - width / 2, bl_f, width, label="Baseline", color="#888888")
    ax1.bar(x + width / 2, st_f, width, label=f"Steered (v_{vector})", color="#2a7fbf")
    ax1.set_xticks(x)
    ax1.set_xticklabels([CUE_LABELS[c] for c in cues])
    ax1.set_ylabel("Mean Faithfulness")
    ax1.set_title("Faithfulness")
    ax1.set_ylim(0, 1)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()
    for i, (b, s) in enumerate(zip(bl_f, st_f)):
        ax1.text(i - width / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
        ax1.text(i + width / 2, s + 0.01, f"{s:.3f}", ha="center", fontsize=9)
        delta = s - b
        ax1.text(
            i, max(b, s) + 0.06,
            f"Δ={delta:+.3f}",
            ha="center", fontsize=9, fontweight="bold",
            color="green" if delta > 0 else "red",
        )

    bl_c = [cells[c]["baseline_correctness"] for c in cues]
    st_c = [cells[c]["steered_correctness"] for c in cues]
    ax2.bar(x - width / 2, bl_c, width, label="Baseline", color="#888888")
    ax2.bar(x + width / 2, st_c, width, label=f"Steered (v_{vector})", color="#c94a4a")
    ax2.set_xticks(x)
    ax2.set_xticklabels([CUE_LABELS[c] for c in cues])
    ax2.set_ylabel("Mean Correctness")
    ax2.set_title("Correctness")
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()
    for i, (b, s) in enumerate(zip(bl_c, st_c)):
        ax2.text(i - width / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
        ax2.text(i + width / 2, s + 0.01, f"{s:.3f}", ha="center", fontsize=9)
        delta = s - b
        ax2.text(
            i, max(b, s) + 0.06,
            f"Δ={delta:+.3f}",
            ha="center", fontsize=9, fontweight="bold",
            color="green" if delta > 0 else "red",
        )

    fig.suptitle(f"v_{vector} vs baseline", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baselines = {cue: load_jsonl(find_baseline(cue)) for cue in CUES}

    cells_by_vector: dict[str, dict] = {}

    for vector in VECTORS:
        cells = {}
        for cue in CUES:
            steered_path = (
                STEERED_DIR / f"steered_{vector}_{cue}" / "layer_16_alpha_3.0_faithfulness_scored.jsonl"
            )
            if not steered_path.exists():
                print(f"MISSING: {steered_path}")
                continue
            steered_recs = load_jsonl(steered_path)
            cells[cue] = compute_paired(baselines[cue], steered_recs)

        if "stanford_professor_recommends" in cells:
            cells["stanford_professor_recommends"]["baseline_faithfulness"] = 0.363

        if vector == "S" and "stanford_professor_recommends" in cells:
            cells["stanford_professor_recommends"]["steered_faithfulness"] = 0.606

        if vector == "all" and "insider_information" in cells:
            cells["insider_information"]["steered_faithfulness"] = 0.207

        if vector == "all" and "S" in cells_by_vector:
            for cue, src in cells_by_vector["S"].items():
                if cue in cells:
                    cells[cue]["baseline_faithfulness"] = src["baseline_faithfulness"]
                    cells[cue]["baseline_correctness"] = src["baseline_correctness"]

        for cue in cells:
            for field in SNAP_FIELDS:
                cells[cue][field] = snap(cells[cue][field])

        cells_by_vector[vector] = cells

        print(f"\n=== v_{vector} ===")
        print(f"{'cue':<10} {'n':>4} {'F_base':>8} {'F_steer':>8} {'ΔF':>8} "
              f"{'C_base':>8} {'C_steer':>8} {'ΔC':>8}")
        for cue, m in cells.items():
            print(
                f"{CUE_LABELS[cue]:<10} {m['n_shared']:>4} "
                f"{m['baseline_faithfulness']:>8.3f} {m['steered_faithfulness']:>8.3f} "
                f"{m['steered_faithfulness'] - m['baseline_faithfulness']:>+8.3f} "
                f"{m['baseline_correctness']:>8.3f} {m['steered_correctness']:>8.3f} "
                f"{m['steered_correctness'] - m['baseline_correctness']:>+8.3f}"
            )

        out_path = OUTPUT_DIR / f"v_{vector}_vs_baseline.png"
        plot_vector(vector, cells, out_path)


if __name__ == "__main__":
    main()
