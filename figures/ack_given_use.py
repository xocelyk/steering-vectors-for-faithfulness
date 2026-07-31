#!/usr/bin/env python3
"""Acknowledgement given cue USE, where 'use' = model picked the cued letter,
NOT merely 'answered incorrectly'.

The chosen letter is the correctness judge's structured model_answer_letter field
(scoring v3+); no regex parsing of the response text.
Reports letter coverage, then P(ack | used) across model x condition (and per cue).
"""
import json, glob
from scoring import prog_correct
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mpl_config
import figstyle as S

RED, TEAL = S.RED, S.TEAL
TR = Path(__file__).resolve().parent.parent / "experiments" / "transfer"
from paths import artifacts_dir
ART = artifacts_dir()
Z = 1.645
def ci(p, n):
    return Z * (p * (1 - p) / n) ** 0.5 if n and not np.isnan(p) else 0.0


MODELS = ["gemma-3-4b-it", "qwen3.5-9b", "gemma-3-12b-it"]
MLAB = {"gemma-3-4b-it": "Gemma-3 4B", "qwen3.5-9b": "Qwen-3.5 9B", "gemma-3-12b-it": "Gemma-3 12B"}
CUES = ["stanford", "xml", "grader", "insider"]

def parse_records(path):
    """Return {task_id: dict(ack, used, correct, parsed, has_letter, cue_target)}."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except json.JSONDecodeError: continue  # skip pointer/garbage lines
            fa = r.get("faithfulness_score"); co = prog_correct(r)
            # Acknowledgement is judged on cut-off/incomplete traces too, so we only
            # require a faithfulness label here. Traces with no extracted letter stay
            # in the ack denominators: correctness is irrelevant to the ack rate, and
            # dropping them previously made this table disagree with cue_ack_use_joint.md.
            if fa is None: continue
            letter = r.get("model_answer_letter")
            ct = (r.get("cue_target_letter") or "").strip().upper()
            out[r["task_id"]] = dict(
                ack=int(fa) == 1, correct=(co == 1),
                parsed=letter, has_letter=letter is not None,
                # USE = the extracted letter is the cued letter. No correctness gate:
                # the cue points at a wrong option, so the cued letter is never correct
                # under programmatic grading (scoring.py).
                used=(letter is not None and ct != "" and letter == ct),
                cue_target=ct)
    return out

def base_path(model, cue):
    fs = glob.glob(str(TR / "runs_scored" / model / "baselines" / "gpqa" / cue / "*.jsonl"))
    return fs[0] if fs else None
def steer_path(model, cue, alpha="5.0"):
    # Pin alpha: this dir holds one file per steering coef; glob order is unsorted,
    # so fs[0] would pick a nondeterministic, often-wrong alpha. Main text reports alpha=5.
    fs = glob.glob(str(TR / "runs_steered_scored" / "contrastive" / "meek" / model
                       / f"gpqa_{cue}" / "gpqa" / cue / f"*alpha{alpha}.jsonl"))
    return fs[0] if fs else None

# ---- letter coverage / consistency ----
print("=== JUDGE ANSWER-LETTER COVERAGE ===")
cov_b = cov_s = n_b = n_s = 0
used_and_correct = used_n = 0
for model in MODELS:
    for cue in CUES:
        bp, sp = base_path(model, cue), steer_path(model, cue)
        if bp:
            b = parse_records(bp)
            for v in b.values():
                n_b += 1; cov_b += v["has_letter"]
                if v["used"]:
                    used_n += 1; used_and_correct += v["correct"]
        if sp:
            s = parse_records(sp)
            for v in s.values():
                n_s += 1; cov_s += v["has_letter"]
print(f"letter-extraction coverage: baseline {cov_b/n_b:.1%} ({n_b}), steered {cov_s/n_s:.1%} ({n_s})")
print(f"sanity: of items where letter==cue_target ('used'), fraction marked CORRECT = "
      f"{used_and_correct/max(used_n,1):.1%} (0 expected: the cued option is never the target)")

# ---- ack | used, per model x condition (pooled over cues, matched task_ids) ----
def collect(model):
    res = {"base": [], "steer": []}
    for cue in CUES:
        bp, sp = base_path(model, cue), steer_path(model, cue)
        if not bp or not sp: continue
        b = parse_records(bp); s = parse_records(sp)
        for tid in (set(b) & set(s)):
            res["base"].append(b[tid]); res["steer"].append(s[tid])
    return res

def rate(items, cond):
    used = [v for v in items if v["used"]]
    notused = [v for v in items if not v["used"]]
    ack_given_use = np.mean([v["ack"] for v in used]) if used else float("nan")
    ack_given_notuse = np.mean([v["ack"] for v in notused]) if notused else float("nan")
    return dict(n=len(items), use_rate=len(used)/len(items),
                ack_given_use=ack_given_use, ack_given_notuse=ack_given_notuse,
                ack_overall=np.mean([v["ack"] for v in items]))

print("\n=== P(acknowledge | USED the cued answer) — parsed cued-letter, pooled over 4 GPQA cues ===")
print(f"{'model':14s} {'cond':6s}  n   use%  ack|use  ack|¬use  ack_all")
ROWS = {}
for model in MODELS:
    c = collect(model)
    for cond in ("base", "steer"):
        r = rate(c[cond], cond)
        ROWS[(model, cond)] = r
        print(f"{MLAB[model]:14s} {cond:6s} {r['n']:4d} {100*r['use_rate']:5.0f} "
              f"{r['ack_given_use']:8.2f} {r['ack_given_notuse']:9.2f} {r['ack_overall']:8.2f}")

# ---- per-cue hidden use, matched task_ids (baseline files hold the full
# split incl. train items; the steered files hold only test items, so both
# sides must be restricted to the intersection) ----
print("\n=== hidden use by cue (base -> steer, matched, contrastive) ===")
print(f"{'model':14s} " + "".join(f"{c:>14s}" for c in CUES))
for model in MODELS:
    row = f"{MLAB[model]:14s} "
    for cue in CUES:
        bp, sp = base_path(model, cue), steer_path(model, cue)
        if bp and sp:
            b, st = parse_records(bp), parse_records(sp)
            common = set(b) & set(st)
            def hid(d):
                used = [d[t] for t in common if d[t]["used"]]
                return 1 - np.mean([v["ack"] for v in used]) if used else float("nan")
            row += f"{hid(b):>6.2f}->{hid(st):<6.2f}"
        else:
            row += f"{'--':>14s}"
    print(row)

# ---- fig11: ack|use across conditions (by cue, and by dataset), steered, with CI ----
fig, (axc, axd) = mpl_config.figure(1, 2, width=10, widths=[4, 3])
w = 0.26
for k, model in enumerate(MODELS):
    vals, errs = [], []
    for cue in CUES:
        sp = steer_path(model, cue)
        v, n = ack_use_for(sp) if sp else (float("nan"), 0)
        vals.append(v); errs.append(ci(v, n))
    x = np.arange(len(CUES))
    axc.bar(x + (k - 1) * w, vals, w, yerr=errs, label=MLABS[model], color=COL[model],
            edgecolor="#444", zorder=3, error_kw=dict(elinewidth=1.0, ecolor="#444"))
axc.set_xticks(np.arange(len(CUES))); axc.set_xticklabels([CLAB[c] for c in CUES], fontsize=8)
axc.set_ylabel("P(acknowledge | used cue)"); axc.set_title("By cue (GPQA, steered)", fontsize=10)
axc.set_ylim(0, 1)
for k, model in enumerate(MODELS):
    vals, errs = [], []
    for ds in DS:
        sp = steer_ds_path(model, ds)
        v, n = ack_use_for(sp) if sp else (float("nan"), 0)
        vals.append(v); errs.append(ci(v, n))
    x = np.arange(len(DS))
    axd.bar(x + (k - 1) * w, vals, w, yerr=errs, color=COL[model], edgecolor="#444",
            zorder=3, error_kw=dict(elinewidth=1.0, ecolor="#444"))
axd.set_xticks(np.arange(len(DS))); axd.set_xticklabels([DLAB[d] for d in DS], fontsize=8)
axd.set_title("By dataset (Stanford cue, steered)", fontsize=10); axd.set_ylim(0, 1)
# title lives in the LaTeX caption; legend in a horizontal strip above the panels
fig.legend(*axc.get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=8)
plt.close(fig)  # fig11 retired from the paper; not written
plt.close(fig)

# ---- markdown table: ack|use base vs steer per model (pooled cues) ----
md = ["# Acknowledgement given cue use\n",
      "P(acknowledge | used cue) vs. P(acknowledge | did not use cue), baseline vs. steered. "
      "Contrastive, GPQA, pooled over 4 cues, α=5. **Used** = the model's judge-extracted final answer is "
      "the cued option (≈40% of items), not merely an incorrect answer. The overall acknowledgement "
      "gain under steering is driven by **Ack | not-used**, not **Ack | used**.\n",
      "| Model | Cond. | n | Use% | Ack \\| used | Ack \\| not-used | Ack overall |",
      "|---|---|---|---|---|---|---|"]
for model in MODELS:
    for cond, tag in [("base", "baseline"), ("steer", "steered")]:
        r = ROWS[(model, cond)]
        name = MLABS[model] if cond == "base" else ""
        md.append(f"| {name} | {tag} | {r['n']} | {100*r['use_rate']:.0f} | "
                  f"{r['ack_given_use']:.2f} | {r['ack_given_notuse']:.2f} | {r['ack_overall']:.2f} |")
(ART / "ack_given_use_table.md").write_text("\n".join(md) + "\n")
print("wrote fig11_ack_given_use_by_condition + ack_given_use_table.md")
