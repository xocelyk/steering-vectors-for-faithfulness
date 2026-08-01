# Paper artifact pipeline

Everything the paper's figures and tables are built from lives here. One
command regenerates all of it from the scored run data in
`experiments/transfer/`:

```bash
figures/generate.sh          # all figures + tables
figures/generate.sh --pdf    # ... then rebuild the paper PDF too
figures/generate.sh probe_auc make_figures   # just specific scripts
```

Run it from anywhere; all paths are resolved relative to this repo. It uses
the repo's `.venv` (set `PYTHON=...` to override).

## Where output goes

Scripts write to **`figures/out/`**, which is tracked in this repo — every
figure and table is versioned alongside the code and data that produced it,
so a regeneration shows up as a reviewable diff. After a default run,
`generate.sh` syncs `figures/out/` into the **paper (Overleaf) repo**, where
`main.tex` reads it.

`paths.py` finds the paper repo automatically: a sibling directory of this
repo (or its parent, in the old nested layout) containing `main.tex`. The
destination folder name is parsed from `main.tex`'s `\graphicspath`
(currently `artifacts_06-25-2026/`), so retargeting the sync only requires
changing it there. If the paper repo isn't present (e.g. a collaborator with
only this repo), the sync is skipped with a warning and `figures/out/` still
has everything.

Overrides (env vars, or in the gitignored `figures/.env`):

- `PAPER_DIR` — absolute path to the Overleaf repo, if auto-detection fails
  (e.g. running from a git worktree).
- `ARTIFACTS_DIR` — write output somewhere else instead. This marks a
  **scratch run**: the paper-repo sync is skipped, and `figures/out/` is
  untouched. E.g. `ARTIFACTS_DIR=/tmp/artifacts-test figures/generate.sh`.

## Pipeline structure

`aggregate.py` runs first: it rebuilds `figures/agg.json` (per-cell stats
from `runs_steered_scored/` vs `runs_scored/` baselines, matched by
`task_id`). `make_figures`, `layer_tables`, `g12_alpha_table`,
`alpha_robustness` and `ackuse_2x2_by_dataset` read that cache — stale
`agg.json` means stale figures, which is why the driver always reruns it.
The remaining scripts glob the raw run trees directly. `agg.json` and
`ack_given_use.json` are tracked too, so the numbers behind the figures are
versioned with them.

Shared style lives in `figstyle.py` (palette/semantics) on top of
`mpl_config.py` (rcParams shim + fonts in `fonts/`). Import `figstyle` in any
new figure script so the set stays consistent.

What each figure/table shows is documented in the paper repo's
`artifacts/README.md` (figure-by-figure notes, metric definitions, caveats).
