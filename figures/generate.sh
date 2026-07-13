#!/usr/bin/env bash
# Regenerate every paper artifact (figures + tables) from the scored run data.
#
# Usage:
#   figures/generate.sh                 # run the full pipeline
#   figures/generate.sh probe_auc ...   # run only the named scripts
#   figures/generate.sh --pdf           # full pipeline, then rebuild the PDF
#
# Artifacts are written to figures/out/ (tracked in this repo), then synced
# into the paper repo's \graphicspath directory. Overrides via PAPER_DIR /
# ARTIFACTS_DIR (see figures/paths.py); an ARTIFACTS_DIR override is treated
# as a scratch run and skips the sync.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="${PYTHON:-$REPO/.venv/bin/python}"

# aggregate must run first: it rebuilds agg.json, which make_figures,
# layer_tables, g12_alpha_table, alpha_robustness and ackuse_2x2_by_dataset
# read. The remaining scripts glob the raw runs_scored/runs_steered_scored
# trees directly.
ALL=(
  aggregate
  make_figures
  baseline_accuracy
  crosscue_cosine
  probe_auc
  cue_ack_following
  ack_given_use
  ackuse_2x2_by_dataset
  layer_tables
  g12_alpha_table
  alpha_robustness
  vector_geometry
  reliance_proxy
)

BUILD_PDF=0
SCRIPTS=()
for arg in "$@"; do
  case "$arg" in
    --pdf) BUILD_PDF=1 ;;
    -h|--help)
      sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) SCRIPTS+=("${arg%.py}") ;;
  esac
done
if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
  SCRIPTS=("${ALL[@]}")
fi

if [[ ! -x "$PY" ]]; then
  echo "Python not found at $PY -- run 'uv sync' in $REPO or set PYTHON." >&2
  exit 1
fi

OUT="$(cd "$HERE" && "$PY" -c 'from paths import artifacts_dir; print(artifacts_dir())')"
echo "Artifacts dir: $OUT"

for s in "${SCRIPTS[@]}"; do
  echo
  echo "==> $s"
  "$PY" "$HERE/$s.py"
done

if [[ "$OUT" == "$HERE/out" ]]; then
  if PAPER_OUT="$(cd "$HERE" && "$PY" -c 'from paths import paper_artifacts_dir; print(paper_artifacts_dir())')"; then
    echo
    echo "==> syncing $OUT/ -> $PAPER_OUT/"
    mkdir -p "$PAPER_OUT"
    rsync -a "$OUT/" "$PAPER_OUT/"
  else
    echo "Paper repo not found -- artifacts are in $OUT, sync skipped." >&2
  fi
else
  echo
  echo "ARTIFACTS_DIR override in effect -- paper-repo sync skipped."
fi

if [[ $BUILD_PDF -eq 1 ]]; then
  PAPER="$(cd "$HERE" && "$PY" -c 'from paths import paper_root; print(paper_root())')"
  if [[ -x "$PAPER/.local/build.sh" ]]; then
    echo
    echo "==> building PDF"
    "$PAPER/.local/build.sh"
  else
    echo "No executable .local/build.sh in $PAPER -- skipping PDF build." >&2
  fi
fi

echo
echo "Done."
