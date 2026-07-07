#!/usr/bin/env bash
set -uo pipefail

# Generate (all models, all cells) + score everything for ONE vector family at a
# given alpha, meek split. Generation is sequential on one GPU; scoring follows.
#
# Usage:
#   ALPHA=2.5 NTFY_TOPIC=steer-mbn-7q2x9k \
#     bash experiments/transfer/run_family_steer_and_score.sh contrastive
#
# Vector types:
#   contrastive | synthetic    -> one group (vectors/<type>, split=meek)
#   optimization               -> two groups (specific + generic), eval on meek
#
# Env knobs: ALPHA(=5.0) SPLIT(=meek) MAX_WORKERS(=72) GPU(=0) NTFY_TOPIC(optional)
# Requires .env with HF_TOKEN (generation) and OPENAI_API_KEY (scoring).

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

VTYPE="${1:?usage: $0 <contrastive|synthetic|optimization>}"
ALPHA_IN="${ALPHA:-5.0}"
SPLIT="${SPLIT:-meek}"
MAX_WORKERS="${MAX_WORKERS:-72}"
GPU="${GPU:-0}"
NTFY_TOPIC="${NTFY_TOPIC:-}"
INSTANCE="${INSTANCE:-$(hostname)}"

# Normalize alpha to the exact float string the generator uses in filenames
# (e.g. "2.5" -> "2.5", "10" -> "10.0") so the scoring glob matches.
ALPHA="$(uv run --no-sync python -c "print(float('${ALPHA_IN}'))")"

notify() {
  [ -z "${NTFY_TOPIC}" ] && return 0
  curl -s -H "Title: ${VTYPE} a=${ALPHA} [${INSTANCE}]" -H "Tags: robot" \
    -d "$1" "ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
}

# Generation groups: "<vectors_dir>|<split_types>|<eval_split>"
case "${VTYPE}" in
  contrastive|synthetic)
    GEN_GROUPS=( "experiments/transfer/vectors/${VTYPE}|${SPLIT}|" )
    SCORE_PREFIX="${VTYPE}"
    ;;
  optimization)
    # SUBSET=specific|generic runs just that sub-method (scoring scoped to its own
    # subtree so two instances don't double-score). SUBSET=both (default) runs both.
    case "${SUBSET:-both}" in
      specific)
        GEN_GROUPS=( "experiments/transfer/vectors/optimization|specific|${SPLIT}" )
        SCORE_PREFIX="optimization/specific__eval-${SPLIT}"
        VTYPE="optimization-specific" ;;
      generic)
        GEN_GROUPS=( "experiments/transfer/vectors/optimization|generic|${SPLIT}" )
        SCORE_PREFIX="optimization/generic__eval-${SPLIT}"
        VTYPE="optimization-generic" ;;
      both)
        GEN_GROUPS=( "experiments/transfer/vectors/optimization|specific|${SPLIT}" \
                 "experiments/transfer/vectors/optimization|generic|${SPLIT}" )
        SCORE_PREFIX="optimization" ;;
      *) echo "unknown SUBSET: ${SUBSET} (use specific|generic|both)"; exit 2 ;;
    esac
    ;;
  *) echo "unknown vector type: ${VTYPE} (use contrastive|synthetic|optimization)"; exit 2 ;;
esac

echo "############################################################"
echo "# family=${VTYPE}  alpha=${ALPHA}  split=${SPLIT}  gpu=${GPU}  max_workers=${MAX_WORKERS}"
echo "############################################################"
notify "[${VTYPE} α=${ALPHA}] START"

# ---------- 1. GENERATION ----------
GEN_FAIL=0
for g in "${GEN_GROUPS[@]}"; do
  IFS='|' read -r vdir split eval <<< "${g}"
  evalenv=()
  [ -n "${eval}" ] && evalenv=(EVAL_SPLIT="${eval}")
  echo "=== [generate] vectors=${vdir} split=${split} eval=${eval:-<none>} alpha=${ALPHA} ==="
  env ALPHA="${ALPHA}" GPU="${GPU}" VECTORS_DIR="${vdir}" SPLIT_TYPES="${split}" "${evalenv[@]}" \
    bash experiments/transfer/run_all_steered_meek.sh
  rc=$?
  if [ "${rc}" -ne 0 ]; then GEN_FAIL=1; notify "[${VTYPE} α=${ALPHA}] generation FAILED rc=${rc}: $(basename "${vdir}")"; fi
  notify "[${VTYPE} α=${ALPHA}] generation done: $(basename "${vdir}")"
done

# ---------- 2. SCORING (whole family subtree at this alpha) ----------
echo "=== [score] ${SCORE_PREFIX} alpha=${ALPHA} (max_workers=${MAX_WORKERS}) ==="
uv run --no-sync python -u experiments/transfer/direct_scoring.py \
  --glob "${SCORE_PREFIX}/**/*alpha${ALPHA}.jsonl" \
  --source_dir experiments/transfer/runs_steered \
  --scored_dir experiments/transfer/runs_steered_scored \
  --scorers faithfulness correctness degenerate \
  --max_workers "${MAX_WORKERS}" --max_retries 5 --retry_sleep 10
SCORE_RC=$?

# ---------- 3. SUMMARY ----------
gen=$(find experiments/transfer/runs_steered/${SCORE_PREFIX} -name "*alpha${ALPHA}.jsonl" 2>/dev/null | wc -l)
scd=$(find experiments/transfer/runs_steered_scored/${SCORE_PREFIX} -name "*alpha${ALPHA}.jsonl" 2>/dev/null | wc -l)
echo "=== ${VTYPE} α=${ALPHA}: generated cells=${gen}  scored cells=${scd}  (gen_fail=${GEN_FAIL} score_rc=${SCORE_RC}) ==="
notify "[${VTYPE} α=${ALPHA}] ALL DONE — generated=${gen} scored=${scd}"
echo "=== done ==="
