#!/bin/bash
# CoreWeave shard parquets -> npz score matrices -> exp89 metric rows -> tables + plots.
#
# fetch_cw_scores.py and build_rollout_rows.py are exp82's, unchanged: the first
# refuses to proceed unless all 554 (dataset, stem) units are present exactly
# once, the second carries exp89's compute_metrics implementation verbatim. Only
# the summary and the two figures are this experiment's own.
set -euo pipefail
set -a; source ~/.config/marin/cw-rno2a.env; set +a
export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'
export FSSPEC_S3_ENDPOINT_URL="$AWS_ENDPOINT_URL"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
EXP82="$REPO_ROOT/experiments/exp82_evals_contacts_v1_contact_prediction"
EXP89="$REPO_ROOT/experiments/exp89_evals_contacts_v1_model_on_eval_set"
PY="$HERE/.venv/bin/python"

S3=s3://marin-us-east-02a/MarinFold/exp169_eval/scores
SCRATCH=${EXP169_SCRATCH:-/home/bizon/exp169_eval}
GT="$SCRATCH/gt_universe.jsonl"

LABELS=(exp117_e16_final_step35679 exp117_e16_early_step33450 exp146_3b_e8_step17839)

for L in "${LABELS[@]}"; do
  echo "=== fetch $L ==="
  "$PY" "$EXP82/fetch_cw_scores.py" --parts "$S3/$L" --out "$SCRATCH/cw_scores/$L"
done

echo "=== metrics (exp89 compute_metrics semantics) ==="
"$PY" "$EXP82/build_rollout_rows.py" --gt "$GT" \
  --model "exp117_e16_final_step35679=$SCRATCH/cw_scores/exp117_e16_final_step35679" \
  --model "exp117_e16_early_step33450=$SCRATCH/cw_scores/exp117_e16_early_step33450" \
  --model "exp146_3b_e8_step17839=$SCRATCH/cw_scores/exp146_3b_e8_step17839" \
  --out "$HERE/data/exp169_rows.csv.gz" --summary "$HERE/data/exp169_rollout_summary.csv"

echo "=== aggregate + paired differences ==="
"$PY" "$HERE/summarize_results.py" \
  --rows "$HERE/data/exp169_rows.csv.gz" \
  --out-summary "$HERE/data/exp169_summary.csv" \
  --out-paired "$HERE/data/exp169_paired.csv"

echo "=== figures ==="
"$PY" "$HERE/plot_results.py" \
  --rows "$HERE/data/exp169_rows.csv.gz" \
  --exp89-csv "$EXP89/data/contact_precision_all.csv" \
  --prior-rows "$EXP82/data/where_we_stand_rows.csv.gz" \
  --out-dir "$HERE/plots"
