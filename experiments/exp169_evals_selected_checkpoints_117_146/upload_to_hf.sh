#!/usr/bin/env bash
# Push the #169 eval results to the open-athena/MarinFold HF bucket.
#
# Under data/contacts-v1-model-eval-exp169/ — the same layout exp89 uses for the
# previous generation of this eval, so the two sit side by side:
#   - exp169_rows.csv.gz / exp169_summary.csv / exp169_paired.csv
#                                    per-protein metrics, aggregates, and the
#                                    paired per-protein differences
#   - gt_universe.jsonl / eval_targets.parquet
#                                    the ground truth and the 554 prompts exactly
#                                    as scored, so a re-scoring needs nothing else
#   - scores/<label>/<dataset>__<stem>.npz
#                                    554 [L,L] vote matrices per checkpoint
#   - plots/*.png
#
# The score matrices are the expensive artifact (~100 sampled rollouts x 554
# proteins x 3 checkpoints of H100 time) and every table here is derivable from
# them plus the GT universe — which is why they are published, not just the
# summary. Same reasoning as exp78 saving its predicted structures.
#
# NOTE: writing the open-athena bucket needs an open-athena-scoped HF token
# (the workstation default may be timodonnell-only -> 403). Uses the system
# `hf` CLI (>=1.x) which has the `buckets` subcommand.
set -euo pipefail
cd "$(dirname "$0")"
BUCKET="hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp169"
SCRATCH="${EXP169_SCRATCH:-$HOME/exp169_eval}"

hf buckets cp data/exp169_rows.csv.gz          "$BUCKET/exp169_rows.csv.gz"
hf buckets cp data/exp169_summary.csv          "$BUCKET/exp169_summary.csv"
hf buckets cp data/exp169_paired.csv           "$BUCKET/exp169_paired.csv"
hf buckets cp data/exp169_rollout_summary.csv  "$BUCKET/exp169_rollout_summary.csv"
hf buckets cp data/BUCKET_README.md            "$BUCKET/README.md"

# The eval inputs, byte-identical to what was scored.
hf buckets cp "$SCRATCH/gt_universe.jsonl"     "$BUCKET/gt_universe.jsonl"
hf buckets cp "$SCRATCH/eval_targets.parquet"  "$BUCKET/eval_targets.parquet"

for f in plots/*.png plots/*.json; do
  [ -f "$f" ] && hf buckets cp "$f" "$BUCKET/plots/$(basename "$f")"
done

# 554 score matrices per checkpoint (idempotent; re-running finishes a partial sync)
for label in exp117_e16_final_step35679 exp117_e16_early_step33450 exp146_3b_e8_step17839; do
  hf buckets sync "$SCRATCH/cw_scores/$label" "$BUCKET/scores/$label"
done

echo "uploaded to $BUCKET"
