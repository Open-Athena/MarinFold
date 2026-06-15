#!/usr/bin/env bash
# Post-run pipeline for exp74: after the Modal run + volume sync, turn the
# raw Protenix outputs into the curated best/ tree, score the four configs
# against pyconfind ground truth, combine with FoldBench-100, and plot.
# Idempotent / resumable.
set -euo pipefail
cd "$(dirname "$0")"
EXP65="$(cd ../exp65_evals_low_msa_depth_proteins && pwd)"
RUNS="${1:-_scratch/exp65_runs}"          # synced Modal outputs ({mode}/{stem}/seed_*)
BEST="_scratch/best_exp65"

echo "== select-best (454 x {single_seq,msa}) =="
uv run python cli.py select-best --runs "$RUNS" --out "$BEST" \
  --modes single_seq,msa --manifest inputs/manifest.csv

echo "== score exp65 (4 configs vs pyconfind GT) =="
uv run python cli.py contact-eval --manifest data/eval_manifest_exp65.csv \
  --best "$BEST" --gt-root "$EXP65" --out data/exp65_scores --modes single_seq,msa

echo "== combine FoldBench-100 + exp65 =="
uv run python combine_scores.py --dirs data/foldbench_scores data/exp65_scores --out data

echo "== plots =="
uv run python plot.py --precision-csv data/contact_precision_all.csv \
  --meta-csv data/contact_eval_meta_all.csv --out plots

echo "== done. outputs: data/contact_precision_all.csv, plots/ =="
