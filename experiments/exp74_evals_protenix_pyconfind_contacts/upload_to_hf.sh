#!/usr/bin/env bash
# Push exp74 raw results to the open-athena/MarinFold HF bucket.
#
# Under data/protenix-contacts-eval-exp74/:
#   - contact_precision_all.csv / contact_eval_meta_all.csv  (the scores)
#   - contacts_raw_all.parquet                               ("save all contacts": every
#                                                             degree>0 pyconfind contact, GT + predicted)
#   - eval_manifest_exp65.csv                                (the eval spec)
#   - best_exp65/{mode}/{stem}/{structure.cif,distogram.npz,confidence.json,provenance.json}
#                                                             (curated top-1 Protenix exp65 outputs)
#
# FoldBench-100 raw outputs are already on the bucket under exp12's prefix
# (data/protenix-foldbench-monomers/), so we don't re-upload them.
#
# NOTE: writing the open-athena bucket needs an open-athena-scoped HF token
# (the workstation default may be timodonnell-only -> 403). Uses the system
# `hf` CLI (>=1.x) which has the `buckets` subcommand.
set -euo pipefail
cd "$(dirname "$0")"
BUCKET="hf://buckets/open-athena/MarinFold/data/protenix-contacts-eval-exp74"

hf buckets cp data/contact_precision_all.csv   "$BUCKET/contact_precision_all.csv"
hf buckets cp data/contact_eval_meta_all.csv   "$BUCKET/contact_eval_meta_all.csv"
hf buckets cp data/contacts_raw_all.parquet    "$BUCKET/contacts_raw_all.parquet"
hf buckets cp data/eval_manifest_exp65.csv     "$BUCKET/eval_manifest_exp65.csv"
hf buckets cp _scratch/hf_readme.md            "$BUCKET/README.md"

# plots + summary.pdf (small; per-file cp is more reliable than folder sync)
for f in plots/*.png plots/summary.pdf; do
  [ -f "$f" ] && hf buckets cp "$f" "$BUCKET/plots/$(basename "$f")"
done

# curated Protenix outputs (~5 GB); idempotent — re-running finishes a partial sync
hf buckets sync _scratch/best_exp65            "$BUCKET/best_exp65"

echo "uploaded to $BUCKET"
