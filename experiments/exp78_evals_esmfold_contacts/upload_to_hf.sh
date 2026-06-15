#!/usr/bin/env bash
# Push exp78 raw results to the open-athena/MarinFold HF bucket.
#
# Under data/esmfold-contacts-eval-exp78/:
#   - contact_precision_all.csv / contact_eval_meta_all.csv  (combined scores:
#                                                             esmfold + esmfold2 + protenix-v2)
#   - contacts_raw_all.parquet                               ("save all contacts": every degree>0
#                                                             pyconfind contact, GT + ESM predicted)
#   - eval_manifest_*.csv                                    (the eval spec; same set as exp74)
#   - structures/esmfold/{stem}/structure.cif               (all ESMFold predicted structures)
#   - structures/esmfold2/{stem}/{structure.cif,provenance.json}  (all ESMFold2 predicted structures
#                                                             + best-of-N provenance)
#
# Saving every predicted structure is an explicit issue #78 requirement: we
# can re-score with different contact criteria without re-running predictions.
#
# Protenix v2 raw outputs are already on the bucket under exp74's prefix
# (data/protenix-contacts-eval-exp74/), so we don't re-upload them.
#
# NOTE: writing the open-athena bucket needs an open-athena-scoped HF token
# (the workstation default may be timodonnell-only -> 403). Uses the system
# `hf` CLI (>=1.x) which has the `buckets` subcommand.
set -euo pipefail
cd "$(dirname "$0")"
BUCKET="hf://buckets/open-athena/MarinFold/data/esmfold-contacts-eval-exp78"

hf buckets cp data/contact_precision_all.csv   "$BUCKET/contact_precision_all.csv"
hf buckets cp data/contact_eval_meta_all.csv   "$BUCKET/contact_eval_meta_all.csv"
hf buckets cp data/contacts_raw_all.parquet    "$BUCKET/contacts_raw_all.parquet"
hf buckets cp data/eval_manifest_foldbench.csv "$BUCKET/eval_manifest_foldbench.csv"
hf buckets cp data/eval_manifest_exp65.csv     "$BUCKET/eval_manifest_exp65.csv"

# plots + summary.pdf (small; per-file cp is more reliable than folder sync)
for f in plots/*.png plots/summary.pdf; do
  [ -f "$f" ] && hf buckets cp "$f" "$BUCKET/plots/$(basename "$f")"
done

# all predicted structures (idempotent; re-running finishes a partial sync)
hf buckets sync _scratch/pred/esmfold   "$BUCKET/structures/esmfold"
hf buckets sync _scratch/pred/esmfold2  "$BUCKET/structures/esmfold2"

echo "uploaded to $BUCKET"
