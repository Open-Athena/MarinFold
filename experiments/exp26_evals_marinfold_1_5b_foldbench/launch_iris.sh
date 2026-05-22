#!/usr/bin/env bash
# Launch the 1.5B FoldBench eval on TRC via iris.
#
# Reads MODELS.yaml off the worker would require shipping the repo
# root, which iris doesn't. Instead the URL + nickname are hard-coded
# below — refresh them if MODELS.yaml's 1.5B entry ever moves.
#
# Usage:
#   ./launch_iris.sh                # full 100 proteins
#   ./launch_iris.sh --limit 3      # smoke; extra args pass through to run_eval.py
set -euo pipefail

MODEL_URL="https://huggingface.co/buckets/open-athena/MarinFold/tree/checkpoints/protein-contacts-1_5b-distance-masked-70f8f5/step-49999"
MODEL_NICKNAME="1.5B"
OUT_GCS="gs://marin-us-east5/protein-structure/MarinFold/exp26/protein-contacts-1_5b-distance-masked-70f8f5-step-49999-foldbench-monomers"

cd "$(dirname "$0")"

# Sanity check: protenix_data/ should be present locally so iris ships
# the GT mmCIFs + manifest alongside the script.
if [[ ! -f protenix_data/data/protenix-foldbench-monomers/manifest.csv ]]; then
    echo "ERROR: protenix_data/ is missing. Run:" >&2
    echo "    uv run --with 'huggingface_hub>=1.5' python fetch_protenix_data.py" >&2
    exit 1
fi

exec uv run iris --cluster=marin job run \
    --tpu v5p-8 \
    --cpu 16 --memory 64GB --disk 64GB \
    --enable-extra-resources \
    --extra vllm --extra tpu \
    --no-wait \
    --max-retries 2 \
    -e MARINFOLD_RUNNER_TAG iris \
    -- python run_eval.py \
        --model-url "$MODEL_URL" \
        --model-nickname "$MODEL_NICKNAME" \
        --out "$OUT_GCS" \
        "$@"
