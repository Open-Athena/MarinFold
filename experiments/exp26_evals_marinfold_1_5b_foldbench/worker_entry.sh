#!/usr/bin/env bash
# Worker-side entry point. Runs inside the iris-provisioned uv env on
# the TPU worker; iris bundles this script + run_eval.py / vocab.py /
# canonical_sequence.py / fetch_protenix_data.py via `git ls-files`,
# so anything not tracked in git (.venv/, outputs/, protenix_data/)
# is NOT shipped — that's why we re-fetch protenix_data here.
#
# Two-step:
#   1. Pull GT mmCIFs + Protenix scores from the open-athena/MarinFold
#      HF bucket. `fetch_protenix_data.py` needs huggingface_hub >= 1.5
#      which conflicts with the worker venv's transformers pin, so we
#      run it in a `uv run --with` ephemeral env layered on top.
#   2. Run the 1.5B distogram eval. Args after `--` flow straight in.
set -euo pipefail

echo "=== exp26 worker_entry.sh ==="
echo "PWD: $(pwd)"
echo "host: $(hostname)"
ls -la

echo "=== Step 1: fetch protenix_data ==="
uv run --with "huggingface_hub>=1.5" python fetch_protenix_data.py

echo "=== Step 2: run_eval ==="
exec python run_eval.py "$@"
