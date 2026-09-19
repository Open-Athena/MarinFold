#!/usr/bin/env bash
# Minimal CPU bootstrap: object-store IO, parquet and Hub upload only.
# The GPU path's bootstrap.sh pulls torch and the Proteina checkpoints, which no
# CPU-side repack, upload or selection stage needs.
set -euo pipefail

export OMP_NUM_THREADS=8
export UV_LINK_MODE=copy
export PYTHONPATH=/tmp/exp278/marinfold:/tmp/exp278
mkdir -p /tmp/exp278
/opt/conda/bin/python -m pip install --quiet uv==0.8.22
uv pip install --python /opt/conda/bin/python --quiet \
  'numpy==1.26.4' 'pyarrow==19.0.1' 'fsspec==2025.3.0' 's3fs==2025.3.0' \
  'huggingface-hub==0.34.4' 'hf-xet==1.1.5'
uv run --no-project /opt/conda/bin/python "$@"
