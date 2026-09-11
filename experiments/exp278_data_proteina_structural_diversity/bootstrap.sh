#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export UV_LINK_MODE=copy
export DATA_PATH=/tmp/proteina-data
export PYTHONPATH=/tmp/proteina:/tmp/proteina/ProteinMPNN:/tmp/exp278/marinfold:/tmp/exp278
mkdir -p "$DATA_PATH" /tmp/exp278
apt-get update -qq
apt-get install -y -qq --no-install-recommends gcc g++
dpkg-query -W gcc g++
/opt/conda/bin/python -m pip install --quiet uv==0.8.22
uv pip install --python /opt/conda/bin/python 'torch==2.4.1' 'numpy==1.26.4' 'scipy==1.13.1' \
  'lightning==2.5.0' 'torchmetrics==1.6.1' 'einops==0.6.1' \
  'dm-tree==0.1.8' 'loguru==0.7.2' 'hydra-core==1.3.2' \
  'biotite==0.41.0' 'biopython==1.85' 'pandas==2.2.3' \
  'jaxtyping==0.2.38' 'ml-collections==1.0.0' 'loralib==0.1.2' \
  'transformers==4.48.3' 'fsspec==2025.3.0' 's3fs==2025.3.0' \
  'pyarrow==19.0.1' 'gemmi==0.7.1' 'pyconfind[fast]==0.5.0' \
  'wandb==0.19.8' 'python-dotenv==1.0.1'
uv pip install --python /opt/conda/bin/python --no-deps torch-scatter==2.1.2 \
  --find-links 'https://data.pyg.org/whl/torch-2.4.0+cu124.html'
uv run --no-project /opt/conda/bin/python /tmp/exp278/prepare_assets.py
uv run --no-project /opt/conda/bin/python "$@"
