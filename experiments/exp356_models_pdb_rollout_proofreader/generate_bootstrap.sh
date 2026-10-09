#!/usr/bin/env bash
set -euo pipefail
unset FSSPEC_S3_CONFIG_KWARGS
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
VLLM_PY=""
for candidate in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do
  if "$candidate" -c 'import torch, vllm' >/dev/null 2>&1; then VLLM_PY="$candidate"; break; fi
done
if [ -z "$VLLM_PY" ]; then echo 'No vLLM interpreter in image'; exit 3; fi
uv pip install --python "$VLLM_PY" --quiet --no-deps \
  fsspec==2026.1.0 s3fs==2026.1.0 aiobotocore==2.26.0 botocore==1.41.5 \
  aiohttp==3.14.3 aiohappyeyeballs==2.7.1 aioitertools==0.13.0 aiosignal==1.4.0 \
  attrs==26.1.0 frozenlist==1.8.0 idna==3.20 jmespath==1.1.0 multidict==6.9.1 \
  propcache==0.5.4 python-dateutil==2.9.0.post0 six==1.17.0 urllib3==2.8.0 \
  wrapt==1.17.4rc1 yarl==1.25.1 pyarrow==23.0.1
export VLLM_PORT=$("$VLLM_PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
exec "$VLLM_PY" generate.py "$@"
