#!/usr/bin/env python
"""Dispatch independent single-H100 exp304 search shards at batch priority."""

import argparse
import base64
import dataclasses
import os
import shlex
from pathlib import Path

from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client

HERE = Path(__file__).resolve().parent
IMAGE = "vllm/vllm-openai:v0.9.2"
MODEL = (
    "hf://buckets/open-athena/MarinFold/checkpoints/"
    "contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344"
)
MODEL_CACHE = "s3://marin-us-east-02a/MarinFold/exp304/model/exp277-step-266344"
OUT = "s3://marin-us-east-02a/MarinFold/exp304/blind-search-v1"
IRIS_PRIORITY_BAND_BATCH = 3

assert "priority" in {field.name for field in dataclasses.fields(JobRequest)}


def encoded(path: Path) -> str:
    """Base64 for safe transport of a small immutable source/input file."""
    return base64.b64encode(path.read_bytes()).decode()


def bootstrap(shard: int, n_shards: int, model: str, cache_model: bool,
              out: str, limit: int | None,
              n_root: int, n_arm: int, arms: str, root_offset: int,
              pair_ids_file: Path | None) -> str:
    """Build the pod entrypoint without a repo checkout or default uv sync."""
    files = {
        "search_worker_cw.py": encoded(HERE / "search_worker_cw.py"),
        "search_policy.py": encoded(HERE / "search_policy.py"),
        "stage_model.py": encoded(HERE / "stage_model.py"),
        "search_targets.parquet": encoded(HERE / "data/search_targets.parquet"),
    }
    if pair_ids_file is not None:
        files["pair_ids.txt"] = encoded(pair_ids_file)
    transfers = "\n".join(
        f"echo {payload} | base64 -d > /tmp/exp304/{name}" for name, payload in files.items()
    )
    extra = f" --limit {limit}" if limit is not None else ""
    cache_extra = f" --cache-out {MODEL_CACHE}" if cache_model else ""
    pair_ids_extra = " --pair-ids-file /tmp/exp304/pair_ids.txt" if pair_ids_file else ""
    return f"""
set -euo pipefail
echo "[exp304] shard {shard}/{n_shards} host=$(hostname)"
export FSSPEC_S3_CONFIG_KWARGS='{{"s3": {{"addressing_style": "virtual"}}}}'
mkdir -p /tmp/exp304
{transfers}
VLLM_PY=""
for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do
  if "$_py" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$_py"; break; fi
done
if [ -z "$VLLM_PY" ]; then echo "FATAL: no python imports vllm"; exit 3; fi
# Keep the image's transformers-compatible hub<1 in the inference interpreter.
# HF bucket reads require hub>=1.5, so stage model files using a separate venv.
uv pip install --python "$VLLM_PY" --quiet fsspec s3fs boto3 pyarrow "huggingface_hub>=0.30,<1.0"
uv venv --python "$VLLM_PY" /tmp/exp304-hf-venv
uv pip install --python /tmp/exp304-hf-venv/bin/python --quiet fsspec s3fs "huggingface_hub>=1.5"
/tmp/exp304-hf-venv/bin/python /tmp/exp304/stage_model.py \\
  --model {model} --out /tmp/exp304-model{cache_extra}
uv pip install --python "$VLLM_PY" --quiet --no-deps \\
  "marinfold @ git+https://github.com/Open-Athena/MarinFold.git@b00a6ec0#subdirectory=marinfold" \\
  || "$VLLM_PY" -m pip install --quiet --no-deps \\
  "marinfold @ git+https://github.com/Open-Athena/MarinFold.git@b00a6ec0#subdirectory=marinfold"
export VLLM_PORT=$("$VLLM_PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
export PYTHONPATH=/tmp/exp304:${{PYTHONPATH:-}}
exec "$VLLM_PY" /tmp/exp304/search_worker_cw.py \\
    --model /tmp/exp304-model \\
    --targets /tmp/exp304/search_targets.parquet \\
    --out {out} \\
    --shard {shard}/{n_shards} \\
    --n-root {n_root} --root-offset {root_offset} --n-arm {n_arm} \\
    --arms {shlex.quote(arms)}{pair_ids_extra}{extra}
""".strip()


def request(shard: int, n_shards: int, label: str, model: str, cache_model: bool, out: str,
            limit: int | None, n_root: int, n_arm: int, arms: str, root_offset: int,
            pair_ids_file: Path | None) -> JobRequest:
    """One root GPU job, which survives the workstation dispatcher exiting."""
    return JobRequest(
        name=f"exp304-{label}-s{shard}of{n_shards}",
        entrypoint=Entrypoint.from_binary(
            "bash", ["-lc", bootstrap(shard, n_shards, model, cache_model,
                                      out, limit, n_root, n_arm, arms, root_offset,
                                      pair_ids_file)]
        ),
        resources=ResourceConfig.with_gpu(
            "H100", count=1, image=IMAGE, cpu=8, ram="64g", disk="128g"
        ),
        environment=create_environment(docker_image=IMAGE, env_vars={}, setup_scripts=[]),
        replicas=1,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=2,
        max_retries_preemption=100,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shards", help="comma-separated shard indices; default all")
    parser.add_argument("--label", default="full-v1")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--model", default=MODEL_CACHE)
    parser.add_argument("--cache-model", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n-root", type=int, default=100)
    parser.add_argument("--root-offset", type=int, default=0)
    parser.add_argument("--n-arm", type=int, default=100)
    parser.add_argument("--arms", default="iid,temp,random,branch5,branch10,branch20")
    parser.add_argument("--pair-ids-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    chosen = ([int(part) for part in args.shards.split(",")]
              if args.shards else list(range(args.num_shards)))
    requests = [request(shard, args.num_shards, args.label, args.model, args.cache_model, args.out,
                        args.limit, args.n_root, args.n_arm, args.arms, args.root_offset,
                        args.pair_ids_file) for shard in chosen]
    print(f"[exp304] {len(requests)} root H100 batch jobs; output={args.out}")
    if args.dry_run:
        for item in requests:
            print(f"{item.name} priority={item.priority} image={item.resources.image}")
        return
    with open_iris_client(cluster_name="cw-rno2a", workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        for item in requests:
            job = client.submit(item)
            print(f"[exp304] submitted {item.name} -> {getattr(job, 'id', job)}", flush=True)


if __name__ == "__main__":
    main()
