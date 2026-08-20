# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror a published HF checkpoint export into CoreWeave S3, in bf16, in-cluster.

The workstation uplink is ~2 MB/s, so pushing the #146 3B export (12 GB fp32 /
5.6 GB bf16) from here costs ~48 minutes and is the whole experiment's critical
path. A CoreWeave pod pulls the same bytes from the HF CDN at tens of MB/s and
writes them to S3 in-region, which turns that into a few minutes.

So the split is: the **small** files that needed real repair (the transformers-5
-> 4.57 config downgrade and the tokenizer-class fix, done by
``prepare_hf_export.py``) are uploaded from here — they are a few hundred kB —
and this job mirrors only the **weights**, casting fp32 -> bf16 on the way. vLLM
loads with ``dtype="bfloat16"`` regardless, so the cast is the rounding it would
do at load time.

The pod runs the vLLM image purely for its baked-in torch/safetensors; it needs
no GPU.

Submitted with the *fresh* marin checkout's client — iris rejects a client older
than 14 days, and this experiment's own venv has no fray/iris at all::

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    KUBECONFIG=~/.kube/coreweave-iris-rno2a \\
    /home/bizon/git/marin-freshiris/.venv/bin/python stage_from_hf_cw.py \\
        --repo open-athena/marinfold-exp146 \\
        --path prot-…-us-east1/hf/step-17839 \\
        --dst s3://marin-us-east-02a/MarinFold/exp169_eval/model_exp146_3b_step17839
"""

import argparse
import base64
import dataclasses
import os
import textwrap

from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

# iris PriorityBand enum (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3).
IRIS_PRIORITY_BAND_BATCH = 3

assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; submit from /home/bizon/git/marin-freshiris."
)

IMAGE = os.environ.get("STAGE_CW_IMAGE", "vllm/vllm-openai:v0.9.2")

# CoreWeave object storage rejects path-style S3. Literal braces on purpose.
FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)

WORKER = '''
"""Runs on the pod: HF repo subdir -> bf16 safetensors -> S3."""
import argparse, json, os, time
from pathlib import Path

import fsspec
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--path", required=True)
ap.add_argument("--dst", required=True)
a = ap.parse_args()

t0 = time.time()
local = snapshot_download(a.repo, allow_patterns=[f"{a.path}/*"],
                          local_dir="/tmp/hf_src", max_workers=8)
src = Path(local) / a.path
print(f"[stage] downloaded {a.repo}/{a.path} in {time.time() - t0:.0f}s: "
      f"{sorted(p.name for p in src.iterdir())}", flush=True)

index = json.loads((src / "model.safetensors.index.json").read_text())
fs, root = fsspec.core.url_to_fs(a.dst)
total = 0
for shard in sorted(set(index["weight_map"].values())):
    t1 = time.time()
    tensors = load_file(src / shard)
    recast = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
              for k, v in tensors.items()}
    out = Path("/tmp/bf16") / shard
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(recast, str(out), metadata={"format": "pt"})
    size = out.stat().st_size
    total += size
    fs.put_file(str(out), f"{root.rstrip('/')}/{shard}")
    out.unlink()
    print(f"[stage]   {shard}: {size / 2**30:.3f} GiB in {time.time() - t1:.0f}s", flush=True)

index["metadata"]["total_size"] = total
with fsspec.open(f"{a.dst.rstrip('/')}/model.safetensors.index.json", "w") as fh:
    json.dump(index, fh)

listing = {os.path.basename(f) for f in fs.ls(root, detail=False)}
print(f"[stage] DONE {total / 2**30:.2f} GiB in {(time.time() - t0) / 60:.1f} min -> {a.dst}")
print(f"[stage] prefix now holds: {sorted(listing)}", flush=True)
'''


def build_bootstrap(*, repo: str, path: str, dst: str) -> str:
    worker_b64 = base64.b64encode(WORKER.encode()).decode()
    return textwrap.dedent(f"""
        set -euo pipefail
        echo "[stage] host=$(hostname) image={IMAGE}"
        {FSSPEC_VIRTUAL_ADDRESSING_EXPORT}

        PY=""
        for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 python3 python; do
          if "$_py" -c "import torch, safetensors" >/dev/null 2>&1; then PY="$_py"; break; fi
        done
        if [ -z "$PY" ]; then echo "[stage] FATAL: no python with torch"; exit 3; fi
        echo "[stage] python: $PY"

        # Unlike the eval worker, this pod never starts vLLM — it only casts and
        # copies tensors — so there is no pinned inference stack to protect and a
        # plain install is fine. (huggingface_hub is already in the image.)
        uv pip install --python "$PY" --quiet fsspec s3fs boto3 huggingface_hub \\
          || "$PY" -m pip install --quiet fsspec s3fs boto3 huggingface_hub

        mkdir -p /tmp/stage
        echo {worker_b64} | base64 -d > /tmp/stage/worker.py
        exec "$PY" /tmp/stage/worker.py --repo {repo} --path {path} --dst {dst}
    """).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id holding the export")
    ap.add_argument("--path", required=True, help="path within the repo to the hf/step-N dir")
    ap.add_argument("--dst", required=True, help="s3:// destination prefix")
    ap.add_argument("--name", default=None, help="iris job name (default derived from --dst)")
    a = ap.parse_args()

    name = a.name or f"exp169-stage-{a.dst.rstrip('/').rsplit('/', 1)[-1].replace('_', '-')}"
    request = JobRequest(
        name=name,
        entrypoint=Entrypoint.from_binary(
            "bash", ["-lc", build_bootstrap(repo=a.repo, path=a.path, dst=a.dst)]),
        # No GPU: this is a download / dtype-cast / upload job. Disk holds the
        # fp32 download plus one bf16 shard at a time.
        resources=ResourceConfig.with_cpu(cpu=8, ram="48g", disk="64g", image=IMAGE),
        environment=create_environment(docker_image=IMAGE, env_vars={}, setup_scripts=[]),
        replicas=1,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=2,
        max_retries_preemption=50,
    )

    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    cluster = os.environ.get("STAGE_CW_CLUSTER", "cw-rno2a")
    print(f"[stage] submitting {name} to {cluster}: {a.repo}/{a.path} -> {a.dst}")
    with open_iris_client(cluster_name=cluster, workspace=None) as iris_client:
        job = FrayIrisClient.from_iris_client(iris_client).submit(request)
        print(f"[stage] submitted {job}")
    print("[stage] root job — it keeps running after this process exits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
