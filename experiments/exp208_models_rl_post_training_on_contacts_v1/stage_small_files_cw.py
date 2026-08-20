# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy an export's small metadata files from an HF repo into CoreWeave S3 — issue #208.

exp169's `stage_from_hf_cw.py` mirrors **weights only**, by design: the small
files that needed repair were uploaded from the workstation, which is a few
hundred kB. exp208 cannot take that route — the stored workstation CoreWeave
credentials are rejected by the object store ("The access key ID you provided
does not exist in our records") while in-cluster writes work fine, as the weight
mirror proved. So the small files travel the same way the weights did.

WHICH CONFIG, AND WHY IT MATTERS. exp208 is re-scoring exp199 on CoreWeave to
isolate the accelerator in a +0.023 R-precision discrepancy
(RPRECISION_STACK_DISCREPANCY.md). The comparison is only single-variable if the
CoreWeave run uses the *same artifact* exp208 evaluated on v5p — which means the
**repaired** config that states rope as top-level `rope_theta` + `rope_scaling`
as well as the transformers-5 `rope_parameters` block.

That is not a nicety. exp82's CoreWeave eval runs the `vllm/vllm-openai:v0.9.2`
image, whose transformers is 4.x, and 4.x does not read `rope_parameters`. It
does not error either: it silently falls back to the architecture default,
loading `rope_theta = 10000` where this model was trained with 500000 (see the
bucket copy's PROVENANCE.md). Staging #199's own model-repo config here would
therefore measure the rope bug and call it an accelerator effect.

    python stage_small_files_cw.py \\
        --repo timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199 \\
        --dst s3://marin-us-east-02a/MarinFold/exp208_eval/model_exp199
"""

import argparse
import base64
import dataclasses
import os
import textwrap

from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

IRIS_PRIORITY_BAND_BATCH = 3
IMAGE = os.environ.get("STAGE_CW_IMAGE", "vllm/vllm-openai:v0.9.2")

assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; submit from /home/bizon/git/marin-freshiris."
)

# CoreWeave object storage rejects path-style S3. Literal braces on purpose.
FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    "export FSSPEC_S3_CONFIG_KWARGS='{\"s3\": {\"addressing_style\": \"virtual\"}}'"
)

# Deliberately NOT the safetensors: those are already mirrored, and re-writing
# them from a different source risks a silent mismatch with the index.
SMALL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")

WORKER = '''
import argparse, os, sys
import fsspec
from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--files", required=True)
a = ap.parse_args()

fs, root = fsspec.core.url_to_fs(a.dst)
existing = {os.path.basename(f) for f in fs.ls(root, detail=False)}
print(f"[small] destination already holds: {sorted(existing)}", flush=True)
if not any(f.endswith(".safetensors") for f in existing):
    sys.exit("[small] FATAL: no safetensors at the destination — mirror the weights first")

for name in a.files.split(","):
    local = hf_hub_download(a.repo, name)
    fs.put_file(local, f"{root.rstrip('/')}/{name}")
    print(f"[small]   wrote {name}", flush=True)

import json
with fsspec.open(f"{a.dst.rstrip('/')}/config.json", "r") as fh:
    cfg = json.load(fh)
theta = cfg.get("rope_theta")
if theta is None:
    sys.exit("[small] FATAL: staged config has no top-level rope_theta; a "
             "transformers 4.x image would silently load default rope")
print(f"[small] verified rope_theta={theta} vocab_size={cfg.get('vocab_size')}", flush=True)
print(f"[small] prefix now holds: {sorted(os.path.basename(f) for f in fs.ls(root, detail=False))}",
      flush=True)
'''


def build_bootstrap(*, repo: str, dst: str, files: str) -> str:
    worker_b64 = base64.b64encode(WORKER.encode()).decode()
    return textwrap.dedent(f"""
        set -euo pipefail
        echo "[small] host=$(hostname) image={IMAGE}"
        {FSSPEC_VIRTUAL_ADDRESSING_EXPORT}

        PY=""
        for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 python3 python; do
          if "$_py" -c "import sys" >/dev/null 2>&1; then PY="$_py"; break; fi
        done
        if [ -z "$PY" ]; then echo "[small] FATAL: no python"; exit 3; fi

        uv pip install --python "$PY" --quiet fsspec s3fs boto3 huggingface_hub \\
          || "$PY" -m pip install --quiet fsspec s3fs boto3 huggingface_hub

        mkdir -p /tmp/small
        echo {worker_b64} | base64 -d > /tmp/small/worker.py
        exec "$PY" /tmp/small/worker.py --repo {repo} --dst {dst} --files {files}
    """).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--files", default=",".join(SMALL_FILES))
    ap.add_argument("--name", default="exp208-stage-small-exp199")
    a = ap.parse_args()

    request = JobRequest(
        name=a.name,
        entrypoint=Entrypoint.from_binary(
            "bash", ["-lc", build_bootstrap(repo=a.repo, dst=a.dst, files=a.files)]),
        resources=ResourceConfig.with_cpu(cpu=2, ram="8g", disk="16g", image=IMAGE),
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
    print(f"[small] submitting {a.name} to {cluster}: {a.repo} -> {a.dst}")
    with open_iris_client(cluster_name=cluster, workspace=None) as iris_client:
        job = FrayIrisClient.from_iris_client(iris_client).submit(request)
        print(f"[small] submitted {job}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
