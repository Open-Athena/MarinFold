# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the CoreWeave re-score in-cluster and publish the rows — issue #208.

The workstation cannot read CoreWeave object storage here: the stored key is
rejected ("The access key ID you provided does not exist in our records") and
this s3fs build also mis-parses the `FSSPEC_S3_CONFIG_KWARGS` blob the pods rely
on. In-cluster reads work — the eval shards wrote there — so the scoring runs
where the data is, and the result comes back via HF.

exp82's `fetch_cw_scores.py` and `build_rollout_rows.py` are **base64-inlined
verbatim**, exactly as the eval worker is. That is deliberate: `build_rollout_rows`
carries exp89's `compute_metrics` unchanged, and this comparison is only worth
anything if the CoreWeave numbers come out of the same scorer as the v5p ones.

    python score_cw_rescore.py --submit
"""

import argparse
import base64
import dataclasses
import os
import textwrap
from pathlib import Path

from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

IRIS_PRIORITY_BAND_BATCH = 3
IMAGE = os.environ.get("SCORE_CW_IMAGE", "vllm/vllm-openai:v0.9.2")
EXP82 = Path(__file__).resolve().parent.parent / "exp82_evals_contacts_v1_contact_prediction"
# #199's own manifest names this as the ground truth it scored against, so both
# sides of the comparison use the same universe.
GT_URL = ("https://huggingface.co/buckets/open-athena/MarinFold/resolve/"
          "data/contacts-v1-model-eval-exp169/gt_universe.jsonl")

assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; submit from /home/bizon/git/marin-freshiris."
)

FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)


def build_bootstrap(*, parts: str, label: str, repo: str) -> str:
    fetch_b64 = base64.b64encode((EXP82 / "fetch_cw_scores.py").read_bytes()).decode()
    rows_b64 = base64.b64encode((EXP82 / "build_rollout_rows.py").read_bytes()).decode()
    return textwrap.dedent(f"""
        set -euo pipefail
        echo "[score] host=$(hostname)"
        {FSSPEC_VIRTUAL_ADDRESSING_EXPORT}

        PY=/usr/bin/python3
        uv pip install --python "$PY" --quiet fsspec s3fs boto3 pyarrow pandas \\
            scikit-learn huggingface_hub \\
          || "$PY" -m pip install --quiet fsspec s3fs boto3 pyarrow pandas \\
            scikit-learn huggingface_hub

        mkdir -p /tmp/score && cd /tmp/score
        echo {fetch_b64} | base64 -d > fetch_cw_scores.py
        echo {rows_b64}  | base64 -d > build_rollout_rows.py
        curl -sSL "{GT_URL}" -o gt_universe.jsonl
        echo "[score] gt_universe.jsonl $(wc -l < gt_universe.jsonl) records"

        "$PY" fetch_cw_scores.py --parts {parts} --out /tmp/score/matrices
        "$PY" build_rollout_rows.py --gt gt_universe.jsonl \\
            --model {label}=/tmp/score/matrices \\
            --out /tmp/score/rows.csv.gz --summary /tmp/score/summary.csv

        "$PY" - <<'EOS'
import os, pandas as pd
from huggingface_hub import HfApi
rows = pd.read_csv("/tmp/score/rows.csv.gz")
for rng in ("all", "long"):
    sel = rows[(rows["cut"] == "R") & (rows["range"] == rng)]
    print(f"[score] R-precision {{rng:5s}} mean {{sel['precision'].mean():.6f}}  n={{len(sel)}}",
          flush=True)
tok = os.environ.get("HF_TOKEN")
if tok:
    HfApi(token=tok).upload_file(
        path_or_fileobj="/tmp/score/rows.csv.gz",
        path_in_repo="exp208_cw_rescore_rows.csv.gz",
        repo_id="{repo}", repo_type="model",
        commit_message="exp208: CoreWeave re-score per-protein rows")
    print("[score] uploaded rows to {repo}", flush=True)
EOS
    """).strip()


def hf_token() -> str:
    import configparser
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    parser = configparser.ConfigParser()
    parser.read(Path.home() / ".cache/huggingface/stored_tokens")
    return parser.get("write2", "hf_token")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="s3://marin-us-east-02a/MarinFold/exp208_eval/scores_full/exp199")
    ap.add_argument("--label", default="exp208_cw_rescore_exp199")
    ap.add_argument("--repo", default="timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199")
    ap.add_argument("--name", default="exp208-score-cw-rescore")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    request = JobRequest(
        name=a.name,
        entrypoint=Entrypoint.from_binary(
            "bash", ["-lc", build_bootstrap(parts=a.parts, label=a.label, repo=a.repo)]),
        resources=ResourceConfig.with_cpu(cpu=4, ram="32g", disk="32g", image=IMAGE),
        environment=create_environment(docker_image=IMAGE,
                                       env_vars={"HF_TOKEN": hf_token()}, setup_scripts=[]),
        replicas=1,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=2,
        max_retries_preemption=50,
    )
    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    cluster = os.environ.get("SCORE_CW_CLUSTER", "cw-rno2a")
    print(f"[score] submitting {a.name} to {cluster}: {a.parts}")
    with open_iris_client(cluster_name=cluster, workspace=None) as iris_client:
        print(f"[score] submitted {FrayIrisClient.from_iris_client(iris_client).submit(request)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
