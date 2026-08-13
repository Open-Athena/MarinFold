# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Fan ``gen_rollouts_worker.py`` out over marin v5p-8 TPUs — exp230's on-policy
rollout generation.

**Why marin TPU and not CoreWeave.**  The standing rule is that GPU work goes to
a CoreWeave H100 cluster at batch priority, and the exp82/exp163 dispatchers this
is forked from do exactly that.  It is not available here: the workstation's
CoreWeave object-storage key
(``~/.config/marin/cw-rno2a.env``) is **revoked** — every request returns *"The
access key ID you provided does not exist in our records"*, against every bucket
and anonymously.  The iris controller is healthy, so jobs could still be
submitted, but neither the model nor the targets could be staged into CW S3 from
here and no result could be read back.  marin's GCS is reachable, has v5p-8
capacity, and is the cluster #163 actually ran this pipeline on after CoreWeave's
batch band starved it for three days.  Restore the key and the CoreWeave path is
a two-line change (``--cluster``, and the S3 prefix).

**TPU submissions go in the INTERACTIVE band** — the opposite of the CoreWeave
rule — because the v5p pool is interactive-dominated and a batch job there never
schedules (#163 §7).

Two placement rules from #163, both of which cost it real time:

* pin the **region**, never the **zone**.  Zone-pinning starved three separate
  jobs; ``with_tpu`` otherwise leaves ``regions`` unset and the scheduler may
  pick a region with no v5p at all.
* a multi-region job needs its **data** mirrored, not just its constraint
  widened.  exp230's model and targets live in ``us-central1`` only, so that is
  the region this pins.

Submitted from the workstation as ROOT jobs (exp82's pattern): ``current_client()``
off-cluster silently falls back to ``LocalClient``, so the iris-backed client is
built explicitly over the CLI's controller tunnel.  Root jobs survive this
process exiting, so there is no driver-must-wait rule and no pod-side ``uv sync``.

    python dispatch_rollouts.py --num-shards 8
    EXP230_DRY_RUN=1 python dispatch_rollouts.py --num-shards 2   # build, don't submit
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import os
from pathlib import Path

from fray.types import Entrypoint, JobRequest, JobStatus, ResourceConfig, create_environment

#: iris ``PRIORITY_BAND_INTERACTIVE``.  See the module docstring — this is
#: deliberately NOT the batch band the CoreWeave rule mandates.
IRIS_PRIORITY_BAND_INTERACTIVE = 1
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "this fray build has no JobRequest.priority — a frozen 0.99.dev wheel has won "
    "the resolution and the band would silently be the default"
)

GCS_PREFIX = os.environ.get(
    "EXP230_GCS_PREFIX",
    "gs://marin-us-central1/protein-structure/MarinFold/exp230_contacts_v1_multi",
)
MODEL = os.environ.get("EXP230_MODEL", f"{GCS_PREFIX}/model/exp199_bf16")
TARGETS = os.environ.get("EXP230_TARGETS", f"{GCS_PREFIX}/targets.parquet")
OUT = os.environ.get("EXP230_OUT", f"{GCS_PREFIX}/rollouts")
REGIONS = os.environ.get("EXP230_TPU_REGIONS", "us-central1").split(",")
TPU_TYPE = os.environ.get("EXP230_TPU_TYPE", "v5p-8")
JOB_PREFIX = os.environ.get("EXP230_JOB_PREFIX", "exp230-rollouts")

N_ROLLOUTS = int(os.environ.get("EXP230_N_ROLLOUTS", "24"))
TOP_K = int(os.environ.get("EXP230_TOP_K", "-1"))
TOP_P = float(os.environ.get("EXP230_TOP_P", "0.95"))
TEMPERATURE = float(os.environ.get("EXP230_TEMPERATURE", "1.0"))
#: v5p-8 is 4 chips; shard the 1.5B across all of them (#163's setting).
TENSOR_PARALLEL = int(os.environ.get("EXP230_TP", "4"))

WORKER = Path(__file__).with_name("gen_rollouts_worker.py")
WORK_DIR = "/tmp/exp230"


def build_bootstrap(shard_i: int, num_shards: int, limit: int | None) -> str:
    worker_b64 = base64.b64encode(WORKER.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit else ""
    return f"""
set -euo pipefail
echo "[exp230] host=$(hostname) shard={shard_i}/{num_shards} tpu={TPU_TYPE}"
mkdir -p {WORK_DIR}
echo {worker_b64} | base64 -d > {WORK_DIR}/gen_rollouts_worker.py

PY=python
# marinfold WITHOUT its dependency set: a plain install repins transformers out
# from under the image's vLLM. The contacts_v1 document generator needs only
# numpy + fsspec on top, both already present.
uv pip install --quiet --no-deps \
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold" \
  || "$PY" -m pip install --quiet --no-deps \
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold"
uv pip install --quiet gcsfs || "$PY" -m pip install --quiet gcsfs
"$PY" -c "from marinfold.document_structures.contacts_v1 import build_document; print('[exp230] marinfold OK')"

export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}
exec "$PY" {WORK_DIR}/gen_rollouts_worker.py \\
    --model {MODEL} \\
    --targets {TARGETS} \\
    --out {OUT} \\
    --shard {shard_i}/{num_shards} \\
    --n-rollouts {N_ROLLOUTS} \\
    --temperature {TEMPERATURE} \\
    --top-p {TOP_P} \\
    --top-k {TOP_K} \\
    --tensor-parallel-size {TENSOR_PARALLEL}{limit_arg}
""".strip()


def build_request(shard_i: int, num_shards: int, limit: int | None, suffix: str) -> JobRequest:
    # with_tpu leaves `regions` unset and the scheduler may then pick a region
    # with no v5p at all — and, worse, one where the data is not mirrored.
    # REGION, never zone: zone-pinning starved three separate #163 jobs.
    resources = ResourceConfig.with_tpu(
        TPU_TYPE, cpu=16, ram="64g", disk="64g", regions=REGIONS,
    )
    assert resources.regions == list(REGIONS) or resources.regions == REGIONS, (
        f"regions did not stick on ResourceConfig: {resources.regions!r}"
    )
    return JobRequest(
        name=f"{JOB_PREFIX}-s{shard_i}of{num_shards}{suffix}",
        entrypoint=Entrypoint.from_binary(
            "bash", ["-lc", build_bootstrap(shard_i, num_shards, limit)]),
        resources=resources,
        environment=create_environment(env_vars={}, setup_scripts=[]),
        replicas=1,
        priority=IRIS_PRIORITY_BAND_INTERACTIVE,
        processes_per_task=1,
        max_retries_failure=3,
        # Preemptible pool; the worker resumes from its own written parts.
        max_retries_preemption=100,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("EXP230_SHARDS", "8")))
    ap.add_argument("--shards", default=None, help="comma-separated subset to (re)submit")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N targets per shard")
    ap.add_argument("--name-suffix", default="", help="iris names are unique; a retry needs one")
    ap.add_argument("--cluster", default=os.environ.get("EXP230_CLUSTER", "marin"))
    a = ap.parse_args()

    which = ([int(x) for x in a.shards.split(",")] if a.shards else list(range(a.num_shards)))
    reqs = [build_request(i, a.num_shards, a.limit, a.name_suffix) for i in which]

    print(f"[exp230] {len(reqs)} job(s), {TPU_TYPE} interactive band, regions={REGIONS}\n"
          f"         n_rollouts={N_ROLLOUTS} T={TEMPERATURE} top_p={TOP_P} top_k={TOP_K} "
          f"tp={TENSOR_PARALLEL} limit={a.limit}\n"
          f"         model={MODEL}\n         targets={TARGETS}\n         out={OUT}")

    if os.environ.get("EXP230_DRY_RUN"):
        print("[exp230] DRY RUN — JobRequests built, not submitting.")
        r = reqs[0]
        print(f"  {r.name}: priority={r.priority} resources={r.resources}")
        print(r.entrypoint.binary_entrypoint.args[1])
        return

    from fray.iris_backend import FrayIrisClient
    from iris.client.client import get_iris_ctx

    if get_iris_ctx() is not None:
        from fray.current_client import current_client
        _submit(current_client(), reqs, must_wait=True)
        return

    from iris.cli.connect import open_iris_client

    print(f"[exp230] submitting from the workstation via the {a.cluster} controller tunnel")
    with open_iris_client(cluster_name=a.cluster, workspace=None) as iris_client:
        _submit(FrayIrisClient.from_iris_client(iris_client), reqs, must_wait=False)


def _submit(client, reqs, *, must_wait: bool) -> None:
    jobs = [client.submit(r) for r in reqs]
    print(f"[exp230] submitted {len(jobs)} job(s)", flush=True)
    for r in reqs:
        print(f"    {r.name}")
    if not must_wait and os.environ.get("EXP230_NO_WAIT", "1") == "1":
        print("[exp230] not waiting — these are root jobs and keep running")
        return
    # j.wait() RAISES on a failed job, abandoning every remaining wait and
    # reporting only the first failure. Catch per job.
    results = []
    for j in jobs:
        try:
            results.append(j.wait())
        except Exception as exc:  # noqa: BLE001 — report, don't abort
            results.append(f"{type(exc).__name__}: {exc}")
    bad = [(r.name, s) for r, s in zip(reqs, results) if s != JobStatus.SUCCEEDED]
    print(f"[exp230] finished: {len(results) - len(bad)}/{len(results)} succeeded")
    for name, status in bad:
        print(f"  FAILED {name}: {status}")


if __name__ == "__main__":
    main()
