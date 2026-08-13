# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Fan ``gen_rollouts_worker.py`` out over marin v5p-8 TPUs - exp230's on-policy
rollout generation.

**Why marin TPU and not CoreWeave.** The standing rule sends GPU work to a
CoreWeave H100 cluster at batch priority, and exp82's dispatcher does exactly
that. It is not available here: the workstation's CoreWeave object-storage key
(``~/.config/marin/cw-rno2a.env``) is **revoked** - every request, against every
bucket and anonymously, returns *"The access key ID you provided does not exist
in our records"*. The iris controller is healthy, so jobs could still be
submitted, but neither the model nor the targets could be staged into CW S3 from
here and no result could be read back. marin's GCS works, v5p-8 capacity is
ready, and it is the cluster #163 actually ran this pipeline on after
CoreWeave's batch band starved it for three days. Restore the key and the
CoreWeave path is exp82's dispatcher with a different ``--targets``.

Structure is exp169's ``dispatch_eval_tpu.py``, which is the validated TPU
shape, with three things it teaches carried over verbatim:

* **Submit from the marin checkout.** ``iris job run`` bundles the CWD and runs
  ``uv sync`` in it on the pod; ``--extra vllm --extra tpu`` are *marin's*
  extras, so the workspace has to BE the marin checkout. An empty scratch dir
  fails on the pod with ``No `pyproject.toml` found``; ``/tmp`` hangs uploading.
  It must also be the **fresh** checkout - iris rejects a client over 14 days
  old.
* **Interactive band, not batch.** The v5p pool is fully subscribed by other
  people's interactive jobs, so a batch-band job yields indefinitely and
  registers no autoscaler demand. This does not contradict the CoreWeave
  always-batch rule, which is about that cluster.
* **``--disk 64GB``.** A v5p-8 VM has 100 GiB of ephemeral disk; asking for more
  is rejected at submit as unschedulable rather than queued.

``marinfold`` is installed on the pod with ``--no-deps`` so it cannot repin
marin's vLLM/transformers; the contacts-v1 generator needs only numpy + fsspec
on top, both already there.

    uv run python dispatch_rollouts.py --num-shards 8
    uv run python dispatch_rollouts.py --num-shards 8 --shards 0 --limit 20  # smoke
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

MARIN = Path(os.environ.get("MARIN_CHECKOUT", "/home/bizon/git/marin-freshiris"))
IRIS = os.environ.get("IRIS_BIN", str(MARIN / ".venv/bin/iris"))
SUBMIT_WORKSPACE = Path(os.environ.get("EXP230_WORKSPACE", str(MARIN)))

GCS_PREFIX = os.environ.get(
    "EXP230_GCS_PREFIX",
    "gs://marin-us-central1/protein-structure/MarinFold/exp230_contacts_v1_multi",
)
MODEL = os.environ.get("EXP230_MODEL", f"{GCS_PREFIX}/model/exp199_bf16")
TARGETS = os.environ.get("EXP230_TARGETS", f"{GCS_PREFIX}/targets.parquet")
OUT = os.environ.get("EXP230_OUT", f"{GCS_PREFIX}/rollouts")

MARINFOLD_GIT = os.environ.get(
    "EXP230_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold",
)

# exp82/exp142's settled sampling recipe. top_k DISABLED: 50 is the HF default
# that rides in from an export's config.json and suppresses contacts (0.67x GT
# instead of 0.96x).
N_ROLLOUTS = int(os.environ.get("EXP230_N_ROLLOUTS", "24"))
TOP_K = int(os.environ.get("EXP230_TOP_K", "-1"))
TOP_P = float(os.environ.get("EXP230_TOP_P", "0.95"))
TEMPERATURE = float(os.environ.get("EXP230_TEMPERATURE", "1.0"))
TENSOR_PARALLEL = int(os.environ.get("EXP230_TP", "4"))  # v5p-8 is 4 chips

WORKER_SCRIPT = Path(__file__).with_name("gen_rollouts_worker.py")
WORKER_LOCAL = "/tmp/exp230/gen_rollouts_worker.py"


def build_bootstrap(*, shard_i: int, num_shards: int, limit: int | None) -> str:
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit else ""
    return f"""
set -euo pipefail
echo "[exp230] host=$(hostname) shard={shard_i}/{num_shards}"

mkdir -p /tmp/exp230
echo {worker_b64} | base64 -d > {WORKER_LOCAL}

# marin's synced venv already has vLLM (its TPU fork), torch, transformers,
# pyarrow, fsspec and gcsfs. marinfold goes in --no-deps so it cannot repin any
# of that.
uv pip install --quiet --no-deps "{MARINFOLD_GIT}"
uv run --no-sync python -c \\
  "from marinfold.document_structures.contacts_v1 import build_document; print('[exp230] marinfold OK')"

exec uv run --no-sync python {WORKER_LOCAL} \\
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


def submit(*, shard_i: int, num_shards: int, limit: int | None, tpu: str, zone: str,
           priority: str, suffix: str, dry_run: bool) -> str:
    name = f"exp230-rollouts-s{shard_i}of{num_shards}{suffix}"
    command = [
        IRIS, "--cluster=marin", "job", "run",
        "--job-name", name, "--no-wait", "--enable-extra-resources",
        "--priority", priority, "--zone", zone, "--tpu", tpu,
        "--extra", "vllm", "--extra", "tpu",
        "--cpu", "16", "--memory", "64GB", "--disk", "64GB",
        "--max-retries", "3",
        "--", "bash", "-lc",
        build_bootstrap(shard_i=shard_i, num_shards=num_shards, limit=limit),
    ]
    if dry_run:
        print(f"[exp230] DRY RUN {name}\n{command[-1][:900]}\n...")
        return name
    subprocess.run(command, cwd=SUBMIT_WORKSPACE, check=True)
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--shards", default=None, help="comma-separated subset to (re)submit")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N targets per shard")
    ap.add_argument("--tpu", default=os.environ.get("EXP230_TPU_TYPE", "v5p-8"))
    # Zone, not just region, because the model and targets live only in
    # us-central1 and a cross-region read of a 2.7 GiB checkpoint per pod is the
    # dominant fixed cost. #163 warns that zone-pinning can starve a job; the
    # mitigation is to check `iris cluster status` for ready capacity in this
    # zone before submitting, not to widen the pin and leave the data behind.
    ap.add_argument("--zone", default=os.environ.get("EXP230_ZONE", "us-central1-a"))
    ap.add_argument("--priority", default="interactive")
    ap.add_argument("--name-suffix", default="", help="iris names are unique; a retry needs one")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    which = [int(x) for x in a.shards.split(",")] if a.shards else list(range(a.num_shards))
    print(f"[exp230] {len(which)} shard(s) on {a.tpu} in {a.zone}, {a.priority} band\n"
          f"         n_rollouts={N_ROLLOUTS} T={TEMPERATURE} top_p={TOP_P} top_k={TOP_K} "
          f"tp={TENSOR_PARALLEL} limit={a.limit}\n"
          f"         model={MODEL}\n         targets={TARGETS}\n         out={OUT}")
    names = [submit(shard_i=i, num_shards=a.num_shards, limit=a.limit, tpu=a.tpu,
                    zone=a.zone, priority=a.priority, suffix=a.name_suffix,
                    dry_run=a.dry_run) for i in which]
    print("[exp230] submitted:")
    for n in names:
        print(f"    {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
