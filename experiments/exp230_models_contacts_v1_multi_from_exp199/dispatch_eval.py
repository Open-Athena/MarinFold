# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Run exp230's two gates on marin TPU, for a checkpoint and for the base.

Two readouts, deliberately different code paths:

* ``--what rprec`` -- **Gate A**.  Runs exp82's reference scorer
  (``_exp82_score_rollout_worker.py``, vendored verbatim) to produce the vote
  matrices exp89's ``compute_metrics.py`` scores.  **Always run the exp199 BASE
  in the same batch as the fine-tune.**  #209 measured a 0.023 gap between
  exp82's worker and #199's own pipeline on identical weights, so comparing a
  new number against the published 0.5873 would manufacture a regression that
  does not exist.  The paired comparison is only meaningful against a base
  scored through this same path.
* ``--what modes`` -- **Gate B** and the multi-mode report, via
  ``eval_modes_worker.py``, which prompts one checkpoint under both mode tokens.

Structure and every placement rule are ``dispatch_rollouts.py``'s; see that
module for why this is marin and not CoreWeave, and for the client-vs-workspace
split.

    python dispatch_eval.py --what rprec --model base=gs://.../exp199_bf16 \\
        --model ft=gs://.../hf/step-2000 --num-shards 8
    python dispatch_eval.py --what modes --model ft=gs://.../hf/step-2000 --num-shards 4
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

MARIN_CLIENT = Path(os.environ.get("MARIN_CLIENT", "/home/bizon/git/marin-freshiris"))
IRIS = os.environ.get("IRIS_BIN", str(MARIN_CLIENT / ".venv/bin/iris"))
SUBMIT_WORKSPACE = Path(os.environ.get("EXP230_WORKSPACE", "/data/exp230_multi/marin_vllm_ws"))

GCS_PREFIX = os.environ.get(
    "EXP230_GCS_PREFIX",
    "gs://marin-us-east5/protein-structure/MarinFold/exp230_contacts_v1_multi",
)
TARGETS = os.environ.get("EXP230_EVAL_TARGETS", f"{GCS_PREFIX}/eval554_targets.parquet")
OUT = os.environ.get("EXP230_EVAL_OUT", f"{GCS_PREFIX}/eval")

MARINFOLD_GIT = os.environ.get(
    "EXP230_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold",
)

#: exp82's settled recipe. 100 rollouts per protein, top-k DISABLED, budget 6L+128.
N_ROLLOUTS = int(os.environ.get("EXP230_EVAL_N_ROLLOUTS", "100"))
TOP_K, TOP_P, TEMPERATURE = -1, 0.95, 1.0
#: Free-generation sample for the mode readouts; #163 used 4 rollouts x 553.
MODE_ROLLOUTS = int(os.environ.get("EXP230_MODE_ROLLOUTS", "4"))
MAX_SECTIONS = int(os.environ.get("EXP230_MAX_SECTIONS", "8"))

WORKERS = {
    "rprec": Path(__file__).with_name("_exp82_score_rollout_worker.py"),
    "modes": Path(__file__).with_name("eval_modes_worker.py"),
}


def build_bootstrap(*, what: str, label: str, model: str, shard_i: int, num_shards: int,
                    mode: str | None, limit: int | None) -> str:
    worker_b64 = base64.b64encode(WORKERS[what].read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit else ""
    local = f"/tmp/exp230_eval/{what}.py"
    if what == "rprec":
        # --no-per-request-seed: the JAX backend rejects SamplingParams.seed
        # outright, and it does so AFTER a clean model load and full warmup, so
        # the discovery costs ~6 min a time. The engine-level seed still applies
        # and the 100 rollouts are independent draws either way.
        args = (f"--model {model} --targets {TARGETS} --out {OUT}/rprec --label {label} "
                f"--shard {shard_i}/{num_shards} --n-rollouts {N_ROLLOUTS} "
                f"--temperature {TEMPERATURE} --top-p {TOP_P} --top-k {TOP_K} "
                f"--no-per-request-seed{limit_arg}")
    else:
        args = (f"--model {model} --targets {TARGETS} --out {OUT}/modes/{label} "
                f"--mode {mode} --shard {shard_i}/{num_shards} "
                f"--n-rollouts {MODE_ROLLOUTS} --max-sections {MAX_SECTIONS} "
                f"--temperature {TEMPERATURE} --top-p {TOP_P} --top-k {TOP_K} "
                f"--tensor-parallel-size 4{limit_arg}")
    return f"""
set -euo pipefail
echo "[exp230-eval] host=$(hostname) what={what} label={label} shard={shard_i}/{num_shards}"
mkdir -p /tmp/exp230_eval
echo {worker_b64} | base64 -d > {local}

uv pip install --quiet --no-deps "{MARINFOLD_GIT}"
uv run --no-sync python -c \\
  "from marinfold.document_structures.contacts_v1 import build_document; print('[exp230-eval] marinfold OK')"

exec uv run --no-sync python {local} {args}
""".strip()


def submit(*, what: str, label: str, model: str, shard_i: int, num_shards: int,
           mode: str | None, limit: int | None, tpu: str, zone: str, mem: str,
           suffix: str, dry_run: bool) -> str:
    tag = f"{what}-{label.replace('_', '-')}" + (f"-{mode}" if mode else "")
    name = f"exp230-eval-{tag}-s{shard_i}of{num_shards}{suffix}"
    command = [
        IRIS, "--cluster=marin", "job", "run",
        "--job-name", name, "--no-wait", "--enable-extra-resources",
        "--priority", "interactive", "--zone", zone, "--tpu", tpu,
        "--extra", "vllm", "--extra", "tpu",
        # Sized to the HOST, not copied from another experiment: a v6e-4
        # advertises ~52GB and a 64GB ask sits pending forever.
        "--cpu", "16", "--memory", mem, "--disk", "64GB",
        "--max-retries", "3",
        "--", "bash", "-lc",
        build_bootstrap(what=what, label=label, model=model, shard_i=shard_i,
                        num_shards=num_shards, mode=mode, limit=limit),
    ]
    if dry_run:
        print(f"[exp230-eval] DRY RUN {name}")
        return name
    subprocess.run(command, cwd=SUBMIT_WORKSPACE, check=True)
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["rprec", "modes"], required=True)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=URI",
                    help="repeatable; for Gate A pass the BASE and the fine-tune together")
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--modes", default="plain,multi",
                    help="--what modes only; Gate B is the plain arm")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tpu", default=os.environ.get("EXP230_TPU_TYPE", "v6e-4"))
    ap.add_argument("--zone", default=os.environ.get("EXP230_ZONE", "us-east5-b"))
    ap.add_argument("--memory", default=os.environ.get("EXP230_MEMORY", "32GB"))
    ap.add_argument("--name-suffix", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    models = [m.split("=", 1) for m in a.model]
    which = [int(x) for x in a.shards.split(",")] if a.shards else list(range(a.num_shards))
    modes = [None] if a.what == "rprec" else [m for m in a.modes.split(",") if m]

    if a.what == "rprec" and len(models) == 1:
        print("[exp230-eval] WARNING: Gate A is a PAIRED comparison. Scoring one model "
              "alone invites comparing it to a published number measured by a different "
              "pipeline -- #209 showed that gap is 0.023 on these very weights.")

    print(f"[exp230-eval] what={a.what} models={[m[0] for m in models]} modes={modes} "
          f"shards={len(which)} on {a.tpu} in {a.zone}\n"
          f"              targets={TARGETS}\n              out={OUT}")
    names = [submit(what=a.what, label=label, model=uri, shard_i=i,
                    num_shards=a.num_shards, mode=mode, limit=a.limit, tpu=a.tpu,
                    zone=a.zone, mem=a.memory, suffix=a.name_suffix, dry_run=a.dry_run)
             for label, uri in models for mode in modes for i in which]
    print(f"[exp230-eval] submitted {len(names)} job(s)")
    for n in names:
        print(f"    {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
