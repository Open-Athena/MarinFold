# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the exp200 RL training pool on an iris CPU pod — issue #200.

Runs cloud-side because the exp53 corpus is ~2,000 shards in us-east5 and the
workstation uplink is ~2.5 MB/s. us-east5 for the same reason: this reads a lot
of shards, so the compute goes to the data.

``marinfold`` is installed ``--no-deps`` at bootstrap rather than being a locked
dependency (exp169's pattern). It needs only fsspec/numpy/pyarrow on top of what
is already here, and keeping it out of the lock means it cannot repin the TPU
stack — marinfold still declares ``transformers<5`` while the vLLM fork wants
5.12.1, and reconciling that in the lock would be a fight with nothing to gain.

Usage::

    uv run python dispatch_prep.py --n 10000 -k 16
    uv run python dispatch_prep.py --dry-run
"""

import argparse

from _submit import check_clean, submit

EXP163_EVAL = "gs://marin-us-east5/MarinFold/exp163/eval554/targets.parquet"
OUT_PREFIX = "gs://marin-us-east5/protein-structure/MarinFold/exp200/train"
MARINFOLD_GIT = "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="proteins in the pool")
    ap.add_argument("-k", "--realizations", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--min-contacts", type=int, default=5)
    ap.add_argument("--pool-mult", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out-prefix", default=OUT_PREFIX)
    ap.add_argument("--region", default="us-east5", help="co-located with the exp53 corpus")
    ap.add_argument("--job-name", default="exp200-prep-pool")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not a.dry_run:
        check_clean()

    out = a.out_prefix.rstrip("/")
    bootstrap = f"""
set -euo pipefail
echo "[prep] host=$(hostname)"
uv pip install --quiet --no-deps "{MARINFOLD_GIT}"
uv run --no-sync python -c "from marinfold.document_structures.contacts_v1 import build_document; print('[prep] marinfold OK')"
exec uv run --no-sync python prep_prompt_pool.py \\
    --n {a.n} -k {a.realizations} --max-len {a.max_len} \\
    --min-contacts {a.min_contacts} --pool-mult {a.pool_mult} --workers {a.workers} \\
    --eval-targets {EXP163_EVAL} \\
    --out-targets {out}/targets.parquet \\
    --out-prompts {out}/prompts
""".strip()

    submit(
        job_name=a.job_name,
        extras=("cpu",),
        cpu=16,
        memory="64GB",
        disk="64GB",
        region=a.region,
        # Reading ~2,000 shards and writing ~10k objects is not a minutes-long
        # interactive shape, but the CPU pool is not contended the way v5p is.
        priority="interactive",
        command=["bash", "-lc", bootstrap],
        raw=True,
        dry_run=a.dry_run,
    )
    print(f"  targets: {a.out_prefix.rstrip('/')}/targets.parquet")
    print(f"  prompts: {a.out_prefix.rstrip('/')}/prompts/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
