# exp105 run notes — cluster/env debugging (2026-07-08)

Chronological notes from taking exp105 from "code merged" to "running on Iris",
so the next person doesn't re-derive the infra problems. The generation logic
(`cli.py`/`generate_rows.py`) was already correct and merged; everything below
is about getting it to actually run against the *current* marin Iris cluster.

## What was wrong and what fixed it

### 1. marin-zephyr branch pin was dead
`pyproject.toml` pinned `marin-zephyr` to the git branch
`alxmrs/stamp-iris-build-date`. That branch is gone (marin PR #5990 closed
unmerged, branch deleted). → had to change the marin source.

### 2. Client-freshness gate rejects stale submitters
The live controller rejects **root** `LaunchJob` submissions whose `marin-iris`
client build is older than **14 days** (`_check_client_freshness`,
`iris/cluster/controller/service.py`, marin PR #5108). The
`marin-*-latest` GitHub release assets are **frozen at 2026-05-29**, so a client
built from them is rejected: `marin-iris client is too old (build 2026-05-29;
minimum 2026-06-24)`. The gate only fires on `job_id.is_root`; child/worker
tasks are exempt.

### 3. Stale worker env can't register with the control plane (the real stall)
This was the one that ate the most time. With the frozen 2026-05-29 wheels in
the **worker** env, worker tasks get *scheduled* (controller marks them RUNNING,
placed on VMs) but never *register* with the zephyr coordinator: finelog's
`register_table` 404s against the upgraded controller
(`finelog.errors.StatsError: Not Found`). Symptom: `zephyr.execution ... 0/N
complete, 2 in-flight, N-2 queued, 2/2 workers alive` and **0 output shards**,
no matter how quiet the cluster is. It looks exactly like a capacity stall but
is not — I confirmed by waiting until the cluster dropped from 1324→311 running
tasks and it still produced 0 shards with 2 live workers.

**Fix for #1–#3: install marin from PyPI, not the GitHub `-latest` assets.**
marin publishes the whole stack to PyPI on ~every push:
`marin-iris`/`zephyr`/`fray`/`rigging` `0.2.40.dev202607080801` (= 2026-07-08,
matches the controller), plus the native `marin-finelog-server` wheel. So
`pyproject.toml` now depends on `marin-zephyr>=0.2.40.dev202607080801` +
`marin-iris>=…` on the default index (no `find-links`). Consequences:
- `uv run iris` from this env is fresh → submits directly (no editable
  `/home/bizon/git/marin` checkout needed — that was an interim workaround).
- Workers built from it register fine (finelog 404s gone: smoke on the PyPI env
  had **0** "Failed to report task status text").
- This is how "run from the marin workspace" (option 1) was realised — same
  current stack, via PyPI wheels instead of a workspace restructure.

### 4. No preemptible CPU pool
`cli.py` originally defaulted to `--preemptible` (exp53's advice). The marin
cluster has **no preemptible CPU scale group** — only
`cpu_vm_e2_highmem_2_ondemand-<region>`. A preemptible request registers zero
autoscaler demand and strands the job. `cli.py` now defaults to
`--no-preemptible`, and `--region` is repeatable so the on-demand pool can span
US regions (us-central1/central2/east1/east5/west1/west4) without exp53's
trans-Atlantic spill.

### 5. Tokenizer goes in the HF bucket, not a model repo
`open-athena/…-tokenizer` model-repo creation 403s, and the corpus-tokenizer
convention is "co-locate in the bucket next to the data" (exp53). So the
tokenizer is built with `--save-local ./tokenizer` and published under
`…/contacts_and_coordinates_v1/tokenizer/` in step 3.

## Remaining constraint: CPU capacity (not a bug)
Even with #1–#4 fixed and the cluster quiet, the CPU pool is small
(~11 ready `e2-highmem-2` VMs, autoscaler `peak_demand`~6, `Demand` stays 0 for
CPU fan-out). A job gets only a handful of workers. That's fine for the small
`val`/`test` splits (22 shards each) but makes the 2067-shard `train` split slow
(~1.1 s/doc/core; ~1,200 core-hours total). **AFDB `gcs_uri` is requester-pays
and the workstation user lacks billing rights, so `train` cannot be generated
locally — it must run on-cluster.** If train needs to finish fast, an admin has
to raise the CPU autoscaler capacity; otherwise it's a multi-day grind that the
per-shard `{shard}` output + `rerun_missing`-style resume makes tractable.

## How to run (current, verified)
See README "Running it". In short, from this directory:
```
uv sync --extra test
uv run iris --cluster=marin job run --cpu 1 --memory 2GB -- \
  python cli.py generate --input <manifest glob> --out <…-{shard:05d}-of-{total:05d}.parquet> \
    --worker-cpu 1 --worker-memory 4g --max-workers 1024 --fetch-concurrency 32 \
    --no-preemptible --region us-central1 --region us-central2 --region us-east1 \
    --region us-east5 --region us-west1 --region us-west4
```

## Smoke results (validation, 2026-07-08)
- Frozen-env smoke: SUCCEEDED, 100/100 docs, 0 drops, `num_tokens`~32,766,
  `sha1==sha1(document)` — generation is byte-correct; but finelog telemetry
  404'd (the #3 symptom, benign at 1 worker).
- PyPI-env smoke (`ccoord-v1-smoke3`): SUCCEEDED, 100/100 docs, **0 finelog
  errors**. `val` then ramped to multiple workers and processed with 0 finelog
  errors — the #3 fix confirmed at multi-worker scale.
