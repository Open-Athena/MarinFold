# Exp277 rollout-v2 evaluation

This snapshot evaluates `contacts-v1-exp277-m2-p06-full-epoch-1.5B` at step
266,344 under the fixed exp82 rollout-and-resample recipe. It scores 670 units:
the legacy 554, 97 `eval-val` natural FoldBench monomers, and 19 `eval-denovo`
designs. It deliberately excludes `eval-test`.

The 5.89 GB float32 HF export already resides in `marin-us-east-02a`. The driver
verifies and reads it in place through CoreWeave's local object-store endpoint.
It first gates on one real protein, then dispatches twelve independent H100
workers at batch priority. Each unit gets 100 freshly resampled rollouts at
temperature 1.0, top-p 0.95, top-k disabled, and token budget `6L+128`.

The scorer is byte-identical to the PR #244 / exp232 validated worker
(`sha256 dd2f76dd5d34d1549e0be1197d9053d6e3aa0ef909f062ff5355d65f31cd571c`).
It records one timing row per evaluation unit. As in the exp82 reference worker,
rollouts that reach the token cap are excluded from contact voting and counted
explicitly. Finalization uses exp89's metric implementation, proves that usable
and unfinished counts cover every requested rollout, and validates all expected
units before reporting.

Run ID `v2-01` writes to:

```text
s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/
  evals/rollout-v2/2026-09-13/v2-01/
```

The batch-priority evaluation was submitted at 2026-09-13 18:08 UTC as
[`/bizon/exp277-eval-v2-01`](https://iris.oa.dev/#/job/%2Fbizon%2Fexp277-eval-v2-01).
The driver validated all 670 expected units and submitted its one-H100 smoke
gate. At launch the smoke task was queued by Kueue because the target cluster
reported no free H100 capacity.

That first driver timed out after six hours without the smoke H100 ever being
allocated; no inference ran. The recovery launcher allows 48 hours for batch
queueing while preserving the same evaluation settings and durable run ID. The
replacement [`/bizon/exp277-eval-v2-01-r01`](https://iris.oa.dev/#/job/%2Fbizon%2Fexp277-eval-v2-01-r01)
was submitted at 2026-09-14 13:19 UTC, revalidated all 670 inputs, and is waiting
at the one-H100 smoke gate with zero failures.

Launch from this directory with:

```bash
uv run --frozen python submit_coreweave.py --run-id v2-01 --suite exp277
```
