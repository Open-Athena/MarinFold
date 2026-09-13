# Exp277 rollout-v2 evaluation

This snapshot evaluates `contacts-v1-exp277-m2-p06-full-epoch-1.5B` at step
266,344 under the fixed exp82 rollout-and-resample recipe. It scores 670 units:
the legacy 554, 97 `eval-val` natural FoldBench monomers, and 19 `eval-denovo`
designs. It deliberately excludes `eval-test`.

The 5.89 GB float32 HF export already resides in `marin-us-east-02a`. The driver
therefore verifies and reads it in place on `cw-us-east-02a`; no weights move.
It first gates on one real protein, then dispatches twelve independent H100
workers at batch priority. Each unit gets 100 freshly resampled rollouts at
temperature 1.0, top-p 0.95, top-k disabled, and token budget `6L+128`.

The scorer is byte-identical to the PR #244 / exp232 validated worker
(`sha256 dd2f76dd5d34d1549e0be1197d9053d6e3aa0ef909f062ff5355d65f31cd571c`).
It records one timing row per evaluation unit and requires zero unfinished
rollouts. Finalization uses exp89's metric implementation and validates all
expected units before reporting.

Run ID `v2-01` writes to:

```text
s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/
  evals/rollout-v2/2026-09-13/v2-01/
```

Launch from this directory with:

```bash
uv run --frozen python submit_coreweave.py --run-id v2-01 --suite exp277
```
