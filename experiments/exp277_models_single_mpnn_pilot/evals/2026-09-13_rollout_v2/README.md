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

The completed batch-priority run is
[`/bizon/exp277-eval-v2-01-r04-rno`](https://iris.oa.dev/#/job/%2Fbizon%2Fexp277-eval-v2-01-r04-rno).
It ran on twelve independent RNO2A H100 workers after US-EAST-02A became fully
saturated, reading the existing 5.89 GB export from the same CoreWeave object
store through RNO2A's local LOTA endpoint. The smoke and all twelve production
workers succeeded; the complete run took 14 minutes 30 seconds. Of 67,000
requested rollouts, 66,999 terminated and contributed to voting. One capped
rollout affected one unit and was explicitly excluded and accounted for.

The resulting exp277 R-precision is 0.62009 / 0.57704 for all / long-range
contacts on legacy 554, 0.55375 / 0.53802 on eval-val, and 0.69582 / 0.67603 on
eval-denovo. Against the native-only decontaminated exp232 winner at step
363,000, the corresponding deltas are +0.01503 / +0.02202, +0.00204 / +0.00211,
and +0.08599 / +0.10375. The eval-val result is a tie under the predeclared
absolute-delta threshold of 0.005; legacy and de novo improve.

The consolidated output is under the run root above at `results/` and is also
published anonymously at
`hf://buckets/open-athena/MarinFold/data/exp277-models-single-mpnn-pilot/evals/rollout-v2/2026-09-13/v2-01/results/`.
The committed copies are under `data/eval_rollout_v2/`, including per-input
timings, the run manifest, aggregate tables, per-protein contact precision, and
the exact exp232 comparison.

Recovery history: the original US-EAST-02A driver timed out after six hours at
the one-H100 capacity gate. `r01` was cancelled when placement moved to RNO2A.
`r02` exposed the one capped rollout; `r03` verified the worker-side accounting
change but retained an overly strict driver smoke check. `r04` accepted a smoke
whose 99 usable plus one capped rollout covered all 100 requests, reused its
durable marker, and completed without a failed child.

Launch from this directory with:

```bash
uv run --frozen python submit_coreweave.py --run-id v2-01 --suite exp277
```
