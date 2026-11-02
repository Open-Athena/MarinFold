# exp343 rollout-v2 evaluation

Scores `contacts-v1-exp343-m2-p06-complex-1.5B` under the **fixed exp82
rollout-and-resample recipe**, on the 670-unit union: the legacy 554, 97
`eval-val` natural FoldBench monomers, and 19 `eval-denovo` designs. It
deliberately excludes `eval-test`.

This is exp277's snapshot with its identity changed and one checkpoint added.
`score_rollout_worker.py` is **byte-identical** to exp277's and exp232's:

```
sha256 dd2f76dd5d34d1549e0be1197d9053d6e3aa0ef909f062ff5355d65f31cd571c
```

That is the point of copying rather than rewriting. Running the same bytes on the
same units is what makes exp343's numbers comparable with exp277's *committed*
per-protein results, so the baseline is not re-scored — re-scoring it would draw
fresh rollouts and produce a second, slightly different exp277 number, leaving
two candidates for "the" baseline. exp277 compared against exp232 this way, and
exp169 against exp232 before that.

Each unit gets 100 freshly resampled rollouts at temperature 1.0, top-p 0.95,
top-k disabled, token budget `6L+128`. Rollouts that reach the token cap are
excluded from contact voting and counted explicitly. Finalization uses exp89's
metric implementation.

## Order of operations

The checkpoint spec is **generated, not written by hand**. The driver verifies
the bytes each worker downloads against the digests in `exp343_checkpoint.json`,
so a placeholder digest would not fail — it would silently turn that check into a
no-op. `checkpoint_specs.suite("exp343")` raises until the manifest exists.

The manifest lives **in this directory**, not under the experiment's `data/`:
`submit_coreweave.py` ships this directory flat as the job workspace, so anything
outside it never reaches the pod.

```bash
# after training completes, against the finished export.
# Reading CoreWeave storage from the workstation needs the credentials the pods
# get injected -- see the module docstring for the FSSPEC_S3 export.
uv run python make_checkpoint_spec.py --step <final step>

# then evaluate
uv run python submit_coreweave.py --run-id v2-01

# then compare, from the experiment root
cd ../.. && uv run python compare_eval.py
```

`--suite exp277` re-scores the baseline deliberately, if that is ever wanted.

## Outputs

```
s3://marin-us-east-02a/MarinFold/exp343_models_complex_corpus_training/
  evals/rollout-v2/2026-09-28/<run-id>/
```

Consolidated results are committed under the experiment's
`data/eval_rollout_v2/` and published to the public HF bucket. Per-input timings
are captured at evaluation time, per the root `AGENTS.md`.
