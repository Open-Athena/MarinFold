---
name: eval-checkpoint
description: >-
  Evaluate a MarinFold contacts-v1 checkpoint with the fixed exp89 contact
  benchmark, scored with exp82's rollout+resample recipe. Use for checkpoint
  scoring, R-precision/AUC requests, comparisons with structure baselines, or
  reproducing contact metrics on local, CUDA, or Iris TPU execution.
---

# Evaluate a contacts-v1 checkpoint

Treat the [exp89 evaluator](https://github.com/Open-Athena/MarinFold/issues/89),
fixed ground truth, candidate-pair universe, and metric implementation as the
measurement specification. Score it with the **rollout + resample** recipe
[exp82](https://github.com/Open-Athena/MarinFold/issues/82) settled on — never
the older pairwise readout. The two are not interchangeable: identical weights
score ~0.086 higher in R-precision under rollout, comparable to two generations
of model progress, so a number filed under the wrong recipe reads as a jump that
never happened. Infer environment-specific commands from the checked-out
revisions and current tooling.

## Establish identity and locality

1. Resolve the W&B run, exact step, checkpoint format, tokenizer/vocabulary,
   storage location, and region. Reuse a complete sibling HF export when one
   exists; otherwise convert Levanter weights and co-locate the tokenizer.
2. Match compute and output storage to checkpoint locality. Alternatively,
   propose a one-time mirror/export to durable HF storage when repeated access
   justifies the egress.
3. **Stop before transfer or submission.** Present the locality-matched and
   one-time-mirror options, including material transfer/cost implications, and
   require the user to choose. Follow repository approval rules for large
   cross-region copies.

## Choose an execution host

Use either approach without changing evaluation semantics:

- **MarinFold-native:** run the evaluator here with pinned published Marin/Iris
  packages and only the compatible MarinFold import surface.
- **Marin-native:** use Marin's workspace runtime, cluster configuration, and
  extras; include the MarinFold evaluator and inputs at a pinned revision.

Prefer the approach with checkpoint-local compute and fewer unpinned
dependencies. Do not install the full MarinFold dependency set into a TPU vLLM
environment when a smaller source/package surface avoids version conflicts.

Size the host for sampling, not for a single forward pass: 100 rollouts per
protein is ~150x the compute of the old pairwise readout, about 80 minutes per
checkpoint on one A5000. Sharded fan-out is the fast path — 12 single-H100
CoreWeave shards at batch priority cover all 554 proteins in ~4 minutes.

## Evaluate

1. Fetch the published exp89 ground-truth universe; do not rebuild it during a
   normal checkpoint evaluation. Verify 554 `(dataset, stem)` units and 552
   unique stems. Require canonical baseline inputs when baseline comparison is
   requested.
2. Score with exp82's rollout+resample workers
   (`score_rollout_vllm.py` for one local GPU;
   `dispatch_rollout_eval_cw.py` + `score_rollout_worker.py` +
   `fetch_cw_scores.py` for sharded fan-out) and exp89's `compute_metrics.py`.
   Do not substitute the pairwise readout, another candidate universe, or
   another metric implementation. Make scoring resumable by `(dataset, stem)`.
   The recipe is fixed:
   - 100 rollouts per protein, each sampled from a **fresh** document
     realization — resampled N-terminus and `<pX> <AA>` statement order. The
     resampling is the cheap half of the recipe; all realizations share a
     prefix length, so they batch.
   - Temperature `1.0`, top-p `0.95`, and **top-k disabled** (`-1` in vLLM, `0`
     in HF `generate`). Top-k is the trap: HF's default of 50 rides in from an
     export's `config.json` when no `generation_config.json` exists, inflates
     `<end>`, and costs ~0.011 R-precision and ~0.020 AUC.
   - Token budget `6L+128`. The older `4L+64` truncates the longer documents
     untruncated sampling produces.
   - Rank pairs by occurrence frequency across the rollouts, voting only the
     contacts still live at the end of each rollout. No pairwise tie-break: it
     moves R-precision by 0.0007 and costs a second inference pass.
   - Write the votes as a symmetric `[L,L]` matrix in input-sequence
     coordinates so `compute_metrics.py` scores it unchanged.
   This needs a **sampling** backend (vLLM or transformers, not MLX) and
   `marinfold`'s contacts-v1 document builder for the resampled realizations —
   install it `--no-deps` into a vLLM image so the image's transformers pin
   survives.
3. Gate the full run on one real protein: load the actual weights and tokenizer,
   sample its full rollout set, parse every completion back to contacts, confirm
   none hit the token cap, and write a valid vote matrix.
4. On TPU vLLM, derive the expected parameter dtype from the model/runtime
   configuration and inspect the tensors in every safetensor shard. If they
   differ—for example, `float32` exported weights with `bfloat16` TPU
   parameters—rewrite
   only floating tensors to the expected dtype before vLLM loads and shards
   them. Preserve integer/bool tensors, names, shapes, shard indexes, config,
   and tokenizer. Run the conversion in a short-lived CPU process, then start
   vLLM fresh so it cannot inherit PyTorch/OpenMP state. Record the source and
   effective dtypes.

## Expected outputs

- `scores/<dataset>__<stem>.npz`: 554 `[L,L]` vote matrices.
- Timing and provenance records: evaluated run/step, source and evaluated
  checkpoint paths, revisions, tokenizer, regions, runtime, topology, and job.
  Record the sampling recipe with them — rollout count, temperature, top-p,
  top-k, token budget, seed — and the unfinished-rollout count, since these
  determine the number as much as the weights do.
- `marinfold_precision.csv` and the unified `contact_precision_all.csv` (or
  equivalent wrapper outputs): per-protein precision at `L`, `L/2`, `L/5`, and
  `R`, plus AUC, for `all`, `short`, `medium`, and `long` ranges—20 rows per
  evaluation unit.
- A concise aggregate table led by all/long R-precision, with AUC and precision
  cuts, completeness counts, output paths, and the checkpoint's W&B train/val
  losses when requested. Record the source W&B metric keys.

## Validate against the E8 reference

Before trusting a new execution path, reproduce the
[exp75](https://github.com/Open-Athena/MarinFold/issues/75) E8 checkpoint
`prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084` at step 35679. Choose the
artifact that best fits the runtime and data locality:

- HF format: [HF mirror](https://huggingface.co/open-athena/marinfold-exp75/tree/main/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/hf/step-35679)
  or `gs://marin-us-east5/checkpoints/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/hf/step-35679/`.
- Levanter format: [HF mirror](https://huggingface.co/open-athena/marinfold-exp75/tree/main/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/checkpoints/step-35679)
  or `gs://marin-us-east5/checkpoints/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/checkpoints/step-35679/`.

With the recipe above and exp89 metrics over all 554 proteins, exp82 reports
R-precision `0.4245` (all) / `0.3656` (long) and AUC `0.9010` (all) / `0.8738`
(long) — row `marinfold-cv1-exp75-rollout` in
`experiments/exp82_evals_contacts_v1_contact_prediction/data/where_we_stand_summary.csv`.

Rollout scoring is stochastic, so match to a tolerance rather than to the digit.
Two independent stacks of this same E8 pass (12 CoreWeave H100s on vLLM 0.9.2,
one workstation A5000 on vLLM 0.11.0) agreed within 0.0015 R-precision and
0.0002 AUC, and [#204](https://github.com/Open-Athena/MarinFold/issues/204)'s
four evaluations of one unchanged checkpoint span 0.0023. Investigate a gap
above ~0.005 before evaluating a new checkpoint.

Reproducing `0.339` (all) / `0.269` (long) with AUC `0.881` instead means the
pairwise readout ran: those are the
[PR #93](https://github.com/Open-Athena/MarinFold/pull/93#issue-4738130859)
numbers for the same checkpoint under the superseded recipe. Fix the scorer, do
not report them.

## Validate completeness

- Account for every expected `(dataset, stem)` unit and report skips or failures
  explicitly. Do not deduplicate on `stem` alone.
- Check that each vote matrix matches its protein length, that every protein got
  its full complement of rollouts, and that no rollout hit the token cap. At
  `6L+128` exp82 saw 0/55,400 unfinished on this eval set; a nonzero count means
  the budget or the sampling knobs are wrong and the scores are truncated.
- Check that metric outputs cover the evaluator's expected ranges and cuts for
  every scored unit, and report valid-value counts where a metric may be
  undefined.

Stop before reporting if any invariant fails. Name every skipped or invalid
unit and preserve partial outputs for diagnosis.

## Debugging ladder

Escalate only as far as needed:

1. Submit the normal Iris job and inspect controller/worker logs.
2. Submit directly to a TPU slice with Iris `--tpu <slice>` when controller
   placement or orchestration obscures the failure.
3. Use `iris task exec` to inspect the live worker without relying on SSH.
4. Use Marin's `scripts/iris/dev_tpu.py` for an interactive slice and SSH when
   worker-level inspection is still necessary; treat SSH/OS Login failures as
   separate from the evaluation runtime.

If no clear supported TPU/Iris path remains—or checkpoint identity, evaluation
inputs, transfer scope, or metric validity is ambiguous—stop and ask the user.
