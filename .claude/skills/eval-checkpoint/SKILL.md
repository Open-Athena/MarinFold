---
name: eval-checkpoint
description: >-
  Evaluate a MarinFold contacts-v1 checkpoint with the fixed exp89 contact
  benchmark. Use for checkpoint scoring, R-precision/AUC requests, comparisons
  with structure baselines, or reproducing contact metrics on local, CUDA, or
  Iris TPU execution.
---

# Evaluate a contacts-v1 checkpoint

Treat the exp89 evaluator, fixed ground truth, candidate-pair universe, and
metric implementation as the measurement specification. Infer environment-
specific commands from the checked-out revisions and current tooling.

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

## Evaluate

1. Fetch the published exp89 ground-truth universe; do not rebuild it during a
   normal checkpoint evaluation. Verify 554 `(dataset, stem)` units and 552
   unique stems. Require canonical baseline inputs when baseline comparison is
   requested.
2. Use exp89's pairwise scorer and `compute_metrics.py`; do not substitute
   another candidate universe or metric implementation. Make scoring resumable
   by `(dataset, stem)`.
3. Gate the full run on one real protein: load the actual weights and tokenizer,
   execute a forward pass, obtain the complete position-token log-probabilities,
   and write a valid score matrix.
4. On TPU vLLM, verify the TPU-targeted runtime and inspect checkpoint,
   parameter, and compute dtypes. Normalize floating weights intentionally
   before sharding; isolate any CPU PyTorch conversion from vLLM engine startup.

## Expected outputs

- `scores/<dataset>__<stem>.npz`: 554 `[L,L]` score matrices.
- Timing and provenance records: evaluated run/step, source and evaluated
  checkpoint paths, revisions, tokenizer, regions, runtime, topology, and job.
- `marinfold_precision.csv` and the unified `contact_precision_all.csv` (or
  equivalent wrapper outputs): per-protein precision at `L`, `L/2`, `L/5`, and
  `R`, plus AUC, for `all`, `short`, `medium`, and `long` ranges—20 rows per
  evaluation unit.
- A concise aggregate table led by all/long R-precision, with AUC and precision
  cuts, completeness counts, output paths, and the checkpoint's W&B train/val
  losses when requested. Record the source W&B metric keys.

Reject silent skips, wrong-shaped matrices, incomplete log-probability vectors,
or a 552-unit result caused by deduplicating on `stem`.

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
