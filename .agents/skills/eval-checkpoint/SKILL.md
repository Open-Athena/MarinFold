---
name: eval-checkpoint
description: >-
  Evaluate a MarinFold contacts-v1 checkpoint on the FoldBench monomer eval sets
  (eval-val for routine work, eval-test as a rarely-read held-out set,
  eval-denovo for designs; exp245) or the legacy exp89 554-protein benchmark,
  scored with exp82's rollout+resample recipe. Use for checkpoint
  scoring, R-precision/AUC requests, comparisons with structure baselines,
  held-out or decontaminated-accuracy claims, designed-vs-natural or viral
  splits, sequence-KNN nulls, or reproducing contact metrics on local, CUDA,
  CoreWeave GPU or Iris TPU execution.
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

## Which eval set — eval-val by default, eval-test only when asked

The 333-unit universe is [exp245](https://github.com/Open-Athena/MarinFold/issues/245)'s
cut of FoldBench's 334 monomers into three sets with **different read budgets**,
and that distinction is the point of the split:

| set | what it is | n | read budget |
|---|---|---:|---|
| **eval-val** | the natural monomers inside the historical FoldBench-100 | 97 | **free.** The working set: checkpoint selection, sweeps, mid-training curves, any routine comparison |
| **eval-test** | every natural FoldBench monomer outside the historical 100 | **217** | **rare and recorded.** A held-out confirmation set. Score it only when the user asks for it or a result is being published, never for selection |
| **eval-denovo** | every de novo designed FoldBench monomer | 19 | free; a sanity check, not a designed-protein benchmark — FoldBench has no more designed monomers, and the bigger exp65 design set is inside the baselines' training data |

**Default: score eval-val + eval-denovo (116 units) and, when the checkpoint needs
placing against earlier generations, the legacy 554.** Leave eval-test out unless
the request is explicitly about held-out or generalisation performance, or the work
is being written up.

**When you do score eval-test, append a row to
`experiments/exp245_evals_foldbench_held_out_monomers/data/eval_test_reads.md`** —
date, checkpoints, why it warranted a read, and the numbers. A held-out set stops
being held out once decisions are fitted to it, and that ledger is the only thing
tracking how much of it has been spent.

**eval-val is a trustworthy stand-in.** #245 scored both sets once and found every
predictor within 0.03 of the same number on them (MarinFold +0.018 to +0.024 in
eval-test's favour, all intervals covering zero), with no extra val→test drop for
the contaminated reference model. That result is what licenses iterating on
eval-val, so do not "check against test" out of caution — it buys nothing and
spends the set.

**The legacy 554** (`gt_universe.jsonl`, unchanged) is where every published
MarinFold number lives, so score it whenever a checkpoint has to be placed against
earlier generations. It is free to look at as often as you like — it has been
selected on for a year already, which is precisely why it cannot answer a
generalisation question. The two universes overlap in 100 proteins but are separate
files; #245 ran 333 units × 3 checkpoints on 12 single-H100 CoreWeave shards in
about four minutes per checkpoint, so cost is never the reason to skip a set —
read budget is.

Everything is public on the bucket (anonymous read):

```
hf://buckets/open-athena/MarinFold/data/contacts-v1-foldbench-monomers-exp245/
    gt_universe_scored.jsonl       # all 333 units; filter to the sets you may read
    eval_targets_foldbench_monomers.parquet   # dataset, stem, L, input_seq
    eval_sets.csv                  # all 334 with eval_set / designed / is_viral /
                                   # kingdom / scorable / exclusion_reason /
                                   # pre-decontamination training identity
    eval_sets.fasta
    per_protein.csv.gz             # 9 predictors x 333 x {all,long} x {R,AUC}
    headline.csv  paired_deltas.csv  val_vs_test.csv  viral_split.csv
    decontamination_check.json  residual_identity.csv  context_budget.csv
hf://buckets/open-athena/MarinFold/data/contacts-v1-model-eval-exp89/
    gt_universe.jsonl              # the legacy 554 units, unchanged
hf://buckets/open-athena/MarinFold/data/contacts-v1-eval2-exp226/
    gt_universe_eval2.jsonl        # the 577-unit superset; eval2 is a column on
    eval2_manifest.csv             # eval2_manifest.csv (307 rows). Superseded by
                                   # eval-test for natural-protein claims.
```

In-repo: `experiments/exp245_evals_foldbench_held_out_monomers/data/`.

**Reporting rules — these change the conclusion, not just the presentation:**

- **Lead with a natural-protein number — eval-val routinely, eval-test when the
  question is generalisation — and never a pooled one.** The legacy 554 is 75 % de
  novo designed and eval2 is 77 %; a pooled mean over either mostly reports how well
  a model folds idealised backbones. Protenix-v2 single-seq scores **0.835** on
  designs and **0.265** on natural monomers — that spread is what pooling hides.
- **Any baseline comparison must use proteins that postdate the baselines' training
  cutoffs.** Decontamination has two sides and we control one. exp65's 396 de novo
  designs (in the legacy 554) look like the designed-protein benchmark — 20×
  eval-denovo, already scored — but **50.5 % were deposited on or before
  Protenix-v2's 2021-09-30 cutoff** and 43 % predate 2020-05, so a
  MarinFold-versus-baseline number there is contaminated *for the baselines*. Use it
  only to compare our own checkpoints to each other, and say so. The FoldBench sets
  satisfy the rule by construction: 0 of eval-test's 218 and 1 of eval-denovo's 19
  predate that cutoff
  (`experiments/exp245_evals_foldbench_held_out_monomers/data/baseline_cutoff_exposure.csv`).
  This is also why eval-denovo stays at 19 — FoldBench holds only 43 designed
  entries across all seven of its tasks — and why it is a sanity check rather than a
  designed-protein benchmark.
- **Designs are much easier than natural proteins you have no homolog for, and
  about as easy as natural proteins in general.** On eval-test's 23 proteins under
  40 % identity to training, designs beat natural by +0.177 [+0.044, +0.306]; over
  all 217 natural monomers the gap is +0.054. Say which comparison you mean.
- **Split viral vs non-viral** (`is_viral` on `eval_sets.csv`). The penalty tracks
  homology dependence: seq-KNN −0.351, ESMFold2 −0.170, MarinFold −0.076 to
  −0.123, Protenix-v2 + MSA −0.045, Protenix-v2 single-seq −0.002. Only 19 of 334
  monomers are viral, so report it as indicative.
- **Quote a sequence-KNN null beside the score, over the corpus the checkpoint
  actually trained on.** On eval-test it is **0.582** out of the un-decontaminated
  AFDB corpus and **0.426** out of #225's decontaminated one. A checkpoint that
  does not clear the null over its own corpus has not demonstrated anything;
  `run_knn_baseline.py` in exp245 rebuilds either null from #94's index.
- **Baselines for all 333 units already exist** in `per_protein.csv.gz` (Protenix-v2
  single-seq and +MSA, ESMFold, ESMFold2, both KNN nulls). Do not re-run them.
- **Differences under ~0.005 are ties** (#204: four evaluations of one unchanged
  checkpoint span 0.0023).
- **`8uxt_A` is excluded** from the 333 and flagged in `eval_sets.csv`: its
  contacts-v1 document truncates at the 8,192-token context, so no rollout can
  produce it in full. Do not silently re-add it.

**Reference values under the rollout recipe**, for sanity-checking a new path
(all-range R-precision):

| checkpoint | eval-val (97) | eval-test (217) | eval-denovo (19) | legacy 554 |
|---|---:|---:|---:|---:|
| `contacts-v1-exp199-cooldown-1.5B` (default; contaminated data) | 0.589 | 0.613 | 0.619 | 0.631 |
| #232 `m2-p06` (decontaminated data) | 0.520 | 0.538 | 0.591 | 0.592 |
| #232 `m1-p02` (decontaminated data) | 0.473 | 0.493 | 0.588 | 0.579 |

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
CoreWeave shards at batch priority cover the 333-unit universe in ~4 minutes per
checkpoint, so three checkpoints in one driver job is routine. exp245's
`rollout/` directory is the current worked example: it verifies each checkpoint in
place against a pinned file manifest, mirrors the public eval inputs into
CoreWeave S3 by URL + digest, runs a one-protein smoke job per checkpoint before
the fan-out, and exports its results to the public bucket from inside the cluster
(the results prefix is not readable from a workstation).

## Evaluate

1. Fetch the published ground-truth universe; do not rebuild it during a normal
   checkpoint evaluation. `gt_universe_scored.jsonl` carries all **333
   `(dataset, stem)` units** under the `foldbench_monomer` label, 333 unique
   stems — filter the *targets* to the sets this run is allowed to read (join
   `eval_sets.csv` on `stem`) rather than scoring eval-test by accident, and verify
   the unit count of whatever subset you submit; 554 units / 552 unique stems for the legacy set;
   577 / 575 for `gt_universe_eval2.jsonl` if that older superset is being scored
   deliberately. Require canonical baseline inputs when baseline comparison is
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

- `scores/<dataset>__<stem>.npz`: one `[L,L]` vote matrix per unit — 333 for the
  default universe, 554 for the legacy set. The sharded worker emits sparse
  parquet parts instead, so `fetch_cw_scores.py --parts <dir> --expect <n>` has
  to run before `build_rollout_rows.py`, which reads npz.
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
  losses when requested. Record the source W&B metric keys. When the universe is
  the 577, report **legacy 554, eval2 pooled (307) and eval2-natural (78)** as
  separate rows — they can rank checkpoints differently, and the pooled eval2
  row is the least informative of the three.

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

**This gate is on the legacy 554 subset**, which is why the default universe
keeps it: run the 577 and restrict to the 554 units to check the gate, rather
than scoring a second time.

## Validate completeness

- Account for every expected `(dataset, stem)` unit and report skips or failures
  explicitly. **Do not deduplicate on `stem` alone** in the legacy or eval2
  universes — they have fewer unique stems than units (554/552, 577/575, eval2
  307/305) because `7ur7_A` and `8ah9_A` recur across datasets with different
  sequences. The 333-unit FoldBench monomer universe is the exception: one dataset
  label, one row per stem, so units and stems are in bijection there. Counting stems
  silently drops proteins and was a real bug in exp226.
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
