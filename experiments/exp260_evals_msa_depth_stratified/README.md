---
marinfold_experiment:
  issue: 260
  title: 'exp: does contact accuracy hold up at low MSA depth? stratify the natural eval set by ColabFold depth for the #232 training checkpoint'
  kind: evals
  branch: exp/260-msa-depth
---

# Contact accuracy vs. MSA depth (issue #260)

**Issue:** [#260](https://github.com/Open-Athena/MarinFold/issues/260) · **Kind:** `evals` · **Branch:** `exp/260-msa-depth`

## Question

**Does the best decontaminated checkpoint we have hold its contact accuracy
where MSA-based methods lose theirs — at low MSA depth?**

[PR #257](https://github.com/Open-Athena/MarinFold/pull/257) evaluated the
[#232](https://github.com/Open-Athena/MarinFold/issues/232) `m2-p06` **training**
checkpoint (step 363,000, decontaminated corpus) on the legacy 554, `eval-val`,
and `eval-denovo`, and deliberately left `eval-test` unscored. This experiment
finishes that read and adds the axis the single-sequence thesis rests on.

Two deliverables:

1. **The usual numbers, completed** — R-precision on `eval-val`, `eval-test`,
   and `eval-denovo`, plus the legacy 554 for continuity.
2. **A depth-stratified table over every natural protein in our eval universe**
   — 372 of them — in tiers `<10`, `10–100`, `100–1000`, `≥1000` sequences, for
   all natural proteins, for the FoldBench half, and for the non-FoldBench half.

"MSA depth" is the depth of the ColabFold MSA that Protenix's `+MSA` arm
actually ran with. MarinFold never sees it; it is a property of the protein, and
it is what an MSA-based competitor had to work with on the same target.

## What the eval universe looks like when you cut it to natural proteins

| subset | n | what it is | MSA source |
|---|---:|---|---|
| `foldbench_natural` | 314 | every natural FoldBench monomer: `eval-val` (97) + `eval-test` (217) | `protenix-foldbench-msa` (#12) |
| `nonfoldbench_natural` | 58 | `cameo_hard` (32) + `casp_fm` (26) — CAMEO hard targets and CASP free-modeling domains, collected in [#65](https://github.com/Open-Athena/MarinFold/issues/65) for exactly this regime | `protenix-exp74-msa` (#74) |
| `foldbench_designed` | 19 | `eval-denovo`, kept as a control, not part of the natural stratification | `protenix-foldbench-msa` |

The 97 `eval-val` proteins are the same ones the legacy set calls
`foldbench100`; they are counted once, under FoldBench, where the eval-set
labels live. The legacy `denovo_pdb` 396 are left out: they are designs, they
would outnumber the natural proteins four to one in any pooled bin, and #74
already published their Neff.

Both MSA volumes were written by the same Protenix pipeline call
(`runner.msa_search.update_seq_msa(..., mode="colabfold")`), so the two halves
are measured on one ruler — see [Consistency checks](#consistency-checks).

## What ran

**Scoring.** PR #257's harness with `eval-test` added: 887 `(dataset, stem)`
units (legacy 554 + all 333 scorable FoldBench monomers), one checkpoint, 100
rollouts each under the fixed [#82](https://github.com/Open-Athena/MarinFold/issues/82)
rollout+resample recipe, 12 single-H100 shards at batch priority on
`cw-us-east-02a`, where the checkpoint already lives. **88,700 usable rollouts,
zero unfinished, 887/887 vote matrices**, 9m41s wall clock.

Nothing about scoring changed: the worker bytes (`sha256 dd2f76dd…`) and #89's
metric script (`sha256 6cbaa1c5…`) are byte-identical to PR #257's, and the
checkpoint is read in place from the HF export PR #257 wrote. No weights moved,
no GCS was touched, and nothing left CoreWeave except the small result tables.

**Depth.** `msa_depth_modal.py` measures raw depth and Neff for all 391 proteins
from the two Modal volumes using #74's pinned `msa_depth.py` — one definition,
one code path, both volumes, no alignment leaving Modal.

## Validation

Re-scoring `eval-val`, `eval-denovo`, and the legacy 554 alongside the new
`eval-test` turns PR #257's published aggregates into a reproduction gate rather
than a claim: same weights, same worker, so a disagreement would mean the
execution path moved, not the model. **All twelve reproduced**, largest absolute
difference **0.0044** (`eval-val` R all: 0.5561 here vs 0.5517 published) —
inside the 0.005 tolerance the `eval-checkpoint` recipe fixes, and consistent
with [#204](https://github.com/Open-Athena/MarinFold/issues/204)'s 0.0023
run-to-run span. Full table:
[`data/coreweave_results/results/published_reference_validation.json`](data/coreweave_results/results/published_reference_validation.json).

This is also why the E8 reference checkpoint is not scored here: PR #257 passed
that gate on this cluster with this worker eight days ago, and reproducing the
checkpoint under test is the stronger check.

## Results — the eval sets

All-range and long-range R-precision, plus AUC, for the #232 `m2-p06` training
checkpoint (step 363,000). Source:
[`data/coreweave_results/results/subset_aggregate_metrics.csv`](data/coreweave_results/results/subset_aggregate_metrics.csv).

| subset | n | R (all) | R (long) | AUC (all) | AUC (long) |
|---|---:|---:|---:|---:|---:|
| **`eval-test`** (first read for this checkpoint) | 217 | **0.5693** | 0.5464 | 0.9455 | 0.9375 |
| `eval-val` | 97 | 0.5561 | 0.5402 | 0.9381 | 0.9258 |
| `eval-denovo` | 19 | 0.6110 | 0.5745 | 0.9668 | 0.9580 |
| legacy 554 | 554 | 0.6059 | 0.5566 | 0.9454 | 0.9312 |

Viral split (`is_viral` on #245's `eval_sets.csv`; indicative only — 19 of 334
monomers are viral):

| subset | n | R (all) |
|---|---:|---:|
| `eval-test` non-viral | 204 | 0.5737 |
| `eval-test` viral | 13 | 0.4993 |
| `eval-val` non-viral | 91 | 0.5604 |
| `eval-val` viral | 6 | 0.4906 |

**Where that sits.** On `eval-test`, against #245's published per-protein
baselines over the same 217 proteins:

| predictor | `eval-test` R (all) |
|---|---:|
| Protenix-v2 + MSA | 0.8446 |
| ESMFold2 | 0.7921 |
| ESMFold | 0.7534 |
| **#232 `m2-p06` training (this run)** | **0.5693** |
| #199 cooldown (contaminated data) | 0.6132 |
| #232 `m2-p06` sweep checkpoint | 0.5377 |
| seq-KNN null, unfiltered corpus | 0.5820 |
| seq-KNN null, decontaminated corpus | 0.4257 |
| Protenix-v2 single-seq | 0.2646 |

Training on past the sweep checkpoint is worth **+0.032** on `eval-test`
(0.5377 → 0.5693), which tracks the +0.013 it gained on the legacy 554. The
checkpoint clears the seq-KNN null over the corpus it actually trained on
(0.4257) by 0.144, so the score is not memorisation; it does **not** clear the
null built from the un-decontaminated corpus (0.5820), which is the right
comparison only for the contaminated models.

`eval-test` scores **0.013 above** `eval-val` here, the same direction and
magnitude #245 measured across nine predictors — the historical FoldBench-100 is
not flattering this checkpoint.

## Results — MSA depth

_In flight: the depth measurements are recomputing after a fail-loud check
discarded the first pass (three `eval-denovo` designs have no a3m on the Modal
volume; they are outside the natural stratification). This section, the tier
tables, and the three figures land in a follow-up commit on this branch._

Preliminary, over the 314 FoldBench natural proteins only, using #247's
already-published depths for the same a3m files — mean all-range R-precision:

| tier | n | MarinFold | Protenix-v2 + MSA | ESMFold2 | Protenix-v2 single-seq | seq-KNN (decontam.) |
|---|---:|---:|---:|---:|---:|---:|
| `<10` | 5 | 0.342 | 0.320 | 0.664 | 0.305 | 0.027 |
| `10–100` | 21 | 0.327 | 0.787 | 0.516 | 0.294 | 0.073 |
| `100–1000` | 63 | 0.457 | 0.837 | 0.745 | 0.303 | 0.287 |
| `≥1000` | 225 | 0.623 | 0.864 | 0.838 | 0.250 | 0.498 |

Read with care — the `<10` bin is five proteins, and the final table adds the 58
CAMEO-hard / CASP-FM targets, which is where most of the shallow-MSA natural
proteins in our universe live.

## Artifacts

**Published, anonymously readable over HTTPS** — root
`https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/contacts-v1-msa-depth-exp260/v1-01`:

| path | what |
|---|---|
| `results/subset_aggregate_metrics.csv` | the eval-set table above, every subset × range × cut |
| `results/aggregate_metrics.csv` | pooled 887-unit aggregates |
| `results/marinfold_precision.csv` | per-protein rows, 887 units × 20 (range × cut) |
| `results/contact_precision_all.csv` | the same in #89's unified schema |
| `results/timings.csv` | per-protein wall time, rollout counts, worker metadata |
| `results/run_manifest.json` | full provenance: checkpoint identity, sampling recipe, job ids, digests |
| `results/published_reference_validation.json` | the PR #257 reproduction gate |
| `inputs/evaluation_subsets.csv` | the 887-unit subset manifest with viral flags |
| `inputs/eval_targets.parquet.validation.json` | unit-count validation for the union |
| `analysis/…` | universe, depths, tiered tables (pending, pushed by `publish_to_hf.py`) |

**CoreWeave, in-cluster only** — root
`s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp260_evals_msa_depth_stratified/rollout-v2/2026-08-31/v1-01`:
`dense_scores/` (887 `[L,L]` npz vote matrices), `rollout/` (sparse parquet
parts + per-shard completion markers), `inputs/` (mirrored, digest-verified eval
inputs), `results/` (the same tables published above).

**Checkpoint under test** — read in place, never copied:

- HF export: `s3://marin-us-east-02a/marin/protein-structure/MarinFold/exp232_sweep_cv1_decontam/evals/rollout-v2/2026-08-24/v2-01/models/exp232-decontam-train-m2-p06-step363000/hf/step-363000`
- Levanter source: `s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/checkpoints/protein/prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-lr005-us-east1/2026.08.21.1/checkpoints/step-363000`
- W&B: [`prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-lr005-us-east1`](https://wandb.ai/eric-czech/marin/runs/prot-exp232-trc-cv1-decontam-train-s01-m2-p06-srcpeak-augcont-lr005-us-east1), step 363,000, contacts-v1 `eval/loss` 2.9681 at step 361,494

**Iris jobs** (cluster `cw-us-east-02a`, batch priority):
`/timodonnell/exp260-msa-depth-eval-v1-01` — one CPU driver, one smoke shard,
twelve H100 shards, all succeeded. Ids are listed in `run_manifest.json`.

**Upstream inputs**, digest-pinned in
[`rollout/checkpoint_specs.py`](rollout/checkpoint_specs.py) and
[`upstream.py`](upstream.py): #169's legacy target table, #245's FoldBench
targets / ground truth / `eval_sets.csv`, #89's legacy ground truth, #245's
`per_protein.csv.gz` baselines, #89's `contact_precision_all.csv`, and the
Modal volumes `protenix-foldbench-msa` and `protenix-exp74-msa`.

**In-repo**: `data/coreweave_results/` (mirror of the published prefix;
`marinfold_precision.csv` committed gzipped), `data/universe.csv`, and — landing
with the depth commit — `data/msa_depth.csv`, `data/per_protein_depth.csv`,
`data/depth_tiers.csv`, `data/paired_deltas.csv`, `data/tier_counts.csv`,
`data/depth_consistency.json`, `plots/*.png`, `plots/summary.pdf`.

The `eval-test` read is recorded as row 3 in
[`experiments/exp245_evals_foldbench_held_out_monomers/data/eval_test_reads.md`](../exp245_evals_foldbench_held_out_monomers/data/eval_test_reads.md).

## Consistency checks

`check_depth_consistency.py` runs two, and both are reported in
`data/depth_consistency.json`:

1. **Against #247** — that experiment counted sequences in the same
   `protenix-foldbench-msa` a3m files with different code. The counts should
   agree exactly.
2. **Across the two volumes** — 11 stems live in both, searched about three
   weeks apart against a database that grows. Their spread bounds how comparable
   the FoldBench and non-FoldBench halves of the tier table are.

## Files

| file | what it does |
|---|---|
| `upstream.py` | Pinned URLs, digests, volume names, tier definitions. |
| `build_universe.py` | Step 1 — the 391-protein universe (`data/universe.csv`). |
| `msa_depth_modal.py` | Step 2 — depth + Neff on Modal (`data/msa_depth.csv`). |
| `check_depth_consistency.py` | The two depth cross-checks. |
| `build_depth_table.py` | Step 3 — join scores to depth; tier means, paired deltas, bootstrap intervals. |
| `plot_depth.py` | Step 4 — the three figures. |
| `publish_to_hf.py` | Push the analysis tables to the public bucket. |
| `rollout/` | The CoreWeave scoring harness, derived from PR #257's. |
| `build_summary.py` | Rebuild `plots/summary.pdf` from `summary_narrative.md` + `plots/`. |

## Reproducing

```bash
cd experiments/exp260_evals_msa_depth_stratified
uv sync
uv run python build_universe.py
uv run modal run msa_depth_modal.py
uv run python check_depth_consistency.py
uv run python build_depth_table.py
uv run python plot_depth.py
uv run python build_summary.py

# the scoring half (needs CoreWeave access; ~10 minutes on 12 H100s at batch)
cd rollout && uv sync
KUBECONFIG=~/.kube/coreweave-iris-gpu uv run python submit_coreweave.py --run-id v1-01
uv run pytest -q test_rollout.py
```
