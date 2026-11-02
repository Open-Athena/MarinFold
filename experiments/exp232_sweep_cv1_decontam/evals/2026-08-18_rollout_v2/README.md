# Exp232 2026-08-18 rollout-v2 checkpoint evaluation

This directory evaluates the final `m2-p06-aug` and `m1-p02-aug` checkpoints from exp232, plus the exp75 E8 checkpoint used to validate the evaluation path. All three checkpoints are verified and evaluated in place from existing CoreWeave S3 objects. No checkpoint is copied to Hugging Face or GCS.

> **Since this run:** `m2-p06`'s export is no longer CoreWeave-only — [#250](https://github.com/Open-Athena/MarinFold/issues/250)
> copied it to the public bucket as `contacts-v1-exp232-m2-p06-1.5B`
> (`checkpoints/prot-exp232-cw-cv1-decontam-s02-m2-p06-aug/hf/step-145199`), weights byte-identical
> and verified against the manifest `checkpoint_specs.py` pins here, with `config.json` rewritten so
> transformers 4.x reads the trained rope. `m1-p02` is still CoreWeave-only. See
> `experiments/exp250_evals_exploration_notebook/publish_exp232_m2_p06.py`.

## Fixed recipe

- 577 `(dataset, stem)` units / 575 unique stems
- 100 fresh contacts-v1 realizations requested per unit
- temperature 1.0, top-p 0.95, top-k disabled (`-1`)
- token budget `min(8192 - prompt_tokens, 6L + 128)`
- occurrence-frequency voting, no pairwise tie-break
- 12 single-H100 shards per checkpoint at batch priority
- exp89 metric implementation and legacy, eval2, natural, and strict reporting cuts

The run uses Iris cluster `marin`, federates to `cw-us-east-02a`, and submits as user `eczech`. `checkpoint_specs.py` pins the three model identities, exact CoreWeave paths, file sizes and digests, and source losses. The float32 exports are evaluated as bfloat16.

## Run

```bash
source ~/marin.env
uv run python submit_coreweave.py \
  --run-id decontam-v2-YYYYMMDD-NN \
  --seed 0
```

The completed run root is:

```text
s3://marin-us-east-02a/marin/protein-structure/MarinFold/
  exp232_sweep_cv1_decontam/evals/rollout_v2/decontam-v2-20260818-01/
```

`results/run_manifest.json` is the completion authority. `data/evaluation_attempts.json` records the resumable execution history, including the diagnostic token-ceiling attempts, and `data/unfinished_rollouts.csv` records every excluded nonterminating sample.

## Results

| Reporting cut | n | m2-p06 R (all) | m2-p06 R (long) | m1-p02 R (all) | m1-p02 R (long) |
|---|---:|---:|---:|---:|---:|
| **eval2 natural (<40%)** | 78 | **0.3450** | **0.2907** | 0.3200 | 0.2621 |
| eval2 pooled (<40%) | 307 | **0.5391** | **0.4822** | 0.5378 | 0.4801 |
| eval2 natural (<30%) | 61 | **0.3074** | **0.2493** | 0.2892 | 0.2279 |
| eval2 pooled (<30%) | 275 | 0.5377 | 0.4801 | **0.5398** | **0.4814** |
| Legacy exp89 | 554 | **0.5916** | **0.5418** | 0.5789 | 0.5292 |

The m2-p06 checkpoint is ahead on the primary eval2-natural subset by 0.0250 all-range and 0.0287 long-range R-precision.

Every checkpoint produced 577 dense matrices and 11,540 metric rows. E8 and m2-p06 used all 57,700 requested rollouts. For m1-p02, 7/57,700 rollouts did not terminate at the fixed token budget and were excluded from voting: `4oza_A` used 98/100 terminating samples and `4ozc_A` used 95/100, while all other 575 units used 100/100. No partial completion was parsed or scored.

## E8 validation

| Metric | Published | Reproduced | Absolute difference | Pass (≤0.005) |
|---|---:|---:|---:|:---:|
| All R | 0.424529 | 0.424323 | 0.000206 | yes |
| Long R | 0.365615 | 0.364153 | 0.001462 | yes |
| All AUC | 0.900963 | 0.899902 | 0.001061 | yes |
| Long AUC | 0.873780 | 0.872352 | 0.001429 | yes |

The validation checkpoint passed all four gates. It was read directly from `s3://marin-us-east-02a/MarinFold/exp163/model/step-35679` after every object was matched to the pinned identity.

## Comparison figure

`build_comparison_data.py` combines the recomputed #75 E8 validation slice and the two new legacy-554 result slices with the prior #146 3B, #166 AA augmentation, CW m1-p06 augmentation, CW m1-p06 cooldown, and Protenix-v2 per-protein rows. `plot_primary_comparison.py` renders `plots/final_checkpoint_rprecision.png`, sorting its boxes by ascending mean R-precision and fitting its descriptive sigmoid to every plotted MarinFold checkpoint with a validation loss. Protenix-v2 is excluded from the fit because it has no contacts-v1 validation loss. Both the derived comparison data and plot have SHA-256 provenance sidecars.
