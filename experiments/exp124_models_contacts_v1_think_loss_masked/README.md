---
marinfold_experiment:
  issue: 124
  title: 'exp: train a 1.5B contacts-v1 model on the pause-token dataset — apples-to-apples vs Eric''s #75'
  kind: models
  branch: exp124/think-loss-masked
---

# exp: train a 1.5B contacts-v1 model on the pause-token dataset — apples-to-apples vs Eric's #75

**Issue:** [#124](https://github.com/Open-Athena/MarinFold/issues/124) · **Kind:** `models` · **Branch:** `exp124/think-loss-masked`

## Question

Does training from scratch on the think-augmented contacts-v1 corpus improve the
contacts-v1 model, when `<think>` tokens are present in the context but masked
out of the training loss?

Operationally, the decisive first check is simpler: can this model preserve
ordinary contacts-v1 validation loss? If it badly regresses the base
contacts-v1 objective, then downstream contact evaluation with inference-time
`<think>` insertion is unlikely to be the next best use of compute.

## Hypothesis

Pause tokens give the model extra autoregressive compute before committing to
contact statements. If the model learns to use those positions as scratch/pause
context without being directly rewarded for predicting them, it may improve
downstream contact prediction under inference-time `<think>` insertion while
preserving ordinary contacts-v1 validation quality.

## Background

- **Dataset:** exp126 published a think-augmented contacts-v1 corpus at
  `hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1_think/`.
  It is a 1:1 transform of exp53 over the same proteins, rounds and splits, with
  randomly inserted `<think>` tokens.
- **Tokenizer:** `timodonnell/contacts-v1-tokenizer@5d68a24a899f`; `<think>` is
  token id `6`.
- **Training target:** in the cache built here, causal positions whose **target**
  token is `<think>` have loss weight 0. The model sees `<think>` in context but
  is not directly optimized to emit it.
- **Baseline scale:** #75 E8 had contacts-v1 validation loss 2.7566; #117 E16
  final is the current nearby 1.5B reference at 2.7037. The exp124 run used the
  exp177/exp117-style TPU recipe rather than literally re-running #75's 8-epoch
  schedule, because that was the validated v5p-128 training path at the time.

## Approach

1. **Build a region-local pretokenized cache.** [`cache.py`](cache.py) reads the
   public exp126 HF-bucket parquet, tokenizes with the contacts-v1 tokenizer, and
   writes Levanter cache fields:

   - `input_ids`
   - `loss_weights`, with weight 0 when the next token is `<think>`.

   The full cache root used by the final run was:

   ```text
   gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/cache/think-masked/2026.07.29.2
   ```

   Exp126 has two missing train shard objects (`00858`, `01423`), so the builder
   skips them. The intended train/val/test sizes are 4,129,682 / 41,954 / 41,567
   documents; the built train cache contains 4,125,682 documents.

2. **Train from scratch on TPU.** [`train.py`](train.py) launches a Qwen3 1.47B
   model with the exp177/exp117 TPU recipe: `seq_len=8192`, global batch 256,
   AdamW LR `3.1623e-3`, WD `0.2`, betas `0.9/0.95`, cosine + 10% warmup,
   `data_seed=0`, block-Feistel shuffle, and a 16-epoch-equivalent step budget.

3. **Evaluate two validation streams.** The run logs both:

   - `eval/contacts-v1-think-masked/loss` on the think validation cache with
     `<think>` targets masked;
   - `eval/tokenized/contacts-v1-val/loss` on the ordinary contacts-v1 validation
     cache, for comparison to prior contacts-v1 CE runs.

   After training, #117 E16 final was also recomputed on the exact exp124
   think-augmented validation cache with `<think>` targets masked, so the native
   think-mode validation result has an apples-to-apples baseline.

## Success criteria

- Think-masked train and validation caches build successfully in `us-east5` and
  load as packed Levanter datasets.
- A TPU smoke run starts from scratch, logs W&B, gets through first batch/JIT,
  and confirms no raw HF/parquet tokenization is in the training hot path.
- Full `v5p-128` run logs train loss, think-val loss, standard contacts-v1 val
  loss, throughput, checkpoint, and HF export with tokenizer co-located.
- Headline: does the think-trained model improve downstream/inference-time think
  behavior without damaging standard contacts-v1 validation?

## Results

All infrastructure criteria were met: cache smokes passed, the full resumed
`v5p-128` run succeeded, and step-35680 checkpoint/HF export completed. The model
result is negative for autonomous contact prediction: exp124 regressed ordinary
no-`<think>` contacts-v1 validation, and forcing `<think>` at inference time did
not improve downstream contact-map metrics. It did improve over #117 when both
are evaluated on the native think-augmented masked validation metric, but that
metric is an oracle-context teacher-forced evaluation rather than evidence that
the model learns to emit useful `<think>` tokens on its own.

**Final W&B run:**
[`exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256_next_token-exp177recipe-v5p128-r3`](https://wandb.ai/open-athena/MarinFold/runs/exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256_next_token-exp177recipe-v5p128-r3)

**Final driver jobs:**

- `/zack/exp124-train-full-20260731-1413-resume-r5`
- `/zack/exp124-train-full-20260731-1744-resume-auto`

**Final W&B summary:**

| metric | value |
|---|---:|
| `global_step` | 35,680 |
| `train/loss` | 3.0132369995117188 |
| `eval/contacts-v1-think-masked/loss` | 3.0855870246887207 |
| `eval/tokenized/contacts-v1-val/loss` | **3.131303071975708** |

The ordinary contacts-v1 validation result is much worse than both reference
contacts-v1 models, but that is not the same metric as the think-augmented
objective exp124 trained on:

![Final validation losses](plots/final_losses.png)

| model/run | metric | val loss | exp124 − model |
|---|---|---:|---:|
| #117 E16 final | ordinary contacts-v1 val | 2.7037 | +0.4276 |
| #75 E8 | ordinary contacts-v1 val | 2.7566 | +0.3747 |
| **exp124 think-masked** | ordinary contacts-v1 val | **3.1313** | — |
| #117 E16 final | think-augmented masked val | 3.0996 | -0.0141 |
| **exp124 think-masked** | think-augmented masked val | **3.0856** | — |

The #117 think-masked baseline was recomputed with
[`recompute_think_masked_val.py`](recompute_think_masked_val.py) on Iris job
`/zack/exp124-think-val-exp117-full-v5e-west4-r1`; the output JSON is stored at
`gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/eval_loss/exp117-full-v5e-west4-r1.json`
and summarized in [`data/exp117_think_masked_eval.json`](data/exp117_think_masked_eval.json).

The exp188 padding-target-loss investigation found sub-0.01-nat objective-scale
issues for comparable contacts-v1 runs. That is too small to explain the ordinary
contacts-v1 regression, which is hundreds of millinats.

### Downstream prompt intervention

We also evaluated the step-35680 checkpoint on the 554-protein contact-prediction
benchmark with the exp82 rollout+resample scorer (`n=100`, temperature 1.0,
top-p 0.95, no top-k), comparing the standard prompt against an intervention
that appends exactly one literal `<think>` after the protein sequence /
`<begin_statements>` prefix. Both conditions completed all 12 CoreWeave shards
and were scored with the same exp89 metric implementation.

| metric | standard prompt | forced `<think>` | forced − standard |
|---|---:|---:|---:|
| all R-precision | 0.3365 | 0.3359 | -0.0006 |
| long R-precision | 0.2856 | 0.2822 | -0.0034 |
| all AUC | 0.8119 | 0.8103 | -0.0016 |
| long AUC | 0.7798 | 0.7757 | -0.0041 |
| all recall@L | 0.3725 | 0.3719 | -0.0006 |
| long recall@L | 0.3834 | 0.3786 | -0.0048 |
| all recall@L/5 | 0.1268 | 0.1278 | +0.0011 |
| long recall@L/5 | 0.1652 | 0.1656 | +0.0004 |

The prompt intervention therefore did not materially change contact-prediction
quality. The small mixed deltas are consistent with sampling noise; the main
ranked metrics are slightly worse with the forced token.

Full summaries are in [`data/prompt_intervention_precision.csv`](data/prompt_intervention_precision.csv)
and [`data/prompt_intervention_recall.csv`](data/prompt_intervention_recall.csv).

## Conclusion

Exp124 answers two different questions differently.

On ordinary no-`<think>` contacts-v1 validation, the result is clearly negative:
training on think-augmented documents while masking `<think>` targets produces a
model with loss **3.1313**, versus 2.7566 for #75 and 2.7037 for #117 final. This
means the run did not preserve the base contacts-v1 language-modeling objective.

On the native think-augmented validation cache with target `<think>` tokens
masked, exp124 is better than #117: **3.0856** vs **3.0996**. This supports only
the narrower claim that the model can use oracle `<think>` tokens when they are
teacher-forced in the context. Because the target `<think>` positions are masked,
there is no direct loss gradient teaching the model to emit those tokens.

The downstream prompt intervention resolves the transfer question: adding one
forced `<think>` token at inference time does **not** improve contact prediction
(all R-precision 0.3359 vs 0.3365 standard; long R-precision 0.2822 vs 0.2856
standard). The native validation gain is therefore best interpreted as an
oracle-context effect, not as an autonomous useful-thinking behavior.

If pause tokens are revisited, stronger designs would include at least one of:

- train/eval recipes that explicitly exercise the same `<think>` protocol at
  training and inference time, then score the downstream contact metric;
- an ablation that **does not** mask `<think>` targets, so generation of pause
  tokens is part of the learned behavior;
- a mixture with ordinary contacts-v1 documents to preserve the base objective;
- shorter pilot runs that compare standard contacts-v1 val loss before spending a
  full 16-epoch-equivalent run.

## Artifacts

- Final W&B run:
  <https://wandb.ai/open-athena/MarinFold/runs/exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256_next_token-exp177recipe-v5p128-r3>
- Output artifact/checkpoint root:
  `gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/checkpoints/exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256_next_token-exp177recipe-v5p128-r3/2026.07.30.4`
- Final checkpoint/HF export: `step-35680`
- Think-masked cache root:
  `gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/cache/think-masked/2026.07.29.2`
- Run history:
  `history/runs/20260730_exp124_models_contacts_v1_think_loss_masked_exp124_cv1_think_masked_e16_lr3p162e_3_wd0p2_bs256_next_token_exp177recipe_v5p128_r3.md`
- Think-masked #117 recompute:
  `data/exp117_think_masked_eval.json`,
  `gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/eval_loss/exp117-full-v5e-west4-r1.json`
- Prompt-intervention contact eval summaries:
  `data/prompt_intervention_precision.csv`, `data/prompt_intervention_recall.csv`,
  `s3://marin-us-east-02a/MarinFold/exp124/standard_prompt_eval/metrics/`,
  `s3://marin-us-east-02a/MarinFold/exp124/think_prompt_eval/metrics/`
- Plot source/output:
  `data/final_losses.csv`, `plots/final_losses.png`, `plots/final_losses.pdf`
