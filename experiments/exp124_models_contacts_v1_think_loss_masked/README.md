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

Does training from scratch on the think-augmented contacts-v1 corpus improve the contacts-v1 model, when `<think>` tokens are present in the context but masked out of the training loss?

## Hypothesis

Pause tokens give the model extra autoregressive compute before committing to contact statements. If the model learns to use those positions as scratch/pause context without being directly rewarded for predicting them, it may improve downstream contact prediction under inference-time `<think>` insertion while preserving ordinary contacts-v1 validation quality.

## Approach

- **Recipe = exp177 successful TPU config.** Start from the exp177/exp117 recipe that reached good I/O and throughput on `v5p-128`: Qwen3 1.47B, `seq_len=8192`, global batch 256, AdamW LR `3.1623e-3`, WD `0.2`, betas `0.9/0.95`, cosine + 10% warmup, `data_seed=0`, block-Feistel shuffle, 16 epoch-equivalent steps.
- **Only intentional training-data change:** use the exp126 think-augmented contacts-v1 corpus, a 1:1 twin of exp53 over the same proteins/rounds/splits:
  - `open-athena/MarinFold/data/document_structures/contacts_v1_think/{train,val,test}/`
  - Exp126 documents are train/val/test = 4,129,682 / 41,954 / 41,567, but two train shard objects (`00858`, `01423`) are absent from the HF bucket. The cache job skips those two slots, so the built train cache contains 4,125,682 documents.
- **Prebuild a region-local Levanter cache before training.** The HF-bucket parquet is public and raw text, so it is not suitable for the TPU hot path. `cache.py` tokenizes it once into `gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/cache/think-masked/...` with fields:
  - `input_ids`
  - `loss_weights`, where causal positions whose target token is `<think>` (id 6) have weight 0.
- **Train from scratch** from the prebuilt cache with Levanter's packed-token path and `loss_weights_key="loss_weights"`; no HF/checkpoint warm start.
- **Evaluate both:**
  - think validation cache with masked `<think>` targets;
  - ordinary exp117 contacts-v1 validation cache (`gs://marin-us-east5/tokenized/contacts-v1-val/2026.07.13.1/validation`) for comparison to prior CE runs.

## Success criteria

- Think-masked train and validation caches build successfully in `us-east5` and load as packed Levanter datasets.
- A small smoke run starts from scratch on TPU, logs W&B, gets through first batch/JIT, and shows reasonable data I/O (no raw HF/parquet tokenization in the training hot path).
- Full `v5p-128` run logs train loss, think-val loss, standard contacts-v1 val loss, throughput, checkpoint, and HF export with tokenizer co-located.
- Headline: does the think-trained model improve downstream/inference-time think behavior without damaging standard contacts-v1 validation?

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
