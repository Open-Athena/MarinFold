---
marinfold_experiment:
  issue: 177
  title: 'exp: compare soft-target contact loss against next-token CE on TPUs'
  kind: models
  branch: exp177/mp-queue-shard-dataset
---

# exp: compare soft-target contact loss against next-token CE on TPUs

**Issue:** [#177](https://github.com/Open-Athena/MarinFold/issues/177) · **Kind:** `models` · **Branch:** `exp177/mp-queue-shard-dataset`

## Question

Does PR #144's contacts-v1 soft-target/document loss improve training over ordinary next-token cross-entropy when both arms train from scratch on TPUs under Eric's best known exp117 recipe?

## Hypothesis

The soft-target objective removes arbitrary contact ordering/orientation from the loss. At matched architecture, token budget, data order, optimizer, and TPU placement, it should reduce validation loss faster than stock CE, or at least preserve downstream contact quality while making the training target less noisy.

## Approach

- Start a new experiment/branch rather than extending abandoned exp156.
- Import the reusable PR #144 document/soft-target machinery into shared code:
  - `marinfold.document_structures.documents`
  - contacts-v1 training-document builders
  - `marinfold_models.document_loss`
  - fixed-quota shard document loading helpers
- Train two from-scratch contacts-v1 Qwen3 1.47B arms on TPUs:
  1. stock next-token CE on canonical serialized contacts-v1 documents;
  2. soft-target contact loss using PR #144's weighted target distributions.
- Use Eric exp117 best-known recipe as the baseline config, without loading exp117 weights:
  - Qwen3 1.47B, seq 8192
  - 16 epochs / 35,680 steps at global batch 256
  - AdamW, LR `3.1623e-3`, WD `0.2`, betas `0.9/0.95`
  - cosine schedule, 10% warmup, `data_seed=0`
- Evaluate both arms against the ordinary tokenized contacts-v1 validation loss. If the soft-target training path cannot use that metric directly, add an auxiliary CE validation hook so the curves are comparable.

## Success criteria

- Both arms launch and train from scratch on TPU with no HF/checkpoint warm start.
- Both arms log matched train loss, validation loss, throughput, and checkpoint/export artifacts.
- The comparison uses the same train corpus, validation corpus, step budget, batch size, optimizer schedule, and TPU shape.
- A short smoke validates the soft-target batch/loss path before the full run.

## Results

### Sparse soft-target shape stats (2026-08-19)

To debug soft-loss training throughput, we computed per-example sparse contact
shape statistics on CoreWeave/S3 without materializing a new training corpus.
The Zephyr job `/zack/exp177-sparse-target-stats-full-r1` read all 3,338 exp139
analyzed shards and wrote 8,845,700 stats rows to:

```
s3://marin-us-east-02a/protein-structure/MarinFold/exp177_soft_target_loss_h2h/stats/sparse_target_shapes_v1/2026.08.19.1/
```

The distribution strongly supports a sparse neighbor-row loss: max incident
degree is tiny compared with the current padded contact budget. Across the
fixed-quota slots, `max_degree` has p50=8, p95=12, p99=13, p999=15, max=19.
Residue counts are also modest: p50=195, p95=533, p99=791, max=1000. Candidate
bucket coverage by `(residue_count, max_degree)` was:

- r256-d32: 67.53%
- r512-d64: 94.30%
- r1024-d128: 100.00%

This suggests we can likely avoid copying/bucketing the whole dataset at first:
a single sparse representation capped around `residue_count<=1024` and
`max_degree<=32` would already cover this run exactly by the observed stats,
with much smaller arrays than the current `(seq_len - 2) // 3` contact padding.

## Conclusion

_(Fill in after results are in.)_
