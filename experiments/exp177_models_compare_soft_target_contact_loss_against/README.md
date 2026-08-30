---
marinfold_experiment:
  issue: 177
  title: 'exp: compare soft-target contact loss against next-token CE on TPUs'
  kind: models
  branch: exp177/augmented-contact-order-ce
---

# exp: compare soft-target contact loss against next-token CE on TPUs

**Issue:** [#177](https://github.com/Open-Athena/MarinFold/issues/177) · **Kind:** `models` · **Branch:** `exp177/augmented-contact-order-ce`

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

## Contact-order augmented CE arm

Branch `exp177/augmented-contact-order-ce` adds a stock next-token CE data path
that rebuilds contacts-v1 documents from analyzed-contact shards in
multiprocessing workers. It preserves the canonical sequence prefix and selected
contact set, then deterministically resamples only the structure suffix order
and endpoint orientation for each `(epoch, shard, row, augmentation)`.

CoreWeave launch knobs:

```bash
EXP177_NEXT_TOKEN_DATA=augmented_contact_order_mp
EXP177_CONTACT_REORDERINGS_PER_ROW=4
EXP177_TRANSFORM_WORKERS=8
EXP177_PREFETCH_SHARDS=8
EXP177_SHARD_CACHE_SIZE=16
EXP177_MP_START_METHOD=fork
EXP177_CONTACTS_SHARD_NAME_TEMPLATE='analyzed-{shard_index:05d}-of-{total_shards:05d}.parquet'
python dispatch_cw_next_token.py
```

The regular validation component remains the tokenized contacts-v1 validation
cache, so validation loss is directly comparable with the ordinary CE baseline.

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

### Sparse soft-target smoke (2026-08-19)

Run `exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20` tested the sparse
loss path directly from the existing exp139 analyzed S3 shards; no new training
corpus was materialized. It used 1 node × 8 H100, TP=8, global batch size 16,
`EXP177_PRECOMPUTED_MP=0`, `EXP177_SOFT_TARGET_BATCH=sparse`,
`EXP177_MAX_SPARSE_CONTACTS=2048`, and `EXP177_MAX_SPARSE_DEGREE=32`.

Iris marked both driver and child jobs succeeded. The first batch loaded in
0.1s, the first train step completed in 43.6s including compilation, and the
post-compile training phase averaged about 9.6s/step by wall time from first
step completion to checkpoint start. The short-run average including compile
was about 11.4s/step over 20 steps. This is substantially faster than the prior
compact padded soft-target smokes: r45 bs16 was ~51.5s/step and r47 bs24 was
~78.8s/step. Final eval loss was 17.172 and the checkpoint was written to:

```
s3://marin-us-east-02a/protein-structure/MarinFold/exp177_soft_target_loss_h2h/checkpoints/exp177-soft-target-cw-r50-sparse-8gpu-bs16-smoke20/2026.08.19.r50/checkpoints/step-19
```

A non-fatal draccus config-encoding exception still appears for the sparse
dataset object during startup, but training continued normally.

### Sparse soft-target batch-size smokes (2026-08-19)

Follow-up single-node 8×H100 smokes swept larger sparse global batches using the
same on-the-fly sparse precomputed path and caps (`max_sparse_contacts=2048`,
`max_sparse_degree=32`):

| Run | Batch | Outcome | Timing signal | Approx tokens/s |
|---|---:|---|---|---:|
| r50 | 16 | succeeded, 20 steps | ~9.6s/step post-compile | ~13.7k |
| r51 | 32 | succeeded, 20 steps | ~18.1-18.6s/step post-compile | ~14.1-14.5k |
| r52 | 64 | reached 15/20, then stopped during interval checkpoint after enough timing signal | ~36.2-36.8s/step | ~14.2-14.5k |
| r53 | 128 | reached 3/8, then stopped after confirming no immediate OOM and poor scaling | ~78-95s/step early/tqdm | ~11-13k |

No OOM was observed up to global batch 128, but throughput plateaus around
batch 32-64 and then regresses at 128. For a larger sparse soft-target run, the
best current single-node setting is likely **global batch 32**: it matches or
slightly beats batch 64 tokens/s with shorter step latency and less checkpoint /
activation risk. Batch 64 is viable if a larger effective batch is preferred,
but it does not improve throughput in these smokes.

### Matched stock next-token smoke (2026-08-19)

Run `exp177-next-token-cw-r57-8gpu-bs128-smoke20` is the current matched
CoreWeave stock baseline: same exp177 Qwen3 1.47B config, seq_len 8192,
1 node × 8 H100, global batch size 128, stock next-token CE. It required two
small launcher fixes first (`ResourceConfig` import from `fray.types`, and a
flat train cache path so Levanter does not try to validate the train component
at a non-existent `contacts-v1/validation` cache).

The successful r57 run loaded the first batch in 13.1s, completed the first
train step in 49.6s including compilation, and reached 15/20 at 10.5s/step.
Wall time from first-step completion to checkpoint start gives ~10.6s/step
post-compile, or about **99k tok/s** (`128 * 8192 / 10.6`). Final validation
loss after 20 steps was 6.179 on `contacts-v1-val`.

Compared directly on the same 1×8 H100 shape, the best sparse soft-target
smokes were ~14-15k tok/s, so the sparse path is still about **6.8-7.0× slower
per token** than current matched stock next-token CE, despite being ~5-6× faster
than the old compact padded soft-target path.

## Conclusion

_(Fill in after results are in.)_
