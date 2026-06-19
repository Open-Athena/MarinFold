---
marinfold_experiment:
  issue: 87
  title: "exp: train a model on combined contacts-v1.sequence_only + contacts-v1 datasets"
  kind: models
  branch: exp/87-combined-contacts-v1-sequence-only
---

# exp: train a model on combined contacts-v1.sequence_only + contacts-v1 datasets

**Issue:** [#87](https://github.com/Open-Athena/MarinFold/issues/87) · **Kind:** `models` · **Branch:** `exp/87-combined-contacts-v1-sequence-only`

## Question

Is there benefit to including unlabelled (sequence-only, no structure) protein
sequences during training? Concretely: does one epoch over the **combined**
`contacts-v1` + `contacts-v1.sequence_only` corpus yield a better
structure-prediction (contacts) eval than training on `contacts-v1` alone?

## Hypothesis

After an epoch through the combined training set we'll see better eval losses on
the with-structure documents than the structure-only runs (#67/#85), because the
~7× larger sequence-only corpus teaches the model better sequence
representations that transfer to the contact-prediction task.

## Background

- **#67** — the quick 1.5B `contacts-v1` run (unmasked next-token loss, shuffled
  Feistel stream, LR 3.5e-4 / cosine, batch 128, seq 8192, v5p-8 @ us-east5-a,
  ~2.7 epochs). Final eval/loss 2.980.
- **#85** — a warm-restart of #67 for ~1 more epoch on v5p-32 (batch 512, LR
  re-heat). Also where the **cache-reader workaround** and the **PyPI
  marin-source-dist** dependency pins were worked out (carried over here).
- **#64** — generated the sequence-only corpus
  (`experiments/exp64_data_contacts_v1_sequence_only/`): 60,004,535 docs /
  ~32.98B tokens from UniRef50, each the **contacts-v1 sequence section only**
  (same `<pX> <AA>` wrap-around indexing, `<n-term>`/`<c-term>` markers,
  shuffled statements) under a new doc-type token `<contacts-v1.sequence_only>`
  (id 2845) and with no structure section. A **unified 2846-token tokenizer**
  (contacts-v1's 2845 + that one appended token, all old ids unchanged) lets the
  two corpora mix under one vocabulary.

We **keep** the distinct `<contacts-v1.sequence_only>` doc-type token (issue #87
decision): the model sees the two kinds as related-but-distinct document types.

## Approach

One epoch over a **token-proportional mixture** of the two corpora, trained from
scratch (fresh init).

- **Train script:** `train_combined_contacts_v1.py`
- **Shared building blocks:** `contacts_v1_train_common.py` (forked from exp85)
- **Export:** `export_combined_contacts_v1.py`

### Data

| component (W&B eval tag) | corpus | role | tokens | cache |
|---|---|---|---|---|
| `contacts-v1` | with-structure | train | ~4.7B | **reused** exp67 `contacts-v1-663ba6` |
| `contacts-v1-val` | with-structure | val | ~165M | **reused** exp67 `contacts-v1-val-92827b` |
| `sequence-only` | sequence-only | train | ~32.65B | fresh `contacts-v1-sequence-only-61cbeb` |
| `sequence-only-val` | sequence-only | val | ~165M | fresh `contacts-v1-sequence-only-val-e71ffa` |

- **Tokenizers.** contacts-v1 tokenize steps keep exp67's exact id
  (`timodonnell/contacts-v1-tokenizer`) + step names + GCS paths so marin
  resolves the **existing** caches (verified: dry-run resolves `…-663ba6` /
  `…-val-92827b`, both `SUCCESS`). The sequence-only steps and the **training /
  export tokenizer** use the unified `timodonnell/contacts-v1-unified-tokenizer`
  (2846 vocab). The reused contacts-v1 caches hold only ids 0–2844 (< 2846), so
  they drop into the 2846-vocab model unchanged.
- **GCS working copies** (region-local; HF buckets are not levanter-addressable):
  - contacts-v1: `…/exp53_contacts_v1_5x/documents/{train,val}/*.parquet`
  - sequence-only: `…/exp64_contacts_v1_sequence_only/documents/{train,val}/*.parquet`
    (uploaded from exp64's local staging as part of this experiment).
- **Mixture weights ∝ train tokens** ⇒ `contacts-v1` ≈ 0.126, `sequence-only` ≈
  0.874, so each batch is **~87% sequence-only** ("most of each batch will tend
  to be the sequence-only data"). The train stream is fully **shuffled** (Feistel
  permutation, `data_seed=0`) — both corpora are physically ordered (contacts-v1
  round-descending; sequence-only length-banded).
- **Per-type eval losses reported separately.** The two validation corpora are
  distinct components; levanter's `tagged_eval_sets` tags each by name ⇒ W&B
  logs `eval/contacts-v1-val/loss` and `eval/sequence-only-val/loss` as
  independent series. Each is evaluated over its full val split.

### Recipe (scaled #67)

v5p-32 (4× #67's v5p-8 — the v5p-8 preemptible pool thrashed for #85), so:

- **batch 512** (4×; per-chip batch stays 32 ⇒ same memory, no OOM)
- **peak LR 7.0e-4** = 3.5e-4 × √(512/128) (standard LR-vs-batch sqrt scaling)
- **seq 8192**, unmasked next-token loss, `pack=True`, cosine + 0.1 warmup
- **~8,909 steps** = one epoch over ~37.36B combined train tokens at 512×8192 =
  4.19M tok/step (mixture sampling is with replacement, so "one epoch" is in
  expectation — each corpus drawn ∝ its size over this many steps).

Then export the final checkpoint to HF (unified tokenizer co-located) and run
the **exp82** contact-prediction harness head-to-head vs #67/#85.

## Success criteria

- We trained on the combined dataset for a single epoch **without diverging**
  (issue's literal bar).
- Per-type eval losses logged separately on W&B.
- **Hypothesis test:** the with-structure eval (`eval/contacts-v1-val/loss`) and
  the exp82 contact-recapitulation numbers beat the structure-only #67/#85
  baselines.

## Launch

```bash
cd experiments/exp87_models_combined_contacts_v1_sequence_only
uv venv && uv sync --extra tpu

# 1) Stage sequence-only parquet to GCS us-east5 (done as part of setup):
#    gsutil -m cp ~/exp64_out/{train,val}/*.parquet \
#      gs://marin-us-east5/protein-structure/MarinFold/exp64_contacts_v1_sequence_only/documents/{train,val}/

# 2) Train (one epoch). WANDB_API_KEY must be in the launching env — it's
#    forwarded into the pod's env_vars by build_train_step.
WANDB_API_KEY=<key> uv run iris --cluster marin job run --no-wait \
    --enable-extra-resources --memory=16GB --disk=16GB --cpu=1 \
    --extra=tpu --zone=us-east5-a \
    -- python -m train_combined_contacts_v1

# The first run also kicks off the fresh sequence-only tokenize (~32.65B tokens)
# as a CPU dependency; the contacts-v1 caches are reused (skipped).

# 3) After step-8908 lands, fill the real -<wandb-runid> suffix + final step into
#    export_combined_contacts_v1.py, then:
uv run iris --cluster marin job run --no-wait --enable-extra-resources \
    --memory=32GB --disk=16GB --cpu=4 -- python -m export_combined_contacts_v1
```

## Status

Code complete and validated locally (step graph constructs; dry-run confirms
contacts-v1 cache reuse and the new sequence-only tokenize steps). Pending:
sequence-only GCS upload completion, then launch.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
