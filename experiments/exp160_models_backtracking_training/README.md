---
marinfold_experiment:
  issue: 160
  title: "exp: does training on backtracking traces let contacts-v1 self-correct at inference?"
  kind: models
  branch: exp160-backtracking-training
---

# exp: does training on backtracking traces let contacts-v1 self-correct at inference?

**Issue:** [#160](https://github.com/Open-Athena/MarinFold/issues/160) · **Kind:** `models` · **Branch:** `exp160-backtracking-training` (stacked on #159)

## Question

Does continue-training on the backtracking corpus (1) preserve standard contact-prediction accuracy, and (2) let retraction-enabled rollouts beat the settled resample+vote recipe at matched inference compute?

## Hypothesis

A model trained on self-correction traces can walk back a bad early commitment mid-rollout. **The bar is high:** #82's resample + pairwise-tiebreak already recovers much of what a single bad rollout loses, so backtracking must beat *that*, not greedy decoding. A null result — with the diagnostics below to explain it — is a clean answer either way.

## Background

Depends on #158 (`<retract>` + `read.py` fold) and #159 (the corpus). Base model / control: `contacts-v1-exp120-1.5B`. Reuses the #120 matched-budget harness and the #89 benchmark (`compute_metrics`).

## Approach

### Status: diagnostics built + smoke-tested. Training pending the scale corpus.

**`retraction_diagnostics.py` (done).** Pure, model-free; computes from a document's edit list + its GT pairs:

- **Is retraction discriminative?** — the experiment's real pass/fail. A 2x2 of {retracted, kept} x {false positive, true contact}, reported as `precision = P(FP | retracted)`, `recall = P(retracted | FP)`, and the headline **`enrichment = precision / P(FP)`** (1.0 = no signal; scale-free, so comparable across proteins with different FP base rates).
- **retract rate** — a model that never retracts has ignored the mechanism.
- **retraction distance** — if it only ever retracts the statement it just emitted, the long-delay signal did not transfer.
- **recovery** — after retracting, does it emit a *true* contact on one of the freed residues?

10 unit tests pin the metric definitions on hand-built edit lists (perfect discrimination -> enrichment = 1/base_rate; retracting at the base rate -> exactly 1.0; no-retraction -> NaN not 0; orientation canonicalisation; re-emission; malformed retracts).

**Smoke test on the #159 370-doc corpus** (`diagnose_corpus.py`) — the reference the trained model is measured against:

```
documents:            370 (97.3% contain a retraction)
mean contacts/doc:    83.5      mean retracts/doc: 31.6
FP base rate:         0.360  (of 30,004 emitted pairs)
retract precision:    0.983     retract recall: 1.000
ENRICHMENT:           2.73x     (ceiling 1/0.360 = 2.78x -> 98% of max)
true contacts retracted: 190    (the deliberate 5% noise retractions)
retract distance:     mean 22.1 / median 10 statements (0.1% immediate)
recovery rate:        0.688
```

Note `recall = 1.000` is **by construction** — the corpus engine's correctness flush retracts every surviving false positive. A trained model has no flush, so its recall will be lower; **precision and enrichment are the meaningful comparisons.**

### Remaining
1. **Training** — continue-train exp120 on the backtracking corpus, matched-budget vs a clean-corpus control, sweeping the clean:backtracking mixing ratio (100:0 control, 75:25, 50:50). Needs the scale corpus (#159's 1M run) — 370 docs is far too small to train a 1.5B. Train under the **superset tokenizer** (crops/ccoord vocab, which now carries `<retract>`) so the model is mixture-native.
2. **Standard eval** — #89 benchmark, retract-aware parsing: did retraction training cost anything?
3. **Decisive eval** — resample+vote baseline vs retraction rollouts vs the retract-probe, at matched **token** compute (retraction lengthens documents, so equal rollout *count* would hand the arm more compute).

### Vocab / checkpoint compatibility (verified)

The exp120 checkpoint predates two appended tokens. Checked directly against
its `tokenizer.json`:

| | |
|---|---|
| exp120 checkpoint vocab | 2,845 (`vocab_size` = 2845) |
| current contacts-v1 tokenizer | 2,847 |
| **id mismatches on the checkpoint's 2,845 tokens** | **0** |
| new tokens needing embedding rows | `<contacts-v1.sequence_only>` (2845), `<retract>` (2846) |

So continue-training is a **+2 embedding resize, not a remap** — every existing
embedding keeps its meaning. This is the append-only vocab discipline from #158
paying off; had `<retract>` been inserted into `NATIVE_TOKENS` instead, all
2,838 shared contacts-and-distances tokens would have shifted and the
checkpoint would have been unusable.

Note the published `timodonnell/contacts-v1-tokenizer` repo (used by the exp120
training path) is the **pre-retract** tokenizer. Training on the backtracking
corpus needs the tokenizer published *with the corpus*
(`.../contacts_v1_backtracking/tokenizer/`), or the crops/ccoord superset if
training a mixture.

## Setup decisions (settled 2026-07-27)

**Base model: `contacts-v1-exp120-1.5B`** (val 2.7213, ppl 15.20). Eric's #117
sweep has better numbers — best **2.7037** on `...wd0p2-bs256-europe-west4` —
but **every exp117 checkpoint has been deleted**: the 23 surviving GCS dirs are
crashed runs holding only `.executor_info`/`.executor_status`, and none are on
the HF bucket. exp120 is the best contacts-v1 model that still loads, and it is
also the model that generated the #159 corpus, so this fine-tunes a model on
corrections of its own mistakes. (Note the true #117 best is **bs256**, not the
bs128/2.7112 run exp137 cites.) *Caveat:* exp120 reports
`eval/contacts-v1-val-orig/loss` while #117 reports
`eval/tokenized/contacts-v1-val/loss` — probably the same split, unproven, so
treat the 0.018 gap as approximate.

**GPU, not TPU.** At decision time the marin TPU cluster was fully subscribed
(0 chips free, 39 jobs pending on "Insufficient TPUs") while `cw-rno2a` had
~239 free H100s. The usual objection — CoreWeave cannot read GCS — is handled
by staging (below); #159 already proved CoreWeave pulls model + corpus from the
HF bucket at 48-worker scale.

**Full fine-tune** (not LoRA), **50:50** backtracking:clean, **superset
tokenizer**.

### Verified before launch

| check | result |
|---|---|
| exp120 ids under the superset tokenizer | **0 mismatches** on its 2,845 tokens → +1004 embedding rows, not a remap |
| superset tokenizer on 200 real backtracking docs | exact token counts, **0 UNKs**, exact round-trip |
| mix composition | 2,047,994 docs, **exactly 50.0%** backtracking, 2.17B tokens |
| mix protein overlap between halves | **0** (asserted) |
| staged Levanter checkpoint in CoreWeave S3 | 13 objects, **17.66 GB**, matches source |

### Data / artifact locations

- mix: `hf://buckets/open-athena/MarinFold/data/document_structures/contacts_v1_backtracking_mix50/` (`train/` + `tokenizer/`)
- init checkpoint: `s3://marin-us-east-02a/protein-structure/MarinFold/exp160_backtracking_training/init/exp120-step-1005`

### Staging notes (cost several attempts)

The Levanter **training-state** checkpoint is required — the HF export is not
enough for `initialize_from_checkpoint_path`. Routing it through the HF bucket
**failed**: that uploader panics on multi-GB files (`is not fully completed:
2257162240/2261618688 bytes`) and this checkpoint has a 2.26 GB shard. It now
goes GCS → CoreWeave S3 directly from a GCS-local marin pod via boto3
multipart — one hop, and training reads CoreWeave-local storage. Credentials
are passed as job env vars only; nothing secret is committed.

### BLOCKER: `marinfold_models` is stale against current marin/levanter

The training launch reached the worker and then failed here:

```
File ".../marinfold_models/defaults.py", line 35
    from levanter.data.text import LmDataConfig
ImportError: cannot import name 'LmDataConfig' from 'levanter.data.text'
```

This is the documented **marin 0.2.57 / levanter 1.2** API move: `levanter.data.text`
and `levanter.optim` became lazy plugin registries with empty `__init__`s, so
imports must name the defining submodule:

| was | now |
|---|---|
| `levanter.data.text.{LmDataConfig, DatasetComponent, UrlDatasetSourceConfig}` | `levanter.data.text.datasets` |
| `levanter.data.text.{TextLmDatasetFormat, PrebuiltLmDatasetFormat}` | `levanter.data.text.formats` |
| `levanter.optim.{OptimizerConfig, AdamConfig}` | `levanter.optim.config` |

`train_backtracking.py` already uses the new paths; **`models/marinfold_models/defaults.py`
does not**. Fixing it is a small library change that unblocks any experiment
re-locking to a recent marin, not just this one.

A second, related constraint forced the env split already committed:
`marin-core` requires `transformers>=5.5.3` while `marinfold` pins `<5`, so the
**training** env carries `marinfold-models` (+ marin stack) and *not* `marinfold`.
Diagnostics that need `marinfold.read` (`diagnose_corpus.py`) run under exp159's
env instead.

**Status: everything else is staged and verified; training is blocked only on
this import fix.**

## Success criteria

- **No regression** on #89 R-precision / AUC vs the matched-budget control.
- **Retraction is used** — non-trivial `<retract>` rate on held-out proteins.
- **Retraction is discriminative** — enrichment clearly above 1.0 (corpus reference: 2.73x). *The real pass/fail.*
- **Headline** — retraction-enabled inference beats resample+vote at matched token compute on long-range R-precision.

## Results

_(Diagnostics built and validated; training pending the #159 scale corpus.)_

## Conclusion

_(Fill in after training.)_
