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

## Success criteria

- **No regression** on #89 R-precision / AUC vs the matched-budget control.
- **Retraction is used** — non-trivial `<retract>` rate on held-out proteins.
- **Retraction is discriminative** — enrichment clearly above 1.0 (corpus reference: 2.73x). *The real pass/fail.*
- **Headline** — retraction-enabled inference beats resample+vote at matched token compute on long-range R-precision.

## Results

_(Diagnostics built and validated; training pending the #159 scale corpus.)_

## Conclusion

_(Fill in after training.)_
