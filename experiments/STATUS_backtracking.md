# Backtracking series — status

**Issues:** [#158](https://github.com/Open-Athena/MarinFold/issues/158) format · [#159](https://github.com/Open-Athena/MarinFold/issues/159) corpus · [#160](https://github.com/Open-Athena/MarinFold/issues/160) training · [#175](https://github.com/Open-Athena/MarinFold/issues/175) mode marker
**PRs:** [#161](https://github.com/Open-Athena/MarinFold/pull/161) → [#171](https://github.com/Open-Athena/MarinFold/pull/171) → [#172](https://github.com/Open-Athena/MarinFold/pull/172) (stacked); exp175 branch pushed, PR not yet opened

## The one-paragraph version

We built a `<retract>` statement for contacts-v1 (#158), a corpus of documents that retract their own mistakes (#159), and trained on it (#160). Retraction turned out to be **discriminative but expensive**: the model uses it, retracts things that really are wrong 90% of the time, and reproduces the corpus's retraction *timing* almost exactly — but folded contact prediction got **worse**. #175 added a mode marker to test whether the model was hedging across two indistinguishable modes; it was, and clean mode recovered most of the loss. Then, inspecting a training document, we found **the corpus itself was broken**: its ground-truth flush emitted contacts in sorted order, ~80% of every document, and the model learned the sort rather than the behaviour. So the #160/#175 accuracy conclusions are void. We are now regenerating.

## Results that stand

### The format and the marker work

| | |
|---|---|
| `<retract>` fold, vocab append-only | ✅ 0 id mismatches on 3,849 pre-existing tokens |
| `<contacts-v1.backtracking>` marker | ✅ **0 vs 45 retracts/rollout** on the same checkpoint, prompt-selected |
| Test suite | 368 marinfold + 20 engine/adapter |

### Retraction is discriminative (#160)

Enrichment **1.134x** [1.111, 1.161], CI excludes 1.0. P(FP | retracted) = 0.902 vs a base rate of 0.796.

Raw enrichment is **not** comparable to the corpus's 5.85x — enrichment is bounded by `1/P(FP)`, so the ceilings are 1.26x and 6.02x. Normalised: **52% of headroom vs the corpus's 97%**.

Retraction **timing** transferred almost exactly: delay mean 19.0 / median 9 statements, 0.1% immediate, against the corpus's 18.0 / 9 / 0.2%.

### The mode marker recovers most of the emission cost (#175)

| arm | Δ R-precision vs exp120-base |
|---|---|
| #160 unconditioned, retraction ignored | −0.0251 |
| **#175 clean mode** | **−0.0068** ±0.0035 |
| #175 retraction mode | −0.0414 ±0.0053 |

73% of the regression recovered by simply telling the model which mode it is in.

## Results that are VOID

Everything about **whether backtracking helps or hurts contact prediction**, in both #160 and #175.

`backtrack_engine.py:315` appended missing ground-truth contacts via `sorted()`. That block is ~80% of every backtracking document:

| | sortedness (adjacent pairs in order) | tails 100% sorted |
|---|---|---|
| corpus, backtracking half | 0.869 | 87% |
| corpus, clean half | 0.500 | 0% |
| **model, retraction mode** | **0.851** | **83%** |
| model, clean mode | 0.500 | 0% |

The inference recipe votes across 100 rollouts. A model doing a sorted sweep emits nearly the same set every time, so the vote signal collapses — consistent with every accuracy number we measured.

## Corpus regeneration — where we are now

`policy.flush` now takes `sorted` / `shuffled` / `none`, and `policy.force_true_prob` forces a fraction of contact steps to draw from ground truth, sampled by the model's own score.

### Measured on 300-document pilots

| | published | **shuffled** | no-flush p=0 | **no-flush p=0.68** |
|---|---|---|---|---|
| sortedness | 0.869 | **0.497** | — | **0.502** |
| contacts/doc | 186.0 | 129.9 | 44.5 | 42.6 |
| retracts/doc | 33.4 | 42.3 | 29.8 | 8.0 |
| **recall vs GT** | 1.000 | **1.000** | 0.202 | **0.419** |
| FPs emitted | — | — | ~30 | **7.4 (100% caught)** |
| true contacts wrongly retracted | 0 | 0 | 0 | **0** |

### ✅ DONE: shuffled corpus at scale

**65/65 workers succeeded, 4,096 parts, 2.75 GB** — the ~1.02M-document corpus. This is a drop-in replacement for the published one: same generation, ordering artifact removed, recall still 1.000.

### The p=0.68 result contradicted my prediction

I predicted ~139 contacts/doc, ~95 forced, recall ≈ 1.0. **Actual: 42.6 contacts, 29.3 forced, recall 0.419.**

My model assumed free draws would stay at 44.5 and forcing would add on top, since only a free draw can stop the loop. Instead **free draws collapsed from 44.5 to 13.3**. Feeding the model true contacts makes the document *look more complete*, so it emits `<end>` much sooner. Forcing displaced free draws rather than adding to them.

What p=0.68 did achieve, and it is not nothing:

- **recall doubled**, 0.202 → 0.419
- **documents became near-perfectly precise**: 7.4 false positives emitted, 100% caught by the trigger, final live set essentially all true
- **the safety check held**: `tp_retracted_by_trigger = 0`, so score-weighted forcing is not planting contacts the model disbelieves

But it did **not** lengthen documents (42.6 vs 44.5 at p=0) and does not reach GT coverage.

## Open decisions

1. **Which corpus feeds the next training run?** The shuffled 1M corpus is ready and is the clean fix of the reported bug. The forced-true corpus is a different object — high precision, 42% recall, much shorter — and would need a higher p or `<end>` suppression to reach coverage.
2. **Does raising p help?** Since forcing shortens the free-draw budget, p and recall are coupled non-linearly. p=0.9 is one pilot away.
3. **#160/#175 need re-running** against whichever corpus wins, to get an uncontaminated answer on backtracking's cost.
