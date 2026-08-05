# Backtracking series — status

**Issues:** [#158](https://github.com/Open-Athena/MarinFold/issues/158) format · [#159](https://github.com/Open-Athena/MarinFold/issues/159) corpus · [#160](https://github.com/Open-Athena/MarinFold/issues/160) training · [#175](https://github.com/Open-Athena/MarinFold/issues/175) mode marker
**PRs:** [#161](https://github.com/Open-Athena/MarinFold/pull/161) → [#171](https://github.com/Open-Athena/MarinFold/pull/171) → [#172](https://github.com/Open-Athena/MarinFold/pull/172) → [#176](https://github.com/Open-Athena/MarinFold/pull/176) (stacked)
**Model:** `hf://buckets/open-athena/MarinFold/checkpoints/exp175-cv1-1_5b-mode50-v2-lr3e-4-e1-cos/hf/step-2070` · [playground notebook](../notebooks/retraction_mode_playground.ipynb)

## The one-paragraph version

We built a `<retract>` statement for contacts-v1 (#158), a corpus of documents
that retract their own mistakes (#159), trained on it (#160), then added a
`<contacts-v1.backtracking>` doc-type marker so the model could be *told* which
mode it was in (#175). Along the way we found the corpus was contaminated — its
ground-truth flush emitted contacts in sorted order, ~80% of every document —
voided the accuracy results, regenerated 1M documents with a shuffled flush, and
re-ran everything. **Final answer: retraction is real but modest, and it does
not pay for itself.** The model retracts when told to and retracts things that
really are wrong more often than chance, but captures only 42% of the available
discrimination, and both modes score *below* the `exp120` model they were
fine-tuned from on the #89 benchmark.

## The headline table

R-precision (all-range), paired per-protein vs `exp120-base` on 554 proteins:

| model | Δ R-precision | retracts/rollout | tokens/rollout |
|---|---|---|---|
| `exp120-base` | — (0.4354) | 0 | 502 |
| #160, unconditioned | −0.0199 ±0.0040 | 24.0 | — |
| #175 v1, clean mode | −0.0068 ±0.0035 | 0.0 | 512 |
| #175 v1, retraction mode | −0.0414 ±0.0053 | 45.0 | 712 |
| **#175 v2, clean mode** | **−0.0063 ±0.0042** | 0.1 | 510 |
| **#175 v2, retraction mode** | **−0.0153 ±0.0043** | 42.0 | 757 |

`v1` trained on the corpus as published; `v2` on the regenerated one. Everything
else identical.

## What is established

### The format and the marker work

| | |
|---|---|
| `<retract>` fold, vocab append-only | ✅ 0 id mismatches on 3,849 pre-existing tokens |
| `<contacts-v1.backtracking>` marker | ✅ **0.1 vs 42.0 retracts/rollout** on one checkpoint, prompt-selected |
| Coordinate-superset id freeze | ✅ `<xyz-000>` pinned at 2847 by test |
| Test suite | 368 marinfold + 20 engine/adapter |

Conditioning is cheap and it works: one appended token, no format change, no
regeneration, and the model splits into two behaviours it previously had to
average over.

### Retraction is discriminative, at 42% of ceiling

Enrichment **1.101x** [1.08, 1.12] — CI excludes 1.0. P(FP | retracted) = 0.888
against a base rate of 0.807.

Raw enrichment is **not** comparable to the #159 corpus's 5.85x. Enrichment is
bounded by `1/P(FP)`, so the ceilings are 1.24x and 6.02x; normalised, the model
captures **42% of headroom against the corpus's 97%**. Always report the
normalised number — the raw one mostly measures how wrong the rollouts are.

Retraction *timing* transferred almost exactly: delay mean 19.6 / median 9
statements, 0.1% immediate, against the corpus's 17.9 / 9 / 0.2%. The model
learned to reach back, not just to undo its last move.

### The corpus artifact was real, is fixed, and cost accuracy — not discrimination

`backtrack_engine.py:315` appended still-missing ground-truth contacts via
`sorted()`. Sortedness of contact-emission order (0.5 = random):

| | v1 corpus / model | v2 corpus / model |
|---|---|---|
| corpus, backtracking half | 0.869 | 0.497 |
| **model, retraction mode** | **0.833** | **0.499** |
| model, clean mode | 0.501 | 0.500 |

The fix propagated end-to-end. What it bought:

| | v1 → v2 |
|---|---|
| retraction-mode Δ R-precision | −0.0414 → **−0.0153** (63% of the gap) |
| clean-mode Δ R-precision | −0.0068 → −0.0063 (nothing) |
| enrichment, fraction of headroom | 43% → 42% (nothing) |

A clean dissociation. The sorted sweep was collapsing the 100-rollout vote — an
*accuracy* mechanism — and was never what taught the model to retract.

### The mode-hedging hypothesis held

#160's unconditioned model scored −0.0199. Its two modes, once separable, sit on
either side: −0.0063 and −0.0153. A model that cannot tell which mode it is in
behaves like a mixture of both, which is what #175 predicted and what the
numbers show.

## What is not established

**Backtracking does not beat the model it was fine-tuned from.** Both modes lose
on #89, clean mode by a small but CI-excludes-zero margin, retraction mode by
2.4x that while spending 51% more tokens. Every variant tried — unconditioned,
mode-marked, contaminated corpus, clean corpus — lands below `exp120-base`.

**The 42% transfer gap is the open problem.** It survived a mode marker *and* a
corpus regeneration. The corpus reaches 97% because a ground-truth flush tells it
which contacts are wrong; the model gets 42% from the same traces. That gap is
between "can be shown the answer" and "can tell from the inside", and no
supervised variant tried so far moves it.

## The corpus generator, as it now stands

`policy.flush` takes `sorted` / `shuffled` / `none` (default `none`);
`policy.force_true_prob` forces a fraction of contact steps to draw from ground
truth, sampled by the model's own score. Measured on 300-document pilots:

| | published | **shuffled** | no-flush p=0 | no-flush p=0.68 |
|---|---|---|---|---|
| sortedness | 0.869 | **0.497** | — | 0.502 |
| contacts/doc | 186.0 | 129.9 | 44.5 | 42.6 |
| retracts/doc | 33.4 | 42.3 | 29.8 | 8.0 |
| **recall vs GT** | 1.000 | **1.000** | 0.202 | 0.419 |
| FPs emitted | — | — | ~30 | 7.4 (100% caught) |
| true contacts wrongly retracted | 0 | 0 | 0 | 0 |

**Shuffled is what shipped** — 65/65 workers, 4,096 parts, 2.75 GB, ~1.02M
documents, a drop-in replacement for the published corpus.

**The forced-true idea contradicted its own prediction.** Forcing was expected to
*add* true contacts on top of the free draws (predicted ~139 contacts/doc,
recall ≈ 1.0). Instead free draws collapsed from 44.5 to 13.3: feeding the model
true contacts makes the document look more complete, so it emits `<end>` sooner.
Forcing **displaced** free draws rather than adding to them — actual 42.6
contacts, recall 0.419. It did double recall and made documents near-perfectly
precise, and the safety check held (`tp_retracted_by_trigger = 0`), but it needs
`<end>` suppression before it could reach coverage, and that is untested at
scale.

## Where to go next

1. **Not another supervised corpus variant.** Two of them (mode marker, corpus
   fix) moved accuracy and left discrimination at 42%. Imitating an oracle's
   judgement gets a model partway to having its own.
2. **Score the outcome instead.** Let the model retract and reward the resulting
   contact set — #98 already collected the rollouts an RFT setup would need.
3. **Untested, cheap, and would sharpen the story:** a 100:0 clean-only control
   at the same budget, which separates "backtracking hurt" from "another epoch
   of fine-tuning on ESM-Atlas hurt". Nothing in this series distinguishes them.
