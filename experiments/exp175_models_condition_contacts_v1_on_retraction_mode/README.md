---
marinfold_experiment:
  issue: 175
  title: "exp: condition contacts-v1 on retraction mode with a <contacts-v1.backtracking> doc-type token"
  kind: models
  branch: exp175-backtracking-mode-token
---

# exp: condition contacts-v1 on retraction mode with a `<contacts-v1.backtracking>` doc-type token

**Issue:** [#175](https://github.com/Open-Athena/MarinFold/issues/175) · **Kind:** `models` · **Branch:** `exp175-backtracking-mode-token` (stacked on #160)

## Question

Does telling the model **at token 0** whether it may retract recover the contact-prediction accuracy [#160](https://github.com/Open-Athena/MarinFold/issues/160) lost, and sharpen the retraction behaviour it learned?

## Hypothesis

**#160 trained on a mixture the model could not condition on.** Both halves of its corpus begin with the identical prefix:

```
backtracking : <contacts-v1> <begin_sequence> <p714>  ...
clean        : <contacts-v1> <begin_sequence> <p1021> ...
```

Worse, **208,363 of the 1,023,997 backtracking documents (20.4%) contain no `<retract>` at all** — measured on the published corpus, not estimated — so a fifth of that half is indistinguishable from clean data in prefix *and* body.

A model in that position must marginalise over "may I take this back later?" at every step. Three of #160's measurements are what that predicts:

| #160 measurement | value |
|---|---|
| rollouts containing a retraction | 43.0% — it *samples* the mode rather than being told |
| enrichment vs its ceiling | 1.134x of 1.26x = **52% of headroom** |
| **emission-quality regression** (retraction ignored) | **−0.0251** R-precision |

That last number is the interesting one, and it is the one this experiment attacks. In retraction mode the optimal emission policy is *more speculative* — you can walk a guess back. With no marker that speculativeness has nowhere to live but the shared policy, so it leaks into clean generation. **A model that cannot tell which mode it is in has to hedge in both.**

## Approach

### Format (`marinfold`, done)

`<contacts-v1.backtracking>` — a variant doc type swapped in as token 0 by `GenerationConfig(backtracking=True)`, following the existing `<contacts-v1.sequence_only>` idiom. Not a new statement type: the statements, sections and fold are untouched.

**Append-only, verified against the tokenizer #160 actually trained under:** 0 id mismatches across all 3,849 pre-existing tokens, `<retract>` still 3848, `<crop>` still 3847, new token at 3849.

⚠️ **The hazard worth naming.** Both coordinate supersets build their inherited block by *filtering* contacts-v1's trailing tokens out and re-appending them at the end. A new trailing token that is not added to that filter lands **inside** the inherited block and shoves the whole xyz/crop block up by one id — silently breaking every published crops/ccoord checkpoint and desyncing the two formats' xyz ids. Both filters are extended, and a test pins `<xyz-000>` at 2847.

### Corpus (`mark_corpus.py`, done)

A **one-token rewrite** of #160's mix — no regeneration. The marker is determined by which generator produced the document, so it is a string operation, not a re-run of the model-in-the-loop job. Runs on a marin CPU pod (GCS → GCS, zero workstation uplink).

**Marked by generator, not by content.** The silent 20.4% keep the marker: they teach the honest conditional *"in this mode, sometimes nothing needs retracting"*. Marking by content would instead teach *"this token implies a retraction follows"* — a different target that destroys the token's use as a mode switch, because at inference we want to enter the mode without promising the model it must use it.

Verified on the output: clean half byte-identical, backtracking tails byte-identical, `live_contacts` unchanged on every sampled document, 0 UNKs and identical token counts under the 3850-token tokenizer.

| | |
|---|---|
| documents | 2,047,994 |
| marked `<contacts-v1.backtracking>` | 1,023,997 |
| ...of which contain no `<retract>` | **208,363 (20.4%)** |
| clean `<contacts-v1>` | 1,023,997 |

### Training

**#160's recipe, unchanged** — full fine-tune of exp120, lr 3e-4, wd 0.2, bs 128, seq 8192, 1-epoch cosine, v5p-32 — so the *only* difference between the two runs is the marker. `dispatch_train.py` and `resize_init_vocab.py` are reused from #160 rather than copied: both are fully parameterized, and reusing them is what makes "same recipe" checkable rather than asserted.

Needs one more +1 offline vocab resize (2845 → 3850) of the **same exp120 base** #160 started from; levanter does not resize on warm start.

### Eval

With a marker the retraction on/off comparison becomes a **generation-time** experiment on one checkpoint, which is strictly more informative than #160's readout-time ablation:

- prompt `<contacts-v1>` → clean mode
- prompt `<contacts-v1.backtracking>` → retraction mode

on the same 554-protein set, through #160's `score_backtracking_worker.py` and exp89 `compute_metrics`.

## Success criteria

- **Format is append-only** — asserted in tests, not assumed. ✅
- **The marker is obeyed** — retract rate under the backtracking marker clearly above the rate under `<contacts-v1>` (#160's unconditioned model: 43%).
- **Emission cost recovered** — clean-mode R-precision closes most of #160's −0.0251 against `exp120-base`.
- **Retraction sharpens** — enrichment beyond #160's 52% of headroom.
- **Headline** — best-of-both-modes beats `exp120-base` on the #89 benchmark at matched token compute. The bar #160 did not clear.

## Results

_(Training pending.)_

## Conclusion

_(Fill in after the run.)_
