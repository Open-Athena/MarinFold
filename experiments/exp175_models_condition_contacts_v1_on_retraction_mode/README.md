---
marinfold_experiment:
  issue: 175
  title: 'exp: condition contacts-v1 on retraction mode with a `<contacts-v1.backtracking>` doc-type token'
  kind: models
  branch: main
---

# exp: condition contacts-v1 on retraction mode with a `<contacts-v1.backtracking>` doc-type token

**Issue:** [#175](https://github.com/Open-Athena/MarinFold/issues/175) · **Kind:** `models` · **Branch:** `main`

## Question

Does giving the model an explicit **document-type marker for retraction mode** — so it knows at token 0 whether it is generating a document that may take contacts back — recover the contact-prediction accuracy that [#160](https://github.com/Open-Athena/MarinFold/issues/160)'s 50:50 fine-tune lost, and sharpen the retraction behaviour it learned?

## Hypothesis

**#160 trained on a mixture the model cannot condition on.** Both halves of its corpus begin with the identical prefix:

```
backtracking : <contacts-v1> <begin_sequence> <p714>  ...
clean        : <contacts-v1> <begin_sequence> <p1021> ...
```

There is no marker anywhere in the document. Worse, **20.1% of the #159 backtracking documents contain zero retractions**, so a fifth of that half is indistinguishable from clean data in prefix *and* body.

A model in that situation must marginalise over "am I in a mode where I can take this back later?" at every step. Three of #160's measurements are what that predicts:

- it retracts on only **43% of rollouts** — it is *sampling* the mode rather than being told it;
- its retraction enrichment reaches only **52% of the achievable headroom** (1.134x of a 1.26x ceiling), against the corpus's 97%;
- and, isolated by the readout ablation, **emission quality itself regressed −0.0251 R-precision** — before any retraction is honoured.

That last one is the interesting hypothesis. In retraction mode the optimal emission policy is *more speculative* — you can walk a guess back. With no marker, that speculativeness has nowhere to live except in the shared policy, so it leaks into clean generation. **A model that cannot tell which mode it is in has to hedge in both.**

Prediction: adding the marker recovers most of the −0.0251, and raises the retract rate and enrichment when the backtracking marker is supplied.

## Background

Depends on [#158](https://github.com/Open-Athena/MarinFold/issues/158) (`<retract>` + the fold), [#159](https://github.com/Open-Athena/MarinFold/issues/159) (the corpus), [#160](https://github.com/Open-Athena/MarinFold/issues/160) (the first training run and the evidence above).

#160's [ablation comment](https://github.com/Open-Athena/MarinFold/issues/160#issuecomment-5117903678) is the direct motivation: it decomposed the headline −0.0199 into a **−0.0251 emission-level cost** and a **+0.0052 gain from honouring retractions**, showing the mechanism pays for itself while the training run does not. This experiment attacks the −0.0251.

## Approach

### 1. Format: a new trailing doc-type token

`<contacts-v1.backtracking>`, following the existing `<contacts-v1.sequence_only>` idiom — a variant doc type swapped in as token 0 by `generate.py`, **not** a new statement type.

**The ordering is the whole safety argument.** `all_domain_tokens()` becomes:

```
[native, contacts-and-distances-v1 block, sequence_only, retract, backtracking]
```

appended **last**, so contacts-v1 goes 2847 → 2848 and every pre-existing id is unchanged.

⚠️ **The superset vocabs need the same treatment or they break.** `contacts_and_crops_v1` / `contacts_and_coordinates_v1` build `inherited_tokens()` by filtering `<retract>` out of contacts-v1's list and re-appending it last. A new trailing contacts-v1 token would otherwise land *inside* the inherited block, shoving the whole xyz/crop block up by one and desyncing the two coordinate formats — the exact hazard #158's `inherited_tokens()` docstring warns about. The new token must be added to that exclusion filter and re-appended last, taking the superset 3849 → 3850.

### 2. Corpus: a one-token rewrite, no regeneration

The marker is determined by **which generator produced the document**, not by its content, and every #159 document came from the backtracking engine. So this is a string rewrite over the published mix, not a re-run of the model-in-the-loop job.

**Mark by generator mode, not by content.** The silent 20.1% keep `<contacts-v1.backtracking>`: they teach the honest conditional *"in this mode, sometimes nothing needs retracting"*. Marking by content would instead teach *"this marker guarantees a retraction"* — a different and probably worse target, and one that would make the marker useless as a mode switch.

### 3. Train

Same recipe as #160 (full fine-tune of exp120, lr 3e-4, wd 0.2, bs 128, seq 8192, 1-epoch cosine) on the marked mix, under a 3850-token superset tokenizer. Needs another +1 offline vocab resize of the exp120 init checkpoint — levanter does not resize on warm start (#160 paid for that lesson).

### 4. Eval — this is where the marker earns its keep

With a marker, the retraction on/off comparison becomes a **generation-time** experiment on one checkpoint rather than a readout-time one:

- prompt with `<contacts-v1>` → clean mode,
- prompt with `<contacts-v1.backtracking>` → retraction mode,

on the same 554-protein eval set, scored through #160's `score_backtracking_worker.py` + exp89 `compute_metrics`. That answers directly whether entering retraction mode is worth it — the question #160 could only approach by re-reading fixed rollouts.

## Success criteria

- **Format is append-only** — every exp120 / #160 id unchanged, both superset vocabs still byte-stable in their xyz/crop block. Asserted in tests, not assumed.
- **The marker is obeyed** — retract rate under `<contacts-v1.backtracking>` clearly above the rate under `<contacts-v1>` (#160's unconditioned model sits at 43%).
- **Emission cost recovered** — clean-mode R-precision closes most of #160's −0.0251 against `exp120-base`.
- **Retraction sharpens** — enrichment captures more than #160's 52% of headroom in backtracking mode.
- **Headline** — best-of-both-modes beats `exp120-base` on the #89 benchmark at matched token compute. This is the bar #160 did not clear.

## Results

_(Fill in after the run completes.)_

## Conclusion

_(Fill in after results are in.)_
