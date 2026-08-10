# Summary slides — exp: condition contacts-v1 on retraction mode with a `<contacts-v1.backtracking>` doc-type token

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Does giving the model an explicit **document-type marker for retraction mode** — so it knows at token 0 whether it is generating a document that may take contacts back — recover the contact-prediction accuracy that [#160](https://github.com/Open-Athena/MarinFold/issues/160)'s 50:50 fine-tune lost, and sharpen the retraction behaviour it learned?

## Why

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

## Results so far

_(Fill in as results come in.)_
