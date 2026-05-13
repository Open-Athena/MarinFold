<!--
Experiment README template. Copy this file to
experiments/exp<N>_<kind>_<name>/README.md and fill in. Prose only —
launchable code lives as .py files alongside this README in the same
directory.

Generate programmatically:
    uv run marinfold scaffold --issue <N> --kind <models|evals|data|document_structures>
-->

---
marinfold_experiment:
  issue: 0
  title: "TEMPLATE: replace me"
  kind: models
  branch: main
---

# TEMPLATE: replace this line with the experiment title

**Issue:** #0 · **Kind:** `models` · **Branch:** `main`

## Question

One sentence — what do we want to learn?

## Hypothesis

What we expect to see, and why.

## Background

Prior runs, papers, conversations this builds on. Link issues/PRs.

## Approach

Bullet points are fine. Reference the launchable code (which lives in
this directory). E.g.:

- Train script: `train.py`
- Helper: imports from `marinfold_models.defaults`
- Compute: TPU v5p-8 × 1, ~12 wall hours, ~$X

## Success criteria

How will we know if the hypothesis held? Concrete metrics + thresholds.

## Results

Headline numbers + plots. Each plot has a source CSV in `data/`.

## Conclusion

Answer the question. A future reader should be able to get the answer
from this section alone; they can scroll up for methods.
