# Summary slides — exp: an interactive evals-exploration notebook — per-protein contact maps and the predictor scoreboard

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

**What does the current contacts-v1 model actually get right and wrong, protein by protein — and can we look at that interactively, from a browser, without a cluster?**

Every eval number we publish today is an aggregate: a mean R-precision over an eval
set, sometimes split by designed / natural / viral. Those aggregates are the right
thing for tracking progress ([#180](https://github.com/Open-Athena/MarinFold/issues/180))
but they are a poor instrument for *understanding* the model. The questions that keep
coming up — which proteins does it fold and which does it lose, does the predicted
contact map look like a wrong fold or like noise, what changed between two
checkpoints on one protein — all need a per-protein view that nothing in the repo
currently offers in one place.

The published artifacts to answer this already exist and are all public
(anonymous read): [#245](https://github.com/Open-Athena/MarinFold/issues/245)'s
per-protein scores for 9 predictors, its eval-set annotation, ground truth for both
universes, and [#247](https://github.com/Open-Athena/MarinFold/issues/247)'s 75
per-protein features. What is missing is a place to put them together and a GPU path
to generate a contact map for an arbitrary protein under the settled inference recipe.

## Why

Not a hypothesis experiment — this is an instrument. The success criterion is
fidelity: a contact map and score produced in the notebook must reproduce the
published number for the same protein and checkpoint, so that anything read off
the notebook is on the same axis as everything we have already filed.

The one substantive prior: per-protein accuracy is *bimodal*, not a smooth
distribution — the model either finds roughly the right fold or produces a map with
no correct long-range structure — and looking at maps will make that visible in a
way the aggregate R-precision cannot.

## Results so far

`notebooks/evals_exploration.ipynb` is in: scoreboard and per-protein browser on CPU,
contact maps and two-checkpoint comparison on a GPU runtime. Everything is read from the
public bucket — no token, no cluster.

## It is calibrated

The scoreboard reproduces the published aggregates to the digit: the #199 cooldown pools to
**0.631** on legacy-554 (0.685 designed / 0.495 natural), and the #232 checkpoints come out
**0.520** / **0.473** on eval-val. Per protein at 100 rollouts, the notebook lands within
rollout noise of the published per-protein score — `1qys_A` 0.684–0.697 against 0.697,
`8ah9_A` 0.909 against 0.894, `7y5r_A` 0.825 against 0.835.

## What it is not

A producer of eval numbers of record. It runs under transformers rather than vLLM and uses
the packaged rollout's pairwise tie-break, both of which move a per-protein score by more
than the aggregate noise floor. Anything worth citing goes through #245's harness.
