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

## The decontaminated checkpoint

#232's `m2-p06` — the best model trained on corpora with every FoldBench homolog removed — was
CoreWeave-only until now; `publish_exp232_m2_p06.py` put both its finals on the public bucket, so
the notebook can fold with them and not just read their scores: the sweep final as
`contacts-v1-exp232-m2-p06-1.5B`, and #232's later step-363000 continuation of the same point as
`contacts-v1-exp232-m2-p06-train-1.5B`. The step-363000 checkpoint is the better model
(0.6051 against 0.5916 R-precision on the legacy 554) and is what every figure here is drawn
from.

Paired over 314 natural monomers it trails the contaminated default by **0.074**
[0.062, 0.086], and is ahead on 13 % of proteins. The gap looks smallest on the 14 proteins with
no training homolog (−0.009) and the 19 viral ones (−0.029) — but those are far harder for both
models, and identity does not rank-order the gap among the 300 proteins that have a homolog
(Spearman +0.001). m2-p06 clears the seq-KNN null over its own corpus by +0.112.

## Scoring the newer checkpoint

#232 evaluated step-363000 on `eval-val` and `eval-denovo` and deliberately left `eval-test`
unscored, so `score_foldbench_rollouts.py` scored all 333 FoldBench monomers with it on 8xA100 —
exp82's recipe, #89's metrics, the dense matrices kept because Helico conditions on them.
**0.5557 / 0.5681 / 0.6123** on eval-val / eval-test / eval-denovo, against the sweep final's
0.520 / 0.538 / 0.591.

The control that makes those numbers usable beside baselines nobody re-ran: the same pipeline on
the checkpoint #245 *did* publish scores 0.5240 against their 0.5198 over the same 97 proteins
(r = 0.995) — an 0.004 offset against a 0.032 change.

## What it is not

A producer of eval numbers of record. It runs under transformers rather than vLLM and uses
the packaged rollout's pairwise tie-break, both of which move a per-protein score by more
than the aggregate noise floor. Anything worth citing goes through #245's harness.
