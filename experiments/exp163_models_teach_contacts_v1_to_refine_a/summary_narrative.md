# Summary slides — exp: teach contacts-v1 to refine a set of candidate rollouts into a better contact set

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can we teach contacts-v1 to take **K candidate rollouts** (its own samples for a protein, of varying quality) and emit a contact set that beats (a) any single rollout, (b) training-free consensus voting over the same K, and (c) its own one-shot calibrated prediction — i.e. learn an **in-context aggregation / refinement operator** over a collection of candidate contact maps?

## Why

**Two measured facts frame the bet.**

*(Step 1)* The base model's per-pair calibrated matrix and a consensus vote over 16 rollouts are a **dead tie** on R-precision (0.221 vs 0.224) — voting is just a Monte-Carlo estimate of the same marginal, so consensus adds nothing the base model doesn't already output. A refiner that learns only consensus is worthless.

*(Step 2 — the decisive probe)* Yet the joint/structural signal the marginal misses is **real and large**: conditioning the base model (zero-shot) on 50% of a protein's *true* contacts lifts R-precision on the *remaining* contacts from 0.145 to **0.556** (+0.41; ΔAUC +0.10; better on 100% of proteins). Knowing part of a real contact map makes the rest highly predictable — the mechanism the refiner would exploit is present and strong, before any training.

**But that signal is precision-gated, which is exactly why training (not prompting) is required.** Conditioning the base model on a *noisy* candidate rollout (~13% precision) instead *hurts* — R drops 0.179 → 0.092 (worse on 91% of proteins) — because the base model was trained on `<begin_statements>` sections that contain only true contacts, so it trusts its context as ground truth and lets ~87% false contacts corrupt its structural prior.

So the refiner's job is precise: **learn that `<begin_candidate>` contacts are noisy hypotheses, not truth — identify the trustworthy ones (contacts recurring across the K candidate blocks are higher-precision, a signal the model can see directly) and use them to trigger the strong joint-completion the oracle probe demonstrated, climbing from ~0.22 toward the 0.556 ceiling.** GT supervision over K *separate* candidate blocks teaches exactly this discrimination; the distinct `<begin_candidate>` marker (never seen by the base model) lets it treat candidates differently from true `<begin_statements>` contacts.

Training uses a **variable candidate count K ∈ {0,…,Kmax}** (K=0 = a plain contacts-v1 doc): the model must produce GT with and without candidates — guarding the twin failure modes of *ignoring* candidates (collapse to base ~0.22) and *blindly trusting* them (the zero-shot noisy poisoning above).

## Results so far

_(Fill in as results come in.)_
