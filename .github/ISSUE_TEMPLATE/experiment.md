---
name: Experiment
about: Propose a research experiment. The dir lives in experiments/; results come back here.
title: "exp: "
labels: ["experiment"]
assignees: []
---

<!--
This template preregisters an experiment. Fill out every section. An
experiment is a directory at experiments/exp<issue#>_<kind>_<slug>/ with a
prose README (question, hypothesis, approach, results) and any
launchable .py files. Library code lives in models/, evals/, data/,
or document_structures/ at the repo root.

See experiments/README.md for the full workflow.
-->

## Kind

<!-- One of: models, evals, data, document_structures.
     - models: trains models
     - evals: runs evals on trained models
     - data: generates training/eval datasets
     - document_structures: defines a generate + evaluate interface for
       a specific protein-document format (overlap with data and evals)
-->

- Kind: <!-- e.g. models -->

## Question

<!-- What do you want to learn? One sentence. -->

## Hypothesis

<!-- What do you expect to see and why? This is what we're preregistering. -->

## Background

<!-- Prior runs, papers, conversations this builds on. Link issues/PRs. -->

## Approach

<!-- Outline what you will do. Bullet points are fine. Reference the
     launchable code (or note that it needs to be written). E.g.:

     - Train script: train.py (uses marinfold_models.defaults.default_train)
     - Eval: evaluate.py
     - Will reuse the protein-docs-tokenizer.
-->

## Compute estimate

<!-- Rough hardware, count, and wall-clock hours. -->

- Accelerator: <!-- e.g. TPU v5p-8 -->
- Count / slice: <!-- e.g. 1 slice -->
- Estimated wall hours: <!-- e.g. 12 -->

## Success criteria

<!-- How will we know if the hypothesis held? Concrete metrics +
     thresholds. Example:
       - eval/protein_dist/macro_loss < <baseline> by >= 0.05
       - eval/protein-docs-cd-val/loss does not regress
-->

## Baselines

<!-- Named prior runs or published numbers to compare against. -->

## Notes

<!-- Anything else the agent or a reviewer should know. -->
