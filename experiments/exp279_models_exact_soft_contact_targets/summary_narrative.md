# Exact soft contact targets — experiment 279

## Question and method

Does averaging over valid contact serializations improve contact prediction?

Use the current decontaminated exp232 recipe for two fresh, matched arms.
Soften contact endpoints with exact generator probabilities.
Keep sequence statements, inputs, positions, attention and packing unchanged.

## Implementation checks

45 CPU tests pass; the full-size accelerator loss/gradient test passes.
An independent oracle checks 8192 positions, 2048 hidden, 2845 vocabulary.
Two virtual CPU devices pass accumulation, resume and SkipStep migration tests.
Both arms train for three GPU steps, save native state, validate and export.
Ordinary MarinFold inference checks the exported model and tokenizer.

## What these checks do not show

There is no accuracy result yet. No full 24-layer production model was trained.
The live corpora were not independently re-audited in this implementation task.
Multi-host GPU/TPU behavior and production memory/throughput still need a pilot.
A successful synthetic smoke is not evidence for the scientific hypothesis.

## Readout

Compare natural-protein eval-val R-precision at matched exposure and compute.
Use identical rollout budgets and paired protein-level uncertainty.
Keep ordinary validation CE comparable; report soft CE with entropy and KL.
Confirm the fresh CE baseline and repeat promising results with another seed.
