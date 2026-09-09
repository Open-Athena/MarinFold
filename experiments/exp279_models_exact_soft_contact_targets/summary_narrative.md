# Exact soft contact targets — experiment 279

## Question and method

Does averaging over valid contact serializations improve contact prediction?

Use the current decontaminated exp232 recipe for two fresh, matched arms.
Soften contact endpoints with exact generator probabilities.
Keep sequence statements, inputs, positions, attention and packing unchanged.

## Implementation checks

53 CPU tests pass; the full-size accelerator loss/gradient test passes.
An independent oracle checks 8192 positions, 2048 hidden, 2845 vocabulary.
Two virtual CPU devices pass accumulation, resume and SkipStep migration tests.
Both arms train for three GPU steps, save native state, validate and export.
Ordinary MarinFold inference checks the exported model and tokenizer.

## What these checks do not show

There is no accuracy result yet; the corpora were not independently re-audited.
A full 1.47B, 32-H100 pilot completed 32 updates at about 3.5 seconds/update.
It saved native state and HF weights/tokenizer, but exposed missing validation.
The cache configuration is corrected and tested with actual validation reads.
The corrected pilot passed: 32 updates, finite validation CE 6.3678, saved state.
The production driver continues that exact checkpoint on 32 H100s.
The soft arm is launched; matched CE and accuracy evaluation remain future work.

## Readout

Compare natural-protein eval-val R-precision at matched exposure and compute.
Use identical rollout budgets and paired protein-level uncertainty.
Keep ordinary validation CE comparable; report soft CE with entropy and KL.
Confirm the fresh CE baseline and repeat promising results with another seed.
