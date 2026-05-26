# Summary slides — Initial port of marin/protein-training-1b modeling experiments

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Bulk-port of the protein-docs modeling work that originated on the
`marin/protein-training-1b` branch into MarinFold. The entire family
(size sweep from 30M to 3B, distance-masked vs unmasked ablation,
all-doc-types mixture, continuation script, and matching HF export
scripts) is collected here as MarinFold's starting state. Each
`train_protein_<size>_<variant>.py` is one ExecutorStep graph,
dispatched to iris via Marin on TPU v5p.

## Why

Reproduce the prior research direction on the new infrastructure:
does training a small Llama on the `contacts-and-distances-v1-5x`
corpus produce a useful protein-structure LM? And what's the best
loss formulation + model scale?

Working hypothesis: distance-bin-only loss masking concentrates
gradient signal on the prediction of interest and beats unmasked
training at matched compute. Scaling should follow Pythia-like
curves at least up to 1.4B; beyond that under-training
(50K steps × batch 128) is the dominant constraint.

## Success criteria

Per size: training reaches the configured step count without
divergence; HF export produces a loadable checkpoint at the
expected path.

For the sweep: loss vs. params follows a Pythia-like curve on
`eval/protein-docs-cd-val/loss`.

For the loss-mask ablation: distance-masked beats unmasked on
`eval/protein_dist/macro_loss` at matched checkpoint steps.

## Results so far

To be filled in as runs complete on the new MarinFold
infrastructure. Pre-MarinFold results live in W&B under the `marin`
project (group `protein-training`); see README.md for the full table
of training scripts and baseline run names.

## Conclusion

Pending — this is the start state, not a result.
