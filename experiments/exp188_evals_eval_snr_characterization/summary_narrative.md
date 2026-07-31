# Summary slides — eval SNR characterization

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Estimate document-level standard error for contacts-v1 validation loss using a
Poisson bootstrap over validation documents.

## Why

Nearby checkpoint comparisons often differ by only ~0.001–0.01 nats. We need to
know whether those deltas are above the finite-validation-set noise floor before
using them as selection signals.

## Results so far

Experiment scaffolded. Shared bootstrap helper lives in PR #187; this experiment
will consume per-document `loss_sum` / `token_count` tables and report eval-loss
stderr.
