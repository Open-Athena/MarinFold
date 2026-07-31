# Summary slides — exp: run contacts-v1 parameter scaling sweep

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Aggregate finished contacts-v1 validation-loss runs from #75, #117, and #146
into one normalized, auditable scaling dataset.

## Why

The sweeps span model sizes and training budgets but use different metric keys,
tag schemas, and sweep subversion conventions. A clean table is the prerequisite
for comparing parameter and token scaling without mixing obsolete runs.

## Results so far

The first fetch contains 129 finished latest-subversion runs: 63 from #75, 50
from #117, and 16 from #146. Every row has the requested parameters, tokens,
epochs, weight decay, learning rate, batch size, and validation loss. The
fetcher records the exact loss key and provenance of each normalized field.

The first view plots every run against epoch count. Rings mark the lowest loss
within each model-size/epoch group; issue-specific marker shapes keep the three
sweeps distinguishable. Losses above 3.2 remain represented at the upper edge
with small upward carets.
