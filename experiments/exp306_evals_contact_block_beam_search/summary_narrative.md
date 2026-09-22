# Summary slides — exp: contact-block beam search for fold discovery and eval-val precision

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Does short, contact-aligned beam search improve contacts-v1 decoding? Specifically, after the model decides to emit `<contact>`, search jointly over its two position tokens before committing the completed `<contact> <pos1> <pos2>` statement. Measure ordinary contact accuracy and whether this exposes both fold-specific modes in the fold-switching testbed.

## Why

Jointly scoring both positions may reject a plausible first position whose best second position is poor. The same beam restriction may also collapse rare alternate contact-map modes, so accuracy and fold diversity must be measured together.

## Development choice

On the 15 primary development fold-switching proteins, width 4 and width 8 each gave 1/15 dual-mode oracle pools and 0/15 dual-mode blind shortlists; iid100 gave 1/15 for both. Width 4 took 6.7 times iid H100 generation time, versus 12.3 times for width 8, and had less loss of minority-mode enrichment. Width 4 is frozen for the held-out run. Full eval-val accuracy and held-out fold-switching results are pending.
