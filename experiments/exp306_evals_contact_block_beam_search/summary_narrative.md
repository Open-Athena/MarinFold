# Summary slides — exp: contact-block beam search for fold discovery and eval-val precision

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Does short, contact-aligned beam search improve contacts-v1 decoding? Specifically, after the model decides to emit `<contact>`, search jointly over its two position tokens before committing the completed `<contact> <pos1> <pos2>` statement. Measure ordinary contact accuracy and whether this exposes both fold-specific modes in the fold-switching testbed.

## Why

Jointly scoring both positions may reject a plausible first position whose best second position is poor. The same beam restriction may also collapse rare alternate contact-map modes, so accuracy and fold diversity must be measured together.

## Development choice

On the 15 primary development fold-switching proteins, width 4 and width 8 each gave 1/15 dual-mode oracle pools and 0/15 dual-mode blind shortlists; iid100 gave 1/15 for both. Width 4 took 6.7 times iid H100 generation time, versus 12.3 times for width 8, and had less loss of minority-mode enrichment. Width 4 was frozen before the held-out run.

## Eval-val result

On all 97 natural eval-val proteins, width 4 scored 0.54182 all-range and 0.52932 long-range R-precision, against 0.55375 and 0.53802 for the 100-rollout baseline. Paired differences were -0.01193 [-0.01741, -0.00641] and -0.00870 [-0.01629, -0.00021] by protein bootstrap. Pure H100 generation took 1.49 times as many total seconds. The held-out fold-switching comparison is still running.
