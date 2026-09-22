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

On all 97 natural eval-val proteins, width 4 scored 0.54182 all-range and 0.52932 long-range R-precision, against 0.55375 and 0.53802 for the 100-rollout baseline. Paired differences were -0.01193 [-0.01741, -0.00641] and -0.00870 [-0.01629, -0.00021] by protein bootstrap. Pure H100 generation took 1.49 times as many total seconds.

## Held-out fold switching

On 29 primary test proteins, the beam pool contained both fold-specific contact modes for 2 proteins, versus 6 for iid100. The blind 16-map shortlist found both for 0, versus 2 for iid. One protein was beam-only, five were iid-only. Beam generation cost 6.28 times as many total H100 seconds. On all 67 non-capped pairs, the counts were 6 versus 23 oracle pools and 2 versus 9 blind shortlists. These are contact-level hits, not validated 3D folds.

## Conclusion

Contact-aligned beam width 4 is technically viable but lowers eval-val R-precision, discovers both held-out contact modes less often, and costs more time. Width 8 gave no development coverage gain and roughly doubled the width-4 cost. Plain iid100 remains better for this testbed.
