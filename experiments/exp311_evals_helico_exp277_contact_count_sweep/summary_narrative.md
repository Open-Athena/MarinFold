# Summary slides — exp: Helico confidence selection across top-0 to top-L exp277 contacts

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

We fold 96 natural eval-val and 19 designed eval-denovo proteins with Helico.
For each target we give Helico 0, 10, 20, ... contacts up to exactly L,
ranked by the latest MarinFold model (exp277 step 266344). Each cut has three
diffusion samples. The full grid has 3,089 cuts and 9,267 predictions.

## Why

The usual top-L cut is only one operating point. Sweeping the number of
contacts reveals the best possible structure available in the grid and tests
whether Helico's own confidence head selects it. We report GDT-TS, lDDT,
TM-score and RMSD, separately for natural and designed proteins.

## Results so far

All 345 reference contact-precision checks matched exp277's recorded values.
One natural target (7pv5_A) is excluded because its prompt-to-token index map
is ambiguous. A one-target smoke run produced all expected samples and metrics.
The complete structural sweep is running on Modal.
