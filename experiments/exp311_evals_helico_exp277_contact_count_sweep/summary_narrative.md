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

All 9,267 structures completed for 96 eval-val and 19 eval-denovo targets.
On eval-val, GDT-TS is 0.6242 oracle / 0.5277 Helico-ranked / 0.5015 top-L;
the ranked sweep beats top-L by +0.0262 [0.0039, 0.0508]. On eval-denovo the
same values are 0.9341 / 0.8990 / 0.8503. Confidence chooses zero contacts for
8/19 designs, showing that top-L is the wrong universal policy.

## What the gap means

The sweep already generates substantially better natural-protein structures:
the oracle is +0.0965 GDT-TS above the confidence choice on eval-val. Helico's
confidence model recovers some of the opportunity but remains the bottleneck.
On designs the oracle gap is smaller (+0.0352), and confidence usually prefers
far fewer contacts (median 0.056L) than it does on natural proteins (0.631L).

## Conclusion

Search the contact count and use Helico confidence instead of always folding
top-L. This improves both natural and designed accuracy. The next useful gain
is better confidence ranking: the structures needed for a much higher
eval-val score are already present in the sweep.
