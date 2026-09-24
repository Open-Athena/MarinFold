# Summary slides — Helico contact-cut sweep

## Question and experiment

Does extending a MarinFold vote-ranked contact list improve Helico folding?

Same checkpoint and settings across cuts: Helico contacts-msafree-01 step 6000,
six trunk recycles, seed 42, no MSA. Three diffusion samples are generated;
the runner scores returned sample zero. This audit reruns only CPU analysis.

97 eval-val targets: 96 have contact-arm results. The original 95-target
comparison additionally excludes one target missing Protenix-derived contact inputs.
Both populations are now reported; no held-out eval-test targets are scored.

## Reproduced result, with uncertainty

On the historical 95 targets, top-L lDDT is 0.6053; 1.5L is 0.6073.
Paired difference +0.0020, 95% interval [-0.0048, +0.0086]. A small gain remains
plausible; this does not establish equivalence or a distinct optimum.

At 5L, difference -0.0135 [-0.0243, -0.0034]. At the union, -0.0245
[-0.0346, -0.0146]. All 96 contact-available targets give the same conclusion.
Independent coordinate rescoring of all 288 top-L/1.5L/union structures agrees
within 0.0000333 per target. All 768 delivered lists match the vote rankings.
Intervals are target-bootstrap, pointwise; inference-seed variation is unmeasured.

## Corrections to the earlier interpretation

Top-L recall is 0.548, not 0.52: R (true-contact count) and L (length) are
different cut budgets. Contact metrics now use the same targets as lDDT.

The union retains 90.40% of top-L's gain above no contacts, not 96%.
The earlier denominator used total lDDT instead of the contact-related gain.

Changing the cut changes precision, recall, and list size together. These
results cannot show recall is worthless or identify the conditioning bottleneck.

## What this rules out, and what remains open

Much longer unweighted lists from this vote ranking did not help this Helico
checkpoint. Modest widening to 1.5L has no demonstrated gain but remains uncertain.

The result does not rule out better ranking, confidence weighting, choosing
complementary constraints, or training that uses the additional true contacts.
Public input extraction and per-file provenance support CPU-only reproduction.
