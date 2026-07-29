# Summary slides — exp174: 3D coordinate-based structural eval for contacts-and-crops-v1

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Every MarinFold eval so far scores contact prediction. contacts-and-crops-v1
(#130) encodes real 3D positions — Pass-1 coarse 10 A boxes plus Pass-2 fine
0.1 A crops — so a model trained on it should be able to emit an actual
structure.

This experiment asks whether it can, scored the way structure predictors are
normally scored: CA-RMSD, all-atom RMSD, lDDT, lDDT-CA and TM-score against the
554-protein eval set from #74/#78.

## Two components, two gates

Component 1 — inference (document to coordinates) — is plans-first by design.
Six approaches are written up in PLANS.md. Plan F is now the plan of record:
neighbour-conditioned iterative refinement. Not yet implemented.

Component 2 — scoring — is done. It takes a directory of predicted coordinate
files in a documented one-PDB-per-protein contract and needs to know nothing
about how they were produced. Ground truth, model predictions and baselines all
speak the same contract.

## Ground truth: 554/554, and one library bug

Full-atom ground truth for all 554 records: 777,459 heavy atoms, zero failures,
minimum sequence-alignment identity 0.95, lengths 30 to 761.

Building it surfaced a real bug in the shared library. The coordinate parse
layer joins the pyconfind residue list to the gemmi atom walk on
(chain, residue number), and the two libraries spell a blank author chain id
differently — gemmi "", pyconfind "_". Structures with no chain id joined
nothing and produced a document with an empty coordinate section, silently.
AFDB is always chain A so the training corpus never hit it; 19 of the 554 eval
proteins did. Fixed, with the parse layer now raising instead of emitting a
coordinate-free document.

## Partial predictions are the normal case

About 96% of contacts-and-crops-v1 documents are truncated by construction, so
the harness reports two families of metric and keeps them apart.

Coverage-penalized (lDDT, lDDT-CA, TM-score) take their denominators from the
ground truth, so an atom the model never placed costs score. These are the
model-comparison numbers.

Covered-only (RMSD, and the "covered" lDDT variants) are computed over the
atoms that were placed. A predictor that emits three atoms perfectly scores
0.0 A RMSD, so these are meaningless without the coverage columns beside them.

A record with no prediction file is scored as a total miss, not skipped.

## The headline so far: the format's ceiling

Before any model number means anything, we measured what a perfect model could
score. Degrade the ground truth to each of the format's resolution tiers and
run it through the same harness.

The 0.1 A digit vocabulary costs nothing: lDDT 1.000, TM 1.000. But every atom
at its correct 10 A box centre scores only lDDT 0.32 / TM 0.51, and one
realistic document — 65% of atoms boxed, 25% refined — tops out at lDDT 0.17 /
TM 0.41.

lDDT falls as coverage squared (a contact needs both its atoms); TM-score falls
linearly (a residue needs only itself).

## E3: the model learned the refinement schedule

The format trains a box's i-th appearance with sigma = 1/(i+1)^2 A of noise.
Whether the model conditions on visit count at all was the gate on the whole
iterative plan.

It does. Emitted error falls from 2.11 A at the first read to 0.13 A by the
sixth, tracking the schedule down to the format's own 0.1 A tenths floor and
then flattening. Re-showing a box is worth doing.

## E2: refinement is not the bottleneck

Teacher-force the correct 10 A boxes and let the model generate only the crops,
and it reaches 96% of the ceiling: lDDT 0.278 against 0.290, TM 0.522 against
0.537, CA-RMSD 4.33 A against 4.16 A.

Given correct coarse boxes, the model's Pass-2 crops are about as good as
ground-truth crops. Pass 2 works.

## Plan F works, and escapes the token budget

Neighbour-conditioned iterative refinement drives atom coverage and refined
fraction to 0.999, against 0.31 for a single document, and doubles Plan A's
lDDT from 0.141 to 0.290 — equal to the single-document ceiling.

Per length it goes past that ceiling, because the ceiling collapses with chain
length while F simply re-prompts: at 201-400 residues F is 1.7x the
one-document ceiling, and past 400 residues 2.6x above it.

## But the fold is wrong, and inference does not fix it

CA-RMSD is about 16.5 A for A, C and F alike, and TM-score never passes 0.28.
Plan F produces a complete, locally precise, wrong structure.

The decisive comparison is E2 against F: same model, same refinement machinery,
less coverage — TM 0.522 versus 0.277 and CA-RMSD 4.33 A versus 16.28 A. The
only difference is whether the coarse boxes are right.

Handing the model 50 true contacts (E1, the format's cap) cuts CA-RMSD by 22%
but gets nowhere near E2. The two checkpoints are indistinguishable.

## Conclusion

contacts-and-crops-v1 at 1.5B is not yet structure-capable de novo, and the
bottleneck is Pass 1 — the coarse fold — not Pass 2 refinement and not the
format's resolution.

The earlier ceiling analysis called the Pass-2 refined fraction "the whole
ballgame". That was right about the format's ceiling and wrong about where this
model sits relative to it: Plan F buys the refined fraction outright and the
fold does not improve. Inference compute is not the lever, and neither is a
bigger fine reserve in a v2 format. The next move belongs to training, aimed at
the coarse spatial layout.

One reporting lesson: lDDT and TM-score come apart sharply here. Plan F is at
the ceiling on lDDT and a third of it on TM, because a local metric rewards a
well-refined wrong fold and a global one does not. Report both, with coverage.
