## exp266 — ProteinMPNN redesign of the decontaminated contacts-v1 set

Take all 3,963,003 backbones of the decontaminated AFDB corpus (#225),
redesign each into 8 new sequences with ProteinMPNN, re-run pyconfind, and
publish the ~31.7 M-document corpus.

This is the cheap null version of "generate novel structures with BoltzGen /
Proteina and inverse-fold them". That version costs GPU-weeks and changes two
things at once — novel folds *and* MPNN-flavoured sequences. This changes only
the second, so it isolates the part that would have to pay off anyway.

## Two things verified before any code was written

**pyconfind ignores input side chains.** confind rebuilds them from the
Dunbrack rotamer library, so a structure stripped to N/CA/C/O gives
bit-identical contact degrees (4/4 structures, max Δ = 0.000). A backbone plus
a residue-name assignment is therefore a complete pyconfind input — the
redesigned corpus is computed under exactly the same contact operator as
contacts-v1, and all-atom generators would buy us nothing here.

**The contact label is strongly sequence-dependent at fixed geometry.**
Shuffling the sequence on an identical backbone keeps only ~43–54 % of the
native contacts (Jaccard 0.31–0.43). So the 8 redesigns are not 8
near-duplicates, and they supply a contrast the corpus cannot: today every
fold appears with exactly one sequence.

## The engineering that fell out of measuring it

ProteinMPNN's decode is L sequential steps, so the GPU is launch-bound rather
than compute-bound and a single CPU core is only ~18x slower per sequence.
That kills the two-cluster design: redesign-on-CoreWeave plus
documents-on-Iris would stage 4 M cifs cross-cloud and fetch every structure
twice, to save CPU hours the Iris pool has. One CPU job does both.

Two more: `temperature` broadcasts, so all 8 designs fit in one `sample()`
call (10.38 s -> 1.30 s at L=154); and `tied_featurize` mis-pads
`omit_AA_mask` for mixed-length batches, so batches are exact-length — which
also means zero padding waste.

## Composition drift is real but small

The worry was that ProteinMPNN's Ala/Glu/Lys/Leu bias would shorten the
documents, since contact count collapses for small side chains (poly-ALA gives
~0 contacts). Measured over 384 designs: the largest AA shift is 2.2
percentage points, and **contacts per residue moves from 0.930 native to 0.900
designed — a ratio of 0.968**.

## Where it stands

Nothing has run on the cluster. The projected full run is ~14,300 CPU
core-hours (~9,700 ProteinMPNN + ~4,600 pyconfind), i.e. ~28 h on 512 Iris
workers — about 5x the largest data job we have run (#139, ~2,850 core-hours).
Next step is a 20 k-backbone Iris smoke to replace the workstation rates with
cluster ones, then a go/no-go on the full run.
