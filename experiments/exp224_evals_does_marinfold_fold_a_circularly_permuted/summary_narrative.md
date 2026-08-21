# Summary slides — exp: does MarinFold fold a circularly permuted protein? (1UN2 CPDsbA-Q100T99 vs wild-type DsbA)

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Does MarinFold predict contacts for a circularly permuted protein as well as
it does for the natural, non-permuted parent — or does its accuracy come from
having memorised the parent's sequence→contact mapping at fixed sequence
separations?

## Why

A circular permutation is close to the ideal probe for this. `1UN2`
(CPDsbA-Q100T99, 2.4 Å) has the same fold, the same residues and essentially
the same 3D contact set as wild-type *E. coli* DsbA, but the contact map in
*sequence* coordinates is completely reorganised: the chain is cut between T99
and Q100 and the two halves are swapped, so every contact that spans the old
N/C-termini changes sequence separation, and the |i−j| distribution the model
has learned as a prior no longer matches the structure.

Two outcomes are informative:

- CP ≈ WT. The model is reasoning about the fold from local sequence, and
  transfers it across a re-ordering it has never seen. Strong evidence against
  the memorisation reading that [#94](https://github.com/Open-Athena/MarinFold/issues/94)
  and [#213](https://github.com/Open-Athena/MarinFold/issues/213) probed from
  other directions.
- CP ≪ WT. The model leans on a learned separation prior / on retrieving the
  parent, and a re-ordering breaks it. That would be a concrete, mechanistic
  instance of the generalisation gap.

Wild-type DsbA (UniProt `P0AEG4`) is in AFDB and therefore almost certainly in
exp199's 70.9M-protein training set. The permutant is an engineered construct
from 1999 and is not — but it is ~96% local-sequence-identical to the parent, so
a pure nearest-neighbour predictor would still retrieve the right *local*
contacts. The discriminating signal is the junction-spanning contacts: pairs
that are in contact in 3D but whose sequence separation the permutation changed.

## Results

The premise holds. CP and WT are the same fold: 88.5% of WT's contacts are
present in CP (Jaccard 0.734), against a crystal-to-crystal floor of 0.85–0.88
from two WT replicates. 47% of WT's contacts cross the cut, and for those the
separation transforms exactly as `CP_sep = 194 − WT_sep`.

The permutation costs ~0.09 R-precision; the extra residues cost nothing.
Mean ± sd over 10 rollout seeds (exp82 recipe, exp89 metrics):

CP 1UN2, permuted — R-precision 0.5247 ± 0.0091, AUC 0.952, 140.7 contacts/rollout

Control (WT + linker + tail, native order) — R-precision 0.6177 ± 0.0104, AUC 0.983, 181.5 contacts/rollout

WT 1FVK / 1DSB / 1A2J — R-precision 0.626 / 0.617 / 0.631, AUC ~0.982, ~179 contacts/rollout

The control has the permutant's exact length and non-native residues in
wild-type order and scores like the real WT crystals, so the −0.0930 gap
(Welch t = −21.3, p = 4.9e-14) is the re-ordering itself.

## The damage lands exactly where the theory says

Same model, same pairs, same universe, split by whether the permutation changed
that pair's sequence separation:

Separation UNCHANGED (within-segment) — CP − WT: R-precision −0.0506, AUC −0.0020

Separation CHANGED (cross-segment) — CP − WT: R-precision −0.1205, AUC −0.0448

Moved pairs lose 2.4x more R-precision and 22x more AUC than unmoved ones.

## Conclusion

MarinFold is not merely retrieving the parent — but a circular permutation
costs it a lot, and the cost is specific to re-ordered pairs.

Keeping R-precision 0.525 (vs 0.626 WT) and AUC 0.952 rules out the strong
memorisation story: a lookup keyed on the parent at fixed separations would
collapse, not degrade by 16%. But the gap is real, caused by the re-ordering,
concentrated on moved pairs, and accompanied by a 22% drop in contacts emitted
per rollout (no truncation — the model is declining to commit, not running out
of budget).

The picture is a model that has learned the fold *and* a strong prior over where
in sequence its contacts should live, leaning on the prior more than a genuine
folding model would. Follow-up: does permutation augmentation at training time
close it, and does that transfer to ordinary proteins? Cheap to test — it needs
no new structures.

Caveat: n = 1 permutant. The internal moved-vs-unmoved contrast carries the
argument; the absolute gap rests on one pair of crystals.
