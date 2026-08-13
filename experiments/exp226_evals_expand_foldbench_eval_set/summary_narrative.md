# Summary slides — exp226: growing the contact eval set with the rest of FoldBench

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Prose only — the renderer reflows each paragraph with textwrap, so
     markdown tables come out as raw pipes. Keep headings under ~45 chars
     or they run off the slide. -->

## The question

We only ever used **100 of FoldBench's 334** monomers. #213 showed the contact
eval set is 58% homologous to exp199's 70.9M training sequences, and that what
survives a homology filter is mostly de novo *designs* — the **natural** count
bottoms out around 50, too thin to measure novel-protein performance rather
than bound it.

So: if we grow the eval set with the FoldBench monomers we never used, how many
of the expanded set survive a sequence-identity filter against our training
data, at 40% and at 30%?

No new inference. exp213's 17 GB MMseqs2 target database is reused as-is; this
appends 222 net-new queries to its 554 and re-runs the search. About seven
minutes on the workstation.

## The set, verified not assumed

FoldBench's `monomer_protein.csv` at exp12's pinned commit `4273f687`,
sha256-verified, has 334 rows, and our 100 are exactly its first 100 — checked
rather than taken on faith. That leaves 234 unused. **Twelve of those are
already eval proteins** under exp65's `denovo_pdb` label and would have been
double-counted, so the real increment is **222 net-new** and the expanded set
is **776**.

Sequences come from RCSB's canonical `entity_poly` field. FoldBench's
`chain_id` is sometimes the mmCIF *label* asym id rather than the auth chain —
ten entries, not the five the issue listed — so chains resolve auth-first with
a label fallback and fail loudly on ambiguity.

Both residue checksums land exactly (66,692 aa and 64,624 aa), and all 100 of
the sequences we already use are reproduced **byte-for-byte** through the new
fetch path.

## Parity with #213 is exact

Re-running exp213's 554 queries through the expanded search reproduces its
**284 and 264** survivors at the two thresholds, and its 273 and 255 without
the coverage gate, with **zero of the 554 rows** changing identity or stratum.
The expanded table is a strict superset of exp213's and joins to it on dataset
and stem.

One correction to the plan on the issue: its quoted command line says
`--max-seqs 2000`, but exp213's published table was built with **5000** — 2000
was its AFDB-only cross-check. Parity means matching the published table.

## Result — 23 survivors, not the 33 predicted

Of the 222 net-new proteins, **23 survive a <40% identity filter and 11 survive
<30%**, taking the expanded eval set to 307 of 776 and 275 of 776.

**The extrapolation on the issue was optimistic at both thresholds**, which is
the question it could not answer for itself. It predicted 33 and 24. The 100
monomers we already use are the *oldest-deposited* rows of a PDB-ID-sorted
file, and the newer entries turn out to be **more** homologous to our training
data, not less: 10.4% survive at <40% against 15.0%, and 5.0% at <30% against
11.0%. Neither gap clears p<0.05 alone (Fisher 0.26 and 0.057), so this reads
as a consistent shortfall rather than a proven rate difference — but the counts
the eval set actually gets are 23 and 11.

Length is not the explanation: median 242 aa against 247.5 aa.

## The natural count is what decides this

Every one of the 23 new survivors is a **natural** protein. The decontaminated
natural count goes from **55 to 78 at <40%**, a gain of 23 or **+42%**, and
from **50 to 61 at <30%**, a gain of 11 or +22%.

The baseline is **55, not the 58** the issue quotes. #213 splits designed from
natural on the dataset label, which cannot see a designed protein sitting in a
FoldBench row — and one can, since 12 FoldBench monomers are themselves in
exp65's de novo set. Resolving each FoldBench entity's RCSB source organism
finds that 3 of #213's 15 FoldBench-100 survivors at <40% are synthetic
constructs. The proxy is calibrated, flagging 12 of 12 known designs, and
deliberately conservative, since it also catches engineered variants of natural
proteins. Only 4 of the 222 trip it, and none of those 4 survive either filter.

## Both arms, and why one is not enough

exp199 trained on **both** corpora, so the union is the filter that counts —
4.13M AFDB sequences with AlphaFold2 labels plus 66.76M ESM-Atlas sequences
with ESMFold2 distillation labels. Every number above uses the union. But every
prior overlap check (#41, #65, #94) only ever looked at AFDB, and that turns
out to matter a great deal.

Against **AFDB alone**, 76 of the 222 net-new monomers would look clean at
<40%. Against **ESM-Atlas alone**, 62. Against **both**, only **23**. A
single-arm check overcounts the clean set by about 3x, and the two arms are
largely complementary rather than redundant: ESM-Atlas removes 53 proteins
AFDB alone would have kept, and AFDB removes 39 ESM-Atlas alone would have kept.

**The pattern reverses between the two FoldBench slices.** Of the 199 net-new
dropped at <40%, 107 are reachable from both arms, 39 from AFDB alone and **53
from ESM-Atlas alone** — the metagenomic half is the larger sole contaminator.
For the existing 554 the same computation gives 183 / 60 / 27, where AFDB
dominates, and reproduces the figures on the issue exactly — a third
independent parity check. The ESMFold2-distillation corpus is doing *more* of
the contaminating on newer PDB entries, which is part of why extrapolating from
the older 100 came out optimistic.

## eval2 — the filtered set

`data/eval2_manifest.csv` is the deliverable that follows from all of this: the
expanded set with **every protein at or above 40% training identity removed**,
leaving **307**, sequences included and annotated so a stricter cut costs no new
compute.

`best_identity` is the coverage-gated maximum over **both** arms, so
`best_identity < 0.30` reproduces the 30% set (**275** proteins) exactly;
`passes_30` is precomputed. Per-arm columns allow the same cut against AFDB or
ESM-Atlas alone, and `best_identity_ungated` is the paranoid bound — 18 of the
307 clear 40% only because of the 50% coverage gate.

One property constrains what eval2 can measure, and it is a column rather than a
caveat in prose. **75% of it (229 of 307) is de novo designed protein** — not a
choice made here, but what survives a homology filter, and exactly the confound
#213 raised; `designed_any` splits it and the natural subset is **78** at 40%
and **61** at 30%.

## Ground truth for all 307

23 of eval2's proteins were outside #89's frozen GT universe and so unscorable.
Their contacts are now computed with **#89's own `compute_contacts`** —
imported, not reimplemented — on the RCSB `-assembly1` mmCIFs exp12 used for the
FoldBench-100, emitted in #89's exact schema. The two files concatenate into a
**577-unit** universe. All 23 come out clean: alignment identity **1.000** for
every one, resolved/L 0.83–1.00, 202–1046 contacts each.

**The control is what makes them usable.** Running the *new* code path on the
100 FoldBench proteins #89 already published reproduces **100/100 records
exactly** — L, n_resolved, chain, alignment identity, the resolved set and every
(i, j, degree) contact. That includes all six label-chain entries, where #89
passed FoldBench's label id and fell back to the longest polymer chain, landing
on the same auth chain this now passes explicitly. So the 23 are scored on the
same definition of "contact" as the 554 and can be pooled with them.

Published to the bucket at `data/contacts-v1-eval2-exp226/`; nothing under #89's
prefix was touched.

## Verdict

**Fold it in at <40%; it changes little at <30%.** A +42% increase in
decontaminated natural proteins is a real gain on exactly the axis #213 said
bottoms out. The +22% at <30% is not enough to change what the eval set can
measure.

FoldBench is confirmed as the dirtiest slice we have. **89.6%** of the net-new
monomers fail a 40% filter, and **all 222** have at least some alignment into
the training set, so the expansion adds only **8** proteins to #213's
"no detectable homolog" stratum, 231 to 239.

Recommendation: add `foldbench_rest` as its own stratum rather than merging it
into `foldbench100`, since the two have measurably different training-set
proximity and any future "first N FoldBench rows" would inherit the same
deposition-date bias.

Not measured here: the fold-novel count for the 222, which needs a Foldseek
pass against exp41's Modal-hosted AFDB representative DB — beyond this issue's
sequence-search budget. And **no model has been scored on eval2 yet**: this
delivers the decontaminated set plus its ground truth, and the 23 new proteins
have no predictions from any comparator either, so a like-for-like baseline over
the full 307 needs those runs first. The 284 subset is comparable today.
