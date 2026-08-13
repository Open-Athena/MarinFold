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

## Verdict

**Fold it in at <40%; it changes little at <30%.** A +42% increase in
decontaminated natural proteins is a real gain on exactly the axis #213 said
bottoms out. The +22% at <30% is not enough to change what the eval set can
measure.

FoldBench is confirmed as the dirtiest slice we have. **89.6%** of the net-new
monomers fail a 40% filter, and **all 222** have at least some alignment into
the training set, so the expansion adds only **8** proteins to #213's
"no detectable homolog" stratum, 231 to 239. Of the 199 dropped at <40%, 174
hit both training arms — the metagenomic half is not the marginal contaminator
here.

Recommendation: add `foldbench_rest` as its own stratum rather than merging it
into `foldbench100`, since the two have measurably different training-set
proximity and any future "first N FoldBench rows" would inherit the same
deposition-date bias.

Not measured here: the fold-novel count for the 222, which needs a Foldseek
pass against exp41's Modal-hosted AFDB representative DB plus 222 structure
downloads — beyond this issue's sequence-search budget.
