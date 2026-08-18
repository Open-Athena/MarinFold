# Summary slides — exp245: FoldBench held-out monomer eval sets

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## The question

Every contact number MarinFold publishes comes from a 554-protein eval set whose
FoldBench half is the **first 100 rows** of `monomer_protein.csv` — chosen before
we understood how much of the eval set our training corpora contained (#213,
#225). #232 then trained the #199 recipe from scratch on corpora decontaminated
against **all** of FoldBench, which makes the other 234 monomers a real held-out
test set for those checkpoints for the first time.

So: cut FoldBench's 334 monomers into three sets, score them, and ask whether the
set we have been reporting on was telling us the truth.

- **eval-val (97)** — the natural monomers inside the historical FoldBench-100.
- **eval-test (217)** — every other natural FoldBench monomer, never scored here.
- **eval-denovo (19)** — every de novo designed FoldBench monomer.

Every protein carries a viral flag, because #241 showed the two strata rank
models differently.

## What was checked before anything was scored

**Decontamination, five links, all verified.** All 334 monomer sequences are in
#225's decontamination reference byte-for-byte; 131,180 training rows match one
of them at the applied rule (≥30 % identity over ≥50 % of the shorter sequence)
and **all 131,180 are in the drop list — zero survive**; the published corpora
match #225's counts; #232's tokenizer pins those exact counts and those exact
bucket prefixes; and both W&B runs read only those caches.

**What the rule does not cover, priced rather than described.** At the applied
gate the highest surviving identity to any of the 334 is 0.299. Relax the
coverage requirement to 40 % and essentially every protein has a surviving
training relative at ≥30 % identity; with no coverage requirement, 65 of them
have one at ≥90 % over a fragment. "Decontaminated at 30 %" means that rule, not
"no shared subsequence".

**Ground truth rebuilt through one path**, with the 126 overlapping frozen units
reproduced byte-identically. **One protein excluded**: `8uxt_A` (1,596 residues)
has no representable contacts-v1 document at an 8,192-token context.

## The evaluation reproduces PR #244

Two of the three checkpoints are the ones #244 scored, and all 97 eval-val
proteins are inside #244's universe, so the path is validated against a
published reference protein by protein rather than against a tolerance on an
aggregate: mean R-precision differs by **−0.0031** (m2-p06) and **−0.0023**
(m1-p02) with per-protein r = 0.996 — inside the 0.0023 spread #204 measured for
one unchanged checkpoint. 333 units × 3 checkpoints, **0 unfinished rollouts**.

The baseline path has its own exact control: re-scoring 12 published proteins
through this experiment's rebuilt ground truth reproduces #213's ESMFold and
ESMFold2 numbers with **max absolute difference 0.0**.

## The result: eval-val was not flattering us

All-range R-precision, quoted as **eval-val (97) / eval-test (217) /
eval-denovo (19)**.

The two decontaminated #232 checkpoints: **m2-p06 0.520 / 0.538 / 0.591** and
**m1-p02 0.473 / 0.493 / 0.588**. The contaminated reference, #199's cooldown:
**0.589 / 0.613 / 0.619**.

Baselines on the same proteins: Protenix-v2 single-seq 0.263 / 0.265 / 0.835;
ESMFold 0.750 / 0.753 / 0.795; ESMFold2 0.802 / 0.792 / 0.864; Protenix-v2 +
MSA 0.846 / 0.845 / 0.844. The sequence-KNN null scores 0.584 / 0.582 / 0.066
over the unfiltered corpus and 0.407 / 0.426 / 0.050 over the decontaminated
one.

**Every predictor scores the same or slightly better on the 217 monomers we had
never touched**, and the contaminated reference moves the same way and by the
same amount as the decontaminated checkpoints: the difference-in-differences is
**−0.006** (m2-p06) and **−0.004** (m1-p02), an order of magnitude inside the
noise. H2 is not supported — the historical FoldBench-100 was not inflating
#199's score through memorised homologs. H1 and H3 hold: every predictor moves
by under 0.03 between the two sets.

## Two things this changes

**The KNN null is the yardstick, and it is corpus-specific.** Copying the contacts
of the ten nearest training sequences scores 0.582 on eval-test out of the
unfiltered corpus and 0.426 out of the decontaminated one — 0.156 of memorisable
contact map per protein, over 0.2 for 99 of the 314 natural monomers. Each model
clears the null over the corpus it trained on (#232 m2-p06 **+0.112**, #199
cooldown **+0.031**) and falls below the null over the richer one. #199's 0.075
lead is therefore not a measurement of what decontamination costs — the runs are
not budget-matched, 290,400 vs 145,199 steps — but it bounds how much of any
contacts-v1 score is reachable by memorisation.

**Protenix-v2 single-sequence is not a comparator on natural proteins.** 0.835 on
the 19 designs, 0.265 on the 314 natural monomers. The "parity with Protenix-SS"
framing came from an eval set that is three-quarters designed protein; on natural
FoldBench monomers both #232 checkpoints beat it by more than 0.27. The real gap
is to ESMFold2 (−0.255) and Protenix + MSA (−0.307).

**Viral proteins are harder for everything except MSA** — eval-test viral vs
non-viral: m2-p06 0.465/0.542, #199 cooldown 0.497/0.621, ESMFold2 0.608/0.804,
seq-KNN 0.262/0.602, Protenix + MSA 0.812/0.847. #241's finding survives on new
proteins, and the gap tracks reachable homology.

## Designs are easier — against the right natural set

Earlier work (#213, #226, #241) found de novo designs much easier than natural
proteins. Here the design advantage looks small: **+0.054** for #232 m2-p06 and
**+0.006** for the #199 cooldown against all of eval-test. Both statements are
true, about different comparisons.

The published contrast used exp65's 396 idealised `denovo_pdb` designs against
the **homology-filtered** natural set (eval2-natural, where MarinFold scores
~0.31-0.36). exp245's eval-denovo is instead the 19 synthetic monomers FoldBench
happens to contain — engineered binders and miniaturised folds, n = 19, interval
+/-0.09 — and eval-test is *every* natural monomer, only 23 of 217 of which are
under 40 % identity to #199's training sequences.

Split eval-test on that axis and the old pattern returns exactly: against natural
proteins with no close training homolog, designs are **+0.177 [+0.044, +0.306]**
easier for m2-p06 and identically +0.177 for the cooldown. Designs are much
easier than natural proteins we have no homolog for; they are about as easy as
natural proteins in general, because most natural proteins have homologs.

Protenix-v2 single-seq is the extreme: **+0.570** designs versus all natural, and
flat across the identity split (0.243 vs 0.267). It looked competitive only on an
eval set that was three-quarters designed protein.

## Use from here

`eval-test` is the default set for a decontaminated-accuracy claim: 217 natural
proteins, four times eval2-natural's audited 63, not three-quarters designed,
ground truth and all five baselines published. `eval-val` keeps continuity with
every published figure. `eval-denovo` keeps designs out of natural-protein means.

Everything is on the bucket under
`data/contacts-v1-foldbench-monomers-exp245/`.
