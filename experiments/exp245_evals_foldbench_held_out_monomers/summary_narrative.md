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

| set | what it is | n scored |
|---|---|---:|
| **eval-val** | the natural monomers inside the historical FoldBench-100 | 97 |
| **eval-test** | every other natural FoldBench monomer, never scored here | 217 |
| **eval-denovo** | every de novo designed FoldBench monomer | 19 |

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
