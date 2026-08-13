---
marinfold_experiment:
  issue: 226
  title: 'exp: expand the contact eval set with the other 234 FoldBench monomers, and measure homology-free survival at 40% / 30% id'
  kind: evals
  branch: claude/github-issue-226-6aef7a
---

# exp: expand the contact eval set with the other 234 FoldBench monomers, and measure homology-free survival at 40% / 30% id

**Issue:** [#226](https://github.com/Open-Athena/MarinFold/issues/226) · **Kind:** `evals` · **Branch:** `claude/github-issue-226-6aef7a`

## Question

**If we grow the contact eval set with the FoldBench monomers we never used, how
many of the expanded set survive a sequence-identity filter against our training
data — at 40 % and at 30 %?**

[#213](https://github.com/Open-Athena/MarinFold/issues/213) closed with a hard
limit: its homology-free subsets are mostly de novo designs, and the *natural*
half bottoms out around n≈50. That is too thin to **measure** novel-protein
performance rather than bound it. FoldBench is the obvious place to look for
more natural proteins, because we have only ever used **100 of its 334**
monomers.

## Hypothesis

Extrapolating the FoldBench-100 survival rate to the rest predicted **~33
survivors at <40 %** and **~24 at <30 %**, taking the decontaminated natural
count from 58 to roughly 91 (+57 %).

The extrapolation's own caveat was the thing worth testing: our 100 are the
**first 100 rows** of a roughly PDB-ID-sorted file — the *oldest-deposited*
entries, not a random sample.

## Approach

Reuse exp213's 70.9 M-sequence MMseqs2 target database. That database is the
union of **both** corpora `contacts-v1-exp199-1.5B` was trained on —
**4,129,682 AFDB** (AlphaFold2 labels, #53) + **66,759,922 ESM-Atlas**
(ESMFold2 distillation labels, #139) — and every training sequence carries its
arm in its FASTA header (`{arm}|{id}`), so each hit is attributable back to one
and the per-protein table has per-arm columns as well as pooled ones. Survival
is reported against the union (the filter that matches how the model was
trained) and against each arm alone.

Append the net-new FoldBench monomers to exp213's 554 queries, search the
union, and reduce with exp213's rule verbatim: a hit counts toward the identity
axis only if `evalue <= 1e-3` **and** `qcov >= 0.50`; the reported identity is
the max `fident` over those hits.

Set construction, all derived and then checked rather than copied from the
issue:

- FoldBench `targets/monomer_protein.csv` at exp12's pinned commit
  [`4273f687`](https://github.com/BEAM-Labs/FoldBench/tree/4273f6877d82bd0b2fa476d1b2f34d121cbccc70),
  sha256 `43c2a5e9…` — **334 rows**, verified.
- Our `foldbench100` is exactly the **first 100 rows** — verified, not assumed.
  → **234 unused**.
- **12** of those 234 are already eval proteins under exp65's `denovo_pdb`
  dataset and must not be double-counted. → **222 net-new**, and an expanded
  set of **776**.
- Sequences are the canonical `entity_poly.pdbx_seq_one_letter_code_can` per
  entity, from RCSB's GraphQL data API.

## Code & how to run

CPU-only, no cluster job, no model inference. The whole thing is ~7 minutes,
almost all of it the MMseqs2 search.

| File | Role |
| --- | --- |
| `exp213_link.py` | The single seam onto exp213's `overlap_lib` / reduction and its committed artifacts. |
| `build_query_set.py` | **Step 1** — pinned FoldBench list → 222 net-new queries + RCSB sequences and source organisms. |
| `search_expanded.py` | **Step 2** — 776-query MMseqs2 search against exp213's existing target DB → the per-protein table. |
| `analyze_survival.py` | **Step 3** — survival counts, designed/natural split, the newer-vs-older test. |
| `plot_survival.py` | **Step 4** — the four figures. |
| `build_eval2.py` | **Step 5** — the homology-filtered eval set + its identity annotation. |
| `build_gt_contacts.py` | **Step 6** — pyconfind ground-truth contacts for the 23 eval2 proteins #89's universe misses. |
| `validate_gt_against_exp89.py` | **Step 6b** — the control: re-derive #89's own FoldBench-100 records through the new path. |
| `publish_gt_to_hf.py` | **Step 6c** — merge into the 577-unit universe and publish to the bucket. |
| `tests/test_expand.py` | Unit tests for chain resolution, the survival predicate and Fisher's exact test. |

```bash
uv sync --extra test
uv run --extra test pytest tests/            # 22 tests, <1 s

uv run python build_query_set.py             # RCSB, ~4 s
uv run python search_expanded.py --work /data/exp213_overlap   # the long step, ~6 min
uv run python analyze_survival.py
uv run python plot_survival.py
uv run python build_eval2.py                 # data/eval2_manifest.csv
uv run python build_summary.py               # plots/summary.pdf
```

**Two corrections to the issue's plan**, both found while implementing:

1. **The search parameters.** The issue's quoted command line says
   `--max-seqs 2000`. exp213's published two-arm table was actually built with
   **`--max-seqs 5000`** (its run log and its `provenance.json` both say so;
   2000 was the AFDB-only cross-check's value). Parity means matching the
   published table, so this uses 5000 — and reproduces it exactly.
2. **The label-chain gotcha is bigger than stated.** The issue lists 5 entries
   whose FoldBench `chain_id` is the mmCIF *label* asym id rather than the auth
   chain. There are **10**: `5sbj_A 8bgb_A 8bke_A 8c4y_A 8ci9_A 8gmy_A 8ork_A
   8qq1_A 8rdd_A 8uds_A`. The 5 it missed are all inside our existing 100, so
   the net-new set is unaffected — and the byte-for-byte check below proves all
   10 resolve correctly.

## Results

### 0. Parameter parity with #213 is exact

The validation anchor passes and then some. Re-running exp213's 554 queries
through the expanded search reproduces **284 / 264** survivors gated and
**273 / 255** ungated — and **0 of the 554 rows** changed their best identity or
their stratum. The expanded table is a strict superset of exp213's, joinable on
`(dataset, stem)`.

The sequence-fetch path is validated the same way: all **100/100** FoldBench
sequences we already use are reproduced **byte-for-byte** from RCSB through the
new code (the issue asked for 8 length comparisons). Both residue checksums
land exactly: 234 unused = **66,692 aa**, 222 net-new = **64,624 aa**.

### 1. The answer: +23 at <40 %, +11 at <30 %

[`data/survival_headline.csv`](data/survival_headline.csv) —

| filter | eval set today (554) | expanded (776) | gain |
| --- | ---: | ---: | ---: |
| **<40 % id** | 284 | **307** | **+23** |
| **<30 % id** | 264 | **275** | **+11** |
| <40 %, ungated | 273 | 289 | +16 |
| <30 %, ungated | 255 | 264 | +9 |

Of the 222 net-new proteins, **23 survive at <40 %** and **11 at <30 %**.

### 2. The newer monomers are dirtier than the ones we already use

This is the question the extrapolation could not answer, and the answer is that
the extrapolation was optimistic — in the same direction at both thresholds
([`data/newer_vs_older.csv`](data/newer_vs_older.csv)):

| filter | our 100 | the other 222 | predicted | actual | Fisher *p* |
| --- | ---: | ---: | ---: | ---: | ---: |
| <40 % | 15.0 % | **10.4 %** | 33.3 | **23** | 0.26 |
| <30 % | 11.0 % | **5.0 %** | 24.4 | **11** | 0.057 |

At <30 % the yield is **less than half** what was predicted, and the deposition-
date effect the issue flagged as "a reason to measure rather than assume" was
worth measuring. Neither difference clears *p* < 0.05 on its own, so the honest
reading is a consistent shortfall at both thresholds rather than a proven rate
difference — but the *counts* are what the eval set gets, and those are 23 and
11, not 33 and 24.

Length is not the explanation: median 242 aa for our 100 vs 247.5 aa for the
234, matching the issue's own check. → [`plots/identity_profile_old_vs_new.png`](plots/identity_profile_old_vs_new.png)

### 3. FoldBench really is the dirtiest slice of the eval set

[`data/survival_by_dataset.csv`](data/survival_by_dataset.csv) —

| dataset | n | survive <40 % | survive <30 % |
| --- | ---: | ---: | ---: |
| `foldbench100` | 100 | 15 (15.0 %) | 11 (11.0 %) |
| **`foldbench_rest`** | **222** | **23 (10.4 %)** | **11 (5.0 %)** |
| `denovo_pdb` (designed) | 396 | 226 (57.1 %) | 212 (53.5 %) |
| `cameo_hard` | 32 | 24 (75.0 %) | 22 (68.8 %) |
| `casp_fm` | 26 | 19 (73.1 %) | 19 (73.1 %) |

**89.6 % of the net-new FoldBench monomers fail a 40 % filter.** And every one
of the 222 has *some* MMseqs2 alignment into the training set (0 have none at
E ≤ 10, against 62/554 in exp213); only 8 have no *significant* homolog. So the
expansion adds just **8** proteins to exp213's pre-registered "no detectable
homolog" stratum (231 → 239) — it moves the <40 %/<30 % filters, not that one.

→ [`plots/survival_by_dataset.png`](plots/survival_by_dataset.png)

### 4. The natural count — the number that decides whether this was worth doing

Every one of the 23 net-new survivors is a **natural** protein
([`plots/natural_gain.png`](plots/natural_gain.png)):

| filter | natural today | natural expanded | gain |
| --- | ---: | ---: | ---: |
| **<40 %** | 55 | **78** | **+23 (+42 %)** |
| **<30 %** | 50 | **61** | **+11 (+22 %)** |

Two things to read carefully here.

**The baseline is 55, not 58.** exp213 splits designed from natural on the
dataset label (`denovo_pdb`), which cannot see a designed protein sitting in a
FoldBench row — and one demonstrably can, since 12 FoldBench monomers are
themselves in exp65's de novo set. Resolving each FoldBench entity's RCSB source
organism finds **3 of exp213's 15 FoldBench-100 survivors at <40 % are synthetic
constructs** (2 of 11 at <30 %). So the pre-existing natural count was 55 / 50,
and the expansion takes it to 78 / 61.

The proxy is deliberately conservative — "no natural source organism" also
catches engineered variants of natural proteins, so it over-flags rather than
under-flags — and it is calibrated: it flags **12/12** of the known de novo
designs. Only **4 of the 222** net-new monomers trip it, and none of those 4
survive either filter.

**The <40 % gain lands, the <30 % gain does not.** +42 % against the predicted
+57 % is a real, useful increase in the axis #213 said bottoms out. At <30 % the
+22 % is small enough that it does not change what the eval set can measure.

### 5. Both training arms, separately — and one arm alone overcounts 3×

exp199 trained on **both** corpora, so the union is the filter that counts and
is what every number above uses. But the two arms are worth separating, because
every prior overlap check (#41, #65, #94) only ever looked at AFDB
([`data/survival_by_arm.csv`](data/survival_by_arm.csv)):

| the 222 net-new, survivors vs… | <40 % | <30 % |
| --- | ---: | ---: |
| AFDB only (4.13 M seqs) | 76 | 40 |
| ESM-Atlas only (66.76 M seqs) | 62 | 29 |
| **both — exp199's actual training set** | **23** | **11** |

**An AFDB-only check would have called 76 of the 222 clean at <40 %. The real
number is 23.** The arms are largely complementary rather than redundant:
ESM-Atlas removes 53 proteins that AFDB alone would have kept, and AFDB removes
39 that ESM-Atlas alone would have kept.

Of the 199 net-new dropped at <40 %
([`data/arm_complementarity.csv`](data/arm_complementarity.csv)): **107 are
reachable from both arms, 39 from AFDB alone, 53 from ESM-Atlas alone.**

**The pattern reverses between the two FoldBench slices.** For the existing 554
the same computation gives **183 / 60 / 27** — reproducing the issue's own
figures exactly, a third independent parity check — where AFDB is the larger
sole contaminator. For the net-new 222 it is **ESM-Atlas** that is the larger
sole contaminator (53 vs 39). The metagenomic ESMFold2-distillation half is
doing *more* of the contaminating on the newer PDB entries, not less, which is
part of why the extrapolation from the older 100 came out optimistic.

→ [`plots/per_arm_survival.png`](plots/per_arm_survival.png)

(The separate [`data/arm_attribution.csv`](data/arm_attribution.csv) reports the
weaker "does this arm have *any* significant hit" statistic — 174 / 9 / 16 for
the net-new set — which is a different question and not the one the identity
filter turns on.)

### 6. eval2 — the homology-filtered eval set

[`data/eval2_manifest.csv`](data/eval2_manifest.csv) is the expanded set with
every protein at or above **40 % training-set identity removed**: **307
proteins**, sequences included, annotated so a stricter cut needs no new
compute.

| | n | natural | scorable today |
| --- | ---: | ---: | ---: |
| **eval2 (<40 % id)** | **307** | 78 | 284 |
| retrospective <30 % (`passes_30`) | 275 | 61 | 264 |
| retrospective <40 % ungated (`passes_40_ungated`) | 289 | 68 | 273 |

`best_identity` is the coverage-gated maximum over **both** training arms, so
`best_identity < 0.30` reproduces the 30 % set exactly; `passes_30` is
precomputed. `afdb_best_identity` and `esm_atlas_best_identity` allow the same
cut against either arm alone. `best_identity_ungated` is the paranoid bound —
18 of the 307 clear 40 % only because of the 50 % coverage gate.

**Two properties of eval2 that constrain what it can measure**, both carried as
columns rather than left in prose:

- **75 % of it (229/307) is de novo designed protein.** That is not a choice
  made here — it is what survives a homology filter, and it is exactly the
  confound #213 raised. `designed_any` splits it; the natural subset is **78**
  at 40 % and **61** at 30 %. Any headline computed on pooled eval2 is mostly a
  statement about designed backbones.
- **All 307 are scorable.** 284 come from #89's frozen GT universe; the other
  23 had no contacts computed, so this experiment computed them — see §7.
  `has_ground_truth` is read off the GT files rather than assumed.

Threshold semantics: "at or above 40 %" is excluded, matching #213/#226's
published counts. Exactly one protein sits on the boundary — `6sa6_A` at
fident 0.400 — and `--boundary keep` includes it (n=308) if the other reading
is wanted.

```bash
uv run python build_eval2.py                              # 307 @ <40%
uv run python build_eval2.py --threshold 0.30 \
    --out data/eval2_strict.csv --out-fasta data/eval2_strict.fasta
```

### 7. Ground truth for the 23, and the control that validates it

The 23 net-new proteins in eval2 had no ground-truth contacts. They do now
([`data/gt_universe_eval2_new.jsonl`](data/gt_universe_eval2_new.jsonl)),
computed with **#89's own `pyconfind_contacts.compute_contacts`** — imported,
not reimplemented — on RCSB `-assembly1` mmCIFs, the same structure source
exp12 used for the FoldBench-100. Records are emitted in #89's exact
`gt_universe.jsonl` schema, so the two files concatenate into a **577-unit**
universe (575 unique stems).

All 23 come out clean: **alignment identity 1.000 for every one**, resolved/L
between 0.83 and 1.00 (median 0.94), 202–1046 contacts each.

**The control is what makes them usable.** Running the *new* code path on the
100 FoldBench proteins #89 already published reproduces **100/100 records
exactly** — `L`, `n_resolved`, `gt_chain`, `gt_align_identity`, the resolved
set, and every `(i, j, degree)` contact
([`data/gt_validation.json`](data/gt_validation.json)). That includes all six
label-chain entries, where #89 passed FoldBench's label id, silently fell back
to the longest polymer chain, and landed on the same auth chain this passes
explicitly (`5sbj_A` → C, `8gmy_A` → D). So the 23 are scored on the same
definition of "contact" as the 554 and can be pooled with them.

Published to the bucket under `data/contacts-v1-eval2-exp226/`, so a downstream
eval needs one prefix and no access to this checkout:

```bash
hf buckets cp hf://buckets/open-athena/MarinFold/data/contacts-v1-eval2-exp226/gt_universe_eval2.jsonl .
# also there: gt_universe_eval2_new_23.jsonl, eval2_manifest.csv, eval2.fasta
```

Nothing under #89's prefix was modified. Rebuild with:

```bash
uv run --extra gt python build_gt_contacts.py          # the 23
uv run --extra gt python validate_gt_against_exp89.py --n 100   # the control
uv run python publish_gt_to_hf.py
```

## Conclusion

**The expansion is worth folding in at <40 %, and not worth much at <30 %.**

- The expanded eval set is **776 proteins**, of which **307 survive a <40 %
  identity filter** and **275 survive <30 %**.
- The decontaminated **natural** count — the axis #213 said bottoms out — goes
  **55 → 78 (+42 %)** at <40 %, and all 23 additions are natural proteins. At
  <30 % it goes 50 → 61 (+22 %).
- The extrapolation on the issue was optimistic at both thresholds because the
  100 monomers we already use are the oldest-deposited rows, and the newer
  entries are *more* homologous to our training data, not less. Predicted 33/24,
  measured 23/11.
- **Both training arms matter and neither is redundant.** Filtering against
  AFDB alone would have left 76 of the 222 looking clean at <40 % instead of 23.
  For the net-new set the ESM-Atlas / ESMFold2 distillation corpus is the
  *larger* sole contaminator (53 proteins vs AFDB's 39) — the reverse of the
  existing 554, where AFDB dominates (60 vs 27).
- The per-protein table for all 776
  ([`data/eval_train_identity_expanded.csv`](data/eval_train_identity_expanded.csv))
  shares exp213's schema, carries per-arm identity columns, and reproduces its
  554 rows exactly, so it is a drop-in replacement for any future eval that
  wants to stratify on training identity — against either arm or both.

- **eval2 is ready to score.** 307 proteins, all with ground truth, published at
  `data/contacts-v1-eval2-exp226/` on the bucket.

**Not done here:** the **fold-novel** count for the 222. That axis needs a
Foldseek pass against exp41's AFDB training-representative DB, which lives on a
Modal volume rather than on this workstation — real compute beyond this issue's
"sequence search plus aggregation" budget. The sequence-novel n is what #226
measured; if the fold-novel n matters for the next eval, that is a separate
short experiment on top of exp41's `query_similarity.py`.

Also not done: **no model has been scored on eval2.** This experiment delivers
the decontaminated set and its ground truth; running a checkpoint over it is the
`eval-checkpoint` path with `--gt gt_universe_eval2.jsonl`, and the 23 new
proteins have no predictions from any comparator (Protenix, ESMFold, ESMFold2)
either — so a like-for-like baseline table over the full 307 needs those runs
first. The 284 subset is comparable today.

**Recommendation:** add `foldbench_rest` to the eval set as its own stratum
rather than merging it into `foldbench100` — the two have measurably different
training-set proximity, and the deposition-date ordering means any future
"take the first N FoldBench rows" would inherit the same bias.

## Artifacts

| File | Contents |
| --- | --- |
| [`data/eval2_manifest.csv`](data/eval2_manifest.csv) · [`eval2.fasta`](data/eval2.fasta) | **eval2** — 307 proteins under 40 % training identity, annotated for a retrospective 30 % cut. |
| [`data/gt_universe_eval2_new.jsonl`](data/gt_universe_eval2_new.jsonl) · [`eval2_new_gt_manifest.csv`](data/eval2_new_gt_manifest.csv) | Ground-truth contacts for the 23 proteins #89's universe misses. |
| [`data/gt_validation.json`](data/gt_validation.json) | The 100/100 control reproducing #89's own records through the new path. |
| [`data/eval_train_identity_expanded.csv`](data/eval_train_identity_expanded.csv) | Per-protein identity table, 776 rows, exp213's schema. |
| [`data/foldbench_targets.csv`](data/foldbench_targets.csv) | All 334 FoldBench monomers: resolved entity, chain-match axis, source organism, sequence. |
| [`data/eval_queries_expanded.fasta`](data/eval_queries_expanded.fasta) | The 776 queries (exp213's 554 verbatim + 222). |
| [`data/foldbench_rest_queries.fasta`](data/foldbench_rest_queries.fasta) | Just the 222 net-new. |
| [`data/query_set_validation.json`](data/query_set_validation.json) | Every set-construction checksum and validation. |
| [`data/survival_headline.csv`](data/survival_headline.csv) · [`survival_by_dataset.csv`](data/survival_by_dataset.csv) · [`newer_vs_older.csv`](data/newer_vs_older.csv) | The result tables. |
| [`data/survival_by_arm.csv`](data/survival_by_arm.csv) · [`arm_complementarity.csv`](data/arm_complementarity.csv) · [`arm_attribution.csv`](data/arm_attribution.csv) | AFDB vs ESM-Atlas vs both. |
| [`plots/summary.pdf`](plots/summary.pdf) | Narrative + plot appendix. |

The MMseqs2 intermediates (`queryDB_expanded`, `alnDB_expanded`,
`aln_expanded.m8`) stay in `/data/exp213_overlap/` alongside exp213's; the
17 GB `targetDB` is shared and was not rebuilt.
