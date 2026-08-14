---
marinfold_experiment:
  issue: 225
  title: 'exp: recreate the training corpora with a real eval-decontamination pass (all 554 eval proteins, sequence + structure)'
  kind: data
  branch: claude/github-issue-225-fead58
---

# exp: recreate the training corpora with a real eval-decontamination pass (all 554 eval proteins, sequence + structure)

**Issue:** [#225](https://github.com/Open-Athena/MarinFold/issues/225) · **Kind:** `data` · **Branch:** `claude/github-issue-225-fead58`

## Question

**Can we rebuild the contacts-v1 training corpora with a decontamination pass
that actually covers the eval set we report on — and what does it cost us in
corpus size and accuracy?**

[#213](https://github.com/Open-Athena/MarinFold/issues/213) measured the
overlap and found it large: **323 / 554 (58 %)** eval proteins have a
significant training homolog, and ESM-Atlas alone hits 296 of them. This is the
other half — the *fix*.

## Background: what is actually filtered today

Two corpora, two very different stories, and neither covers the benchmark we
report on.

**AFDB (`contacts_v1`, 4,129,682 train docs, [#53](https://github.com/Open-Athena/MarinFold/issues/53))
— no eval decontamination whatsoever.** `selection.py` filters on `seq_len`,
cluster size and pLDDT round only. Its train/val/test split is `afdb-24M`'s own
hash of `struct_cluster_id`, so no leakage *across the AFDB split* — but the
554 eval proteins were never purged from it. The consequences were measured and
never acted on: [#41](https://github.com/Open-Athena/MarinFold/issues/41) found
**99/100** FoldBench monomers fall in a fold the model trained on and 48 are
structurally near-identical to a *train* representative;
[#65](https://github.com/Open-Athena/MarinFold/issues/65) labelled 194
`redundant` / 215 `same_fold` / 48 `novel_fold` out of 457. Those verdicts exist
only as reporting strata. `exp65/notes/eval-strategy-summary.md` proposed the
right fix — cluster everything, choose eval clusters, delete those clusters from
train — and it was never implemented.

**ESM-Atlas (`contacts_v1_esm_atlas`, 66,759,922 docs,
[#91](https://github.com/Open-Athena/MarinFold/issues/91) →
[#139](https://github.com/Open-Athena/MarinFold/issues/139)) — one filter, with
four gaps.** #91's funnel dropped 41,517 sequences against an eval reference at
≥40 % identity / ≥50 % coverage, before clustering so dropped sequences could
not return as representatives. The gaps:

1. **FoldBench-100 was never in the reference.** `--eval-ref` pointed at exp65's
   `candidate_sequences.csv` — 457 rows / 454 proteins. The benchmark is 554 =
   those 454 **+ FoldBench-100**, so ~18 % of the eval set got no filtering at
   all, and per #41 it is the *most* contaminated slice we have.
2. **The threshold is looser than our own definition of leakage.** 40 % vs
   exp65's `REDUNDANT_ID = 0.30`. The 30–40 % band was kept by design.
3. **Sequence-only, on the axis #41 showed is the wrong one.** Of the 99
   FoldBench monomers with a same-fold-or-closer training match, **65 sit below
   30 % sequence identity**. #213 confirms it from the other side: of its 231
   *sequence-novel* eval proteins, 194 are still `same_fold`/`redundant` against
   the AFDB training folds.
4. **#91's own stated follow-up never happened** — re-running exp65's dedup with
   the Atlas in the training reference.

## Hypothesis

- **H1 (the corpora are recoverable):** a proper drop list removes a small
  enough fraction of both corpora that a retrained model is materially
  unchanged on the full 554, while the homology-free subset number becomes
  honest rather than something we back out post-hoc.
- **H0 (the fold-level purge is too expensive):** decontaminating on the
  structural axis at fold level (TM ≥ 0.5) deletes so much of the corpus that
  the resulting model is worse for reasons unrelated to leakage, and the honest
  fix is a better eval set, not a smaller training set.

The tiered design is built to distinguish these **before** committing to a
retrain.

## Approach

Both corpora are already materialised, so **decontamination is a row filter on
`entry_id` plus a re-shard, not a regeneration** — no pyconfind, no document
rebuild. What that turns into concretely:

### Stage 1 — one pinned decontamination reference

`build_reference.py` freezes the 554-protein benchmark as exp78's manifests
define it, in two halves:

- **Sequences** — `data/reference/eval_queries.fasta`, committed, and asserted
  byte-identical to #213's copy so both experiments provably search the same
  queries.
- **Structures** — the evaluated chain of each GT structure, extracted to a
  single-chain mmCIF named after the reference key. Committed alongside is
  `data/reference/eval_structures.csv` (sha256, chain, resolved length,
  source path); the 96 MB of mmCIFs go to the public bucket via
  `publish_reference.py`.

Chain selection is exhaustive rather than heuristic: 533 of the 554 files hold
exactly one polymer chain and it is taken whatever it is called (10 of them are
named differently from the manifest's `gt_chain` — `5sbj_A` is chain `C` inside
its own file); the other 21 are whole CAMEO/CASP entries and are resolved by
`gt_chain`. Anything unresolvable raises.

### Stage 2 — the drop lists

**Sequence axis** (`sequence_droplist.py`). #213 asked which *eval* proteins
have a training homolog and kept one row per query; this asks which *training
rows* are a homolog of some eval protein, so it keeps every hit. It reuses
#213's 70,889,604-sequence MMseqs2 database directly — every target in it is
headed `{arm}|{shard}_{row}_{entry_id}`, so a hit names the corpus row it came
from and inverting hits into a drop list needs no join against the corpus.

The search is re-run rather than reduced from #213's alignments because #213
capped the prefilter at `--max-seqs 2000` and 96 queries hit the cap.

**Structure axis** (`structure_droplist.py`). For AFDB this is nearly free:
`afdb-24M` is already Foldseek-clustered, every corpus row carries its
`struct_cluster_id`, and #41 published a Foldseek DB of the 1,331,330 cluster
representatives with a rep → split manifest. A hit purges a **whole cluster** —
exp65's Step 3, and the only version that makes train and eval fold-disjoint
rather than merely sequence-disjoint. Two properties of that shortcut are
stated rather than assumed: the search is at *representative* granularity (we
have structures for the 1.33 M reps, not all 4.13 M documents), and purging is
at *cluster* granularity for both structural tiers.

For ESM-Atlas there is no structural database and no structural cluster id —
its 66.76 M rows are clustered at 40 % *sequence* identity only. Building one is
the ~$1 k Foldseek job the issue gates on Stage 3, and until it exists the
ESM-Atlas structural tiers are reported as **unmeasured, not zero**.

### Stage 3 — three tiers, priced before anything is retrained

| Tier | Rule | What it buys |
|---|---|---|
| **A — sequence** | identity ≥ 30 % over ≥ 50 % query coverage, **or** E ≤ 1e-3 | Fixes gaps 1, 2 and 4. Cheap. |
| **B — + structurally redundant** | A, plus TM ≥ 0.90 to any eval structure | Removes near-identical structures a sequence search cannot see. |
| **C — + fold-level purge** | B, plus whole clusters at TM ≥ 0.50 | Fold-disjoint by construction. |

`survival.py` reports documents kept and dropped per (arm, tier), plus the
**sequence-only / structure-only / both** decomposition — the structure-only
column is the direct answer to whether #41's warning was worth acting on.

## Code & how to run

Everything below is CPU-only and local: no cluster job, no model inference. The
two expensive inputs already exist — #213's MMseqs2 database on this
workstation, and #41's Foldseek DB on the HF bucket.

| File | Role |
| --- | --- |
| `decontam_lib.py` | The contracts: tier ladder, thresholds, the `{arm}\|{shard}_{row}_{entry_id}` grammar, the filterable-corpus registry, binary installers. |
| `build_reference.py` | **Stage 1** — freeze the 554 sequences + single-chain structures, with checksums. |
| `publish_reference.py` | Push the reference and the drop lists to the public bucket under a versioned prefix. |
| `build_corpus_index.py` | Stream a corpus's id columns (`entry_id`, `struct_cluster_id`) over `HfFileSystem` — column projection, so the AFDB index costs ~200 MB of transfer instead of 13 GB. |
| `sequence_droplist.py` | **Stage 2a** — all-hits MMseqs2 search → per-training-row drop list. |
| `sweep_evalue.py` | Prices Tier A's dependence on the reporting threshold off the same search. |
| `foldbench_reference.py` | Builds a query FASTA for *all* of FoldBench (1,940 protein chains across every task), not just the 100 monomers the benchmark scores. |
| `identity_droplist.py` | Prices a pure "≥ 30 % identity" rule, over one reference or the union of several. |
| `structure_droplist.py` | **Stage 2b** — Foldseek the 554 against #41's representative DB → per-cluster drop list. |
| `survival.py` | **Stage 3** — per-tier survival and the per-axis decomposition. |
| `plot_decontam.py` | The four figures. |
| `tests/test_decontam.py` | Pure unit tests for the three places a bug would silently *under*-filter. |

```bash
uv sync --extra test
uv run --extra test pytest tests/                       # pure logic, <1 s

WORK=/data/exp225_decontam
uv run python build_reference.py --structures-out $WORK/eval_structures
hf buckets sync hf://buckets/silterra/afdb-24M-foldseek-train-reps $WORK/afdb_reps_db
uv run python build_corpus_index.py --arm afdb --workers 24
uv run python sequence_droplist.py  --work $WORK        # ~8 min
uv run python structure_droplist.py --work $WORK        # hours; TM-align over 1.33 M reps
uv run python sweep_evalue.py       --work $WORK
uv run python survival.py           --work $WORK
uv run python plot_decontam.py && uv run python build_summary.py
```

`build_reference.py` cross-checks its FASTA against #213's committed copy.
#213 is still on a branch ([PR #216](https://github.com/Open-Athena/MarinFold/pull/216)),
so until it lands, extract that file and pass it explicitly:

```bash
git show origin/claude/eval-sequence-overlap-analysis-33a77a:experiments/exp213_evals_train_sequence_overlap_audit/data/eval_queries.fasta > /tmp/exp213_queries.fasta
```

## Results

Stages 1–3 are complete. Stage 4 (republish) and Stage 5 (retrain) are not
started — they were gated on the numbers below, and the numbers change what is
worth doing.

### 0. The reference is pinned, and the drop list is verified rather than trusted

554 proteins (100 FoldBench + 396 de novo + 32 CAMEO-hard + 26 CASP-FM);
sequences byte-identical to #213's, structures reduced to the evaluated chain
with sha256s in [`data/reference/eval_structures.csv`](data/reference/eval_structures.csv).

The drop list is produced by a four-step chain — document decoded back to a
sequence (#213), written into a FASTA header, keyed by MMseqs2, parsed back out
— and every step is a place it could silently name the wrong rows.
[`validate_droplist.py`](validate_droplist.py) checks both directions: all
**77,887** AFDB entry_ids exist in the corpus, and the `(shard, row)`
coordinates the header carries independently resolve to the **same** entry_id
for every one of them. Same for a 60-shard ESM-Atlas sample (18,082 rows).
`entry_id` is unique in both corpora, so a filter keyed on it is exact.

### 1. Tier A is cheap, and the threshold is not doing the work

| arm | documents | dropped | % |
| --- | ---: | ---: | ---: |
| AFDB | 4,129,682 | **77,887** | **1.886 %** |
| ESM-Atlas | 66,759,922 | **1,047,096** | **1.569 %** |

No censoring: the busiest of the 554 queries drew 399,763 alignments against a
`--max-seqs` of 1,000,000.

Tier A's identity arm has no significance floor of its own, so how deep MMseqs2
is asked to report could in principle set the answer.
[`sweep_evalue.py`](sweep_evalue.py) prices that off the same search
([`data/evalue_sensitivity.csv`](data/evalue_sensitivity.csv)):

| reporting ceiling | AFDB | ESM-Atlas |
| --- | ---: | ---: |
| E ≤ 1e-3 (E-value arm alone) | 1.727 % | 1.373 % |
| E ≤ 1 | 1.845 % | 1.513 % |
| **E ≤ 10** (exp65 / #213) | **1.886 %** | **1.569 %** |
| E ≤ 100 | 1.927 % | 1.623 % |
| E ≤ 1000 | 1.970 % | 1.676 % |

Six decades of reporting depth move the answer by a quarter of a percentage
point. **"Tier A costs under 2 % of either corpus" is a statement about the
contamination, not about an mmseqs flag.**

→ [`plots/evalue_sensitivity.png`](plots/evalue_sensitivity.png)

### 2. All four of #91's gaps are real, and its sensitivity setting cost 124k rows

Every contaminated row re-derived under **#91's own rule against #91's own
reference** — 40 % identity, 50 % coverage of the eval protein, exp65's 454 —
and assigned the first category that applies
([`data/residual_attribution.csv`](data/residual_attribution.csv)):

| why it survived #91's funnel | ESM-Atlas rows | share | AFDB rows |
| --- | ---: | ---: | ---: |
| 30–40 % identity band (#91 cut at 40 %) | 491,634 | 47 % | 28,780 |
| remote homology, below any identity bar | 249,232 | 24 % | 15,979 |
| only a FoldBench-100 protein reaches it | 182,517 | 17 % | 20,384 |
| **clears #91's own 40 %/50 % rule** | **123,713** | **12 %** | 12,744 |

The last row is not a rule error. #91's funnel ran
`mmseqs search --alignment-mode 3 --min-seq-id 0.40 -c 0.50 --cov-mode 1`, with
the coverage gate on the same side as ours — but at `SEARCH_SENSITIVITY="4.0"`,
commented in `create_dataset.sh` as *"4.0 = faster, may miss a few near-40 %-id
hits"*. At exp65/#213's `-s 7.5` those misses are 123,713 published documents.
The trade-off was made knowingly; this is its price.

(The AFDB column is descriptive only — that corpus was never filtered at all,
so nothing there "survived" anything.)

### 3. #41 was right about the axis, and the fold-level purge is unaffordable

15,915,243 Foldseek TM alignments (96 min, 64 cores, no censoring — busiest
query 108,183 hits against a 1,000,000 cap). **945,861 of the 1,304,911 AFDB
train cluster representatives — 72 % — are within Foldseek's reach of some eval
structure at all.**

| arm | Tier A | Tier B (+ TM ≥ 0.9) | Tier C (+ fold purge TM ≥ 0.5) |
| --- | ---: | ---: | ---: |
| AFDB | 77,887 (**1.89 %**) | 100,207 (**2.43 %**) | 1,540,597 (**37.31 %**) |
| ESM-Atlas | 1,047,096 (**1.57 %**) | *not measurable* | *not measurable* |

Distinct structural clusters in the AFDB corpus fall 941,028 → 936,076 (A) →
929,284 (B) → **601,840** (C).

The per-axis decomposition ([`data/survival_by_axis.csv`](data/survival_by_axis.csv))
is what #41's warning asked for:

| tier | sequence only | **structure only** | both |
| --- | ---: | ---: | ---: |
| A | 77,887 | 0 | 0 |
| B | 62,965 | **22,320** | 14,922 |
| C | 22,316 | **1,462,710** | 55,571 |

**Tier B buys 22,320 documents of real structural decontamination for 0.54 % of
the corpus** — training structures near-identical to an eval structure that no
sequence search sees. That is #41's warning, confirmed and cheap to act on.

**Tier C costs 37.31 % of AFDB, and 95 % of that cost is structure-only.** The
reason is visible in the TM histogram: the mode of "best TM to any of the 554"
sits essentially *at* the 0.5 same-fold boundary, so fold-disjointness is not
failing on an unlucky threshold — a third of AFDB's structural clusters simply
are the same fold as something in a 554-protein eval set.

→ [`plots/survival_by_tier.png`](plots/survival_by_tier.png),
[`plots/axis_decomposition.png`](plots/axis_decomposition.png),
[`plots/tm_distribution.png`](plots/tm_distribution.png)

### 4. Three quarters of Tier C's cost protects the 396 de novo designs

Splitting the purge by which half of the eval set drives it
([`data/tier_scope.csv`](data/tier_scope.csv)):

| eval scope | Tier B | Tier C |
| --- | ---: | ---: |
| all 554 | 37,242 (0.90 %) | 1,518,281 (**36.77 %**) |
| natural 158 | 4,068 (0.10 %) | 387,396 (**9.38 %**) |
| designed 396 | 33,174 (0.80 %) | 1,130,885 (**27.38 %**) |

A de novo design is usually a small idealised bundle, so it is the same fold as
an enormous share of AFDB and purges far more of the corpus per query than a
natural protein does. Scoping the fold purge to the 158 natural eval proteins
costs **9.38 %** instead of 36.77 % — a 4× reduction — and it removes the folds
of exactly the proteins that can actually leak through evolutionary homology.
#213's finding that its homology-free subset is 80 % designs is the same fact
seen from the eval side: designs have no relatives to leak through.

This is not a recommendation to skip fold-decontaminating the designs. It is
the price of doing so, stated separately, because the two halves of Tier C's
third-of-the-corpus are not equally worth paying for.

### 5. A wider reference and a symmetric coverage gate: still under 2 %

Two variants worth pricing, because both are things one would plausibly want
instead of Tier A as specified.

**A wider reference.** The pinned v1 reference is the 554 we report on, of
which 100 are FoldBench monomers. FoldBench itself is far larger — its
protein-protein, antibody-antigen, protein-peptide, protein-ligand, protein-DNA
and protein-RNA tasks carry protein chains too. [`foldbench_reference.py`](foldbench_reference.py)
assembles all of them: **1,940 protein chains** (1,449 unique sequences) from
1,493 entries, selected by each task file's explicit chain-type column, with
assembly copies (`A-2`) folded back to their source chain. All 100 scored
monomers are inside it. Three entries (8C0H, 8PNI, 8XNH → 6 chains) no longer
exist in RCSB and are recorded as missing rather than silently skipped.

**A symmetric coverage gate.** Tier A gates on `qcov` — coverage of the *eval*
protein — following exp65 and #213. That misses a short training protein
aligning to one domain of a long eval protein. Gating on the **shorter of the
two sequences** (`max(qcov, tcov)`, since for a fixed aligned region the
shorter sequence has the larger coverage) closes that.

Under **≥ 30 % identity over ≥ 50 % of the shorter sequence, with no E-value
arm at all** ([`data/identity_droplist.csv`](data/identity_droplist.csv)):

| reference | AFDB | ESM-Atlas |
| --- | ---: | ---: |
| the 554 alone | 57,482 (1.39 %) | 782,179 (1.17 %) |
| all of FoldBench alone | 130,409 (3.16 %) | 573,382 (0.86 %) |
| **union** | **166,679 (4.04 %)** | **1,206,744 (1.81 %)** |

1,373,423 of 70,889,604 training proteins — **1.94 %** across both corpora.

Two things the table says. **All of FoldBench costs more AFDB than our own eval
set does** (3.16 % vs 1.39 %), which is what you would expect: its 1,940 chains
are all natural PDB proteins with real evolutionary families, where 396 of our
554 are de novo designs with almost nothing to purge. And **the union is well
below the sum** — 86 % of it in both arms — because the FoldBench monomers sit
in both references and a training protein is routinely homologous to several
eval proteins at once.

What each choice is worth, on the union:

| variant | AFDB | ESM-Atlas |
| --- | ---: | ---: |
| coverage of the **shorter** sequence | 4.04 % | 1.81 % |
| coverage of the reference only (Tier A's gate) | 2.92 % | 1.46 % |
| coverage of the training protein only | 2.39 % | 0.95 % |
| coverage of **both** (near-global) | 1.26 % | 0.59 % |
| shorter-gate **plus** Tier A's `E <= 1e-3` arm | 5.69 %¹ | 3.03 %¹ |

¹ reference-side gate, reduced at the `E <= 10` ceiling
([`data/identity_droplist_with_evalue.csv`](data/identity_droplist_with_evalue.csv)).

**A caveat on "no E-value threshold".** The rule applies none, but the search
only reported alignments to `E <= 1000`, and identity-plus-coverage has no
significance floor of its own, so it keeps accreting weak alignments the deeper
mmseqs reports. Reducing the same alignments only to `E <= 10` gives 3.63 % /
1.52 % rather than 4.04 % / 1.81 %. At 30 % identity over half of a short
sequence, chance alignments are reachable — the corpora contain sequences down
to 10 residues — so the number is a function of the reporting depth in a way
the E-value-armed tiers are not.

Either way the conclusion is unchanged: **even the widest reference and the
most permissive coverage gate leave 96–98 % of the training data intact**,
against the 37 % Tier C alone would delete from AFDB.

## Conclusion

**H1 holds through Tier B. H0 holds for Tier C, and the number that settles it
is 37.31 %.**

A decontamination pass that covers the whole benchmark — all 554, at our own
30 % identity bar rather than #91's 40 %, with the remote-homology catch-all —
costs **under 2 % of either corpus**, and that figure is stable across six
decades of the search's reporting threshold. Adding the structural-redundancy
rule takes AFDB to 2.43 % and buys 22,320 documents that no sequence filter can
see. Both are cheap enough that there is no reason not to do them. **The
recommended published tier is B for AFDB and A for ESM-Atlas.**

The fold-level purge is a different matter. It removes **37.31 %** of the AFDB
corpus and 36 % of its structural clusters, 95 % of that from the structural
axis alone, and the TM histogram shows why: the same-fold boundary cuts through
the mode of the distribution. #41's "nearly every FoldBench fold is represented
in AFDB train" is quantitatively right, and the consequence is that a
fold-disjoint training set is not a filtered version of this corpus — it is a
different, much smaller corpus, and a model trained on it would be worse for
reasons that have nothing to do with leakage. **Tier C is declined, with
37.31 % as the number that justifies declining it.** The 9.38 % natural-only
variant is a real middle option and is the one worth revisiting if the
fold-novelty question becomes load-bearing.

**The ESM-Atlas Foldseek build (~$1k) should not be paid for yet.** It was
gated on this table, and the table argues against it: the only tier it could
serve is B, since C is declined, and on the arm where we *can* measure it, Tier
B's structural increment is 0.54 % of the corpus. Spending ~$1k and a day of
cluster time to find an estimated half a percent more ESM-Atlas rows is poor
value while Stage 5 (the retrain) has not run. If the retrain shows the
decontaminated model moving materially, the build becomes worth revisiting.
The one caveat is that ESM-Atlas is metagenomic and its fold distribution
against this eval set need not match AFDB's — the AFDB number bounds the shape
of the answer, not the answer.

**What this does not settle.** Stage 5 is the actual test of H1: whether a
#199-recipe model retrained on the decontaminated mixture is unchanged on the
full 554 while #213's 0.611 → 0.549 homology-free drop mostly does not
reappear. Nothing here measures accuracy. And #213's closing limit still binds
on the eval side: its homology-free subsets are 80 % de novo designs, bottoming
out at n ≈ 19 natural proteins novel in both sequence and fold. A
decontaminated corpus makes the headline honest; it does not make that corner
measurable, which is what #65 was opened for.

### Two approximations worth carrying forward

- **The structural axis is measured at representative granularity.** We have
  structures for the 1.33 M AFDB cluster representatives, not for all 4.13 M
  documents, so a cluster is judged by its representative and purged whole. AFDB
  clusters *are* Foldseek clusters, so members are close to their
  representative by construction — but this is an approximation, not an exact
  per-row TM.
- **`qtmscore` drives the verdict**, matching the `fold_verdict` convention of
  #41/#65. The conservative `max(qtmscore, ttmscore)` is carried in the drop
  list and moves Tier C's cluster count from 487,969 to 543,666 (+11 %), so the
  choice does not change the conclusion.
