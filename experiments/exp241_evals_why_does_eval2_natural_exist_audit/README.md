---
marinfold_experiment:
  issue: 241
  title: "exp: why does eval2-natural exist? audit the mechanisms that let 78 recent natural proteins escape a 40% identity filter against 70.9M training sequences"
  kind: evals
  branch: claude/eval2-natural-analysis-932188
---

# exp: why does eval2-natural exist?

**Issue:** [#241](https://github.com/Open-Athena/MarinFold/issues/241) · **Kind:** `evals` · **Branch:** `claude/eval2-natural-analysis-932188`

## Question

`eval2` ([#226](https://github.com/Open-Athena/MarinFold/issues/226)) keeps only
eval proteins whose sequence identity to our training data is **under 40 %**.
307 survive; **78 are labelled natural**, and that 78 is the subset every
novel-protein claim now rests on.

The prior that motivates the issue: almost every *natural* protein deposited to
the PDB recently has a sequence that was determined years earlier, is in UniProt,
and therefore has an AlphaFold model in AFDB and/or a close relative in the ESM
Atlas. On that prior `eval2-natural` should be close to **empty**. It is 78 — and
**53 of the 78 have zero significant hits anywhere in 70.9 M training
sequences**, at a median length of 148 aa. A 400 aa natural protein with no
detectable homolog in 70 M sequences is not a thing that should happen.

## eval2's exact definition (the question asked in passing)

**40 %, not 30 %.** A protein is in eval2 iff

```
max{ fident : evalue <= 1e-3  AND  qcov >= 0.50 }  <  0.40
```

over the union of **both** training arms, or it has no hit meeting those
conditions at all. Alignments come from `mmseqs search -s 7.5 --max-seqs 5000
-e 10`. 30 % is carried as the retrospective `passes_30` column (275 proteins),
not the definition. `upstream.EVAL2_THRESHOLD` re-derives eval2 membership from
that inequality and raises if it disagrees with the published manifest, so the
threshold reported here is demonstrably the one that built the set
([`tests/test_audit.py`](tests/test_audit.py)).

The comparison target is **not** "everything known". It is
**70,889,604 sequences** assembled as:

| arm | n | what it is | share of its parent DB |
| --- | ---: | --- | ---: |
| AFDB (#53) | 4,129,682 | AFDB v4 → `afdb-24M` → exp53 cluster selection | **1.9 %** of AFDB v4 |
| ESM-Atlas (#139) | 66,759,922 | ESM Atlas 40 %-linclust representatives | 10.8 % of ESM Atlas v0 |

And the AFDB arm's provenance chain contains **two independent filters against
structurally singular proteins** — which is the crux of the answer:

```
AFDB v4 (214 M)
  --[full-length only; mean pLDDT >= 70; length <= 2048]-->  ~30 M
  --[must be in BOTH the AFDB50 sequence-cluster file AND the structural
     cluster file with cluFlag=2; "fragments, singletons and sequence-only
     entries are excluded"]-->                               24,009,002
  --[exp53: top 5 members per struct_cluster_id by pLDDT, and DROP any
     cluster with fewer than 3 usable members]-->             4,129,682
```

## Approach

Analysis only — CPU, no cluster job, no model inference. It reuses exp226's
committed identity table, exp213's arm FASTAs and 17 GB target database on
`/data/exp213_overlap`, and exp65's source manifests, plus light RCSB / UniProt /
EBI-AFDB REST traffic. One new MMseqs2 search (§6, the base rate), 260 s.

| File | Role |
| --- | --- |
| `upstream.py` | The single seam onto exp226 / exp213 / exp65, and the place eval2's definition and the arm provenance are asserted rather than restated. |
| `annotate_rcsb.py` | **Step 1** — resolve all 776 eval proteins to an RCSB entity; four independent designed signals, UniProt xrefs, taxonomy, dates. |
| `check_training_reachability.py` | **Step 2** — is the sequence unknown or just unsampled? AFDB API, exact accession membership in the arm, UniRef50/90 cluster intersection. Also the positive control. |
| `measure_base_rate.py` | **Step 3** — the unconditioned rate: 585 random recent PDB chains through eval2's own filter and target DB. |
| `analyze.py` | **Step 4** — the mechanism ladder and the cross-tabs. |
| `plot_mechanisms.py` | **Step 5** — the four mechanism figures. |
| `apply_correction.py` | **Step 6** — the corrected manifest, and exp226's scoreboard recomputed on it. |
| `plot_eval2_natural_scoreboard.py` | **Step 7** — where MarinFold stands on the audited n=63. |
| `tests/test_audit.py` | 22 unit tests, no network. |

```bash
uv sync --extra test
uv run --extra test pytest tests/          # 22 tests, <1 s

uv run python annotate_rcsb.py --cohort all       # RCSB, ~3 min
uv run python check_training_reachability.py      # UniProt/AFDB, ~4 min
uv run python measure_base_rate.py --n 1400       # RCSB + one mmseqs search, ~15 min
uv run python analyze.py
uv run python plot_mechanisms.py
uv run python apply_correction.py             # manifest v2 + rescored headline
uv run python plot_eval2_natural_scoreboard.py
```

## Results

### 1. 15 of the 78 are not natural proteins at all

exp226 resolved RCSB source organisms **only for the FoldBench rows**. All 24
`cameo_hard` and 19 `casp_fm` rows inside the 78 carry `designed_any = 0` as a
*default* — nothing had ever looked. Looking finds
([`data/label_audit.csv`](data/label_audit.csv)):

| signal | in the 78 |
| --- | ---: |
| source organism is `synthetic construct` (taxon 32630) | 14 |
| entry keyword `DE NOVO PROTEIN` | 10 |
| **either** | **15** |

All 15 are `cameo_hard`, and every one is unambiguous from its own title —
*"De novo Design of Near Infrared Fluorescent Proteins"*, *"De novo designed
cholic acid binder"*, *"The designed serine hydrolase known as win1"*,
*"Efficient and scalable protein design using a relaxed sequence space"*. CAMEO
draws from weekly PDB pre-releases, which are full of design-lab depositions.
Four more designs (`8gac_A`, `8k83_A`, `8k84_A`, `8oyy_A`) hide in
`foldbench_rest` outside eval2.

The correction runs one way only: **0 of the 396 `denovo_pdb` rows look natural**
under the same three tests, so nothing moves back.

> **eval2-natural is 63, not 78** — and eval2 is **77 % designed** (244/307), not
> 75 %.

Chain resolution is controlled: 756/776 eval proteins match their resolved RCSB
entity exactly, as a substring, or as a gapped subsequence; the 20 that do not
are 16 `denovo_pdb` and 4 CASP domains whose labels do not turn on the entity.
Spot checks confirm the hard cases (`9had_C` → the designed binder, not the mite
allergen; `9hac_B` → `BBF-14_binder4`; `9b7d_D` → the metagenomic Tad3).

→ [`plots/mechanism_ladder.png`](plots/mechanism_ladder.png)

### 2. The premise is right: these sequences are old, and AlphaFold folded most of them

Of the 63 audited-natural proteins ([`data/training_reachability.csv`](data/training_reachability.csv)):

| check | n | of 63 |
| --- | ---: | ---: |
| has a UniProt sequence entry | **60** | 95 % |
| AlphaFold DB has a model for that accession | **45** | 71 % |
| that accession is in **our AFDB training arm** | **0** | **0 %** |

The sequences are not new. UniProt first published them a **median of 15 years
ago** (median 2011, range 1987–2026): `P12255` (*Bordetella pertussis*) has been
public since **1989-10-01**, `P36291` since **1994-06-01**, `P61825` since 2004.
The user's intuition about sequence age is exactly right.

What is wrong is the step "in AFDB ⇒ in our training set". Our AFDB arm is
**1.9 % of AFDB**, filtered twice against structurally singular proteins. So
**45 of the 63 are proteins AlphaFold folded and we simply did not train on**.

They also have relatives that exist and that we do not hold: the 56 with UniRef
statistics sit in UniRef50 clusters totalling **3,651 sequences** (median cluster
6, up to 717) of which **0** are in the arm. That last number is partly
tautological — eval2 selects on low identity to the arm — so it is reported as
corroboration, not as the proof; §6 is the uncircular measurement.

→ [`plots/known_but_unsampled.png`](plots/known_but_unsampled.png)

### 3. The mechanism ladder

Each of the 78 is charged to the earliest reason that applies, every rung a check
that was run ([`data/mechanism_counts.csv`](data/mechanism_counts.csv),
[`data/mechanism_table.csv`](data/mechanism_table.csv)):

| mechanism | n | share | from CASP/CAMEO |
| --- | ---: | ---: | ---: |
| **not natural** (designed protein) | 15 | 19 % | 15 |
| natural, but not a UniProt sequence | 3 | 4 % | 2 |
| in UniProt, but AlphaFold never folded it | 15 | 19 % | 10 |
| **folded by AlphaFold — we just did not train on it** | **45** | **58 %** | 16 |
| search miss (pipeline defect) | **0** | 0 % | 0 |

The three `not_in_uniprot` cases are each explicable and none is a new sequence:
`9b7d_D` is metagenomic (Tad3, never deposited as a UniProt entry), `9dl1_D` is
**TRACeR-I**, an engineered peptide-MHC receptor scaffold (arguably designed, so
63 is still an overcount), and `8qnf_A` is an NRPS condensation domain RCSB never
cross-referenced.

### 4. Viral proteins are the hole in both corpora

The single largest taxonomic driver ([`data/kingdom_by_arm.csv`](data/kingdom_by_arm.csv)):

| kingdom | n | has AFDB-arm hit | has ESM-Atlas hit | **survives into eval2** |
| --- | ---: | ---: | ---: | ---: |
| **virus** | 41 | **22 %** | **41 %** | **66 %** |
| bacteria | 227 | 88 % | 84 % | 15 % |
| eukaryote | 114 | 86 % | 79 % | 13 % |
| archaea | 8 | 75 % | 75 % | 25 % |

**27 of the 63 audited-natural eval2 proteins are viral** — 43 % of the set. Both
arms miss them for different reasons: AFDB's pLDDT ≥ 70 and cluster-membership
filters strip poorly-modelled and structurally singular viral proteins, and the
ESM Atlas is metagenomic (MGnify), which carries phage but not eukaryotic
viruses. 10 of the 15 `afdb_absent` proteins are viral.

→ [`plots/kingdom_gap.png`](plots/kingdom_gap.png)

### 5. The search is not broken — the positive control passes

12 eval proteins have their exact UniProt accession in the AFDB training arm.
**11 of 12 are found at ≥ 90 % identity** (nine at ≥ 94 %, two at 1.000); the twelfth
(`5ys7_A`, a designed protein) comes back at 0.875, an alignment difference, not
a lost hit ([`data/arm_membership_control.csv`](data/arm_membership_control.csv)).
No protein in the 78 has its accession in the arm, so the `search_miss` count is
0 by measurement rather than by assumption.

That only **12 of 776** eval proteins have an exact accession in the arm is
itself the finding restated: the corpus rarely contains the specific protein, it
contains a relative.

### 6. The base rate — the uncircular measurement

Everything above conditions on eval2 membership, and eval2 *is* the filter. So:
585 protein chains sampled at random offsets from the 183,327 RCSB polymer
entities deposited since 2022, one per entry, deduplicated by sequence, put
through **eval2's own filter against exp213's own target database**
([`data/base_rate_summary.csv`](data/base_rate_summary.csv)):

| population | n | no ≥ 40 % training relative |
| --- | ---: | ---: |
| **random recent PDB natural chains** | 576 | **7.1 %** (41) — 95 % CI [5.2 %, 9.5 %] |
| ...at < 30 % | 576 | 5.0 % (29) |
| FoldBench-100 | 100 | 15 % |
| FoldBench rest (+222) | 222 | 10 % |
| CASP-FM + CAMEO-hard | 58 | **74 %** |

**About one natural protein in fourteen escapes this filter, before any eval-set
curation at all.** The eval universe holds **358** audited-natural proteins and
**63** of them are in eval2 — 17.6 %, against a 7.1 % unconditioned base rate.
Curation explains the gap, not the existence: CASP-FM and CAMEO-hard select for
template-free difficulty and run at 74 %, while the FoldBench slices — a plain
cut of recent PDB — run at 15 % and 10 %, within a factor of two of the base
rate.

The kingdom effect reproduces on this unconditioned sample, independently of the
eval sets: **viral 31.4 % (16/51) vs bacterial 1.8 % (3/171)** — Fisher OR 25.6,
*p* = 4.3 × 10⁻⁹.

→ [`plots/base_rate.png`](plots/base_rate.png)

### 7. The ESM-Atlas arm cuts at the same 40 % eval2 does

The ESM-Atlas arm is 66.76 M **linclust representatives at 40 % identity**
(#91/#139) — the same number eval2 thresholds on. Its identity profile over the
776 shows exactly the pile-up that predicts
([`data/arm_identity_histogram.csv`](data/arm_identity_histogram.csv)):

| arm | share of hits in [0.40, 0.55) | share ≥ 0.90 |
| --- | ---: | ---: |
| AFDB | 31 % | 9.9 % |
| **ESM-Atlas** | **44 %** | **4.5 %** |

A query can be a near-duplicate of an ESM Atlas *member* and still fall below
40 % against that member's *representative*. The mode of the ESM-Atlas
distribution sits directly on eval2's cut, which makes the filter knife-edge
sensitive on that arm — worth knowing before anyone tightens the threshold.

### 8. Recency is not the explanation

Among audited-natural eval proteins with a deposit date
([`data/survival_by_deposit_year.csv`](data/survival_by_deposit_year.csv)),
survival is 16 % (2022), 12 % (2023), 23 % (2024) — no monotone trend. Newer
depositions are not more novel; exp226 found the same thing from the other
direction (its newer FoldBench monomers were *more* homologous, not less).

### 9. Applying the correction changes the scoreboard — in MarinFold's favour

[`apply_correction.py`](apply_correction.py) emits
[`data/eval2_manifest_v2.csv`](data/eval2_manifest_v2.csv): exp226's manifest,
every original column preserved so it is a drop-in replacement, with
`designed_any` corrected and the evidence carried beside it
(`designed_any_exp226`, `designed_source`, `kingdom`, `is_viral`,
`escape_mechanism`, `entry_title`). eval2 becomes **244 designed / 63 natural**.

Moving the 15 changes the numbers, so they are recomputed — importing exp226's
own `aggregate` and `paired_deltas` (same seed, same 10,000 resamples, same
estimator), so the only thing that differs is membership
([`data/correction_effect.csv`](data/correction_effect.csv)):

| R-precision (all) | published, n=78 | audited, n=63 | change |
| --- | ---: | ---: | ---: |
| MarinFold #199 | 0.3372 | **0.3133** | −0.024 |
| Protenix-v2 single-seq | 0.3259 | **0.2303** | **−0.096** |
| ESMFold | 0.4623 | 0.3980 | −0.064 |
| ESMFold2 | 0.5293 | 0.4845 | −0.045 |
| Protenix-v2 + MSA | 0.6979 | 0.6909 | −0.007 |
| seq-KNN (null) | 0.1478 | 0.1754 | +0.028 |

The 15 designs were where the *baselines* were strong, not MarinFold — so
removing them **strengthens** #226's headline rather than softening it:

| paired delta (R, all) | published, n=78 | audited, n=63 |
| --- | --- | --- |
| **MarinFold − Protenix-v2 single-seq** | +0.011 [−0.044, +0.069] — tie | **+0.083 [+0.031, +0.136] — MarinFold wins** |
| MarinFold − ESMFold | −0.125 [−0.173, −0.080] | −0.085 [−0.135, −0.039] |
| MarinFold − ESMFold2 | −0.192 [−0.239, −0.146] | −0.171 [−0.224, −0.121] |
| MarinFold − Protenix-v2 + MSA | −0.361 [−0.418, −0.301] | −0.378 [−0.443, −0.309] |
| MarinFold − seq-KNN (null) | +0.189 [+0.141, +0.236] | +0.138 [+0.088, +0.184] |

#226 reported that MarinFold's parity with Protenix-v2 single-seq "comes back" on
the natural half. On the *audited* natural half it is no longer a tie:
**MarinFold beats Protenix-v2 single-seq by +0.083, significant.** At <30 %
(n=46) the sign flips from −0.041 to +0.040 but stays a tie.

Everything else #226 concluded holds: MarinFold still loses to ESMFold, ESMFold2
and Protenix+MSA on eval2-natural, all significant, and still beats the seq-KNN
null by a wide margin.

### 10. The viral half and the non-viral half do not rank alike

Because 27 of the 63 are viral, the stratification is not cosmetic
([`data/eval2_headline_v2.csv`](data/eval2_headline_v2.csv)):

| R-precision (all) | viral (27) | non-viral (36) |
| --- | ---: | ---: |
| Protenix-v2 + MSA | 0.621 | 0.743 |
| ESMFold2 | 0.358 | 0.580 |
| ESMFold | 0.257 | 0.504 |
| **MarinFold #199** | **0.253** | 0.359 |
| Protenix-v2 single-seq | 0.210 | 0.246 |
| seq-KNN (null) | 0.073 | 0.252 |

**On viral proteins MarinFold ties ESMFold** — paired delta −0.004 [−0.059,
+0.045], not significant — while on non-viral it loses by 0.145 [−0.216,
−0.080]. MarinFold's margin over Protenix single-seq is the mirror image: +0.113
[+0.036, +0.192] non-viral, +0.043 [−0.017, +0.102] viral. A single pooled
eval2-natural number averages two regimes with different rankings.

→ [`plots/eval2_natural_scoreboard.png`](plots/eval2_natural_scoreboard.png)

**Which MarinFold checkpoint.** The bars are `contacts-v1-exp199-1.5B` (CoreWeave
p06) — the checkpoint every baseline in `eval2_per_protein.csv.gz` was scored
beside. The current default is the **p06 cooldown**, which scores **0.3579**
against p06's 0.3372 on the *published* n=78 (directly comparable on that
subset). Its per-protein eval2 rows live on CoreWeave S3, which is not reachable
from the workstation, so re-cutting it to n=63 needs one in-cluster job — worth
doing before any eval2-natural number is published for the current default.

## Conclusion

**eval2-natural exists because "our training corpus" is not "everything known" —
it is a 1.9 % sample of AFDB plus 67 M metagenomic cluster representatives — and
because a fifth of the set was never natural.**

The premise in the issue is correct on every point it makes about the databases,
and wrong on the one step it takes for granted:

- **The sequences are old.** Median UniProt first-public year 2011, some from
  1987–1994. None of this set is a newly determined sequence.
- **AlphaFold folded most of them.** 45 of 63 have an AFDB model. We did not
  train on a single one.
- **15 of the 78 are de novo designs** that a proxy exp226 never ran on
  CAMEO/CASP rows silently passed as natural. The honest count is 63, and eval2
  is 77 % designed.
- **7 % is the base rate**, measured on a random unconditioned sample of recent
  PDB. eval2-natural is not an anomaly needing explanation; it is that rate times
  an eval universe, amplified by two novelty-curated sources running at 74 %.
- **43 % of it is viral**, because both corpora systematically miss viruses.

**What this changes for how eval2-natural is used** — for
[`reference_eval2_default_eval_set`](../../.agents/skills/eval-checkpoint/SKILL.md)
and the [#180](https://github.com/Open-Athena/MarinFold/issues/180) tracker:

1. **The n is 63, not 78 — done.** `data/eval2_manifest_v2.csv` is the drop-in
   replacement, the scoreboard is recomputed on it (§9), and the
   `eval-checkpoint` skill now points at it. Applying the correction *raises*
   MarinFold's standing: it beats Protenix-v2 single-seq by +0.083 on the
   audited natural set, where the published n=78 said tie.
2. **"No homolog in the training set" ≠ "novel protein".** It means unsampled.
   A claim about generalisation to *novel* proteins needs the fold-novelty axis
   (#41's Foldseek verdict), not this filter alone.
3. **eval2-natural is 43 % viral, and the halves rank differently** (§10) — on
   viral proteins MarinFold ties ESMFold; on non-viral it loses by 0.145. The
   skill now requires the `is_viral` split.

**Cheapest way to grow the set**, now that the mechanism is known: the 7 % base
rate over 183 k recent PDB entities is ~13 k candidate natural chains. Sampling
that population directly — rather than filtering curated benchmarks — is the
route to an eval2-natural of several hundred, and unlike #226's FoldBench
expansion it does not run out.

**Not done here:** the fold-novelty axis for the 63 (needs exp41's Foldseek DB,
which lives on a Modal volume), and **the current default checkpoint's
eval2-natural score on n=63** — the p06 cooldown's per-protein eval2 rows are on
CoreWeave S3 and need one in-cluster job to re-cut. No model was run for this
experiment; §9's numbers are exp226's existing per-protein scores re-aggregated
under the corrected split.

## Artifacts

| File | Contents |
| --- | --- |
| [`data/eval2_manifest_v2.csv`](data/eval2_manifest_v2.csv) | **The corrected eval2 manifest** — drop-in replacement for exp226's, `designed_any` fixed, evidence columns attached. |
| [`data/eval2_headline_v2.csv`](data/eval2_headline_v2.csv) · [`eval2_paired_deltas_v2.csv`](data/eval2_paired_deltas_v2.csv) · [`correction_effect.csv`](data/correction_effect.csv) | exp226's scoreboard on the audited split, with viral / non-viral strata, and old-vs-new side by side. |
| [`data/mechanism_table.csv`](data/mechanism_table.csv) | The 78, one row each: mechanism, kingdom, UniProt/AFDB status, UniRef cluster stats, dates, title. |
| [`data/mechanism_counts.csv`](data/mechanism_counts.csv) | The ladder, with the CASP/CAMEO split. |
| [`data/label_audit.csv`](data/label_audit.csv) | The 19 proteins whose designed/natural label changed, with the evidence and the entry title. |
| [`data/training_reachability.csv`](data/training_reachability.csv) | Per-protein UniProt xref, AFDB-full, arm membership, UniRef50/90 sizes and intersections. |
| [`data/arm_membership_control.csv`](data/arm_membership_control.csv) | The positive control — 12 eval proteins whose exact accession is in the arm. |
| [`data/base_rate_per_protein.csv`](data/base_rate_per_protein.csv) · [`base_rate_summary.csv`](data/base_rate_summary.csv) | The 585-chain unconditioned sample and its survival rates. |
| [`data/rcsb_annotation.csv`](data/rcsb_annotation.csv) | All 776 eval proteins: entity, organism, taxonomy, keywords, xrefs, dates, chain-resolution control. |
| [`data/kingdom_by_arm.csv`](data/kingdom_by_arm.csv) · [`arm_identity_histogram.csv`](data/arm_identity_histogram.csv) · [`survival_by_deposit_year.csv`](data/survival_by_deposit_year.csv) | The cross-tabs. |
| [`plots/summary.pdf`](plots/summary.pdf) | Narrative + plot appendix. |

The MMseqs2 intermediates (`query_exp241base.fasta`, `alnDB_exp241base`,
`aln_exp241base.m8`) stay in `/data/exp213_overlap/` alongside exp213's and
exp226's; the 17 GB `targetDB` is shared and was not rebuilt.
