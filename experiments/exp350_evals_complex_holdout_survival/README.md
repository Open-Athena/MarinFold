---
marinfold_experiment:
  issue: 350
  title: 'exp: curate 30%-held-out experimental complexes and evaluate contacts plus Helico structures'
  kind: evals
  branch: exp350/complex-survival-audit
---

# exp: curate 30%-held-out experimental complexes and evaluate contacts plus Helico structures

**Issue:** [#350](https://github.com/Open-Athena/MarinFold/issues/350) · **Kind:** `evals` · **Branch:** `exp350/complex-survival-audit`

## Question

Can we construct a useful evaluation set of experimental protein complexes in which **every constituent protein chain is below 30% sequence identity to the training data**, and measure exp343's inter-chain contact R-precision and structural accuracy after Helico folding?

The first deliverable is a **survival table and an evidence-backed feasibility decision**, before running predictors.

## Hypothesis

A freshly audited subset of FoldBench protein–protein assemblies is the best starting point because existing monomer training data was filtered against all FoldBench chains. However, the complex-training corpus used a narrower reference, so the surviving count is unknown. PINDER test dimers and, if needed, newer experimental PDB assemblies may add independent clean complexes.

## Background

- #343 / #349 trained `contacts-v1-exp343-m2-p06-complex-1.5B`, final step 280154, on exp277's native + ProteinMPNN mixture and #294's AFCDB/PINDER complexes. Complex LM loss improved, but experimental interface R-precision and assembly quality have not been measured.
- #225 intended to filter native AFDB and ESM-Atlas against the legacy eval union and all FoldBench protein chains at >=30% identity over >=50% of the shorter sequence. Its search used MMseqs' default alignment mode, whose score-derived identity estimate can disagree with an exact backtrace at this boundary; this audit checks the corpus the checkpoint actually consumed.
- #266 generated ProteinMPNN redesigns of retained native backbones; their actual sequences also need auditing.
- #294 filtered its complexes against the 577-target monomer eval reference and excluded PINDER val/test entries. This does not certify FoldBench complex chains against the final training mixture.
- FoldBench's protein–protein CSV contains 279 interfaces from 239 assemblies at local revision `90a6033fed4fcb74d4f1304e932fd93b52d48a6c`. PINDER-XL advertises 1,955 representative dimers. Neither is a post-filter count.
- Helico supports chain-aware contact inputs, but its benchmark runner currently selects the best ground-truth score across seeds and fixes DockQ chain mapping by chain names. These paths require adaptation before reporting confidence-selected, symmetry-aware end-to-end results.

## Approach

### Stage A — survival audit (immediate scope)

1. Pin source versions and the exact exp343 training manifest, including the held-out complex shard. Inventory reusable local sequence indexes and published metadata before building or transferring anything large.
2. Census FoldBench protein–protein assemblies and PINDER test dimers. Lead with natural, protein-only biological dimers; distinguish homo/heterodimers, designs, antibodies/peptides, higher assemblies, unsupported chemistry, incomplete interfaces and context-ineligible inputs. Preserve every candidate with a named terminal status.
3. Define a contaminating sequence hit as **identity >=0.30 and aligned coverage >=0.50 of the shorter sequence**. Reject a complex if any constituent chain hits any actual training chain. Use sensitive alignment searches with pinned parameters, explicit identity/coverage conventions, adequate hit limits and threshold checks. Preserve supporting alignments; an unsearched component is unknown, never a pass.
4. Audit native AFDB, native ESM-Atlas, both actual ProteinMPNN sequence corpora, and both complex-source arms (AFCDB/PINDER). Reuse complete earlier alignments only with verified sequence and corpus provenance. Keep source-backbone lineage for redesigns and distinguish exact entries, homologs, and structural relatives.
5. Audit Helico's inherited Protenix and contact-conditioning training exposure separately. Record MarinFold-clean and end-to-end-clean eligibility independently; temporal separation alone is not 30% sequence separation. Identify targets previously used for Helico development.
6. Produce per-source and cumulative survival counts, by homo/heterodimer, plus per-chain evidence and per-complex rejection reasons. Report independent sequence/interface groups rather than treating multiple interfaces or repeated structures as independent targets.
7. Decide whether FoldBench suffices, whether PINDER adds useful clean targets, or whether newer experimental PDB assemblies are needed. Do not weaken the identity rule to increase yield. For the existing checkpoint, filter evaluation candidates; removing training rows retrospectively cannot repair contamination.

### Stage B — freeze the benchmark

After the survival audit is reviewed, freeze biological-assembly identities, sequences, coordinate snapshots, residue/chain mappings, experimental-quality criteria, context eligibility and all exclusion reasons. Split development/test by connected homology groups and keep repeated interfaces/assemblies together. Previously used development targets belong in development. Keep antibodies, peptides, designs and higher-order complexes as explicitly separate extensions.

### Stage C — contact evaluation

Use the established 100-rollout/resampling recipe, adapted and validated for multi-chain contacts-v1. Rank all eligible resolved cross-chain residue pairs, with no intra-chain sequence-separation exclusion applied across chains. Ground truth uses the pinned pyconfind contacts-v1 operator and threshold. Primary metric: inter-chain R-precision, where R is the number of true interface contacts. Report intra-chain accuracy, completeness, uncertainty, and appropriate null/control predictions separately. Handle equivalent-chain permutations consistently.

### Stage D — Helico structural evaluation

Compare no contacts, predicted intra-chain contacts only, predicted intra+inter-chain contacts, and oracle contacts as a diagnostic ceiling. Select contact budgets on development data; true R must never set inference inputs. Pin Helico weights/code and use confidence-based sample selection; label ground-truth best-of-N only as oracle. Primary structural metric: DockQ and success at DockQ >=0.23, with interface RMSD and per-chain quality. Use symmetry-aware chain mapping and matched baselines with explicit training exposure. Aggregate within assemblies and bootstrap independent complex groups.

## Success criteria

- A reproducible survival table with exact candidate denominators, per-corpus losses, independent-group counts, explicit unknowns and actionable conclusions about dataset feasibility.
- Frozen FASTA/manifests, sequence-search provenance and per-chain supporting alignments; no unsupported claim that a candidate passed an unsearched training component.
- Clear separation of MarinFold-only holdout from end-to-end Helico holdout and prior development exposure.
- If a viable dataset exists, a frozen development/test split and later per-complex R-precision and structural results with confidence selection and uncertainty.
- Every eventual predictor run records per-input timing, worker metadata, exact model/run/step and code revisions. Small CSVs/plots live in git; large artifacts go to the public MarinFold HF bucket, with working compute/storage co-located.
- No predictions are required for Stage A. No cross-region copy above 10 GB or shared-cluster disruption is authorized.

## Results

The requested settings do **not** yield a useful complex benchmark from
FoldBench PPI plus PINDER test. The locally auditable MarinFold arms leave zero
FoldBench complexes and one PINDER homodimer, before either ProteinMPNN redesign
arm is considered.

| cumulative stage | FoldBench PPI | PINDER test |
| --- | ---: | ---: |
| source assemblies/dimers | 239 | 1,955 |
| natural protein-only dimers passing scope/quality | 184 | 1,929 |
| after AFCDB complex training | 14 | 40 |
| after PINDER complex training | 12 | 34 |
| after native AFDB | 4 | 8 |
| after native ESM-Atlas | **0** | **1** |

The FoldBench scope survivors are 150 homodimers and 34 heterodimers. No
heterodimer survives AFCDB. The sole local MarinFold survivor is PINDER target
`6xnr__A1_UNDEFINED--6xnr__A2_UNDEFINED`, a 290-residue homodimer released
2020-08-26 at 2.05 A. It has not been cleared against the two ProteinMPNN
sequence corpora, so **one is an upper bound, not a certified survivor**.

The conservative Helico fine-tuning exposure screen contains a qualifying hit
for 6XNR. It therefore leaves zero candidates for a claim that the whole
MarinFold-to-Helico path is sequence-held-out. Helico's complete inherited
Protenix pretraining set remains unreconstructed, but that unknown cannot
restore 6XNR after the observed fine-tuning exposure.

The exact identity calculation matters. For example, exp225 recorded the
8AXI-A versus ESM-Atlas target
`2762233d21f8e9b60694382259bc6b80` at estimated `fident=0.297`; the same
alignment with an MMseqs backtrace has 200 identical positions over 620 aligned
positions (32.26%). The retained checkpoint corpus therefore contains hits that
the earlier score-derived estimate placed just below 30%. This audit runs
`--alignment-mode 3 -a 1`, preserves `nident`, and applies the boundary with
integer arithmetic.

The complex-corpus search reached its 100,000-candidate median prefilter cap.
Its apparent no-hit calls are consequently a high-sensitivity screen, while
all reported exclusions have concrete alignment witnesses. This limitation can
only reduce the one-candidate upper bound. Native searches were narrowed to all
105 full/resolved sequence representations of the 46 complex-clean candidates;
their median prefilter list was 20,565, below the 100,000 cap. Helico used a
500,000 cap against a 454,536-chain reference.

Reproduce the committed tables and plot from the prepared local databases:

```bash
uv run python build_candidates.py
uv run python analyze_survival.py
uv run python select_survivor_queries.py
uv run python search_sequences.py --arm native \
  --queries data/native_queries.fasta --threads 48
uv run python analyze_survival.py
uv run python plot_survival.py
uv run python build_summary.py
```

The one-time complex decoder and Helico reference builders are
`build_complex_sequences.py` and `prepare_helico_reference.py`. Search commands,
hashes, database counts and source provenance are committed under `data/`; large
alignment databases and logs remain under `/data/exp350`.

## Conclusion

FoldBench is not viable for a 30%-identity-held-out evaluation of the current
exp343 checkpoint, and PINDER test does not rescue it. Auditing the 320 GB of
ProteinMPNN redesign sequences is unnecessary for this feasibility decision:
those arms cannot increase an upper bound of one, and that one is already
exposed to Helico fine-tuning.

Stage B should not freeze either source as the benchmark. The next curation
pass should start from experimental complexes outside FoldBench/PINDER and
search them against these same checkpoint-exact indexes before manual review.
Release date can prioritize candidates but cannot substitute for the sequence
test. If that search still has poor yield, the practical alternative is to
define the evaluation set first and train a new complex model with every arm,
including redesigns and complex documents, decontaminated against the frozen
reference. No R-precision or structure predictions should be launched from
this candidate pool.
