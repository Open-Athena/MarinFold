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

Can we construct a useful evaluation set of experimental protein complexes under either of two holdout definitions: **component-held-out**, where every chain is below 30% identity to every training chain, or **pair-held-out**, where no training complex contains homologs of both chains? Can it support exp343 inter-chain contact R-precision and structural evaluation with Helico?

The first deliverable is a **survival table and an evidence-backed feasibility decision**, before running predictors.

## Hypothesis

A freshly audited subset of FoldBench protein–protein assemblies is the best starting point because existing monomer training data was filtered against all FoldBench chains. The strict component rule may have poor yield, while the pair rule should retain complexes whose component families were seen separately but whose relationship was not. PINDER test dimers can add independent clean pairs.

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
3. Define a contaminating sequence hit as **identity >=0.30 and aligned coverage >=0.50 of the shorter sequence**. Evaluate two rules. The strict component rule rejects a complex if either chain hits any training chain. The pair rule rejects it only if one training complex document has a one-to-one assignment of both candidate chains to homologous chain instances. Use sensitive alignment searches with pinned parameters, explicit identity/coverage conventions, adequate hit limits and threshold checks. Preserve supporting alignments; an unsearched component is unknown, never a pass.
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

The answer depends on the holdout claim. The strict component-held-out rule is
not viable: it leaves zero FoldBench complexes and one PINDER homodimer after
the locally auditable MarinFold arms, and that last target is exposed to
Helico's documented fine-tuning pool. The proposed pair-held-out rule is viable
for a benchmark of unseen chain-family pairings.

| cumulative stage or definition | FoldBench PPI | PINDER test |
| --- | ---: | ---: |
| source assemblies/dimers | 239 | 1,955 |
| natural protein-only dimers passing scope/quality | 184 | 1,929 |
| strict component-held-out after local MarinFold arms | **0** | **1** |
| pair-held-out after AFCDB complex training | 40 | 187 |
| pair-held-out after PINDER complex training | **35** | **183** |
| pair-held-out plus conservative Helico fine-tuning screen | **32** | **65** |

For the pair rule, a complex training document is contaminating only when its
two chain instances can be assigned one-to-one to the two candidate chains and
both exact alignments pass 30% identity over 50% of the shorter sequence. An
AFCDB homodimer has two chain instances even though its decoded FASTA stores one
deduplicated sequence. PINDER training documents require two distinct stored
chains. The single-chain native AFDB, ESM-Atlas and ProteinMPNN documents cannot
contain a paired witness and therefore do not exclude candidates under this
definition.

The final MarinFold pair-held-out pool contains 218 dimers. FoldBench contributes
15 homodimers and 20 heterodimers; PINDER contributes 40 homodimers and 143
heterodimers. A conservative Helico screen rejects a candidate when two distinct
chains from the same documented fine-tuning PDB entry match the candidate pair.
It leaves 97 dimers: 13/19 FoldBench homo/heterodimers and 13/52 PINDER
homo/heterodimers. Helico's inherited Protenix pretraining exposure remains
unreconstructed, so 97 supports a fine-tuning-pair-clean claim rather than a
fully certified end-to-end holdout claim.

The broad complex-corpus search reached its 100,000-result median prefilter cap.
All 220 apparent survivors were therefore searched again with a 1,000,000-result
cap. The confirmation median was 86,868, below the cap, and two additional
PINDER AFCDB pair witnesses were found, producing the final count of 218. Exact
MMseqs backtraces (`--alignment-mode 3 -a 1`) provide `nident`; the 30% boundary
is applied with integer arithmetic. This avoids the earlier score-derived
identity error, exemplified by an 8AXI-A/ESM-Atlas alignment estimated at 29.7%
but containing 200 identities over 620 aligned positions (32.26%).

Reproduce the pair analysis and figure after preparing the local indexes:

```bash
uv run python analyze_pair_survival.py
uv run python select_pair_queries.py
uv run python search_sequences.py --arm complex \
  --queries data/pair_queries.fasta --work /data/exp350_pair_confirm \
  --target-db /data/exp350/complexDB --max-seqs 1000000 --threads 48
uv run python analyze_pair_survival.py --complex-alignments \
  /data/exp350/complex_query_alignments.tsv \
  /data/exp350/complex_target_alignments.tsv \
  /data/exp350_pair_confirm/complex_query_alignments.tsv \
  /data/exp350_pair_confirm/complex_target_alignments.tsv
uv run python plot_pair_survival.py
uv run python build_summary.py
```

Search commands, hashes, counts and compact witness tables are committed under
`data/`. The complete compressed alignment evidence and logs are public in the
[MarinFold HF bucket](https://huggingface.co/buckets/open-athena/MarinFold/tree/main/data/evals/exp350_complex_holdout_survival/evidence).

## Conclusion

Use the 218 candidates as the input to Stage B if the intended claim is
**pair-held-out**: the model may have seen each component family separately,
but no observed complex-training example contains homologs of both partners.
For Helico structural reporting, use the 97-candidate conservative subset and
state that the screen covers documented Helico fine-tuning PDBs while inherited
Protenix pretraining remains unknown.

Before launching predictors, freeze connected homology groups, collapse
redundant assemblies/interfaces, remove prior development targets, and split
development/test by group. This will determine the final independent test
count. Keep the strict component-held-out result as a separate negative finding;
it does not support a useful benchmark for this checkpoint from these sources.
