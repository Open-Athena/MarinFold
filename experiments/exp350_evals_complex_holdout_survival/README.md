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

The first deliverable was a **survival table and an evidence-backed feasibility
decision**. The experiment now also freezes the selected FoldBench benchmark and
its contact-scoring universe before running predictors.

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

### Stage A — survival audit

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

### Frozen FoldBench benchmark

The final benchmark uses FoldBench alone. All targets postdate FoldBench's
2023-01-13 cutoff, which keeps the Helico comparison after the inherited
Protenix v1 pretraining window. The date cutoff does not protect against
Helico's later fine-tuning data, so the three observed same-PDB pair-homology
hits remain excluded. Manual citation review also found two de novo binder
targets missed by the original title heuristic; those are recorded as designed
extensions rather than pooled with natural complexes.

| freeze stage | complexes | homodimer | heterodimer |
| --- | ---: | ---: | ---: |
| MarinFold pair-clean FoldBench candidates | 35 | 15 | 20 |
| after Helico fine-tuning pair screen | 32 | 13 | 19 |
| natural frozen benchmark | **30** | **13** | **17** |

The 30 targets form 26 connected homology groups under the same exact 30%
identity / 50%-of-shorter rule. Twenty-five groups are singletons; one
five-target group shares a ubiquitin-family partner and stays entirely in test.
The deterministic metadata-only split assigns 8 targets to development and 22
to test. Development has 3 homodimers and test has 10; mean total length is
448.25 versus 448.27 residues, respectively.

For contact scoring, each target has a full canonical two-chain input, explicit
FoldBench label-chain to mmCIF author-chain mapping, resolved-residue mask and
contacts-v1 pyconfind interface truth in zero-based concatenated-sequence
coordinates. The candidate universe is the Cartesian product of resolved
positions across the two chains. R is the number of inter-chain contacts with
degree >=0.001; the within-chain sequence-separation rule is not applied across
chains. Across the set, R ranges from 10 to 200 contacts and the candidate
universes contain 3,042 to 326,612 pairs.

The structural half preserves the original FoldBench coordinates and native
target CSV layout for Helico/DockQ, plus exact two-protein AF3 inputs. This is
important for assemblies containing extra symmetry copies and for cases where
FoldBench label chain IDs differ from author chain IDs. The 12.3 MB bundle is
public at the [MarinFold HF bucket](https://huggingface.co/buckets/open-athena/MarinFold/tree/main/data/evals/exp350_foldbench_pair_holdout/v1).

Rebuild and publish the frozen set with:

```bash
uv run python freeze_foldbench_eval.py --threads 24
uv run python publish_foldbench_eval.py
```

`foldbench_complex_eval_targets.parquet` is ready for a multi-chain adaptation
of the exp82 100-rollout evaluator and carries the conventional `dataset`,
`stem`, `L`, `input_seq`, `resolved`, `contacts`, and `gt_contacts` fields plus
chain boundaries. `score_foldbench_contacts.py` implements stable top-R scoring
over resolved cross-chain pairs and bootstraps independent homology groups.

The chain-aware rollout worker preserves exp82's 100-sample settings while
giving each chain independent termini and applying the six-residue separation
filter only within a chain. Its complex-specific generation cap is `12L+128`,
bounded by the model's remaining 8,192-token context; the exp82 monomer cap of
`6L+128` truncated dense complex rollouts during the launch smoke test. Its
CoreWeave dispatcher is pinned to
`contacts-v1-exp343-m2-p06-complex-1.5B-step-280154`, validates the existing
in-region checkpoint mirror, runs at batch priority and records per-target
timings. Each root job retrieves the 183 KB frozen target parquet anonymously
from the public HF bundle, avoiding any workstation-to-object-store dependency:

```bash
set -a; source ~/.config/marin/cw-rno2a.env; set +a
/home/bizon/git/marin-freshiris/.venv/bin/python \
  dispatch_foldbench_contacts_cw.py
```

After collecting the score parts, compute the primary metric and prepare the
structural conditioning arms with:

```bash
uv run python score_foldbench_contacts.py \
  --scores <score-dir> --per-target data/contact_r_precision.csv \
  --aggregate data/contact_r_precision_summary.csv
uv run python export_helico_contacts.py \
  --scores <score-dir> --out-dir <helico-arms-dir>
```

The exporter produces top-L/5, top-L/2 and top-L arms both with all predicted
contacts and with intra-chain contacts only. Budgets depend on input length,
never true R. Choose the budget on development, then run Helico on test with
four arms: contacts withheld, predicted intra-chain, predicted intra+inter-chain
and oracle. Use one trunk seed so Helico's built-in confidence ranking selects
among diffusion samples without the benchmark runner selecting a seed by
ground-truth DockQ. Report DockQ, DockQ >=0.23 success, iRMSD and lRMSD.

## Conclusion

Use the frozen 30-target FoldBench benchmark for the exp343 evaluation under the
explicit **pair-held-out** claim: the model may have seen each component family
separately, but no observed complex-training example contains homologs of both
partners. FoldBench's temporal cutoff addresses inherited Protenix v1 exposure;
the separate Helico fine-tuning pair screen addresses its later training pool.

Tune contact budgets and structural settings on the 8-target development cut,
then report the 22-target test cut once. Keep the strict component-held-out
result as a separate negative finding; it does not yield a useful benchmark for
this checkpoint. PINDER is not needed for this evaluation.
