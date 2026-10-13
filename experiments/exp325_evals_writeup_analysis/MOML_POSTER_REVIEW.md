# MoML poster: review decisions and provenance

The revised poster keeps the original 36 × 24 inch, three-column format,
author list, logo and QR code. Text, tables and the Helico diagram are editable
in PowerPoint. The numerical panels use the existing exp325 summaries; no
inference, rescoring, exclusions or confidence intervals changed.

[Editable PowerPoint](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/moml-poster-2026-10-13/MarinFold_MoML_revised.pptx) ·
[Print PDF](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/moml-poster-2026-10-13/MarinFold_MoML_revised.pdf) ·
[300-dpi PNG](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/moml-poster-2026-10-13/MarinFold_MoML_revised.png).

## Decisions on the supplied review

| Suggestion | Decision and change |
|---|---|
| 1. Replace “does not depend on evolutionary information” | **Agree.** State “without MSAs or pretrained protein-language-model embeddings at inference,” and explicitly acknowledge information inherited from natural sequences and teacher models. Removed the blanket claim about existing predictors instead of substituting another broad generalization. ESMFold and ESMFold2 are labeled as single-sequence inference here. |
| 2. Soften the difficult-protein claim | **Agree.** Removed the categorical examples and causal “consequently.” Report the observed association with **Neff**, the quantity actually on these plots. Different bins contain different proteins; this does not identify why accuracy changes. |
| 3. Check which Atlas release supplied training structures | **Checked; retain ESMFold2.** exp139 records the ESMFold2 Atlas distillation corpus, published in July 2026. The poster now explicitly identifies the **2026 ESM Atlas**, with Candido et al. as reference 1. Replacing the training teacher with ESMFold would be incorrect. |
| 4. Reconcile counts, examples and evaluation populations | **Agree, with a larger correction.** The evaluated first-epoch checkpoint saw **232,090,905 documents and 248,583,762,834 raw tokens**, not 235M/260B. Table entries are training examples, not unique structures. Explain 334 → 314 → 305 below. |
| 5. Define held-out; identify Top7 | **Agree.** Describe the parent-sequence exclusion and its limits instead of using an unqualified “held-out” heading. No structural-similarity filtering was applied. ProteinMPNN redesigns use retained parents. Helico has a separate temporal training rule; teachers do not inherit MarinFold's exclusion guarantee. Top7 is a separate designed-protein illustration outside FoldBench. Its maps are redrawn from the documented exp277 step266344 votes, replacing an illustration whose checkpoint was not identified in the supplied poster. |
| 6. Add evaluation methods | **Agree.** Added cohort, contact definition, length/cutoff, ranking, 100-rollout voting, KNN source, Neff, equal protein weighting, bootstrap intervals, baseline MSA use and Helico conditioning details. Removed the outdated asterisk and use “Helico + L/2 ground truth contacts.” |
| 7. Emphasize the findings | **Agree, with qualification.** Shortened the Summary and lead with 0.414 → 0.714 sampling precision, the Neff association and the gap between predicted and ground truth contact conditioning. Ground truth performance **suggests** a contact-quality bottleneck; it is not a causal decomposition of all errors. Define oracle selection as selecting the best individual sample; it need not outperform aggregation. |
| 8. Qualify the structure comparison; add a Protenix-v1 control | **Agree on the qualification; defer the extra baseline.** At Neff <10, MarinFold + Helico is 0.285 versus 0.290 for single-sequence Protenix-v2. Gains appear in higher bins. Existing Helico-without-contacts results are retained. A pretrained Protenix-v1 comparison on the same 305 targets would help isolate fine-tuning, but no matched panel is available in exp325. The poster states that limitation; no results from a different cohort are inserted. |
| Copyedits | **Agree where the wording remains.** Use single-sequence, decoder-only, fine-tune, residue–residue, protein–protein, openly developed, and “Context length: 8,192 tokens.” Removed the rapidly evolving / highly conserved examples and the inconsistent “focus … developed” construction. Use “Each bracketed symbol is one token.” |
| Shorter Summary, larger plots, explicit GDT scale | **Agree.** Summary is reduced to four short paragraphs. Plot axis and legend text are enlarged; GDT-TS explicitly reads “GDT-TS (0–1).” The bar plot retains all seven standard-predictor arms and the original 95% intervals. |
| References | **Agree after checking versions.** Replaced the Simpson/lorem-ipsum placeholders with twelve real, linked citations. Keep the 2026 Protenix-v1 paper: the source checkpoint is `protenix_base_default_v1.0.0`, not the older 0.5 release. Cite FoldBench as Nat. Commun. **17, 442 (2026)**; its early online date was December 2025. Poster citations use short model names; full titles appear below. |

## Exact model and corpus

Checkpoint: `contacts-v1-exp277-m2-p06-full-epoch-1.5B`, step **266344**;
1.47B parameters, trained from scratch. All three numerical plots and the
replacement Top7 contact map use this checkpoint.

| Component | Training documents |
|---|---:|
| Native ESMFold2 Atlas | 65,553,178 |
| Native AlphaFold DB | 3,963,003 |
| ProteinMPNN AFDB redesigns | 31,702,680 |
| ProteinMPNN Atlas redesigns | 130,872,044 |
| **Total** | **232,090,905** |

Redesign total: **162,574,724**. The poster rounds to millions and says entries
are rounded independently. These are additional sequences paired with retained
backbones, not 162.6M new independent structures. Raw token count is
**248,583,762,834**; padded training slots are a different quantity.

Sources: [epoch counts](../exp277_models_single_mpnn_pilot/data/epoch_corpus_counts.csv),
[training source table](data/training_sources.csv),
[Atlas extraction](../exp139_data_contacts_v1_esm_atlas/README.md),
[redesign parent reconciliation](../exp266_data_mpnn_redesign_contacts_v1/README.md).

## Population and exclusion scope

- **334 FoldBench monomers** minus **19 designs** minus `8uxt_A` (its full
  contact document exceeds the 8,192-token context) = **314 natural proteins**.
  The pooled panel contains 97 validation and 217 test proteins.
- Structural comparisons retain the **305 common complete cases** already
  fixed by the archived exp250 structural table. Nine excluded natural targets:
  `8es6_A`, `8q79_A`, `8por_A`, `8qle_A`, `8ra0_A`, `8u0i_A`, `9b8e_A`,
  `7pv5_A`, `7t9r_A`. These lack a full set of successful/usable archived
  structural results; missing values are not zero scores. One documented
  example is `7pv5_A`, whose prompt-to-structure mapping is ambiguous around
  a modified cysteine. The per-target inventory is [structure_exclusions.csv](data/structure_exclusions.csv).
- **MarinFold**: MMseqs2 matches at ≥30% identity over ≥50% of the shorter
  sequence to the legacy evaluation set plus all FoldBench protein chains are
  removed from the native training corpus. No structural-similarity exclusion
  is applied. Redesigns are generated only for retained parent backbones;
  this is not a claim that every new sequence underwent an independent
  structural or homology screen. See [the actual filtering rule](../exp225_data_decontaminate_training_corpora/README.md).
- **Helico**: PDB training is restricted to structures released before
  2021-09-30. This is a temporal restriction, not MarinFold's homology filter.
  Homologous proteins and similar folds may remain. See
  [Helico training and limitations](https://github.com/Open-Athena/helico/blob/main/RESULTS_contact_conditioning.md).
- **Teachers and external baselines**: no equivalent training-overlap guarantee
  is established. Sequence-only inference does not imply evolutionary novelty.
- **Top7**: [PDB 1QYS](https://www.rcsb.org/structure/1QYS) is a computationally
  designed protein and predates the baselines' training cutoffs. It is an
  illustration, not an additional clean baseline comparison or FoldBench data
  point. Its cached votes and checkpoint are in
  [top7_provenance.json](../exp277_models_single_mpnn_pilot/data/eval_rollout_v2/top7_provenance.json).

## Methods behind the displayed panels

Contact truth is ConFind degree ≥0.001 with its native-amino-acid rotamer
protocol (`native_only=True`, 3 Å contact criterion, 2 Å clash criterion,
25 Å candidate cutoff). These are not Cβ-distance contacts. Candidates use
resolved reference residues with sequence separation ≥6; L is full frozen
input-sequence length and k = floor(L/2). Missing ranked contacts retain the
denominator and receive zero credit.

Structure-derived contacts are ranked by **qᵢ × qⱼ × degree**, over all
ConFind-emitted pairs with degree >0. q is mean deposited non-hydrogen-atom
pLDDT normalized to [0,1]. Ground truth never enters baseline ranking. See
[CONTACT_CONFIDENCE.md](CONTACT_CONFIDENCE.md) and [POSTER_NEFF.md](POSTER_NEFF.md).

MarinFold consensus ranks occurrence votes over 100 saved rollouts. Individual
samples retain contact emission order after filtering and deduplication;
unfinished/invalid individual samples score zero. The sampling panel includes
all parsed maps in its consensus (0.714128); the main depth panel omits
unfinished maps (overall 0.713291). This existing protocol distinction is now
explicit; we did not silently reconcile the two by rescoring. “Oracle best”
selects the best individual P@L/2 using ground truth, independently per protein.

Sequence KNN searches the **native AFDB corpus only**, filtering out the exp225
dropped rows, and uses ten MMseqs2 hits excluding exact self-hits. It maps
neighbor contacts through local alignments and ranks votes, with bitscore used
only to break vote ties. It does not index the Atlas or redesign corpus.
Source: [run_knn_baseline.py](../exp245_evals_foldbench_held_out_monomers/run_knn_baseline.py).

Neff80 sums inverse neighbor counts at ≥80% identity, including each sequence
itself; identity is computed over jointly non-gap columns. Repeated sequences
are downweighted. The bins have contact counts 15 / 32 / 79 / 188 and structure
counts 14 / 31 / 76 / 184. Upper bounds are exclusive. All means weight proteins
equally; intervals are percentile 95% intervals from 5,000 protein bootstraps.
These estimates do not include training-seed uncertainty. See [NEFF_ANALYSIS.md](NEFF_ANALYSIS.md).

Helico uses `contacts-msafree-01`, step6000, initialized from Protenix-v1.
It receives no MSA. Top-L/2 predicted pairs and random ground truth pairs are
encoded PRESENT; all unselected pairs remain UNKNOWN, not ABSENT. Each map
gets three diffusion samples, six recycles and seed42; the main structural
plot uses the highest original `ranking_score`, not pTM-only selection.
The ground truth arm averages two independently sampled subsets within each
protein; one target has only 109 positives for 110 requested contacts. Thus
contact budgets are nearly matched but sampling budgets differ. See
[MARINFOLD_L2.md](MARINFOLD_L2.md) and [ORACLE_L2.md](ORACLE_L2.md).

ESMFold and ESMFold2 use single-sequence inputs here. The +MSA arms use frozen
benchmark alignments; AF2/AF3/Boltz-2 disable templates. Predictor sampling
budgets are not compute matched. The matched Helico-without-contacts arm is a
control for conditioning in the fine-tuned model, not a substitute for a
pretrained Protenix-v1 ablation.

## References checked

1. Candido et al. (2026). [Language Modeling Materializes a World Model of Protein Biology](https://doi.org/10.64898/2026.06.03.729735). bioRxiv. Citation and ESMFold2 Atlas provenance corroborated by the [official ESM repository](https://github.com/biohub/esm).
2. Varadi et al. (2024). [AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences](https://academic.oup.com/nar/article/52/D1/D368/7337620). Nucleic Acids Research 52(D1), D368–D375.
3. Dauparas et al. (2022). [Robust deep learning–based protein sequence design using ProteinMPNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC9997061/). Science 378, 49–56. doi:10.1126/science.add2187.
4. Zhang et al. (2026). [Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction](https://doi.org/10.64898/2026.02.05.703733). bioRxiv; version and citation confirmed in the [official Protenix repository](https://github.com/bytedance/Protenix).
5. Xu et al. (2026). [Benchmarking all-atom biomolecular structure prediction with FoldBench](https://www.nature.com/articles/s41467-025-67127-3). Nature Communications 17, 442.
6. Jumper et al. (2021). [Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2). Nature 596, 583–589.
7. Abramson et al. (2024). [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w). Nature 630, 493–500.
8. Passaro et al. (2025). [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707). bioRxiv; citation confirmed in the [official Boltz repository](https://github.com/jwohlwend/boltz).
9. Lin et al. (2023). [Evolutionary-scale prediction of atomic-level protein structure with a language model](https://doi.org/10.1126/science.ade2574). Science 379, 1123–1130. This is the ESMFold baseline citation, not the training Atlas version.
10. Zhang et al. (2026). [Protenix-v2: Broadening the Reach of Structure Prediction and Biomolecular Design](https://doi.org/10.64898/2026.04.10.717613). bioRxiv; citation confirmed in the [official Protenix repository](https://github.com/bytedance/Protenix).
11. Yang et al. (2025). [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). arXiv:2505.09388.
12. Hall et al. (2025). [Introducing Marin: An Open Lab for Building Foundation Models](https://marin.community/blog/2025/05/19/announcement/). Marin Community, May 19.

## Reproduce and trace

```bash
uv run --with python-pptx python revise_moml_poster.py --input '/path/to/MarinFold MoML.pptx'
uv run python publish_moml_poster.py --upload
```

Outputs are under `scratch/moml_poster/revised/`: editable PowerPoint, a
36 × 24 inch PDF, a 10,800 × 7,200 pixel PNG at 300 dpi, and a smaller preview.
Individual panels are in `plots/moml/`; `--panels-only` regenerates them without
needing the original PowerPoint.
The PDF keeps native poster text/diagrams; figure PNGs are embedded at 400 dpi.

[moml_poster_figure_rows.csv](data/moml_poster_figure_rows.csv) maps every plot
mean/error bar to an unchanged source-summary row.
[moml_poster_manifest.json](data/moml_poster_manifest.json) hashes the supplied
PowerPoint, source tables, Top7 data and final output files. The generator
retains the user's original input file without modification.
