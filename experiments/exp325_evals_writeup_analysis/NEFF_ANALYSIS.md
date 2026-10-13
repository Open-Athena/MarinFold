# Accuracy versus effective MSA depth

The depth plots now have **Neff80** versions: the same predictions, score
definitions, checkpoint, contact budgets and confidence selections, regrouped
by alignment redundancy. The raw-count versions remain available.

![Precision at L/2 versus Neff](plots/04_contacts_pl2_neff.png)

[P@L/2 PDF](plots/04_contacts_pl2_neff.pdf) ·
[White poster PDF](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/poster-2026-10-08/poster_plots/pdf/04_contacts_pl2_neff.pdf).

![Folding accuracy versus Neff](plots/05_folding_neff.png)

[GDT-TS PDF](plots/05_folding_neff.pdf) · [lDDT PDF](plots/05_folding_neff_lddt.pdf) ·
[White poster PDF](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/poster-2026-10-08/poster_plots/pdf/05_folding_neff.pdf).

## Definition and alignment identity

For each sequence i, count sequences j with **at least 80% identity** to i,
including itself. Its weight is the inverse of that count; **Neff is the sum
of those weights**. Identity is matching non-gap symbols divided by columns
that are non-gap in both sequences. Lowercase A3M insertions are removed.
The definition and parser are from exp74's `msa_depth.py`; both code hashes
and every alignment hash are retained.

We recomputed Neff for **all 333 frozen predictor alignments** under
`scratch/alphafold/inputs/`, verifying the original input-manifest hash,
per-MSA hash, query sequence, query length and raw count. These are the exact
alignments underlying the existing exp325 depth axis and shared-MSA baselines.
No search was rerun. The older exp260 Neff table uses different alignments
and queries, so it was not used for these plots.

All original A3M records contribute, including repeated queries. Computation
deduplicates identical rows with multiplicity weights to preserve the exact
definition. Binary CUDA matrix products count matching symbols and overlap;
the >=80% test uses `5*matches >= 4*overlap`, with nonzero overlap required.
Synthetic duplicate/gap/threshold tests reproduce the NumPy reference, and
up to eight rows per alignment are checked against direct integer comparisons.
Per-record neighbor counts are archived, so each Neff can be reconstructed.
The full measurement took 74.1 seconds on a local RTX A5000; per-input timings
and worker metadata are in `data/msa_neff.csv`.

## What changes

The same prespecified boundaries, **10 / 100 / 1,000**, now apply to a fractional
effective count. Plot labels use `<10`, `10–100`, `100–1000`, and `≥1000`;
finite upper bounds are exclusive.

| Population | Raw-depth tier sizes | Neff80 tier sizes |
|---|---|---|
| Contacts, 314 natural proteins | 5 / 21 / 62 / 226 | **15 / 32 / 79 / 188** |
| Structures, 305 natural proteins | 5 / 20 / 60 / 220 | **14 / 31 / 76 / 184** |

`9b8e_A` is in the low-Neff contact tier but retains its previous structural
exclusion. No new protein exclusions are introduced. Designs and split/viral
views remain separate. Every predictor in a panel still covers exactly the
same proteins. Means and 95% intervals use 5,000 protein bootstraps, seed325.

The larger low-Neff group contains proteins with up to 49 raw alignment records.
Its means therefore describe a different population from the original five:

| Predictor | P@L/2, Neff<10 (n=15) | GDT-TS, Neff<10 (n=14) |
|---|---:|---:|
| AlphaFold3 + MSA | 0.684 | 0.576 |
| AlphaFold2 + MSA | 0.629 | 0.527 |
| Boltz-2 + MSA | 0.623 | 0.575 |
| Protenix-v2 + MSA | 0.643 | 0.598 |
| ESMFold2 | 0.662 | 0.582 |
| ESMFold | 0.528 | 0.390 |
| MarinFold contacts / Helico + MarinFold top-L/2 | 0.401 | 0.285 |
| Sequence-KNN | 0.069 | — |
| Protenix-v2 single sequence | 0.336 | 0.290 |
| Helico, no contacts | — | 0.271 |
| Helico + L/2 oracle contacts | — | 0.872 |

Overall values do not change: MarinFold P@L/2 is **0.713**, and Helico with
MarinFold L/2 contacts gives **0.478 GDT-TS**, versus **0.843** for L/2 oracle
contacts. The main folding panel still contains only the L/2 MarinFold variant.

## Scope and provenance

Seventeen new static plots and eight interactive panels cover existing
predictors, L/2 oracle, R-precision, P@L/5, P@L/2, focused KNN, folding, and
the AF3 context heatmap. PDF, SVG and PNG versions include long-range/lDDT
alternatives. [All poster files](POSTER.md) have white backgrounds.

The AF3 extended-sampling heatmap keeps the **original five raw-depth<10
proteins**. Its low-Neff cells explicitly say `n=5 subset`; they are not
estimates for all 14 structural proteins in that tier. The per-protein
confidence, sparse-oracle and seeded-completion diagnostics also retain their
original five-protein selection.

| Figure element | Data / source |
|---|---|
| Neff, original count and query length | `msa_neff.csv`; alignment SHA256, per-record count hash, audit count and timing |
| Every protein's old/new tier | `neff_targets.csv`; original target annotations and measurement row |
| Dot and interval | `neff_summary.csv`; method, metric, cohort, tier, contributing stems |
| Per-protein accuracy | `neff_figure_rows.csv`; unchanged original score and source row, plus `analysis_source` / `analysis_source_row` |
| Tier migration | `neff_tier_migration.csv`; old/new bins, counts and explicit members |
| AF3 heatmap cell | `neff_af3_context_summary.csv` → `neff_af3_context_rows.csv`; per-cell n and original context-row IDs |
| Integrity and definitions | `neff_analysis.json`; source, output and preprocessing hashes |

The [public bundle](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp325-writeup-analysis/neff-2026-10-13)
includes the frozen alignment inputs and all per-record neighbor counts.

```bash
# Once: compute Neff from the frozen predictor inputs, then regroup scores.
uv run --project /home/bizon/git/helico --no-sync python generation/measure_neff.py
uv run python prepare_neff.py

# Appearance-only changes read cached measurements and intervals.
uv run python render.py
uv run python build_summary.py
uv run python export_poster.py --upload
uv run python publish_neff.py --upload
```

The test-split regrouping was explicitly requested after the earlier results;
it is an exploratory stratification comparison, recorded in exp245's ledger.
