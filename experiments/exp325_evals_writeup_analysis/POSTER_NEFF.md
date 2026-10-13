# White poster package: Neff and three-way contact ranking

[Download the complete ZIP](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/poster-neff-product-2026-10-13/poster_neff_product.zip) ·
[Browse individual files](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp325-writeup-analysis/poster-neff-product-2026-10-13/poster_neff_product) ·
[All plots in one PDF](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/poster-neff-product-2026-10-13/poster_neff_product/all_plots.pdf).

This is the curated poster collection: **44 figures**, each with a plain white
background, in `pdf/`, `png/`, and `svg/`. Open `index.html` after extracting the
ZIP to browse the collection. PDF is native vector artwork with embedded fonts;
PNG is **7,200 pixels wide, tagged 300 dpi** (24 inches / 61 cm wide). Editable
SVG and the Lato fonts/license are included too.

All depth-stratified comparisons use **Neff at 80% identity**, with labels
`<10`, `10–100`, `100–1000`, `≥1000`. Finite upper bounds remain exclusive.
There are no raw-depth counterparts, degree-only contact comparisons,
confidence-only comparisons, or alternative-ranking delta plots in this package.

Every standard-predictor contact comparison—**R-precision, precision @ L/5,
and precision @ L/2, all-range and long-range**—uses:

`residue pLDDT(i) × residue pLDDT(j) × ConFind contact degree(i,j)`

The seven structure arms are AF3, AF2, Boltz-2, ESMFold2, ESMFold and Protenix-v2
with/without MSA. MarinFold still ranks consensus votes; Sequence KNN still uses
its transferred scores. Structural accuracy (GDT-TS, lDDT, TM-score) retains the
same selected structures and results: contact ranking affects contact precision,
not those already-computed folds. Main folding panels retain MarinFold top-L/2
and random L/2 oracle contacts.

## Files to start with

| Figure stem | Contents |
|---|---|
| `01_predictors_neff` | Existing predictors: GDT-TS; `_lddt` counterpart |
| `01d_af3_depth_context_neff` | TM-score and AF3 sampling context |
| `02_oracle_neff` | Helico with L/2 oracle contacts: GDT-TS; `_lddt` counterpart |
| `04_contacts_neff` | Three-way-ranked R-precision; `_long` counterpart |
| `04_contacts_pl5_neff` | Three-way-ranked precision @ L/5; `_long` counterpart |
| `04_contacts_pl2_neff` | Three-way-ranked precision @ L/2; `_long` counterpart |
| `04b_knn_neff` | MarinFold versus Sequence KNN; both ranges and metrics |
| `05_folding_neff` | Helico + MarinFold L/2 contacts: GDT-TS; `_lddt` counterpart |

The other figures are the method schematic, sampling/consensus diagnostics,
per-protein confidence scatter plots, AF3 sampling, sparse-oracle and seeded
completion studies. Studies originally run on five raw-depth<10 proteins retain
that explicitly labeled fixed cohort; they are **not** results for all 15 natural
proteins with Neff<10 (14 in the structural comparison). No new inference ran.

## Ranking and reproducibility

This extends the validated [confidence-ranking analysis](CONTACT_CONFIDENCE.md)
from L/5 to R and L/2. Use all emitted ConFind degree>0 contacts whose endpoints
are resolved in the frozen truth. Filter sequence separation before ranking:
≥6 for all-range, ≥24 for long-range. Residue confidence is mean heavy-atom
pLDDT, normalized to [0,1]. Ties use input-sequence `(i,j)` order. Ground-truth
contacts retain the degree≥0.001 definition and are consulted only for scoring.

The denominator is R (the number of true eligible contacts), floor(L/5), or
floor(L/2); L is the frozen input length. Unfilled ranks receive zero credit,
including when a predictor emits too few contacts. On 314 natural proteins,
there are 34/69 R shortfalls, 1/19 L/5 shortfalls and 5/193 L/2 shortfalls across
2,198 structure/protein pairs (all/long). No zero-score noncontacts are inserted.
Designs are retained separately in the analysis, never pooled in the poster.

`analysis/poster_product_summary.csv` supplies the six contact panels. Each
summary row lists its proteins; `poster_product_figure_rows.csv` links each
protein to `poster_product_metrics.csv`, then `poster_product_inputs.csv` and the
archived structure/pair hashes. All 4,662 existing three-way L/5 scores reproduce.
The [full selected pair ordering](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/poster-neff-product-2026-10-13/reproduce/ranked_pairs.parquet)
is published separately (29 MB), so the artwork ZIP stays small. Keep `rank<=k`
for a metric's requested contact count to recover its exact selected contacts.
The [original pair/input archive](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/contact-confidence-2026-10-13/contact_confidence_inputs_and_pairs.tar.gz)
contains every eligible emitted pair and mapped residue confidence.

`manifest.json` identifies every included figure and its hashes. Unchanged
figures trace through `analysis/original_poster_manifest.json` to the existing
public analysis bundle. The predictor structures, model and cohort splits are
unchanged; this is an exploratory reanalysis of previously examined test data.

```bash
# Once, using the restored confidence pair archive:
uv run python prepare_poster_product.py
# Base white artwork, if scratch/poster_public is absent or stale:
uv run python export_poster.py
# Fast figure/packaging iteration from cached analysis and base artwork:
uv run python export_poster_neff.py --upload
uv run python build_summary.py
```

The new contact-preview PDFs/PNGs live in `plots/poster_neff_product/` on the
experiment branch. Their basenames intentionally match the original metric names
inside this separate package; **the older uncurated ZIP contains different ranking
variants**, so use the download link at the top of this page for the poster.
