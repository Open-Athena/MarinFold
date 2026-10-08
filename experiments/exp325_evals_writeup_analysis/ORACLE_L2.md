# L/2 oracle contacts across all MSA depths

The oracle point in **Figure 02 and Figure 05** now uses randomly sampled
true contacts: **floor(L/2) per map**, with **every unselected pair unknown**.
No negative contacts or MSA are supplied. L is frozen input sequence length.
These are ground-truth conditioning diagnostics, not deployable predictors.

The cohort is the same **305 natural proteins** as the main structure comparison:
95 validation and 210 test; MSA tiers contain **5 / 20 / 60 / 220 proteins**.
Only L/2 is run in this extension. One protein, `8f8n_A`, has 109 eligible true
contacts versus the requested 110, so it receives all 109. The complete
requested/effective counts are in [oracle_l2_maps.csv](data/oracle_l2_maps.csv).

## Results

All **610 maps / 1,830 structures** are complete. Mean GDT-TS is **0.843**
overall, compared with **0.893** for the archived full positive/negative oracle.
L/2 remains strong at low MSA depth; in the deepest tier it falls below the
strongest MSA-based predictors. Different tiers contain different proteins.

| MSA depth | Proteins | L/2 GDT-TS | 95% protein-bootstrap interval | L/2 lDDT | Archived full-map GDT-TS |
|---|---:|---:|---:|---:|---:|
| <10 | 5 | 0.906 | 0.873–0.939 | 0.847 | 0.932 |
| 10–99 | 20 | 0.841 | 0.798–0.883 | 0.835 | 0.875 |
| 100–999 | 60 | 0.853 | 0.817–0.879 | 0.828 | 0.881 |
| ≥1000 | 220 | 0.839 | 0.822–0.854 | 0.825 | 0.897 |
| **All depths** | **305** | **0.843** | **0.829–0.856** | **0.826** | **0.893** |

The 210-protein test mean is **0.842202 GDT-TS**. Pure inference totals
**139.11 H100 minutes**, including the reused five-protein results; direct
per-map times and worker metadata are in
[helico_oracle_l2_timings.csv](data/helico_oracle_l2_timings.csv).
The archived full-map comparison uses its original sample selection; the
five-protein diagnostic separately retains fresh full-map controls.

## Protocol

Each protein has two independently sampled contact subsets, uniformly without
replacement among upper-triangle oracle positives. Sampling uses the same
stable per-protein/per-replicate seeds as the [five-protein budget study](ORACLE_BUDGET.md).
The exact ten L/2 maps and 30 diffusion predictions for those five proteins
are reused; the other 300 proteins are newly folded.

Helico is `contacts-msafree-01-step-6000`, source
`b10385d736673c81b10e70d1099962af6f2573c0`, checkpoint SHA-256
`779d540e5bb45cd26bb970188da2f1937ed3079942923a1356c29d06f20fe644`.
Each map gets three diffusion samples, six recycles and seed 42, with no MSA.
Contact geometry is unchanged: Helico/pyconfind, 3 Å, degree ≥0.001,
sequence separation ≥6. All benchmark entities are retained.

Choose the highest `ranking_score` diffusion sample **within each map**,
then average the two subset accuracies **within each protein**. No subset
is chosen using accuracy or confidence. Tier means weight proteins equally;
95% intervals bootstrap proteins, not maps or diffusion samples. Separate
pTM-selected values are cached as a sensitivity analysis.

The user chose L/2 after seeing the five low-depth test proteins. This is an
exploratory extension of that diagnostic, with all predictor checkpoints fixed.
The full-map oracle remains in the separate low-depth budget and confidence
experiments; it is no longer the oracle dot in the main GDT-TS/lDDT depth panels.
The AF3 sampling TM-score context retains its explicitly labeled full-map control.

## Figures and lineage

![All methods across MSA depth](plots/05_folding.png)

[PDF](plots/05_folding.pdf) · [lDDT](plots/05_folding_lddt.pdf) ·
[Oracle introduction panel](plots/02_oracle.pdf) · [Poster files](POSTER.md)

[White poster PDF](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/poster-2026-10-08/poster_plots/pdf/05_folding.pdf)

1. `generation/prepare_oracle_budgets.py --full-cohort-l2` freezes all 305
   targets, hashes every CIF/map and records the sampling seeds. It writes
   `data/oracle_l2_{targets.csv,maps.csv,protocol.json}` and the raw inputs in
   `scratch/helico/oracle_l2/`.
2. `generation/run_helico.py` with `WRITEUP_PHASE=oracle_l2` verifies input
   and model hashes, runs only the two L/2 maps, and records all 1,830 samples
   plus direct timings for the 610 maps. `helico_oracle_l2_run.json` records
   the inference protocol. Coordinates and conditioning states are durable.
3. `prepare_oracle_l2_analysis.py` selects by confidence, averages subsets and
   caches `oracle_l2_per_protein.csv`, with exact selected raw CSV row indices.
   `oracle_l2_summary.csv` stores the tier means/intervals; `oracle_l2_vs_full.csv`
   pairs these with the archived full oracle. It verifies exact reuse of all
   prior low-depth sample rows.
4. `prepare.py` joins those protein scores to the same archived baselines and
   248B-token MarinFold results. The method key is `oracle_L2`. Every figure
   value in `figure_rows.csv` points into `oracle_l2_per_protein.csv`, which in
   turn points into `helico_oracle_l2_samples.csv`. `summary.csv` records the
   means, bootstrap intervals and contributing proteins.
5. `render.py` draws Figures 02/05 and their interactive/lDDT variants from
   the cached tables. `export_poster.py --upload` generates plain-white vector
   PDFs/SVGs and 7,200-pixel PNGs. Styling changes require no inference or analysis.

```bash
uv run --project /home/bizon/git/helico --no-sync python generation/prepare_oracle_budgets.py --full-cohort-l2
WRITEUP_PHASE=oracle_l2 HELICO_DRY_RUN=1 PYTHONPATH=. uv run --project generation python generation/run_helico.py
WRITEUP_PHASE=oracle_l2 PYTHONPATH=. uv run --project generation python generation/run_helico.py
uv run python prepare_oracle_l2_analysis.py
uv run python prepare.py
uv run python prepare_oracle_budget_analysis.py
uv run python prepare_af3_context.py
uv run python prepare_pl5.py
uv run python render.py
uv run python build_summary.py
uv run python export_poster.py --upload
uv run python publish_oracle_budget.py --l2 --archive --upload
```

[Modal run](https://modal.com/apps/open-athena/main/ap-gaLnEYu7HnCVkYmOoerQe9) ·
[Public exact inputs, coordinates, tables and figures](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp325-writeup-analysis/oracle-l2-2026-10-08)
