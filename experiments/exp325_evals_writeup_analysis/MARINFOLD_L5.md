# MarinFold top-L/5 contacts through Helico

Figure 05 compares **top-L, top-L/2 and top-L/5 predicted contacts** from the
same 248B-token MarinFold model, alongside the existing L/2 oracle and all
seven structure predictors. Every arm covers the same **305 natural proteins**:
95 validation and 210 test, with 5 / 20 / 60 / 220 in the four MSA-depth tiers.

![GDT-TS across MSA depths](plots/05_folding.png)

[PDF](plots/05_folding.pdf) · [lDDT PDF](plots/05_folding_lddt.pdf) ·
[Interactive figure](site/index.html#section-05_folding) ·
[White poster PDF](https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp325-writeup-analysis/poster-2026-10-08/poster_plots/pdf/05_folding.pdf).

## Result

L/5 reduces overall mean GDT-TS to **0.376**, versus **0.478 with L/2** and
**0.504 with L**. Relative to L/2, the paired change is **−0.103**
(95% protein-bootstrap interval **[−0.123, −0.083]**); relative to L it is
**−0.129 [−0.151, −0.108]**. The decrease is concentrated in the deeper tiers.

| MSA depth | Proteins | MarinFold L | MarinFold L/2 | MarinFold L/5 | Oracle L/2 |
|---|---:|---:|---:|---:|---:|
| <10 | 5 | 0.261 | 0.297 | **0.245** | 0.906 |
| 10–99 | 20 | 0.325 | 0.341 | **0.331** | 0.841 |
| 100–999 | 60 | 0.369 | 0.370 | **0.304** | 0.853 |
| ≥1000 | 220 | 0.563 | 0.524 | **0.402** | 0.839 |
| All | 305 | 0.504 | 0.478 | **0.376** | 0.843 |

Values are equal-weight protein means of GDT-TS. Both shallow tiers' paired
intervals include zero for L/5 versus L and versus L/2. There are only five
proteins in the lowest tier. L/5 still exceeds no-contact Helico's overall 0.150,
but fewer predicted contacts do not improve overall folding in this comparison.
Mean lDDT likewise declines from **0.637 (L)** to **0.623 (L/2)** to
**0.563 (L/5)**. Selecting L/5 structures by pTM gives mean GDT-TS **0.378**,
close to the primary ranking-score result.

## Fixed comparison

The new arm takes **floor(L/5) predicted positive contacts**, where L is the
frozen input sequence length, and marks every other pair unknown. It uses no
oracle seed contacts, negative contacts, or MSA. The checkpoint remains
`contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344`, trained on
248,583,762,834 raw tokens.

Reuse the same archived 100-rollout consensus rankings as Figure 05's top-L
and [top-L/2](MARINFOLD_L2.md): exp311 for validation, exp325's publication run
for test. No MarinFold inference is repeated. The validation ranking file is
checked against its published hash; every test list is independently rebuilt
from the original sparse votes. Filter mapped residue separation <6 **before**
truncating to L/5. Every L/5 map is verified to be a subset of its frozen L/2
map, retaining the same pair order and vote ties. No correctness labels enter
contact selection. Exactly 22 short mapped pairs are skipped across 11 proteins.

Helico uses the same source `b10385d` and `contacts-msafree-01-step-6000.pt`:
six recycles, seed42, three diffusion samples per map. Highest `ranking_score`
selects the primary sample, as in Figure 05. pTM-selected outcomes are saved
separately. Per-input inference time and worker metadata are recorded.
[Inference run](https://modal.com/apps/open-athena/main/ap-NTbsDIpCYt9oTRCisvlr8r).
All 305 maps and 915 structures completed. Measured model inference totals
4,199.2 H100-seconds, with a median of 12.8 seconds per protein, excluding loading.

The oracle remains **L/2**, averaging two random subsets after confidence
selection within each subset. It is not an L/5 oracle. The original top-L and
L/2 results and all other predictor points are reused unchanged.
The user requested L/5 after inspecting earlier results; this is an exploratory
budget comparison, not untouched held-out confirmation.

## Data and reproduction

- `marinfold_l5_pairs.csv`: each selected pair and its original consensus rank.
- `marinfold_l5_maps.csv`: exact effective counts and frozen map hashes.
- `helico_marinfold_l5_samples.csv`: all 915 diffusion samples and confidence scores.
- `marinfold_l5_selected.csv` and `marinfold_l5_per_protein.csv`: selections by
  ranking score and pTM, with raw source filenames and zero-based source rows.
- `marinfold_l5_summary.csv` and `marinfold_l5_deltas.csv`: protein-weighted means
  and paired differences, using 5,000 protein-bootstrap replicates.
- `figure_rows.csv` → `marinfold_l5_per_protein.csv` → raw samples traces every
  new Figure 05 dot; `summary.csv` stores the exact plotted tier aggregates.

The [public bundle](https://huggingface.co/buckets/open-athena/MarinFold/tree/data/exp325-writeup-analysis/marinfold-l5-2026-10-12)
contains every frozen input and all raw coordinates, metrics and timings.
The publication manifest pins the artifact hashes.

From this experiment directory, with the existing frozen exp325 inputs:

```bash
uv run --project /home/bizon/git/helico --no-sync --with pyarrow python generation/prepare_marinfold_l2.py --divisor 5
WRITEUP_PHASE=marinfold_l5 HELICO_DRY_RUN=1 PYTHONPATH=. uv run --project generation python generation/run_helico.py
WRITEUP_PHASE=marinfold_l5 PYTHONPATH=. uv run --project generation python generation/run_helico.py
uv run --project generation modal run generation/archive_helico.py --marinfold-l5
uv run python prepare_marinfold_l2_analysis.py --divisor 5
uv run python prepare.py
```

The shared preparation and analysis scripts accept `--divisor 2` or `5`.
They write separate artifacts and do not replace the earlier budget's results.
After preprocessing, appearance changes only require the cached-data renderers:

```bash
uv run python render.py
uv run python build_summary.py
uv run python export_poster.py --upload
uv run python publish_oracle_budget.py --marinfold-l5 --upload
```
