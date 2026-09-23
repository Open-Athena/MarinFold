# Figure-to-analysis map

Every numerical point is auditable through committed tables. Plotting does no
inference, joins, confidence selection, or resampling.

| Figure | Question | Prepared tables | Analysis / source |
|---|---|---|---|
| 01_predictors | How do existing predictors fare across MSA depth? | `figure_rows.csv`, `summary.csv` (`figure=01_predictors`) | `prepare_structure`; exp250 archived per-target GDT-TS and lDDT plus new AF2/3 scores; natural proteins matched across methods |
| 02_oracle | How well can Helico realize the true map? | Same, `figure=02_oracle` | `prepare_structure`; full three-state oracle and no-contact Helico, plus AF2/3 and Protenix + MSA context |
| 02b_confidence | Does confidence prefer the oracle to random maps? | `confidence_per_map.csv`, `confidence_per_protein.csv`, `confidence_summary.csv` | `prepare_confidence`; new Helico samples, equal known mask/contact counts and diffusion budgets |
| 03_method | What was trained, and how does inference work? | `training_sources.csv`, `manifest.json` | Exp277 corpus inventory and fixed model identity; diagram arrows have no quantitative width |
| 04_contacts | How accurate are the predicted contacts? | `figure_rows.csv`, `summary.csv` (`figure=04_contacts`) | `prepare_contacts`; exp277 validation/design results plus new exp325 test results; exp245 baseline scores and new AF2/3 pyconfind contacts |
| 05_folding | Do predicted contacts improve structure accuracy? | Same, `figure=05_folding` | `prepare_folding`; exact top-L, select the highest-confidence of three diffusion samples; exp311 validation/design and new exp325 test results |
| 06_sampling | Do better individual maps occur among 100 samples? | Same, `figure=06_sampling`; `sampling_diagnostics.csv` | `prepare_sampling`; exp321 ordinary iid validation control plus new exp325 test rollouts; same sample pool for each protein's three comparisons |

For Figures 01, 02, 04 and 05, a dot is the `mean` from `summary.csv` keyed by
`(figure, metric, cohort, tier, method)`. Its error bar is `ci_low..ci_high`.
`n` and `stems` identify the exact proteins. Filter `figure_rows.csv` by that key
to recover individual values. `source` is the repository-relative input file;
`source_row` is its zero-based CSV row excluding the header. `manifest.json`
records source and prepared-file SHA256 digests. The Plotly hover shows the key.

For Figure 06, each scatter dot is one protein's consensus/oracle pair in
`figure_rows.csv`. Hover identifies the protein and split. The mean panel uses
`summary.csv`, and `paired_deltas.csv` stores the within-protein oracle-minus-
consensus estimate. Contact-map uniqueness, pairwise Jaccard and true-contact
union recall are separate diagnostics, not evidence of distinct folds.

For Figure 02b, `confidence_per_map.csv` identifies the confidence-selected
diffusion sample and original row for each map. The per-protein table records
the oracle row and all five random-map rows used in each comparison. Win rates
give ties half credit; confidence ranks also retain ties. Intervals bootstrap
proteins. The scatter compares oracle GDT-TS with the mean of the five random
maps' confidence-selected structures. pLDDT is recorded for the structure
selected by ranking_score, not maximized independently across diffusion samples.

All main figures use natural proteins. Plot menus expose validation, test,
viral/nonviral and designed cohorts where applicable; designs are never pooled
with natural proteins. MSA depth is the query-inclusive alignment sequence
count, not Neff. Its four fixed bins are [1,10), [10,100), [100,1000), and ≥1000.
Intervals use 5,000 protein bootstraps, seed 325. Paired deltas resample the
within-protein difference. These are descriptive comparisons, not a causal
estimate of increasing MSA depth for a fixed sequence.

`coverage.csv` lists every target's availability per figure. Structural
comparisons use complete cases; missing proteins do not receive zero scores.
The complete 333-target contact universe is 97 natural validation, 217 natural
test, and 19 designs. `8uxt_A` was excluded by the original context limit.

AlphaFold2/3 are included in Figures 01, 02, 04 and 05. Their new source tables
are `af{2,3}_structure_metrics.csv` and `af{2,3}_contact_metrics.csv`.
`generation/score_alphafold.py` verifies confidence selection, matches query
positions to experimental protein atoms, calls the existing Helico metrics,
and computes pyconfind contacts with exp89's frozen candidate universe.
The input and sampling contract is `alphafold_inputs.json`; per-protein timings
also retain the selected candidate identity. Structural populations stay at the
original 305 matched natural proteins, and contact populations stay at 314.
