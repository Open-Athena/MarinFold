# Figure-to-analysis map

Every numerical point is auditable through committed tables. Plotting does no
inference, joins, confidence selection, or resampling.

| Figure | Question | Prepared tables | Analysis / source |
|---|---|---|---|
| 01_predictors | How do existing predictors fare across MSA depth? | `figure_rows.csv`, `summary.csv` (`figure=01_predictors`) | `prepare_structure`; exp250 archived per-target GDT-TS and lDDT plus new AF2/3 and Boltz-2 scores; natural proteins matched across methods |
| 02_oracle | How well can Helico realize the true map? | Same, `figure=02_oracle` | `prepare_structure`; full three-state oracle and no-contact Helico, plus AF2/3, Boltz-2 and Protenix + MSA context |
| 02b_confidence | Where does the oracle rank among 100 ESMFold2 maps on low-depth proteins? | `structured_confidence_per_map.csv`, `structured_confidence_ranks.csv` | `prepare_structured_confidence`; 505 full maps, same known mask and three Helico samples/map; positive counts may vary |
| 02c_accuracy_confidence | Does Helico confidence track original ESMFold2 structure accuracy? | `structured_accuracy_confidence.csv` | `generation/score_esmfold2_decoys.py` scores all 500 original structures with the AF2/3/Boltz-2 scorer; `prepare_structured_confidence` joins by protein, seed, arm and structure hash |
| 03_method | What was trained, and how does inference work? | `training_sources.csv`, `manifest.json` | Exp277 corpus inventory and fixed model identity; diagram arrows have no quantitative width |
| 04_contacts | How accurate are the predicted contacts? | `figure_rows.csv`, `summary.csv` (`figure=04_contacts`) | `prepare_contacts`; exp277 validation/design results plus new exp325 test results; exp245 baseline scores and new AF2/3 and Boltz-2 pyconfind contacts |
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

For Figure 02b, `structured_confidence_per_map.csv` identifies the selected
Helico sample and its original row in `helico_structured_samples.csv`.
`map_source_row` points into `structured_decoy_maps.csv`, which retains the
ESMFold2 seed, structure digest, contact-state digest, contact counts and
agreement with the oracle. `generation/prepare_structured_decoys.py` extracts
all maps using the same Helico/pyconfind geometry and oracle eligible-pair mask.
The rank table records all contributing source rows, counts above/equal to the
oracle, its best/worst/mid rank, and the number of distinct decoy maps.
Dots show the prepared confidence; diamonds identify the oracle. The ranking-score
axis removes the empty gap between one clash-penalized score (−99.40625) and the
ordinary scores. Both remaining segments are linear, the break is labeled //,
and tick/hover labels retain the actual scores. No point is removed. The scatter
uses the same display transform; it does not change the cached ranks or correlations.
Vertical jitter is purely visual. pLDDT uses the same structure selected by
ranking_score, without a second selection. The earlier random control remains
in `confidence_*.csv` for audit and is superseded in this panel.

For Figure 02c, x is `source_structure_gdt_ts` from the original ESMFold2
prediction, y is the corresponding contact-conditioned Helico `ranking_score`.
`accuracy_source_row` points to `esmfold2_decoy_structure_metrics.csv`, while
`source_row` retains the selected Helico sample. Oracle source accuracy is one
by definition (experimental structure against itself), explicitly marked by
`accuracy_rule`; it does not assert perfect Helico reconstruction. Alternate
views use measured Helico `gdt_ts`/`lddt` instead of source-structure accuracy.
Colors identify proteins, circles identify predictions, and diamonds identify
oracle maps. The menu can isolate each protein, including oracle diamonds that
overlap in the combined view. Every prediction is retained in the scatter, with no filtering.
`structured_accuracy_summary.csv` reports the 100-prediction accuracy range and
median for each protein and view, plus descriptive within-protein Spearman
correlation with Helico confidence. Oracle reference points are excluded from
those correlations, and there is no correlation pooled across proteins.

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

Boltz-2 occupies the same four panels, without changing any original population.
`generation/score_boltz2.py` verifies all 25 candidates, their saved confidence
JSON and coordinate digests, and the selected confidence argmax before scoring.
It imports the same `structure_scores` and contact metric functions used for
AF2/3; `boltz2_{structure_metrics,contact_metrics}.csv` are its source tables.
`boltz2_inputs.json` pins code, weight hashes, sampling and MSA processing;
`boltz2_run.json` records scorer and input hashes; `boltz2_timings.csv` records
GPU time, setup time, runtime packages and the selected candidate per protein.
