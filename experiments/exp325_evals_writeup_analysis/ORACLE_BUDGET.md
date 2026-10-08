# Sparse oracle-contact sweep

Status: all 305 inputs prepared and the one-protein validation passed. The full
sweep is awaiting approval of its $176.45 estimate (requested ceiling: $200).

## First protein: 8ii8_A (MSA depth 3)

This validates the inference path on one protein; it is **not a depth-tier
estimate**. Sparse rows average two random subsets after selecting one of three
diffusion samples by ranking score for each subset.

| Supplied information | Positive contacts | GDT-TS |
|---|---:|---:|
| None | 0 | 0.422 |
| Random 5 true contacts | 5 | 0.410 |
| Random 10 true contacts | 10 | 0.720 |
| Random L/5 true contacts | 45 | 0.786 |
| Random L/2 true contacts | 113 | 0.964 |
| All true contacts; others unknown | 138 | 0.961 |
| Full oracle map, including non-contacts | 138 | 0.959 |

[Raw samples](data/oracle_budget_smoke_samples.csv),
[timings](data/oracle_budget_smoke_timings.csv),
[run manifest](data/oracle_budget_smoke_run_manifest.json), and
[Modal run](https://modal.com/apps/open-athena/main/ap-giciAzRnDuLejhmgLJs7ZA).
All 33 predictions completed; all 33 unit tests pass. Exact 5/10 counts,
positive-only conditioning, nested draws, and confidence-only selection are
checked before the full run. Only one protein, `8f8n_A`, caps L/2 (110 requested,
109 available); both random draws record that cap explicitly.

## Frozen full-sweep protocol

The population is exactly Figure 02's **305 natural proteins**, including the
authorized test split: 5 / 20 / 60 / 220 in the four MSA-depth tiers.
This is a ground-truth conditioning diagnostic.

For each protein, sample two independent random permutations of its true
contacts, without replacement. Each permutation supplies nested subsets of
**5**, **10**, **floor(L/5)** and **floor(L/2)** pairs. L is the frozen FoldBench
input sequence length. Relative budgets cap at the available positive count;
fixed 5/10 budgets must be exact. All unselected pairs are **unknown**, including
pairs that happen to be true non-contacts. Contact geometry is unchanged:
Helico's pyconfind oracle, 3 Å contact distance, degree ≥0.001, separation ≥6.

Three fresh controls use the identical inference setup: no contacts, all true
contacts with other pairs unknown, and the full oracle map including true
non-contacts. This separates positive-contact coverage from negative information.

Helico stays at `contacts-msafree-01-step-6000`, checkpoint SHA-256
`779d540e5bb45cd26bb970188da2f1937ed3079942923a1356c29d06f20fe644`, source
`b10385d736673c81b10e70d1099962af6f2573c0`. No MSA; three diffusion samples,
six recycles, seed 42 per map. The main comparison keeps Figure 02's original
highest-ranking-score sample selection. pTM selection is a separate sensitivity
view. Neither uses accuracy to choose a sample. Average the two map results
within each protein; never select the better random subset. Bootstrap proteins
for 95% intervals. The shallowest tier still contains only five proteins.

The complete sweep is **3,355 maps / 10,065 structures**. Its dry-run estimate is
**$176.45**, including a 1.7× runtime margin, worker startup, the current 1.75×
us-east region multiplier, and 10% for CPU/memory. The existing runner requires
explicit approval above its $100 gate. Modal rates were checked against
[pricing](https://modal.com/pricing) and
[region selection](https://modal.com/docs/guide/region-selection) on 2026-10-08.

## Reproduction and figure lineage

1. `generation/prepare_oracle_budgets.py` extracts and freezes the contact maps
   once. `data/oracle_budget_targets.csv`, `oracle_budget_maps.csv` and
   `oracle_budget_protocol.json` record the cohort, requested/effective counts,
   random seeds and exact input hashes. Raw maps and input CIFs live in
   `scratch/helico/oracle_budget/` before publication.
2. `generation/run_helico.py`, with `WRITEUP_PHASE=oracle_budget`, verifies
   checkpoint and input hashes, predicts every map and records all samples and
   direct per-map timings. `SWEEP_TARGET_LIMIT=1` runs the first low-depth protein
   as a resumable smoke test. `HELICO_DRY_RUN=1` performs the cost check only.
3. `prepare_oracle_budget_analysis.py` creates `oracle_budget_selected.csv`,
   `oracle_budget_per_protein.csv`, `oracle_budget_summary.csv` and paired deltas.
   Every per-protein value points back to its selected raw sample rows.
   `oracle_budget_control_check.csv` compares fresh and archived controls.
4. `render_oracle_budget.py`, called by `render.py`, consumes those small tables
   for `02e_oracle_budget` and `02f_oracle_budget_context` (GDT-TS and lDDT).
   Interactive menus expose split and pTM-selection views. Poster exports use
   the same data with the requested plain white background.
