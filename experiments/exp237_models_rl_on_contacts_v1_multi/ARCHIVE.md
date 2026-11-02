# Where exp237's artifacts live

The GPU host this experiment ran on was **decommissioned on 2026-08-20**. Nothing
below depends on it. Sizes are what was actually kept, not what existed.

## In this repository

| what | where | why it is here |
|---|---|---|
| every scored checkpoint's metrics | [`data/agg_modes_*.json`](data/) — **57 files** (56 exp237 checkpoints + the #230 baseline) | the numbers every table and figure is built from |
| paired-bootstrap comparisons | `data/compare_*.json` | the CIs quoted in RESULTS.md |
| training trajectories | `data/training_steps*.csv.gz` | per-batch diagnostics for the reward/diagnostic plots |
| the offline analyses | `data/phase0_*`, `data/section_count_*`, `data/oracle_headroom.json`, `data/pooled_vs_plain100.json` | Phase 0, the scale-correctness measurement, the pooled comparison |
| all code | [`skyrl/`](skyrl/) and the analysis/plot scripts | 45 tests, `python -m pytest skyrl/tests -q` |

**Everything in the write-ups can be regenerated from what is committed here.**

## Archived off-host (local, not in git)

`/data/tim/exp237_archive/` — 1.4 GB:

| what | size | why kept |
|---|---:|---|
| `logs/` | 71 MB | all 18 raw training logs, including the rotated `.partN` continuations |
| `per_rollout/` | 14 MB | 57 `*_per_rollout.parquet` — **required to re-run `compare_arms.py`**; per-protein scores are not otherwise recoverable |
| `agg_sections/` | 624 MB | the raw generated contact sets for every scored checkpoint — lets any re-scoring or re-pooling run without regenerating on a GPU |
| `checkpoints/` | 11 GB | the four checkpoints the write-ups cite, as HF exports |
| `scripts/` | — | the exp237 tree exactly as it ran on the host, for provenance |

Checkpoints kept, and why those four:

| directory | result |
|---|---|
| `mk_step36_best_consensus_0.5806` | best consensus in the experiment |
| `mks2_step24_best_oracle_0.5677` | best ORACLE-best in the experiment |
| `m_b_step18_oracle_0.5663` | the previous oracle record, cited throughout |
| `m_b_lr3e6_step90_consensus_0.5775` | second-best consensus; the resume point for MBLONG and M-BP |

## Deliberately not kept

* **FSDP training checkpoints (1.2 TB).** `ckpts_*/global_step_N` is optimiser
  state for resuming a run. Every one of them was already exported to the HF
  format above before scoring, and no result depends on resuming.
* **The other 52 HF exports (~143 GB).** Their metrics, per-protein scores and
  raw generations are all archived; only the weights are gone, and none of them
  is a cited result.
