# `exp199_rollout_rows.csv.gz` — provenance

Per-protein contact metrics for **`contacts-v1-exp199-1.5B`**
(`prot-exp199-cw-cv1-s02-m1-p06-aug` @ step 145199), the checkpoint this
experiment stratifies. 11,080 rows = 554 `(dataset, stem)` units × 4 ranges ×
5 cuts, the same schema and the same frozen candidate universe as exp89's
`contact_precision_all.csv`, so the two tables join directly.

| | |
| --- | --- |
| Source run | [#212](https://github.com/Open-Athena/MarinFold/issues/212)'s independent re-measurement (a tie-break for [#209](https://github.com/Open-Athena/MarinFold/issues/209)) |
| Scorer | `experiments/exp82_evals_contacts_v1_contact_prediction/score_rollout_worker.py`, unmodified, at MarinFold `dd7670d` |
| Recipe | n=100 rollouts, T=1.0, top-p 0.95, top-k off, `--contact-mult 6`, `--seed 0`, `--max-num-seqs 128`, per-request seed on |
| Weights | `open-athena/marinfold-exp199` @ `ed7103b`, `prot-exp199-cw-cv1-s02-m1-p06-aug/hf/step-145199`, loaded bf16 (shard sha256 verified) |
| Hardware | 1× RTX A5000, 554 × 100 rollouts in 100 min, 0/55,400 unfinished |
| Ground truth | exp89's `gt_universe.jsonl` (554 units, 552 unique stems) |
| Result | **R-precision 0.611032 (all) / 0.564536 (long)** |
| Local working copy | `/data/exp208_replication/` (`PROVENANCE.md` there has the full audit) |

**Use this table, not #199's published one.** #199's own eval pipeline reports
0.587348 / 0.542181 for the same weights — 0.023 low, roughly 10× the 0.0023
span of #180's four control replicates. #209 traced the gap to that pipeline
(same weights, same metric, same GT; reproduced across two accelerators and two
seeds under exp82's worker) and named `score_rollout_worker.py` the reference
scorer. The stratification here compares MarinFold against baselines protein by
protein, so using the understated table would bias every comparison in the same
direction.
