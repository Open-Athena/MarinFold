# eval-test read ledger

`eval-test` (217 natural FoldBench monomers) is a held-out confirmation set. It
stops being held out the moment a decision is fitted to it, so every read is
recorded here: **date, who/what was scored, why it warranted a read, and the
numbers that came out.**

The rules this ledger enforces are in the repository
[README](../../../README.md#using-eval-test-sparingly). In short: select on
eval-val, publish on eval-test, and if this file grows a long tail of routine
entries then the set is spent and needs replacing — sample recent PDB directly
rather than re-filtering a curated benchmark
([#241](https://github.com/Open-Athena/MarinFold/issues/241)).

| # | date | scored | why | result (metric named per row) |
|---|---|---|---|---|
| 1 | 2026-08-18 | #232 `m2-p06` decontam, #232 `m1-p02` decontam, #199 CW cooldown (contaminated reference) — plus all five baselines | The set's construction read ([#245](https://github.com/Open-Athena/MarinFold/issues/245)): establishing whether the historical FoldBench-100 (now eval-val) was over-reporting, which is what licenses using eval-val freely from here on. Nothing was selected on the result. | **R-precision (all)**: 0.538 / 0.493 / 0.613; Protenix-v2 single-seq 0.265, ESMFold 0.753, ESMFold2 0.792, Protenix-v2 + MSA 0.845, seq-KNN 0.582 (unfiltered corpus) / 0.426 (decontaminated) |
| 2 | 2026-08-19 | Helico `contacts-msafree-01` step 6000, seven conditioning arms, plus Protenix-v2 single-seq and +MSA as folding baselines | The folding half of #245's question, filed as [Open-Athena/helico#14](https://github.com/Open-Athena/helico/issues/14): does contact quality carry into structure accuracy? Reported in lDDT, not R-precision. Both eval sets were scored together and nothing was selected on eval-test — every arm was fixed before the run, and eval-val agrees with eval-test to within 0.014 for all nine predictors. | **lDDT** (eval-test, 210 units common to all arms): Helico + `m2-p06` contacts top-L **0.619**, top-L/2 0.603, top-L/5 0.558; Helico + oracle contacts 0.860; Helico + Protenix-v2 +MSA contacts 0.828; Helico + Protenix-v2 single-seq contacts 0.394; Helico, no contacts 0.364. Baselines: Protenix-v2 + MSA 0.860, Protenix-v2 single sequence 0.400. |
| 3 | 2026-08-31 | #232 `m2-p06` **training** checkpoint (step 363,000) — the PR #257 checkpoint, which that PR deliberately left unscored on eval-test | Requested for [#260](https://github.com/Open-Athena/MarinFold/issues/260): the MSA-depth stratification needs per-protein scores over the whole natural eval universe, and eval-test is 217 of its 314 natural FoldBench proteins — including 5 of the 6 whose ColabFold MSA is shallower than 10 sequences. One checkpoint, fixed before the run; nothing was selected on the result. | **R-precision (all)** 0.5693 (long 0.5464), **AUC (all)** 0.9455. Scored in the same run: eval-val 0.5561, eval-denovo 0.6110, legacy-554 0.6059 — PR #257's published aggregates for those three reproduced within 0.0044. |

## What a read costs

Nothing in compute — 217 units is ~3 minutes on twelve single-H100 CoreWeave
shards. The cost is statistical: every look is an opportunity to select, even
informally ("that direction looked worse on test, drop it"), and the set has no
replacement queued.
