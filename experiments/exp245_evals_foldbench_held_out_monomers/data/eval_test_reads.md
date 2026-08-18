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

| # | date | scored | why | R-precision (all) |
|---|---|---|---|---|
| 1 | 2026-08-18 | #232 `m2-p06` decontam, #232 `m1-p02` decontam, #199 CW cooldown (contaminated reference) — plus all five baselines | The set's construction read ([#245](https://github.com/Open-Athena/MarinFold/issues/245)): establishing whether the historical FoldBench-100 (now eval-val) was over-reporting, which is what licenses using eval-val freely from here on. Nothing was selected on the result. | 0.538 / 0.493 / 0.613; Protenix-v2 single-seq 0.265, ESMFold 0.753, ESMFold2 0.792, Protenix-v2 + MSA 0.845, seq-KNN 0.582 (unfiltered corpus) / 0.426 (decontaminated) |

## What a read costs

Nothing in compute — 217 units is ~3 minutes on twelve single-H100 CoreWeave
shards. The cost is statistical: every look is an opportunity to select, even
informally ("that direction looked worse on test, drop it"), and the set has no
replacement queued.
