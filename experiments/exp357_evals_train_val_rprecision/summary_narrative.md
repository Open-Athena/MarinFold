# Training versus validation R-precision — exp357

## Training accuracy is not unusually high

Current MarinFold: 50.9% all-range R-precision on 256 native AFDB training proteins; 51.0% on 256 matched original AFDB validation proteins; 55.5% on 97 natural FoldBench eval-val proteins.

The matched train-minus-validation gap is -0.12 percentage points, with a 95% bootstrap interval of [-4.13, +3.83]. Only one training protein reaches 90% R-precision; none reaches 95%.

Long-range R-precision is 46.5%, 47.0%, and 53.7%, respectively. This is contact-ranking accuracy, not the percentage of proteins folded correctly.

## An additional training pass does not fix it

We compared exp277 step 266344 with continuation step 479417. All-range changes are -0.21 points on training proteins, +0.26 on AFDB validation, and +0.19 on eval-val. All paired intervals include zero.

The continuation starts from step 213072, before the first run's cooldown, then adds a fresh shuffled pass and a new cooldown. This tests that specific increase in training exposure and schedule, not every possible way of training more.

The model is not reproducing even its training contacts very accurately under deployment-style inference. One more similar pass through this corpus has not improved that behavior.

## Ten times as many samples gives small gains

On 32 fixed diagnostic proteins per cohort, increasing from 100 to 1000 unconditioned rollouts improves current-checkpoint all-range R-precision by 0.21, 0.32, and 0.77 points for training, AFDB validation, and eval-val.

Long-range gains are 0.63, 2.32, and 1.33 points. The sampling curve is mostly flat by 100 samples. Some sampling gains are real, but this does not explain the large gap from near-perfect training accuracy.

The diagnostic subset is held fixed across budgets. Its absolute averages need not equal the averages of the full cohorts.

## True partial structure is much more useful

Supplying half the true contacts raises all-range R-precision on the remaining task from 47.2% to 71.8% for training proteins, 46.6% to 68.2% for AFDB validation, and 49.8% to 70.1% for eval-val.

Both conditions use 100 rollouts. Given contacts are removed from both candidate universes: copying the prefix earns no credit. All three paired gain intervals exclude zero. The continuation checkpoint behaves similarly.

This demonstrates useful conditional structure knowledge. The supplied contacts contain information unavailable to normal sequence-only inference, so this is an oracle diagnostic, not deployable accuracy or proof of an inherent autoregressive ceiling.

## What the experiment supports

We see no strong native-AFDB training advantage and no near-perfect memorization. The extra training pass is flat, and a tenfold sampling budget helps only modestly. Conditioning on true structural context helps substantially.

Missing sequence-to-structure information, model capacity, objective alignment, optimization, and inference/readout remain possible causes. A controlled small-set overfit experiment would be a stronger next test of whether this model and inference setup can attain high training R-precision at all; it has not been run here.

The sampled training component is native AFDB, not the entire native/MPNN/ESM mixture. Original AFDB validation is not certified free of homologs in later ESM additions. AFDB labels are predicted structures; FoldBench labels are experimental.

## Completeness and reproducibility

All 609 proteins were evaluated at both checkpoints. The reviewed metric tables contain 313,800 terminated samples. Three original samples hit the usual output cap; all samples of those three units were repeated with full remaining context, with zero caps. No headline cohort mean changed by 0.02 points.

Both checkpoints reproduce the previous eval-val all/long R-precision within 0.1 points. The exact exp89 metric, frozen inputs, checkpoint identities, raw rollouts, timing records, bootstrap intervals, and correction deltas are preserved.

Public artifacts: open-athena/MarinFold bucket, data/exp357-train-val-rprecision/v1/ and v1-capcheck/. The final tables, figures, and PDF are under v1/report/. See the experiment README for full paths and reproduction steps. Eval-test was not scored.
