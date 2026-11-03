## Five arms, 500 structures

Ten lowest-Neff natural eval-val proteins; ten independent samples per arm.

Default MarinFold and MPNN-only two-epoch lr5e-4 each supply contacts to Helico.

Added 100 fresh predictions each: ESMFold2, AF3 single sequence, AF3 with MSA.

All arms use the same resolved sequences, native references and scorer.

All 500 predictions are complete; every structure has full native CA coverage.

## AF3 + MSA leads on this pilot

Confidence-selected macro means: TM-score / all-heavy-atom lDDT / GDT-TS.

AF3 + MSA: 0.799 / 0.730 / 0.780.

ESMFold2: 0.640 / 0.565 / 0.605.

MPNN-only MarinFold -> Helico: 0.440 / 0.397 / 0.373.

Default MarinFold -> Helico: 0.433 / 0.388 / 0.378.

AF3 single sequence: 0.403 / 0.370 / 0.358.

Compared with default Helico, paired TM-score differences (95% protein bootstrap):

ESMFold2 +0.207 [0.034, 0.370]; AF3 + MSA +0.366 [0.169, 0.541].

AF3 single sequence -0.030 [-0.099, 0.044]; no clear difference in ten proteins.

## Sampling helps Helico, but leaves a gap

Oracle / median / confidence / individual-mean TM-score at ten samples:

Default -> Helico: 0.483 / 0.340 / 0.433 / 0.347.

MPNN-only -> Helico: 0.481 / 0.350 / 0.440 / 0.348.

ESMFold2: 0.672 / 0.625 / 0.640 / 0.626.

AF3 single sequence: 0.465 / 0.413 / 0.403 / 0.413.

AF3 + MSA: 0.838 / 0.801 / 0.799 / 0.793.

Helico's oracle-over-mean gain is about 0.135, versus 0.045-0.052 for baselines.

Yet ESMFold2's individual mean exceeds Helico's observed oracle best-of-ten.

The two MarinFold checkpoints still have no clear aggregate winner.

## Confidence helps some models more

TM-score gain over the individual-sample mean, with 95% protein intervals:

Default -> Helico: +0.086 [0.017, 0.159].

MPNN-only -> Helico: +0.092 [0.037, 0.154].

ESMFold2: +0.015 [0.002, 0.031].

AF3 single sequence: -0.010 [-0.032, 0.010].

AF3 + MSA: +0.006 [-0.016, 0.034].

This pilot does not demonstrate a confidence-selection gain for either AF3 arm.

ESMFold2 uses pTM; AF3 and Helico use their own native ranking scores.

Each confidence choice supplies all metrics; oracle is separately best per metric.

## The lowest-Neff target is an exception

8eb9_A, Neff 1.34: confidence-selected TM-score is 0.728 for MPNN -> Helico.

Default Helico 0.531; AF3 single sequence 0.519; ESMFold2 0.255; AF3 + MSA 0.233.

8bux_A: all single-sequence arms score 0.157-0.218, but AF3 + MSA scores 0.922.

Low Neff does not mean that the alignment is useless, or that it always helps.

AF3 MSA-minus-single-sequence mean TM-score: +0.396 [0.196, 0.565].

Ten proteins cannot establish full eval-val performance or a universal ranking.

## Matched inputs and structural scoring

Baseline seeds: 381000 + 100*cohort_index + sample_index; one sample per seed.

AF3: official pinned source, ten recycles, templates off in both arms.

MSA: archived ColabFold rows projected onto the same resolved residues.

ESMFold2: released pinned snapshot, 20 loops, 200 diffusion steps, LM dropout .3.

TMscore: reference-normalized TM-score and optimized GDT-TS, fixed residue IDs.

lDDT: all native heavy atoms, 15 A neighborhoods, within-residue pairs excluded.

Missing atoms receive identical penalties; minimum baseline coverage is 99.57%.

This matches sample budgets, not compute or model training exposure.

## Reproduction and compute

README, per-sample CSVs, all prefix summaries and plotting scripts are in exp381.

Public HF v1 retains the original 200 Helico predictions and native references.

Public v2-baselines adds all 300 baseline CIFs, mapped arrays and input alignments.

Run analyze_baselines.py and build_summary.py for figures/PDF without GPU access.

Measured new inference: 1,407.6 H100-seconds; shared setup: 240.3 seconds.

AF3 includes cold-shape JIT; ESM folding includes preparation. Not a speed test.

Eleven workers including smoke; all full-budget prediction calls succeeded.

See README for pinned revisions, Modal runs, source hashes and limitations.
