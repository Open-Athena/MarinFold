# Summary slides — exp: does MarinFold represent both folds of fold-switching proteins?

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Does MarinFold's distribution contain **both** conformations of fold-switching
proteins — and if it does not sample them spontaneously, how much explicit
contact evidence does it take to force the alternative fold?

## Why fold switchers, and why MarinFold

Fold switchers adopt two stable folds from one sequence, which makes them the
field's sharpest generalisation probe. AlphaFold2 predicts only one conformation
for 94% of them, and its successes are driven by training-set memorization.

MarinFold is a different instrument in three ways that matter. `contacts-v1` is
generative over contact sets, so "is the other fold in the distribution?" is
answered by sampling rather than by MSA tricks. It is promptable, so "how much
evidence moves the model" is a number. And we can read our own training data, so
the memorization test Porter's group had to infer can be run directly.

## The eval set, and its premise

93 fold-switching pairs from `ncbi/AF2_benchmark` TableS1; 68 pass a three-part
gate (≥10 contacts unique to each fold, ≥50% chain overlap, switching region
located). 177 of 178 PDB entries were already in the local mirror.

The two folds share a median Jaccard of 0.515. The floor — two chains of the same
sequence in the same fold inside one crystal — is 0.873. All 68 sit below the
replicate median. That floor independently reproduces #224's DsbA figure of
0.85–0.88 on different proteins.

## Result 1 — the model reaches one fold, and it is Fold1

52 of 68 pairs, mean φ = +0.112 [+0.071, +0.155], p = 3.9e-6; paired R-precision
margin +0.078. Strongly asymmetric: Fold1 preferences reach +0.6, Fold2 stops at
−0.2. KaiB — the protein AF-Cluster was built for — sits at +0.47.

The preference is *larger* on tier A (both folds from one crystal, +0.176) and
*larger* again restricted to the switching region (+0.153), so it is the fold
switch being measured, not crystallisation or domain motion.

## The three ways it could have been an artifact

Resolution: Fold1 structures really are sharper (2.52 Å vs 2.80 Å, p = 0.031),
but φ is uncorrelated with that advantage (ρ = +0.011) and is the same size when
Fold1 is the blunter structure (+0.103) as when it is sharper (+0.091).

Set size: |A| exceeds |B| (92.7 vs 78.2), but φ normalises by each and
size-balanced pairs give +0.105.

Method: X-ray/X-ray only gives +0.102 (38/51, p = 1.1e-4).

## Result 2 — sampling does not find the other fold

Per-rollout φ spread against a binomial null built from each pair's own recalls:
median dispersion 1.067. The rollout ensemble is one mode.

That is the finding that distinguishes this from AlphaFold's failure. AF2 needs
MSA tricks to manufacture a second conformation. MarinFold has a native sampling
mechanism and still does not find one.

## Result 3 — but the other fold is improbable, not impossible

Teacher-forced NLL of both folds under identical realizations and matched
contact counts: mean ΔNLL/token −0.019 (p = 0.27), Fold1 favoured in 37 of 68.
A coin flip.

The two readouts agree per protein (ρ = −0.54), so they measure the same thing.
The gap is between scoring and reaching: Fold2 is inside the distribution and
sampling never goes there. That is a steering problem, not a representation one.

## Result 4 — what the training data encoded

The corpora are AFDB and ESM-Atlas, i.e. AF2 and ESMFold predictions, so
whatever conformer those chose is the only one MarinFold ever saw. Recovering the
actual training document for each pair: Fold1 30, Fold2 4, neither 26.

88% Fold1 among decided pairs — an independent reproduction of Porter's ~81%
AF2 Fold1 rate, from a different direction.

## Result 5 — the memorization test is null, and why

The model agrees with the training fold 76% of the time. The two marginals alone
predict 75% (Fisher p = 0.56), and Spearman(φ, training identity) is −0.001.

We cannot tell whether the model follows its training data or independently
shares its bias, because only 4 pairs have a training document encoding Fold2 —
the predictors that generated the corpus almost never produce the alternative
fold. Powering the test needs a PDB-trained model, where both folds exist.

## Result 6 — forcing the fold

Putting k of Fold2's contacts in the prompt moves the model monotonically, with
the symmetric Fold1 control moving the other way from an identical baseline
(+0.110 vs +0.111 at k=0). Recovery is scored on the remaining sets, so the
prompt cannot inflate it, and the echo rate is 0.000 — the model never repeats
what it was handed.

k* = 10 contacts, and 39 of the 51 pairs needing a flip get one. The cost is a
fixed COUNT, not a fixed fraction: k* is uncorrelated with |B| (rho +0.145,
p = 0.38) while the fraction is strongly anticorrelated (-0.413, p = 0.009). A
pair with 20 fold2-unique contacts and one with 200 both need about ten. What
scales instead is prior commitment (rho +0.364 with unconditioned phi).

## Caveats

n = 68, of which 6 are tier A. The calibration gate lands at +0.0056 ± 0.0029
rather than a clean zero — inside one seed's own spread, but not nothing. Several
pairs switch by domain swap or subunit exchange, where the monomer map moves
least. And this scores contact maps, not structures: #174 showed the contact set
is necessary but not sufficient for the fold.
