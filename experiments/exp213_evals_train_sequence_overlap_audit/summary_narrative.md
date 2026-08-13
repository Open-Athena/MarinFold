# Summary slides — exp213: train-set sequence overlap and the homology-free re-eval

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## The question

How much of our headline contact-prediction number survives if we throw out
every eval protein that has a sequence homolog in the training set?

The worry is concrete. Our training labels are contacts read off **AlphaFold2
(AFDB) and ESMFold2 (ESM Atlas) structures** — predictors that had MSAs or a
large PLM. MarinFold sees only a single sequence at inference, but if an eval
protein has a close relative in the training corpus, the coevolution signal
that produced that relative's structure is baked into the weights and reachable
by homology. In that regime "single-sequence contact prediction" is partly an
MSA-derived retrieval.

## What was new here

Two prior experiments covered half of this. #65 ran an MMseqs2 leakage check,
but only against the 1.33M AFDB cluster *representatives*, and never tied it to
an accuracy number. #94 built the identity-stratified accuracy curve, but
against the AFDB corpus alone and for the #61 model — 0.26 R-precision weaker
than today's.

Neither had ever looked at the **66.8M ESM-Atlas sequences**, which are 94% of
exp199's training set by count and are metagenomic — exactly the sequences most
likely to supply remote homologs for natural eval proteins.

So: decode all 70.9M training sequences out of the two published corpora
(neither carries a `sequence` column), search all 554 eval proteins against
them, and re-aggregate the per-protein scores that already exist for six
predictors. No new model inference.

## Result 1 — the eval set is homology-rich

58% of the 554 eval proteins have a significant training homolog; only 11% have
no MMseqs2 alignment at all. ESM-Atlas hits **296** proteins, more than AFDB's
275, and 48 proteins have a training relative that exists *only* in the
metagenomic corpus. The half nobody had checked does matter.

But sequence novelty is not fold novelty: of the 231 sequence-novel proteins,
133 are still `same_fold` and 61 `redundant` against the AFDB training folds.
Only **37** are novel in both sequence and fold.

## Result 2 — accuracy does not track training proximity

Spearman rho between best training identity and R-precision:

- MarinFold #199: **−0.12** pooled, **+0.04** on natural proteins
- seq-KNN k=10 (the copy-the-neighbour null): **+0.53**

seq-KNN is the calibration. A model whose score came from retrieving memorised
homologs would look like that last number. MarinFold does not. Restricting to
the 231 homology-free proteins moves R-precision 0.611 → 0.549, and against
ESMFold/ESMFold2 on natural proteins MarinFold actually *gains* ground.

This is #94's finding, extended to the full training set and a far stronger model.

## Result 3 — but the Protenix single-seq parity does not survive

#180 records MarinFold at parity with Protenix-v2 single-sequence: +0.008
[−0.017, +0.033] over 554 proteins. That parity is homology-dependent.

- On proteins **with** a training homolog, MarinFold leads by **+0.13**
- On proteins **without** one, it trails by **−0.17**
- Difference of differences: **−0.303 [−0.346, −0.259]**, ~100x the tracker's
  0.0023 noise floor

On the 37 proteins novel in both sequence and fold, MarinFold scores **0.328**
against Protenix-SS's 0.547 and ESMFold2's 0.620.

The effect is specific to Protenix-SS — MarinFold's standing against ESMFold,
ESMFold2 and Protenix+MSA does not degrade. That asymmetry is the opposite of
what uniform homology leakage predicts, and points at Protenix-SS being
differentially strong in the novel-fold regime rather than at MarinFold leaking.

## What to do with it

1. **The headline number is not substantially inflated by homology leakage** —
   but **"we have caught up with single-sequence Protenix" should be qualified
   as "on proteins resembling our training data"** until the novel-fold gap
   closes.
2. **Report the homology-free subset alongside the headline.**
   `data/eval_train_identity.csv` is committed per protein and joins on
   `(dataset, stem)` — one merge.
3. **This eval set cannot answer the question much better than it just did.**
   The homology-free subsets are 80% de novo designs; the natural,
   homology-free, fold-novel corner is n≈19. Measuring novel-protein
   performance rather than bounding it needs more natural low-homology
   proteins — the gap #65 was originally opened to fill.
