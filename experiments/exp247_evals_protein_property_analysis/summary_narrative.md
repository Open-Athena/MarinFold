# Summary slides — exp247: what makes a protein hard

<!-- Feeds plots/summary.pdf via build_summary.py. Prose only — the renderer
     reflows each paragraph, so markdown tables come out as raw pipes. -->

## The question

#245 produced per-protein contact scores for nine predictors over 314 natural
FoldBench monomers, and never explained them. On eval-test every predictor's
scores run from 0 to 1. What property of a protein decides where it lands — and
is it the same property for MarinFold as for the baselines?

The comparative half is the useful half. If our accuracy tracks training-set
proximity while ESMFold2's tracks MSA depth, that says something concrete about
what a from-scratch sequence LM has learned, and it tells us which proteins to
add to training.

60 features over four families: size and shape, contact geometry, secondary
structure, homology and family abundance, plus domains, function, localisation
and taxonomy from RCSB and UniProt. Analysis only — every score already existed.

## One answer, for every predictor

**Family abundance.** How many relatives a protein has — MSA depth, homologs in
our corpus, KNN neighbours, alignments surviving decontamination — is the only
block of features that predicts contact accuracy, and it predicts every predictor.

Spearman ρ with MSA depth: #199 cooldown **0.67**, **#232 m2-p06 0.50**, ESMFold
0.46, ESMFold2 0.41, seq-KNN 0.37, Protenix + MSA 0.30, Protenix single-seq
**-0.06**. All q < 0.001 and stable across eval-val and eval-test.

Everything else is noise by comparison. Length, relative contact order and
fraction of long-range contacts are all |rho| <= 0.04 for MarinFold. The
difficulty axes everyone assumes matter do not move our model.

## We are the homology-dependent one

The hypothesis going in was that a single-sequence model should hold up where
MSAs are thin. **The opposite is true.** Between the deepest quartile (7,413 to
19,393 sequences) and the shallowest (2 to 784, median 160), MarinFold loses
**0.25** R-precision (0.631 -> 0.378).
ESMFold2 loses 0.18 and Protenix-v2 + MSA loses 0.08. The gap to ESMFold2 widens
as the family shrinks (rho = +0.18, p = 0.0014).

The same ordering appears in how *predictable* each predictor is from protein
properties alone: cross-validated R-squared runs seq-KNN 0.86, #199 cooldown 0.49,
#232 m2-p06 0.34, ESMFold 0.29, ESMFold2 0.24, Protenix + MSA about zero. The more
a method leans on homology, the more its per-protein accuracy is a property of the
protein rather than of the method. MarinFold sits closer to the KNN null than
ESMFold2 does.

A sequence LM trained on a sequence database inherits that database's family
statistics. That is the finding.

## Biology barely matters

Controlling for length, contact order and training identity, nothing in the
biology block reaches half the strength of the homology block: UniProt domain
count +0.21, bacterial +0.19, cytoplasmic -0.12, nuclear -0.11, viral -0.12,
and membrane / secreted / enzyme all under 0.06. Secondary structure is small and
points the unintuitive way -- sheet fraction +0.11, helix -0.09, so beta-rich
proteins are marginally easier for us.

Viral survives the control, which matches #241 and #245: viral proteins are hard
partly because their families are thin, and partly for a residue that is not just
family size.

## What to do about it

**Training data.** The lever is not more tokens of the same distribution, it is
coverage of small families. Both corpus arms are built from clustered databases --
AFDB at struct-cluster level, ESM-Atlas at 40 % linclust -- which systematically
downweight exactly the proteins we fail on. #241 found both arms miss viruses;
this generalises it to thin families of any kind.

**Evaluation.** MSA depth belongs in the reporting as a stratum. A comparison on a
set with median depth 3,000 says little about depth 100, and the ordering is not
preserved: on the shallowest quartile MarinFold reaches 56 % of ESMFold2's score,
on the deepest 74 %.

**Open.** This is correlational. A training-data intervention -- upsample small
families, or train on a de-duplicated corpus -- is what would test causality.
