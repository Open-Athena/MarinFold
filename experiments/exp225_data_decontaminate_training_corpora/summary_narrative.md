# Summary slides — a real eval-decontamination pass over the contacts-v1 corpora

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     The renderer does not process markdown, so keep this plain prose —
     no ** emphasis, no tables.
     Keep this current as the experiment progresses. -->

## The question

Can we rebuild the contacts-v1 training corpora with a decontamination pass
that actually covers the eval set we report on — and what does it cost?

Issue #213 measured the overlap: 323 of the 554 eval proteins (58%) have a
significant training homolog. This is the other half — the fix, and its price.

## What is filtered today

Two corpora, two stories.

AFDB (4,129,682 documents, #53) has NO eval decontamination at all. Its split
is afdb-24M's own cluster hash, so nothing leaks across the AFDB split, but the
554 eval proteins were never purged from it. #41 found 99 of 100 FoldBench
monomers fall in a fold the model trained on; those verdicts became reporting
strata and nothing else.

ESM-Atlas (66,759,922 documents, #91 then #139) has one filter, with four gaps.
It dropped 41,517 sequences at 40% identity or above, but: FoldBench-100 was
never in the reference, which is 18% of the benchmark; 40% is looser than our
own 30% bar; the filter is sequence-only, on the axis #41 showed is the wrong
one; and #91's own stated follow-up never happened.

## The design

Both corpora are already materialised, so decontamination is a row filter on
entry_id, not a regeneration. Three tiers, priced before anything is retrained:

  A — sequence. Identity at or above 30% over at least half the eval protein,
      or E at or below 1e-3.
  B — plus structurally redundant. A, plus TM 0.90 or above to any eval
      structure.
  C — plus fold-level purge. B, plus whole clusters at TM 0.50 or above.

The expensive inputs already existed: #213's 70.9M-sequence MMseqs2 database,
and #41's Foldseek database of the 1.33M AFDB cluster representatives. No new
pyconfind, no document rebuild, no cluster job.

## What we found

Tier A is cheap: 1.89% of AFDB, 1.57% of ESM-Atlas. And it is not an artefact
of a threshold — sweeping the search's reporting depth over six decades moves
it by a quarter of a percentage point.

All four of #91's gaps are real. Of the surviving contaminated ESM-Atlas rows,
47% sit in the 30-40% band, 24% are remote homologs below any identity bar, 17%
are reachable only from FoldBench-100 — and 12% (123,713 rows) clear #91's own
40%/50% rule. That last group is the measured price of its -s 4.0 search
sensitivity, a trade-off its own script flagged in a comment.

#41 was right about the axis, and acting on it is cheap at Tier B: 22,320 AFDB
documents are structurally near-identical to an eval structure while being
invisible to any sequence search. That is 0.54% of the corpus.

Tier C is unaffordable: 37.31% of AFDB, and 95% of that is structure-only. The
reason is not an unlucky threshold. The mode of "best TM to any of the 554"
sits essentially at the 0.5 same-fold boundary — a third of AFDB's structural
clusters simply are the same fold as something in a 554-protein eval set.

Three quarters of that cost protects the 396 de novo designs. Scoping the fold
purge to the 158 natural eval proteins costs 9.38% instead of 36.77%. Designs
are small idealised bundles that share a fold with an enormous share of AFDB,
and they are the proteins with no evolutionary relatives to leak through in the
first place.

## Where it lands

H1 holds through Tier B; H0 holds for Tier C. The recommendation is to publish
Tier B for AFDB and Tier A for ESM-Atlas, and to decline Tier C with 37.31% as
the number that justifies declining it. The 9.38% natural-only variant is a
real middle option if fold novelty later becomes load-bearing.

Do not pay for the ESM-Atlas Foldseek build (about $1k) yet. It was gated on
this table. The only tier it could serve is B, since C is declined, and on the
arm we can measure, B's structural increment is 0.54%. Revisit if the retrain
moves. The caveat: ESM-Atlas is metagenomic, so its fold distribution against
this eval set need not match AFDB's.

Stage 5 — retraining the #199 recipe on the decontaminated mixture and
re-scoring against #213's homology-free subset — is the actual test of H1.
Nothing here measures accuracy.
