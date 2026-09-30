# Summary slides — exp: quantify structure-aware sequence deduplication of the training corpus

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

For MarinFold's current decontaminated AFDB + ESM-Atlas training corpus, how many documents and source tokens would remain after deduplication at several sequence-identity thresholds if we remove a protein only when a retained protein also has the same structure, and how many similar-sequence/different-structure examples does that rule rescue?

## Why

The answer will differ sharply by source. ESM-Atlas was already reduced to one representative per 40%-identity Linclust cluster, so within-ESM loss above 40% should be small; most removable redundancy should come from AFDB's multiple selected members and overlap between AFDB and ESM-Atlas. A sequence-only calculation will overstate the removable fraction, but the structurally discordant rescue rate should be small rather than zero. The #292 supplement should have much more removable sequence redundancy because it deliberately adds homologous cluster members; its structurally selected minority should survive a joint rule more often than its quality-fill majority.

## Full-corpus sequence ceiling

A full 69,516,181-document Linclust pass followed by direct Smith-Waterman
verification and a no-chaining selector removes 5,022,836 documents (7.23%) at
50% identity and 80% coverage of both chains. It removes 587,171 (0.84%) at
70%, 276,929 (0.40%) at 90%, and 176,767 (0.25%) at 100% aligned identity.
Because the common rule requires 80% rather than 100% coverage, that last cell
is not the still-outstanding exact-whole-sequence duplicate count.

At 50%, 667,093 AFDB rows (16.83%) and 4,355,743 ESM rows (6.64%) are removed.
The full local AFDB ledger puts its exact loss at 738,657,274 source tokens
(16.66%). The ESM token total still needs a source-local scan of all 3,338
shards; no incomplete-shard extrapolation is presented as exact.

## What does TM 0.7 change?

In #292's curated, measured AFDB anchor/anchor gallery, 44 pairs clear 50%
identity and 80% bidirectional coverage. Requiring both directional TM-scores
to clear 0.7 rescues one pair (2.3%). Raising TM to 0.8 rescues 7 (15.9%); at
0.9 it rescues 24 (54.5%). The operating threshold matters much more than the
phrase "structure-aware dedup" suggests.

## Current interpretation

The 50% sequence-only result is an upper bound on what the requested joint rule
may remove: applying TM>=0.7 can only rescue rows from those 5.02M removals.
Within-source ESM redundancy is unexpectedly the largest component (3.75M),
showing that one representative per earlier 40%-Linclust cluster was not an
all-pairs guarantee. Cross-source witnesses account for 718k removals.

The curated TM gallery suggests a small rescue at 0.7, but it is not a random
sample and cannot be multiplied into 5.02M responsibly. The remaining gate is
a stratified source-local structure/contact sample, followed by full scoring
only if the smoke-run cost is acceptable. No training corpus has been replaced
or deleted.
