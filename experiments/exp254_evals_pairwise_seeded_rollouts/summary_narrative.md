# Summary slides — exp254: seeding rollouts with top-ranked pairwise contacts

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     The renderer wraps blank-line-separated paragraphs and does not render
     markdown lists or emphasis — write plain sentences, one per paragraph.
     Keep this current as the experiment progresses. -->

## What we're doing

Instead of drawing N contacts-v1 rollouts i.i.d., first compute contact
probabilities with the pairwise P(contact) readout, take the top N pairs, and
start rollout r by prompting the model with pair r.

Score both ways on eval-val (97 natural FoldBench monomers, #245) with the
decontaminated #232 m2-p06 checkpoint: consensus R-precision as usual, and the
oracle best rollout per structure as a headroom diagnostic.

## Why we expected a tie, and a diversity win

Conditioning on true partial structure is enormously informative: #163 lifted
R-precision from 0.145 to 0.556 by prompting with ground-truth partial contacts.
The catch has always been that there is no oracle at inference time.

The pairwise readout is a weak stand-in for one — under exp82 the same weights
score about 0.086 lower in R-precision under pairwise than under rollout voting.

So consensus was predicted to be a wash: a wrong seed poisons a whole rollout.
Oracle best-of-N was predicted to improve, because forcing N distinct starting
pairs decorrelates the rollouts, and a broader search is what best-of-N rewards.

## Results

Gate first: the unseeded control reproduces #245's published m2-p06 eval-val
number, 0.5217 against 0.520, so these arms sit on the same axis as every other
MarinFold contact number.

Consensus R-precision on eval-val (n=97), all-range, against a 0.5217 control.

Seeded with the top 100 pairs overall — 0.5234, paired gain +0.0017.

Seeded with 100 long-range pairs only — 0.5244, paired gain +0.0028.

Seeded with the best 33 in each separation range — 0.5247, paired gain +0.0030.

Every one is a tie: all three intervals straddle zero and sit inside #204's
0.005 noise floor. Both preregistered predictions held.

The only significant effect anywhere is oracle best-of-100 under top-100
seeding, +0.0142 [+0.0055, +0.0247] — and that is the arm that does not
concentrate its seeds. Restricting all 100 seeds to long range gives the gain up.

## Biasing toward long range does not work

The top 100 pairs overall are already 56.8% long-range, because long-separation
pairs dominate the candidate universe. Equal thirds therefore lowers the
long-range share to 34% rather than raising it.

Neither helps, and the arm aimed at long range is behind the unaimed one at long
range: 0.5068 against 0.5081.

Long-range seeds are not intrinsically worse. Inside the equal-thirds arm they
are the most accurate at 63.5%, against 49.0% medium and 46.0% short. Depth into
the pairwise ranking is what costs a seed its accuracy, not separation range.

## Why every arm ties: one contact is almost no conditioning

The rollout index is the pairwise rank of its seed, so the run contains its own
dose-response curve, and the curve is flat.

Seeds ranked 1-10 are true contacts 79.2% of the time and the rollouts they
produced score 0.3931. Seeds ranked 71-100 are true only 46.0% of the time and
their rollouts score 0.3890. Rollouts given no seed at all score 0.3868.

Within a protein a true seed beats a false one by +0.0124 [+0.0082, +0.0168],
and a false seed costs nothing against no seed at all. Against #163's 0.145 to
0.556 from conditioning on true partial contact sets, the signal is in the joint
structure of many contacts, not in an anchor.

Beware the pooled true-versus-false split, which reads +0.18 and is protein
difficulty, not conditioning: good proteins yield both more correct seeds and
better rollouts.

## The bottleneck is ranking, not diversity

The 100 rollouts already propose 92% of the true contacts, from only 15.7x R
distinct pairs, with a mean pairwise Jaccard of 0.257 between any two of them.
All three seeded arms match that to within 0.002.

Vote rank recovers 0.52 of the true contacts at the R cut, 0.67 at 2R, 0.79 at
5R and 0.92 at the union. The 0.52 to 0.92 gap is ranking loss, and it is five
times larger than anything more diversity could buy.

Re-ranking the pooled pairs does not close it. Cross-validated over proteins,
the best R-precision from any combination of vote count, pairwise probability,
emission rank and rollout self-consistency is +0.0015 — a tie. Fitting over all
candidates instead buys +0.020 AUC and costs 0.0065 R: you can have one or the
other, because 99.9% of candidates are pairs no rollout ever proposed.

## What to do with it

Seeding is not a decoding win, and neither is biasing where the seeds come from.
The one thing it bought was diversity, and the rollouts are already diverse.

Pointwise re-ranking is exhausted too — votes, pairwise probability, emission
rank and self-consistency all measure the same marginal confidence and agree.

Reaching the 0.92 the sampler already proposes needs something that scores sets
rather than pairs: geometric realizability (#211), a folding model in the loop,
or joint conditioning on partial maps of size k greater than one, which is where
#163 says the signal is. The downstream half — how many of these pairs a folding
model actually wants — is MarinFold #256.
