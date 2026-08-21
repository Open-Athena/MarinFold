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

R-precision on eval-val (n=97), all-range then long-range.

pairwise P(contact), the seed source — 0.4585 / 0.4407.

i.i.d. consensus, the control — 0.5217 / 0.5042.

seeded consensus — 0.5234 / 0.5081.

i.i.d. oracle best-of-100 — 0.5199 / 0.5415.

seeded oracle best-of-100 — 0.5341 / 0.5433.

Both preregistered predictions held. Consensus is a tie: +0.0017 [-0.0011,
+0.0046] paired over proteins, inside #204's 0.005 noise floor. Oracle
best-of-100 is a small real win: +0.0142 [+0.0055, +0.0247].

## Why it is a tie: one contact is almost no conditioning

The rollout index is the pairwise rank of the seed it was handed, so the run
contains its own dose-response curve — and the curve is flat.

Seeds ranked 1-10 are true contacts 79.2% of the time and the rollouts they
produced score 0.3931.

Seeds ranked 71-100 are true only 46.0% of the time and their rollouts score
0.3890.

Rollouts given no seed at all score 0.3868.

Within a protein a true seed beats a false one by +0.0124 [+0.0082, +0.0168],
and a false seed costs nothing against no seed at all. Against #163's 0.145 to
0.556, the signal is in the joint structure of many contacts, not in an anchor.

Beware the pooled true-versus-false split, which reads +0.182 and is protein
difficulty, not conditioning: good proteins yield both more correct seeds and
better rollouts.

## What to do with it

Seeding is not a decoding win. But the lever that moved was diversity, not
accuracy — 100 distinct starts raised the best of 100 without raising the
average.

The follow-up is seeding with k > 1 contacts, a partial map rather than an
anchor, which is where #163 says the signal lives. At 58% seed accuracy a
k-contact seed is right only 0.58^k of the time, so it needs either the sharper
top-10 prior (79% accurate) or a retraction-trained model (#158 / #175).

Two side findings worth keeping: consensus over 100 rollouts already matches the
oracle best single rollout at all-range on this set, and the remaining best-of-N
headroom is entirely long-range.
