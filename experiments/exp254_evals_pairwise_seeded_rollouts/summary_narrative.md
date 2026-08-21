# Summary slides — exp: seed each rollout with a top-ranked pairwise contact — does conditioning on our own high-confidence predictions beat i.i.d. sampling?

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Does **seeding** each of N contacts-v1 rollouts with a distinct high-confidence
contact — taken from the *pairwise* `P(contact)` readout, one seed per rollout —
beat plain i.i.d. rollout sampling, measured both by consensus R-precision and by
oracle best-of-N?

## Why

Two things are known and they pull in opposite directions.

1. **Conditioning on true partial structure is enormously informative.**
   [#163](https://github.com/Open-Athena/MarinFold/issues/163) found that
   prompting a rollout with *ground-truth* partial contacts lifted R-precision
   from 0.145 to 0.556. The joint signal is there; the problem has always been
   that we have no oracle at inference time.
2. **The pairwise readout is a weak stand-in for that oracle.** Under exp82's
   comparison the same weights score ~0.086 *lower* in R-precision under pairwise
   than under rollout voting, so the pairwise top-N is a noisy prior — good
   enough to be better than chance, not good enough to be trusted.

So the pre-registered expectation is:

- **Consensus:** roughly a wash, possibly slightly *worse* than i.i.d. A wrong
  seed poisons a whole rollout, and the seeded pairs additionally get a
  structural +1 vote each, which imports pairwise's ranking errors into the
  consensus. I do not expect this arm to clear i.i.d. by more than noise
  (0.005).
- **Oracle best-of-N:** seeded should be **better**. Forcing N *distinct*
  starting pairs decorrelates the rollouts, which is a broader search of the
  contact-map posterior — exactly what a best-of-N readout rewards. If seeding
  buys anything, it should show up here first, and that would make it a
  candidate reranking/RFT signal rather than a deployable decoder on its own.

The interesting negative result is also informative: if seeded oracle best-of-N
does *not* beat i.i.d. oracle best-of-N, then the per-rollout diversity we
already get from realization resampling is saturating the search, and the
headroom #163 identified is not reachable by conditioning on our own noisy
predictions.

## Results so far

_(Fill in as results come in.)_
