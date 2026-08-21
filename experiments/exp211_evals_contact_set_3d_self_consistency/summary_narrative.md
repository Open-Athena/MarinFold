# Summary slides — exp: is a contacts-v1 rollout a geometrically self-consistent contact set?

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.

     RENDERING NOTE: render_text_slide runs textwrap.fill(p, width=90) over
     blank-line-separated paragraphs. There is no markdown — tables collapse
     into run-on text and **bold** shows literally. Keep every row/bullet as
     its own blank-line-separated paragraph, under ~90 characters. -->

## The question

When contacts-v1-exp199-1.5B emits a contact set in one rollout, is that set jointly
realizable as a single 3D structure — more so than a contact set carrying the same
per-pair marginals but assembled across different rollouts?

Plainly: does autoregressive generation produce a coherent structural hypothesis, or a
bag of independently-drawn contacts that happen to share one document?

Issue #211 — github.com/Open-Athena/MarinFold/issues/211

## Why it is worth asking

Every contacts-v1 eval we have (#82, #89, #180) collapses the 100 rollouts per protein
into per-pair vote counts and scores the marginals. The per-rollout partition is
discarded at the moment of the increment, so nothing has ever measured whether a single
rollout is internally coherent.

Two prior results say the joint is where the information is, and neither tests
generation. #201 Phase 0: 77% of the val loss is nuisance permutation entropy, so
next-token CE barely scores the joint. #163: conditioning on TRUE partial contact maps
lifts R-precision 0.145 to 0.556.

If consistency ranks rollouts without ground truth it is also a best-of-N selector and a
candidate RL reward for #200 / #208 — usable on sequences with no known structure, which
is the entire ESM-Atlas half of #199's training mixture.

## The method that did not work

The natural approach — Floyd-Warshall bound smoothing over the distances implied by the
contacts plus the 3.8 A CA(i)-CA(i+1) bond — has no discriminative power. It reports
zero violations for a true contact set AND zero for a separation-matched random one, at
four different bound pairs.

The reason is structural, not a tuning failure. A violation needs a path of upper bounds
summing below some lower bound. Any two-hop contact path is 2U, about 20-24 A, and a
contact plus k backbone steps is U + 3.8k — so nothing reaches under a ~10 A lower bound
once min_seq_separation = 6 has excluded the close-in-sequence pairs.

Triangle smoothing tests feasibility in an arbitrary METRIC SPACE, which a contact graph
satisfies almost for free. It does not test feasibility in R^3.

## What works instead

The 3D embedding residual: minimise bond, contact, non-contact and steric violation over
x in R^(Lx3), and report the leftover contact excess, minimised over restarts.

Floyd-Warshall is kept as step 1 of 3 — it is the bound-smoothing step of the classic
Crippen-Havel EMBED algorithm and the right preconditioner for the embedding — but it is
not the metric.

Bounds are measured on all 554 ground-truth proteins, not assumed (91,525 contact pairs,
11.1M non-contact pairs), reusing exp174's published GT bundle:

bond = 3.809 A (median CA(i)-CA(i+1))

u_contact = 13.02 A (contact CA-CA p99.5)

l_noncontact = 6.46 A (non-contact CA-CA p0.5)

d_min = 3.53 A (closest non-bonded CA pair, p1)

## The number that governs every claim

10.7% of non-contact pairs sit closer than u_contact.

pyconfind contacts are SIDE-CHAIN contacts, so CA-CA distance is only a proxy for them
and no threshold pair separates the two populations. A nonzero residual therefore means
"less geometrically consistent", never "provably unrealizable".

The arm comparison is unaffected: every arm is scored under identical bounds, so the
scale cancels. But the absolute number is not a proof of anything, and the writeup does
not treat it as one.

## The calibration gate passed, and corrected the issue

Ground truth beats a separation-matched random set of the same size and the same |i-j|
profile by 5.6x in median per-contact excess, on 89.6% of the 470 chain-break-free
proteins — 95.4% at L 100-200, 88.3% at L 200-350, and 100% at L >= 350.

Two of the three criteria written into the issue were wrong. "GT scores about 0" is
incoherent against the bounds it was paired with: u_contact is a p99.5, so ~0.5% of real
contacts exceed it by construction and ground truth carries a structural nonzero floor.
The gate is now relative.

The decoy arm is not a floor. A different real protein's contact map scores the same as
the true one (0.0384 vs 0.0337; truth wins on 49.6%, a coin flip). That is correct — the
score is sequence-blind — but it bounds the claim: this experiment detects "not a fold at
all", not "wrong fold copied from a real one".

Scope limit: below L = 100 the metric is near-blind (GT 0.0000 vs random 0.0011) — a
short chain embeds almost anything. A further 15% of the eval set has a chain break and
is not embeddable as the continuous chain a contacts-v1 document asserts. Both subsets
are reported separately, never silently dropped.

## How sensitive is it where the model actually lives?

The gate contrast is easy. The real contrast is a rollout against a chimera built from
the same rollouts with the same marginals — both roughly 60/40 true/false mixtures of the
same contacts. Sweeping corruption across #199's operating band (R-precision about 0.59,
so about 40% wrong), over 60 proteins at L >= 100:

a 0.05 step in corruption: sign consistency 53-65%, p 0.04-0.35 — NOT reliably separable
at n = 60.

a 0.10 step: sign consistency 58.3%, p = 0.006 — separable.

a 0.20 step: sign consistency 73.3%, p = 2.3e-05 — strongly separable.

So the resolvable effect is bounded and the experiment's power has to come from scale.
This sweep is also an UPPER bound on the real effect: the chimera keeps every pair the
model proposed and only breaks their co-occurrence, a gentler perturbation than swapping
pairs for random ones.

## The arms

Every arm is size-matched. #142 measured rollouts emitting about 0.70x the ground-truth
contact count, and a sparser set is trivially easier to embed, so an unmatched comparison
would read under-generation as consistency.

ground truth — calibration ceiling. GT size-matched — removes the count confound.

rollout — the treatment.

marginal-matched chimera — THE KEY NULL: pairs drawn from the pooled rollout votes with
probability proportional to vote count.

splice chimera — half of rollout a plus half of rollout b. Decoy protein —
sequence-blindness ceiling. Separation-matched random — floor.

Why the marginal chimera is the sharp test: it shares the model, the protein, the
per-pair marginals and the set size with the rollout. The only difference is whether the
contacts were drawn jointly in one autoregressive pass or independently from the pooled
vote distribution. Any gap is joint structure the model put there while generating, and
cannot be explained by marginal accuracy — which is all any existing eval measures.

## Result: yes, and the effect is large

554 proteins x 100 rollouts, 30 replicates per arm, 51,890 scored contact sets.
Headline on the 394 chain-break-free proteins at L >= 100.

Rollout vs marginal-matched chimera: mean delta +0.0655 per contact, 95% CI
[+0.0561, +0.0752], rollout more consistent on 89.8% of proteins, Wilcoxon
p = 1.4e-58.

Rollout vs splice chimera: +0.0339, rollout better on 93.1%, p = 2.4e-62.

For scale: the entire ground-truth-to-random range of this metric is 0.064 to
0.272. The rollout-vs-chimera effect is 0.0655, about 31% of that full range. This
is not a marginal result.

A rollout is about as 3D-consistent as the real contact map (0.0562 against 0.0639
for ground truth, 0.0630 for size-matched ground truth), while the chimera built
from the same contacts with the same marginals sits at 0.1216.

## The effect grows with protein length

L 100-200: delta +0.0473, rollout better on 88.3% of 240 proteins.

L 200-350: delta +0.0771, rollout better on 90.6% of 128 proteins.

L 350-761: delta +0.1763, rollout better on 100.0% of 26 proteins.

That is the direction #180 predicts, and it is the opposite of an artifact: if the
effect were driven by the metric's slack it would shrink as constraints multiply.

Below L = 100 the effect is +0.0029 (70.4%) — the gate already established the
metric is near-blind there, and it duly reports nothing. The 84 chain-break
proteins, excluded from the headline, show +0.0915 on 95.2%.

## But consistency does NOT rank rollouts by accuracy

The secondary question was whether this buys a reference-free selector. It does
not, or barely.

Spearman rho(excess, precision) within a protein: mean -0.0175, useful on 51.8% of
proteins — a coin flip.

Picking the most-consistent of 30 rollouts gains +0.0110 precision (95% CI
[+0.0032, +0.0188]) against an oracle headroom of +0.1299. That is about 8% of the
available headroom: statistically nonzero, practically weak.

The calibration gate already explained why. A decoy protein's contact map scores
the same as the truth (0.0668 vs 0.0639) because the score is sequence-blind. It
cannot tell a coherent WRONG fold from a coherent RIGHT one — and a rollout can be
highly self-consistent and still wrong.

## Conclusion

The model generates a coherent structural hypothesis, not a bag of independently
drawn contacts. Sampling the same contacts with the same per-pair marginals but
independently costs 0.0655 per contact, on 89.8% of proteins, and the gap widens
with length. Autoregressive generation is doing real joint work that every
marginals-only eval (#82, #89, #180) discards at the vote-counting step.

The coherence is not accuracy-aligned. Self-consistency is nearly uncorrelated with
whether a rollout is right, so it is not a best-of-N selector and, on its own, not
an RL reward — it would reinforce coherence the model already has. For #200 / #208
that is the useful negative: pair it with an accuracy signal or skip it.

Read together with #163 — where conditioning on TRUE partial contact maps lifts
R-precision 0.145 to 0.556 — the picture is that the model's remaining gap is about
CORRECTNESS, not COHERENCE. It already commits to a single self-consistent
structure; it commits to the wrong one.

## Caveats, stated plainly

The score is statistical, not a proof. 10.7% of real non-contact pairs sit closer
than the contact upper bound, so a nonzero residual means "less geometrically
consistent", never "provably unrealizable".

The rollout-beats-ground-truth comparison is confounded and is not claimed. The
model preferentially emits short-range contacts (precision 0.679 short vs 0.566
long), and short-range contacts are geometrically easier to satisfy. The primary
contrast is NOT confounded this way: the chimera is drawn from the same pooled
rollout contacts with the same marginals, so it carries the same separation
profile.

Below L = 100 the metric is uninformative, and 15% of the eval set has a chain
break. Both subsets are reported separately rather than dropped.

Cost: 95.8 min to generate 55,400 rollouts and 494 min to score 51,890 contact
sets, both on one RTX A5000. The CoreWeave fan-out was written but unusable — the
workstation's credentials are dead.
