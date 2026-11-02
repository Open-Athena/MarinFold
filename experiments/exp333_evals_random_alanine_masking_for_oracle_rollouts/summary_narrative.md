# Summary slides — exp: random alanine masking for oracle rollouts

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can independently replacing random subsets of a protein sequence with alanine produce complementary contacts-v1 rollouts that improve oracle best-of-100 accuracy on eval-val?

## Why

Poly-alanine retains much of MarinFold's learned contact-map prior, and bounded
native/poly-alanine contrastive guidance improved oracle best@100 in #321.
Sampling directly from independently partially alanine-mutated sequences is a
much simpler way to perturb the conditional.  The experiment asks whether the
resulting hypotheses complement native rollouts at a fixed pool of 100.

## Results so far

The intervention fails its development gate, so the untouched 81 proteins
remain unread.  Against the matched zero-mask worker, 5% masking lowers oracle
best@100 by 0.0286 all-range and 0.0238 long-range; every larger rate is worse.

The best fixed-budget mixture (50 iid + 25 each at 5% and 10%) is +0.0079
all-range against the published iid100 pool but -0.0199 long-range.  It therefore
fails the co-primary long-range safety threshold.  Masking lowers map overlap,
but true-contact union recall and individual-map accuracy fall with it.

## Conclusion

Do not advance random alanine masking to the 81-protein confirmation set.  It
creates novelty by erasing useful sequence evidence, not complementary accurate
hypotheses.  Full poly-alanine is degenerate: most rollouts emit only an end
token and oracle accuracy is effectively zero.
