# Summary slides — exp: null-sequence contrastive guidance for contacts-v1 rollouts

<!-- Feeds plots/summary.pdf via build_summary.py.
     One ## heading per slide; body text becomes the slide. -->

## Outcome

Bounded null-sequence guidance improves the metric we ultimately care about: oracle best-of-100 contact accuracy.

On 81 eval-val proteins untouched during method selection, all-token poly-Ala guidance at gamma=0.5 raises validity-gated oracle R-precision from 0.5331 to 0.5554 all-range and from 0.5279 to 0.5474 long-range. Paired bootstrap intervals exclude zero.

The literal native/null probability ratio does not work: it falls to 0.4774/0.4713 and becomes unreliable. The useful decoder retains native logits and adds a bounded contrast.

## Method and amended endpoint

For each rollout, native and same-length poly-Ala prompts share the exact emitted suffix. Only the native stream samples:

guided = native + 0.5 * (native - polyAla)

Development used 16 natural proteins and 15 fold-switch pairs. The full comparison used the other 81 eval-val proteins; no eval-test data was read.

After the original full arms launched, the owner clarified that consensus accuracy was secondary and oracle best-of-100 was primary. We froze that amendment before adding pure-ratio and T=1.1 controls. An unfinished rollout or any rollout with a malformed contact statement scores zero.

## Primary result

At N=100 on the untouched 81 proteins:

Bounded guidance: 0.5554 all / 0.5474 long.

Ordinary iid T=1.0: 0.5331 / 0.5279. Overlap-matched T=0.8: 0.5384 / 0.5343. High-diversity T=1.1: 0.5169 / 0.5112. Pure native/null ratio: 0.4774 / 0.4713.

Guidance minus iid is +0.02235 all-range (95% CI +0.01588 to +0.02899) and +0.01952 long-range (+0.01119 to +0.02800).

The gamma=0 path reproduces the established exp277 score across all 97 eval-val proteins within 0.0007, validating the implementation.

## Why it works

The oracle gain comes from stronger individual samples, not a broader pool.

Guidance raises mean valid single-rollout R-precision from 0.4073 to 0.4340, while pairwise Jaccard rises from 0.2846 to 0.3334 and true-contact union recall falls from 0.9364 to 0.9175.

Pure ratio does create breadth—union recall 0.9612—but over-generates: 249/8,100 invalid rollouts, 558 malformed statements, and 353 contacts per rollout. Its best map is much worse.

So the background stream is a useful contrastive quality signal only while the native distribution remains the base.

## Compute and implications

One hundred guided rollouts fit the same H100 time as a median 132 iid rollouts. Guidance still wins that equal-time comparison by +0.01714 all / +0.01415 long, with both intervals excluding zero. It even beats 200 iid rollouts all-range; the long-range 200-rollout interval includes zero.

Fold switching is secondary: guidance scores 6/29 oracle and 3/29 blind versus iid 6/29 and 2/29. Pure ratio reaches 7/29 and 4/29 but emits 5,329 malformed statements, so this is not a usable win.

Conclusion: keep bounded gamma=0.5 guidance for oracle-oriented sampling, reject pure-ratio decoding, and next build a truth-free selector that can realize the improved upper tail.
