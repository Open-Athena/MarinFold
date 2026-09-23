# Summary slides — exp: null-sequence contrastive guidance for contacts-v1 rollouts

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can a paired, null-sequence inference pass be used to subtract the sequence-independent contact-map prior during `contacts-v1` decoding, yielding more useful rollout diversity without sacrificing contact accuracy?

## Why

Yes, at moderate guidance strength. At decoding step `t`, run the same model on:

- the native sequence `x` followed by the generated contact-token prefix `y_<t`; and
- a same-length null sequence `x0` followed by the **identical** `y_<t`.

If their next-token logits are `l_x` and `l_0`, sample the native stream from

```text
g_gamma(v) = l_x(v) + gamma * (l_x(v) - l_0(v)).
```

`gamma=0` is ordinary decoding. Positive `gamma` amplifies tokens supported by the real sequence more than by the already-emitted contact map. This is CFG-like, but not literally classifier-free guidance: the second prompt is a sequence ablation, not a learned unconditional condition.

The raw ratio proposed in the motivating idea, `l_x - l_0`, is worth measuring as a diagnostic, but is not the safest primary decoder. A pure ratio can promote a token that is extremely unlikely under both streams merely because the null stream dislikes it slightly more. The primary implementation therefore retains the native logits as the base distribution and applies a native-probability plausibility gate before guidance.

Preregistered expectation:

- moderate guidance will increase **true-contact coverage across rollouts** and reduce rollout overlap while preserving consensus R-precision;
- very large `gamma`, pure-ratio decoding, or guidance on control tokens such as `<end>` will over-generate, increase malformed/low-probability output, or trade precision for indiscriminate novelty;
- the gain, if real, should survive both a homopolymer null and a composition-preserving shuffled-sequence null. A result unique to one null is a null-prompt artifact, not evidence for sequence-specific guidance.

## Results so far

_(Fill in as results come in.)_
