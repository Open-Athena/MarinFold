# Summary slides — exp: generate contacts-v1 training dataset that includes pause tokens

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can we improve contact-prediction accuracy by adding `<think>` (pause) tokens at inference time to the **contacts-v1** document structure, assuming the model was trained on documents that contain them?

## Why

Same mechanism as #34: at inference time `<think>` tokens let the model spend extra compute before it has to commit to the next token, without conditioning on additional real tokens — potentially boosting accuracy. This issue is the **contacts-v1 analog of #34**, which did this for `contacts-and-distances-v1` (producing `contacts-and-distances-v2`).

## What was built

A `think` flag on the existing contacts-v1 library generator
(`GenerationConfig(think=True)` / CLI `--think`): `<think>` runs are spliced
between `<contact>` statements, drawn from #34's exact distributions. No vocab
or tokenizer change (`<think>` was already reserved); `think=False` is
byte-identical to the pre-think generator, so existing checkpoints are safe.

## Results

168 contacts-v1 tests pass (144 old + 24 new). Over 10k synthetic think docs,
empirical stats match spec: initial-run gate 0.75, initial-run mean length 7.6
(~1/0.13), extra-run count mean 0.75, extra-run mean length 4.0 (~1/0.25).
Every doc fit 8192 tokens, ended with `<end>`, and kept `<think>` strictly
between contacts. Real generator on 1QYS: think adds 31 tokens (420→451),
still fits.

## Next

Follow-up issues: at-scale generation → training with `<think>` loss-masked →
inference-time evaluation vs. the best contacts-v1 model.
