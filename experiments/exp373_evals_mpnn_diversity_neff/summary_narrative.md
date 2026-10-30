# ProteinMPNN-only diversity and Neff dependence

## Question and design

Does redesign-only training broaden contact predictions, and does it reduce dependence on available evolutionary information?

Two pinned 1.5B checkpoints; 97 natural eval-val proteins and 19 designs reported separately. Three temperatures, one fixed prompt plus a matched randomized prompt bank. 278,400 individual samples planned. Eval-test excluded.

## Frozen primary comparisons

Paired fixed-prompt Jaccard diversity at temperature 1, controlled for contact count. True versus false discovery and ensemble accuracy distinguish useful alternatives from additional errors.

Neff analysis uses 100 randomized prompts. Fixed tiers contain 4, 10, 28 and 55 natural proteins. Compare high-minus-low accuracy drops and the slope of paired accuracy differences against log10(Neff).

## Current state

Plan and inputs frozen before inference. The canonical parser passes five tests of offset/orientation invariance, duplicates, ordered retraction, malformed outputs and deterministic seeds.

Both checkpoint-local smokes passed: 200/200 completed samples, matching prompts and tokenizer hashes. Twelve H100 production shards are active. Upload rate-limit failures were recovered with a direct bucket API and bounded retries; completed conditions are reused. Final results pending.
