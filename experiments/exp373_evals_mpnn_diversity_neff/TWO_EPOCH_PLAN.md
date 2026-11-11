# Two-epoch checkpoint follow-up

Frozen before inference on 2026-11-02. Add both completed redesign-only
checkpoints from issue 364: learning rates 5e-4 and 1e-3, step 373768.
The 5e-4 model is the primary follow-up selected in the checkpoint handoff;
1e-3 is a secondary learning-rate comparison. Preserve the default and
one-epoch results and their immutable v1/v2 artifacts.

Reuse the exact 97 natural eval-val and 19 eval-denovo inputs, truth,
metadata, candidate universe, parser, tokenizer, paired prompts and seeds.
No eval-test. For each new model and protein run fixed prompts with 300
samples at temperatures 0.7 and 1.3, 1000 at temperature 1, and 100 randomized
prompts at each temperature. Generate the temperature-1 fixed samples as
300 then 700, matching the earlier generation call boundaries. This adds
440800 individual samples across 1392 conditions. Keep malformed and capped
samples, and report their counts. Preserve the original prefix-continuation
policy and sampling/runtime parameters.

Run beside the verified exports in cw-us-east-02a, at most 12 batch-priority
single-H100 workers (six interleaved shards per model). No model egress.
Verify all six checkpoint files against the supplied SHA256 manifest inside
each worker. Gate production on 100 randomized-prompt smoke samples from
one real protein per checkpoint. Capture per-input timings and hardware.
Publish raw samples and reproducible sources under v3-two-epoch; do not
overwrite v1 or v2-1000.

Analysis uses the original definitions: first 300 fixed samples for diversity,
entropy, contact-count thinning, seed-block checks and exact-margin shuffles
(10 chains with 3000 trades, independent 1500-trade mixing checks); randomized
100 at temperature 1 for the primary Neff analysis; fixed temperature 1
through 1000 for ensemble accuracy, discovery, oracle prefix-best F1 and
mean individual F1. Oracle selection uses truth independently for all/long
ranges. Mean individual F1 averages the 1000 single-rollout scores within
each protein, then averages proteins equally. All/long use separations >=6/24.

Report all four models and paired protein-bootstrap differences: each
redesign model minus default, each two-epoch model minus the original
one-epoch model, and 1e-3 minus 5e-4 at two epochs. Use the original 10000
bootstrap draws and seed 373; pointwise 95% intervals, unadjusted for multiple
comparisons. Report natural proteins first and designs separately. Repeat
Neff tier/extreme-drop/continuous interaction, raw-depth, length/viral
adjustment, nonviral and seed-block sensitivities. Preserve the eight
prespecified contact-map examples. No matched redesign sequence-KNN null is
available; make no new mechanism or generalization claim from these comparisons.

The checkpoints differ in training exposure and learning-rate schedules;
epoch contrasts do not isolate training duration causally. The default has
not changed. Compare new randomized-prompt accuracy to issue 364's independent
evaluation as a sanity check, allowing stochastic differences from its seeds.
