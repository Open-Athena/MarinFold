---
marinfold_experiment:
  issue: 368
  title: 'exp: proofreader guided inference for accurate contact maps'
  kind: evals
  branch: codex/exp368-proofreader-guidance
---

# exp: proofreader guided inference for accurate contact maps

**Issue:** [#368](https://github.com/Open-Athena/MarinFold/issues/368) · **Kind:** `evals` · **Branch:** `codex/exp368-proofreader-guidance`

## Question

Can the released exp356 proofreader improve complete MarinFold contact maps by guiding candidate selection, lookahead, population search, repair, or aggregation at inference time?

## Hypothesis

Proofreader guidance will improve precision and recall at matched compute, with lookahead benefiting from the model's demonstrated retrospective contact judgments. Strong guidance may sacrifice candidate coverage or exploit score errors, so actual contact accuracy and diversity must be measured.

## Background

Builds on #356 and PR #358. Freeze generator exp277 step 266344 and proofreader exp356-exp277-frozen-contact-pdb50k-v2 step 18000. The user authorized autonomous implementation, CoreWeave execution, and job monitoring. Existing checkpoints and targets are co-located in CoreWeave us-east-02a.

## Approach

1. Implement and verify cached, batched contact proposals and proofreader readout against the released inference implementation. Preserve actual prompts and original contact semantics.
2. Use a deterministic group-aware validation pilot for candidate-ranking diagnostics and end-to-end policy comparisons: top-percentile filtering; soft probability weighting; short lookahead; population search; one-pass suffix repair; and larger independent rollout pools with alternative aggregators.
3. Measure all/long-range R-precision, average precision over the full eligible pair universe, precision at fixed L-based budgets, recall/precision curves, candidate coverage, diversity, and per-input GPU time. Include a fast ordinary-generation baseline and matched-compute comparisons; distinguish compute envelopes from exact matching.
4. Extend promising policies to a larger validation panel. Freeze selection and stopping rules before one final reserved-test comparison. Never select policies on test results.
5. Save resumable per-input artifacts, source hashes, timing CSVs, job and recovery metadata. Publish figures and a written result report, including negative results.

## Success criteria

At least one guided policy should improve final contact-map quality over ordinary sampling and proofreader-weighted aggregation at comparable measured GPU cost, with paired group-bootstrap uncertainty. A complete null result is acceptable if the planned policy families are tested and the failure mode is measured.

## Results

Implementation and validation are in progress on 2026-10-21. Four CPU tests
cover cache isolation, full-forward equivalence, proposal resampling, lookahead
commit boundaries, repair-prefix reconstruction, and full-universe AP. The first
H100 smoke test verified that every frozen backbone weight matches the original
generator. It also caught a BF16 incremental/full-forward discrepancy (maximum
0.02394 contact probability); the experiment therefore tests FP32 cached search
and rescores final maps through the released BF16 full-forward implementation.
No guided-policy accuracy result is available yet.

### Frozen pilot protocol

`pilot.json` fixes 32 validation proteins from 32 distinct homology groups,
spanning 33–937 residues. One representative per group is chosen by a SHA256
key, then 32 uniformly spaced length ranks are selected. Neither contact labels
nor previous model quality enters panel selection. Four independent prompts and
sampler seeds are used per guided policy and protein. The same prompt identities
seed the ordinary rollout pool.

The pilot includes an ordinary grammar-constrained control, uniform selection
from 16 proposals, top 10/25/50% of 16 proposals, top 10% of 64, soft weights
q^beta for beta 1/2/4, top-10% selection with 10% unguided exploration, 8/32-contact
lookahead with two continuations per proposal, ranking by total expected correct
contacts, eight-particle search in blocks of eight, and one-pass suffix repair.
Percentile gates retain ceil(fraction*K) proposals: the nominal 10% gate keeps
2/16 or 7/64. Generator probability is already represented by proposal sampling;
it is not multiplied into resampling weights again.

The constrained sampler permits valid contacts with sequence separation at least
six and no duplicate pairs. It samples the first endpoint, then the second,
using temperature 1 and top-p .95 after each grammar mask. It is consequently a
grammar-conditioned generator, with its own matched control. EOS is sampled
once at each contact boundary, without proofreader intervention. Lookahead uses
a fixed contact horizon and does not commit its future tokens; this is
length-conditioned evidence, not a trained terminal-value estimate. Population
search ranks current prefixes by predicted F1 and applies a 0.1 Jaccard penalty
while selecting survivors; descendants contribute one final map per run.
Repair uses future evidence to locate the lowest-confidence contact in the
first half, discards that contact and its suffix, then regenerates with guidance.

Independent ordinary generation uses the original unconstrained vLLM recipe.
Nested pools of 8, 32 and 128 rollouts are generated in waves of 8, 24 and 96,
so their cumulative generation and proofreading costs are measured directly.
Aggregation compares frequency, sum q, mean q, sqrt(frequency)*mean q, and sum q
with inverse-neighbor weights for trajectories whose Jaccard similarity is at
least 0.8. These pool comparisons retain the actual generator's invalid and
duplicate diagnostics; map metrics consider only eligible unique pairs.

Primary outcomes are all-contact and long-range (separation at least 24)
R-precision and AP over the complete eligible pair universe, including all
unproposed pairs at score zero. We also report precision at L, L/2 and L/5,
candidate-pool oracle recall, and measured H100 inference time. Timing includes
discarded proposals, lookahead, repair and final rescoring; model loading is
reported separately. Cost claims distinguish measured envelopes from exact
matching. Group bootstrap intervals retain a mean-over-structures estimand.

After the pilot, choose at most three methods on the validation accuracy/cost
frontier, with all-contact R-precision primary and long-range R-precision used
to reject gains limited to local contacts. Extend these methods to the remaining
validation groups before writing a fixed test plan. No reserved-test scores are
read for tuning, panel selection or stopping decisions. The existing test set
was previously used for the proofreader release, which is disclosed in the
eventual report; generator pretraining decontamination is not established.

### Execution and recovery

Jobs are independent batch H100 root jobs on `cw-us-east-02a`, submitted through
the central `marin` controller. Inputs and weights are already in its regional
S3 store. Outputs live under
`s3://marin-us-east-02a/MarinFold/exp368/`. The standalone Iris submission client
is pinned to `af9c6789e8` because the older workstation client is rejected by the
current controller. `data/dispatches.jsonl` records exact commands, source hashes,
job IDs and monitoring ownership. Per-rollout commit markers permit restart
without losing completed trajectories. A changed code or plan requires a new
output prefix. No new model training is part of this experiment.

## Conclusion

_(Fill in after results are in.)_
