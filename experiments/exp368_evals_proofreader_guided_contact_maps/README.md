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

Implementation and validation are in progress on 2026-10-21. Seven CPU tests
cover cache isolation, full-forward equivalence, proposal resampling, lookahead
commit boundaries, repair-prefix reconstruction, and full-universe AP. The first
H100 smoke test verified that every frozen backbone weight matches the original
generator. It also caught a BF16 incremental/full-forward discrepancy (maximum
0.02394 contact probability); the experiment therefore tests FP32 cached search
and rescores final maps through the released BF16 full-forward implementation.
The complete guided pilot is still pending; partial cohorts are not used for
policy selection. The ordinary nested-pool baseline is complete on all 32
pilot structures. Sum-q R-precision rises from 0.5792 at eight rollouts to 0.5993
at 128; long-range R-precision rises from 0.5537 to 0.5806. The paired all-contact
gain is 0.0200 (95% group-bootstrap interval 0.0050–0.0329). Oracle candidate
coverage rises from 0.7765 to 0.9509. At 128 rollouts, frequency alone gives
0.6040 all-contact and 0.5864 long-range R-precision. These pilot means should
not be compared with means on the larger validation cohort.

Two aggregation variants were added after observing the pilot: confidence as a
tie-breaker for integer vote counts, and votes weighted by predicted rollout F1.
Their all-contact gains over frequency at 128 rollouts are 0.0004 and 0.0018;
both paired 95% intervals include zero. These are exploratory validation results.
CSV inputs and timings are in `data/pilot_pool/`; the figure is
`plots/pilot_pool.png`. An independent-pool latency benchmark requests each
pool in a single vLLM call, avoiding the staged baseline's repeated batch costs.

Ordinary-pool diagnostics now include fixed sequence-length-budget precision/recall
curves, emitted-contact calibration and trajectory overlap. On the 128-rollout
pilot, mean absolute error in per-rollout predicted precision is 0.0606 and in
predicted recall is 0.0547, averaged within each structure and then across the
32 structures. Mean pairwise trajectory Jaccard overlap is 0.3473 (at most 128
deterministically sampled trajectory pairs per protein). These measurements are
for ordinary generation; guided-rollout calibration is evaluated separately.
The calibration curve pools contacts, whereas the reported error means weight
structures equally. Figures and diagnostic CSVs are generated by
`diagnose_maps.py` and `plot_diagnostics.py`.

![Ordinary rollout pool precision, recall, calibration and diversity](plots/pilot_pool_diagnostics.png)

The separate single-batch pool benchmark is also complete on all 32 pilot
structures. At 128 rollouts, frequency ranking gives all-contact R-precision
0.6036 and long-range R-precision 0.5894, using 7.165 mean warm H100 seconds per
map. Relative to eight rollouts with sum-q aggregation (0.5792 all-contact,
0.5537 long-range, 2.649 seconds), the paired gains are 0.0244 (95% interval
0.0101–0.0373) and 0.0357 (0.0154–0.0519). At 128 rollouts, sum q gives 0.6006
all-contact and 0.5876 long-range R-precision at 9.581 seconds. The candidate
pool contains 95.39% of native contacts. These single-batch measurements are the
preferred practical cost baseline; nested pools remain a separate experiment.
Neither score nor timing comparisons treat stochastic completions from different
batch sizes or deployments as identical. Data are in `data/pilot_direct/`.

![Contact-map accuracy from pools requested in a single batch](plots/pilot_direct_pool.png)

On the same complete pilot, `analyze_reranking.py` identifies exactly which
contacts enter and leave the top R when frequency ranking is replaced by sum q.
At eight rollouts, sum q changes 15.37% of those prediction slots on average and
improves all-contact R-precision by 0.0163 (95% paired interval 0.0090–0.0248).
At 128 rollouts, it changes 7.62% of slots and the mean effect is −0.0031
(−0.0099–0.0039), which does not establish a difference. True contacts gained
and lost by these swaps exactly reproduce the per-protein metric differences.
Thus the pilot supports proofreading small pools, but does not establish an
additional ranking benefit at 128 rollouts. This is a retrospective diagnostic
of existing complete pools; the larger validation comparison remains pending.

![True contacts gained and lost by proofreader reranking](plots/pilot_reranking.png)

### Larger validation frequency baseline

Ordinary generation is complete for all 1,375 structures in the 255 validation
groups outside the pilot. With frequency ranking, increasing from eight to 128
rollouts improves all-contact R-precision from 0.5014 to 0.5477. The paired gain
is 0.0463 (95% group-bootstrap interval 0.0403–0.0526). Long-range R-precision
increases from 0.4709 to 0.5215, a gain of 0.0506 (0.0430–0.0586), over the
1,371 structures with at least one long-range native contact. Mean warm generation
cost increases from 2.431 to 6.599 H100 seconds per map (2.714 times the cost).

Oracle candidate coverage increases from 0.7392 to 0.9523, while full-universe AP
increases from 0.4565 to 0.5632. At 128 rollouts, precision at L, L/2 and L/5 is
0.5058, 0.6709 and 0.7932 respectively. This comparison uses frequency for both
pool sizes. Fresh proofreader-weighted scoring on the same generated pools is
still running; the older 1,552-structure reanalysis uses a different cohort and
cannot substitute for that comparison. Data and all 2,750 contemporaneous
per-input generation timings are in `data/validation_frequency/`.

![Larger validation comparison of frequency-ranked pools](plots/validation_frequency.png)

Across the larger validation panel, 1,196 structures improve with 128 votes,
63 tie and 116 worsen; the median R-precision gain is 0.0392. Descriptive
stratification at 128, 256 and 512 residues finds positive mean gains in every
length band, for both all-contact and long-range R-precision. The 128-rollout
median warm latency is 4.345 seconds, its 95th percentile is 17.024 seconds,
and the maximum is 63.331 seconds. These measured times exclude model loading
and output writes. Fixed length strata and underlying values are saved beside
the full-panel metrics.

![Latency and protein-level effects of larger rollout pools](plots/validation_pool_scaling.png)

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

### Exploratory fixed-length control

An additional compute-envelope baseline, `pilot_large.json`, requests independent
ordinary pools of 512 and 2,048 rollouts on the same 32 pilot structures. It was
added before the full guided pilot was available because observed lookahead cost
already substantially exceeded the 128-rollout baseline. Pool sizes were fixed
before these runs; frequency ranking can be evaluated directly from raw outputs.
This adaptive baseline measures the benefit of spending more compute on ordinary
sampling and does not claim exact matching to every guided policy's cost.

While the original pilot was running, five short proteins had complete results
for every policy. On this shared, incomplete cohort, top-10% filtering increased
mean per-rollout precision from 0.5862 to 0.6956, but emitted 26.10 rather than
40.85 contacts and reduced recall from 0.5803 to 0.5193. Every trajectory in
these two groups ended with a sampled EOS. These observations motivate a
stopping-rule test; they do not select a winner or estimate full-pilot performance.

`length_probe.json` freezes an additional exploratory comparison on the same
32 pilot proteins: ordinary constrained sampling and top-10%-of-16 sampling,
each evaluated at exactly L and 2L unique contacts with EOS disabled. Four
independent continuations are generated per sampler and protein. The L map is
a prefix of its corresponding 2L continuation, giving 256 underlying trajectories
and 512 prefix maps. No artificial EOS is appended. Both methods receive the
same fixed budgets, which depend only on sequence length. Timings accumulate
generation and prefix scoring, including the intermediate L score in the 2L
cost, while excluding artifact writes. This addition is explicitly adaptive and
uses no new reserved-test scores. The original 15-policy protocol remains frozen.

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

A temporary eight-GPU RNO server staged a shared 6.014 GB model/input copy to
use available capacity. Iris preempted it four times; `--max-retries=0` only
disabled failure retries, while preemption retries defaulted to 1000. Local
state was lost, the staging counter reset, and repeated input transfers exceeded
the intended 9.5 GB campaign ceiling. The server was canceled when this was
detected at 15:20 UTC. `data/preemption_incident.json` records the evidence.
Initial recovery launches were restricted to storage-colocated `cw-us-east-02a`.
`recover_task.py` downloads and verifies each original task's frozen source
bundle and resumes directly from its durable S3 trajectories. Twelve independent
H100 recovery jobs retain the original sampling plan and sharding. The ordinary
baseline is also queued on all 1,375 structures from the 255 remaining validation
groups, excluding every pilot group. No guided winner or test policy is selected.

After a prolonged East capacity wait, a task-owned object-store bucket was
created in `RNO2A`. `prepare_regional_mirror.py` made one 6,014,167,823-byte copy
of the inputs to
`s3://marinfold-exp368-rno2a/MarinFold/exp368/inputs-v1/`. It persists multipart
progress and reserves each 32 MiB source read before transfer, with a cumulative
7 GB cap that survives preemption. A server-side copy attempt was rejected at
its first part; its empty upload was explicitly aborted before the streaming
copy. The streaming mirror completed successfully in about 70 seconds.
`regional_inputs.py` checks the full reader, head, reconstructed-generator and
target hashes against the original pilot. Regional recovery jobs use identical
H100 hardware and original frozen task bundles, with large weight reads now
colocated. New larger phases also write to the regional bucket. After East result writes began timing out at 16:55 UTC, five guided workers
exhausted their failure retries and the remaining seven were canceled. The
14,227,449-byte snapshot (4,446 objects) was copied to
`s3://marinfold-exp368-rno2a/MarinFold/exp368/results/pilot-guided/`.
Every local object matched its source ETag and every destination object was
read back and checked against SHA256. The original immutable tasks resume from
this regional copy. `data/storage_write_incident.json` and
`data/guided_results_migration.json` retain the evidence and complete file audit.

The regional ordinary-generation bootstrap now uses the vLLM container's existing
interpreter instead of installing a second CUDA/PyTorch environment solely for
staging. A separate two-protein smoke run verified identical input and generator
hashes and completed all 8/32/128 rollout waves. Scientific recovery still executes
each task's original immutable source bundle. `watch_scoring.py` is a finite local
dependency watcher: it submits the five declared scoring tasks only after their
matching generation jobs succeed, records every launch, and exits after submission.
The active experiment monitor continues to check the scoring jobs themselves.
When the regional controller stopped reporting after 17:14 UTC, the watcher
was restarted with an artifact-based dependency check: all four source/plan-pinned
completion markers and all 2,750 planned generation groups must exist before
it queues scoring against stale scheduler status. The shared cluster was not
modified. Frequency-only analysis can run directly on raw generated contacts;
its results exactly reproduce all fields for 96 previously scored pilot maps.

The East controller and eight schedulable H100s became available during the RNO
controller outage. A fresh East storage write/read check passed. The saved
guided snapshot (14.2 MB) and complete validation-generation outputs (404.6 MB)
were copied to `s3://marin-us-east-02a/MarinFold/exp368/recovery-east-v2/results/`,
with source ETags and every destination SHA256 readback verified. The queued
RNO replacement jobs were canceled and their terminal states checked before
East replacements were submitted. A new H100 fixed-length smoke passed all
model hashes, exact L and 2L counts, nested-prefix invariants and FP32 cache checks
(maximum probability error below 9e-7). The full fixed-length pilot is now queued
alongside the resumed original pilot and larger validation scoring. Large model
reads in East use the existing East checkpoints; no new cross-region model copy
was needed. Recovery manifests and cancellation audits are retained in `data/`.

### GB200 fallback checks

After the active East workers were preempted at 18:15 UTC, all replacement pods
waited for batch admission; the East controller also stopped reporting briefly.
The healthy `cw-us-east-08a` fleet admitted a small ARM/GB200 runtime check promptly.
It passed on PyTorch 2.10.0+cu130: maximum cached/full-forward probability and
recall errors were 5.96e-8, and lookahead committed only one contact. This used a
tiny random Qwen3 model and transferred zero trained-weight bytes. It validates
the runtime, not the production model. The exact result and timing are in
`data/runtime_smoke_gb200.json` and `data/runtime_smoke_gb200_timings.csv`.

The CUDA wheel is hash-pinned from the [official PyTorch CUDA 13 index](https://download.pytorch.org/whl/cu130/torch/).
A production-model check will use one 6,014,167,823-byte mirror to the existing
`marin-us-east-08a` bucket, with the same durable 7 GB reservation cap and full
input hash checks as the RNO mirror. GB200 results require separate hardware
provenance and timing comparisons. The existing vLLM 0.9.2 image is amd64-only;
the pinned official 0.14.1-cu130 ARM image would constitute a separately verified
ordinary-generation backend, not a bitwise reproduction of the H100 baseline.

## Conclusion

The completed 32-protein pilot supports larger ordinary rollout pools: 128
frequency-ranked rollouts improve both all-contact and long-range R-precision
relative to eight proofreader-weighted rollouts. A separate 1,375-structure
validation comparison confirms that increasing frequency-ranked pools from eight
to 128 improves R-precision by 0.0463 (95% interval 0.0403–0.0526). Fresh
proofreader-weighted validation scoring is still pending. The guided-policy and fixed-length
comparisons remain incomplete, so there is no conclusion yet about whether
proofreader-guided sampling improves the final accuracy/cost tradeoff.
