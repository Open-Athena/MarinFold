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

The [released proofreader](https://huggingface.co/buckets/open-athena/MarinFold/tree/checkpoints/exp356-exp277-frozen-contact-pdb50k-v2/step-18000)
uses the frozen causal Qwen3 contact generator as its feature extractor. A
four-layer bidirectional transformer, width 512 with eight attention heads,
reads the three token states for every emitted contact, emission-order features,
and a global rollout token. The readout has 22,181,378 trainable parameters;
the complete frozen-backbone model has 1,487,731,202 parameters. Contact sigmoid
outputs estimate correctness, and a pooled sigmoid output estimates rollout
recall. All supplied contacts can inform each contact judgment. This experiment
keeps every model weight fixed and changes inference and map aggregation only.

The reader was trained on actual generator rollouts from 50,000 experimental
PDB chain structures (46,933 PDB entries), with 34,894 distinct supplied sequences
in 9,074 connected related groups. Its validation split has 1,552 structures in
287 groups. The larger inference-validation panel here excludes all 32 pilot
groups, leaving 1,375 structures in 255 groups; it is not simply the validation
split minus 32 individual proteins. Split construction and the original model
release are documented in [exp356](../exp356_models_pdb_rollout_proofreader/README.md).

Reference labels inherit exp356's audited experimental targets from exp222:
native-only pyconfind contacts, a 3 Å side-chain threshold, minimum contact
degree 0.001, and sequence separation at least six. Targets exclude internal
sequence gaps and truncated contact lists; short terminal omissions are accepted
only when the resolved sequence maps to one contiguous canonical span under
the frozen preparation rules. Recall uses the complete reference contact set
for that retained sequence. Long-range results further require separation at
least 24. The contact definition and target preparation remain fixed throughout.

## Approach

1. Implement and verify cached, batched contact proposals and proofreader readout against the released inference implementation. Preserve actual prompts and original contact semantics.
2. Use a deterministic group-aware validation pilot for candidate-ranking diagnostics and end-to-end policy comparisons: top-percentile filtering; soft probability weighting; short lookahead; population search; one-pass suffix repair; and larger independent rollout pools with alternative aggregators.
3. Measure all/long-range R-precision, average precision over the full eligible pair universe, precision at fixed L-based budgets, recall/precision curves, candidate coverage, diversity, and per-input GPU time. Include a fast ordinary-generation baseline and matched-compute comparisons; distinguish compute envelopes from exact matching.
4. Extend promising policies to a larger validation panel. Freeze selection and stopping rules before one final reserved-test comparison. Never select policies on test results.
5. Save resumable per-input artifacts, source hashes, timing CSVs, job and recovery metadata. Publish figures and a written result report, including negative results.

## Success criteria

At least one guided policy should improve final contact-map quality over ordinary sampling and proofreader-weighted aggregation at comparable measured GPU cost, with paired group-bootstrap uncertainty. A complete null result is acceptable if the planned policy families are tested and the failure mode is measured.

## Results

Implementation and validation are in progress on 2026-10-21. Eleven CPU tests
cover cache isolation, full-forward equivalence, proposal resampling, lookahead
commit boundaries, repair-prefix reconstruction, full-universe AP, exact
diversity-discount equivalence to a set-based reference, and restriction of
reserved-test analysis to one frozen pair on a complete cohort. The first
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
of existing complete pools; the larger validation comparison is reported below.

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
now complete and reported below; the older 1,552-structure reanalysis uses a
different cohort. Data and all 2,750 contemporaneous
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

### Complete proofreader aggregation validation

All 2,750 pools have also been scored by the released full-forward reader.
The table uses all 1,375 structures; long-range metrics use the 1,371 with
at least one eligible native contact. Costs include generation and, where
needed, proofreading, excluding model load and artifact writes.
CPU map aggregation and metric computation are outside these GPU-stage timings;
they are not end-to-end serving latencies.

| Rollouts | Map ranking | R-precision | Long-range R-precision | Mean warm H100 seconds |
| ---: | --- | ---: | ---: | ---: |
| 8 | Frequency | 0.50137 | 0.47089 | 2.431 |
| 8 | Sum contact confidence | 0.51737 | 0.48799 | 2.580 |
| 128 | Frequency | 0.54769 | 0.52153 | 6.599 |
| 128 | Sum contact confidence | 0.54762 | 0.52242 | 8.921 |
| 128 | Confidence breaks vote ties | 0.54820 | 0.52216 | 8.921 |
| 128 | Votes weighted by predicted rollout F1 | 0.55195 | 0.52565 | 8.921 |

At eight rollouts, sum q improves R-precision over frequency by 0.01601
(95% paired group-bootstrap interval 0.01239–0.02033) and long-range
R-precision by 0.01710 (0.01240–0.02223). At 128, the corresponding changes
are −0.00008 (−0.00386–0.00424) and +0.00089 (−0.00371–0.00599), which do
not establish an additional benefit. Averaging confidence alone performs
poorly: R-precision is 0.47455 at eight and 0.45504 at 128 rollouts.

The exploratory rollout-F1 weighting is more promising at 128: relative to
frequency, its gains are 0.00426 (0.00173–0.00763) overall and 0.00412
(0.00129–0.00781) for long-range contacts, at 1.352 times the measured cost.
Every unique eligible contact in a rollout receives that rollout's predicted
F1 as its vote weight. Predicted precision is the mean confidence over all
unique parsed contact pairs, including invalid pairs; predicted recall is the
reader's rollout-level output. The trajectory weight is
$2\widehat P\widehat R/(\widehat P+\widehat R)$, with a numerical floor of
$10^{-8}$. Only eligible pairs vote in the final map, and native labels are not
used to set weights. This variant was added after the ordinary-pool pilot, and these
intervals are not adjusted for selecting among aggregation rules. This is
validation evidence, not a reserved-test confirmation.

Data, per-input timings, and all seven aggregation rules are in
`data/validation_pool/`. Paired comparisons use identical generated pools.

![Paired validation effects of proofreader aggregation](plots/validation_aggregation_effects.png)

![Validation aggregation accuracy and candidate coverage](plots/validation_pool.png)

The contact-level audit over the full panel explains this pattern. Sum q
replaces 18.29% of top-R slots at eight rollouts, and 8.70% at 128. True
contacts entering minus true contacts leaving exactly reproduce each
protein's R-precision difference. At 128 the gained and lost fractions nearly
cancel. See `data/validation_pool/reranking_per_protein.csv.gz`.

![Contacts gained and lost on the full validation panel](plots/validation_reranking.png)

For 128-rollout ordinary pools, the per-structure mean absolute error in
predicted rollout precision is 0.05238 and in recall is 0.05768. Mean
predicted recall is 0.38840 versus observed recall 0.40864; the contact-level
calibration curve nevertheless closely follows the diagonal. This illustrates
why aggregate calibration does not guarantee optimal ranking of candidate
contacts or accurate rollout recall. Mean sampled trajectory-pair Jaccard
overlap is 0.26679. All trajectories in this panel ended with sampled EOS.
With 128-rollout F1 weighting, selecting L contacts gives mean precision
0.50908 and recall 0.59423; selecting L/2 gives precision 0.67623 and recall
0.40055. These fixed prediction budgets depend only on sequence length.
The diagnostic compares the eight-rollout sum-q and 128-rollout F1-weighted
configurations, so their difference combines pool size and aggregation.
Data are in `data/validation_diagnostics/`.

![Full-panel precision, recall, calibration and diversity](plots/validation_diagnostics.png)

The following contact maps show the three proteins already chosen for numerical
checks: first, middle and last length ranks of the pilot (33, 193 and 937
residues). Both columns rank the same 128 H100 rollouts and select L contacts.
They were chosen without accuracy selection and illustrate individual maps;
the full-panel paired comparisons above establish the average effects.
The coordinate CSV in `data/contact_examples.csv.gz` reproduces the figure
without access to model weights or private rollout storage.

![Reference and predicted contact maps on three prespecified examples](plots/contact_examples.png)

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

At a prefix $S$, the generator samples proposal contacts $c_1,\ldots,c_K$.
The bidirectional reader scores each hypothetical prefix $S+c_j$, giving
$q_j=\widehat P(\text{contact }c_j\text{ is correct}\mid S+c_j,\text{sequence})$.
Hard filtering samples uniformly among the highest-scoring $\lceil fK\rceil$
proposal slots. Soft filtering instead samples with weights
$w_j=\max(q_j,10^{-8})^\beta$. The exploration variant mixes the gated
distribution with 10% uniform probability over all proposal slots. Repeated
proposals remain separate slots, preserving the generator's sampled multiplicity;
previously committed contacts are excluded. This does not enumerate or rank
the complete residue-pair universe, or guarantee that retained contacts have
90% correctness probability.

Lookahead replaces $q_j$ by its mean after two separately sampled futures of
eight or 32 contacts, truncated only at the context/contact limit. Later
contacts may revise the judgment of $c_j$, but only $c_j$ is committed.
The total-score variant ranks candidates by the sum of the reader's revised
probabilities for **all** contacts in $S+c_j$, allowing a proposal to change
confidence in earlier contacts. These are validation experiments with the
existing reader; its probabilities are not assumed to remain calibrated under
the shifted distribution caused by guidance.

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

The cost comparison evaluates the implementations actually run: ordinary pools
use vLLM batching, while guided search uses the verified PyTorch cache and
explicit branching. Search timings include discarded candidate and lookahead
work. They do not establish a lower bound on the achievable cost of guidance
with a different inference engine. Completed-trajectory timings exclude queue
waits and work lost to preemption; job and recovery ledgers separately record
those operational interruptions.

Primary outcomes are all-contact and long-range (separation at least 24)
R-precision and AP over the complete eligible pair universe, including all
unproposed pairs at score zero. We also report precision at L, L/2 and L/5,
candidate-pool oracle recall, and measured H100 inference time. Timing includes
discarded proposals, lookahead, repair and final rescoring; model loading is
reported separately. Cost claims distinguish measured envelopes from exact
matching. Group bootstrap intervals retain a mean-over-structures estimand.
R-precision uses R equal to the number of eligible native contacts solely for
evaluation; inference does not know R. The emitted map is a ranked contact-score
map, and deployable fixed-size outputs can instead select L, L/2 or L/5 pairs
using sequence length alone. Summed or F1-weighted votes are ranking scores,
not calibrated probabilities that a residue pair is in contact. Unfilled
prediction slots count as misses, and no native labels determine an inference
budget or a contact ranking.

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

### Execution and numerical validation

The primary guided and fixed-length experiments use H100 GPUs. FP32 cached
search passes the unchanged 1e-5 full-forward check, and final trajectories are
rescored through the released BF16 reader. The initial BF16 cache failed and
was rejected before the main experiment. Both source and plan are hash-pinned;
each completed trajectory has an independent durable commit marker and timing.

A separate ordinary-generation and full-forward-scoring replication runs on
GB200. The verified PyTorch 2.10.0+cu128 reader passes the original replay guard
on three length-selected reference proteins: exact contact-score agreement for
193 and 937 residues, and maximum difference 0.001523 at 33. GB200 cached search
fails the original FP32 tolerance and is excluded. vLLM differs between hardware
phases (0.9.2 on H100, 0.14.1 on GB200), so timings are reported separately and
do not isolate GPU speed alone.

The shared Iris controllers and batch capacity were intermittently unavailable.
The H100 campaign ultimately resumed as owned Kubernetes Jobs in the existing
Kueue batch queue, without changing shared cluster configuration or priority.
Completed outputs were preserved through source/ETag/SHA256-verified migrations;
original scientific bundles and seeds remained unchanged. Native workers retain
the inherited `runner_tag=iris` field, so `data/native_dispatches.jsonl` is the
authoritative submission-backend record.

An early preempted server repeated input staging and exceeded its intended
9.5 GB transfer ceiling. It was stopped when detected. Subsequent regional
mirrors reserve reads durably under a 7 GB cap, and large model reads are
colocated. The incident, failed numerical checks, exact storage locations,
recoveries and source audits are retained in [execution notes](execution_notes.md)
and the linked `data/` manifests. The scientific cost tables measure successful
warm inference work; they exclude queue time, staging and work lost to preemption.

### Separate GB200 ordinary-pool replication

The complete 32-protein GB200 replication has eight-rollout sum-q R-precision
0.5783 and long-range R-precision 0.5591. At 128 rollouts, frequency achieves
0.6025 and 0.5860, while sum-q achieves 0.5995 and 0.5846. Mean warm costs are
1.511 seconds for eight-rollout sum q, 3.471 seconds for 128-rollout frequency
and 5.204 seconds for 128-rollout sum q. Data are in `data/pilot_direct_gb200/`.
These are a separate hardware/runtime cohort: vLLM 0.14.1 on GB200 differs from
the H100 vLLM 0.9.2 baseline, so the timing difference does not isolate GPU speed.
The 512- and 2,048-rollout pools are also complete for all 32 structures:

| Independent rollouts | Frequency R-precision | Long-range R-precision | Native contacts present | Mean warm GB200 seconds |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.60248 | 0.58597 | 95.20% | 3.471 |
| 512 | 0.60319 | 0.58579 | 98.60% | 18.404 |
| 2,048 | 0.60257 | 0.58449 | 99.33% | 82.198 |

Relative to 128 rollouts, 2,048 changes all-contact R-precision by +0.00009
(95% paired interval −0.00426–0.00503) and long-range R-precision by −0.00148
(−0.00687–0.00411), at 23.68 times the measured cost. This does not establish
an accuracy benefit. Candidate coverage approaches saturation while frequency
ranking plateaus; recovering additional true contacts requires better ranking
of the candidate pool. The frequency-only comparison is followed by an adaptive proofreading check below.
Data and paired comparisons are in `data/pilot_large_gb200/`; the combined
frequency curve is in `data/pilot_pool_scaling_gb200/`.

At 20:15 UTC, after observing the frequency plateau and the full-panel
128-rollout F1-weighting gains, an additional exploratory scoring phase was
frozen in `data/large_pool_proofreading_extension.json`. It applies the same
released full-forward reader to all 64 existing GB200 pools at 512 and 2,048
rollouts. Its primary contrast is 2,048 versus 128 with F1-weighted votes;
all seven existing aggregation rules will also be reported. It generates no
new rollouts, changes no weights, and uses no new reserved-test metrics.
This is an adaptive validation extension, not a prespecified confirmation.
All 64 pools have now been scored. F1-weighted R-precision is 0.60378,
0.60503 and 0.60380 at 128, 512 and 2,048 rollouts. The primary 2,048-minus-128
difference is +0.00002 (95% paired interval −0.00463–0.00500); long-range
changes by −0.00269 (−0.00911–0.00317). Mean warm generation-plus-proofreading
time rises from 5.204 to 108.403 GB200 seconds, a factor of 20.83. Thus larger
pools do not establish an R-precision gain even with this weighting. Secondary
full-universe average precision improves by 0.00897 (0.00480–0.01397), from
0.63904 to 0.64801. This is consistent with improved recovery below the top-R
cutoff, rather than a better top-R map. All seven aggregators and both same-pool
and across-pool comparisons are in `data/pilot_large_scored_gb200/`; the combined
curve is in `data/pilot_pool_scaling_scored_gb200/`.

![Proofreader aggregation through 2,048 ordinary rollouts](plots/pilot_large_scored_gb200.png)

![Frequency ranking and candidate coverage through 2,048 rollouts](plots/pilot_large_gb200.png)

The separate larger GB200 validation panel is complete for frequency ranking:
1,375 structures from 255 groups, with 1,371 contributing long-range metrics.
Eight versus 128 rollouts gives R-precision 0.50086 versus 0.54779, a paired
gain of 0.04693 (95% interval 0.04052–0.05353). Long-range R-precision rises
from 0.47099 to 0.52194, a gain of 0.05096 (0.04338–0.05915). Mean warm cost
is 1.196 versus 3.054 GB200 seconds. These reproduce the H100 frequency result
under a separate runtime; the timings are not pooled across hardware. Data are in
`data/validation_frequency_gb200/`.

![GB200 larger validation frequency comparison](plots/validation_frequency_gb200.png)

Full-forward proofreading is also complete for every GB200 validation pool.
At 128 rollouts, F1-weighted votes give all-contact R-precision 0.55239 and
long-range R-precision 0.52647. Relative to frequency, paired gains are
0.00459 (95% interval 0.00207–0.00790) and 0.00453 (0.00164–0.00813).
Mean warm generation-plus-proofreading time is 4.622 GB200 seconds, versus
3.054 for frequency alone. Sum q gives 0.54750 and 0.52249; its differences
from frequency again have intervals spanning zero. The same structures and
groups are used as in the H100 phase, so this is a stochastic/runtime
replication, not an independent biological validation cohort. Results and
all 5,500 stage timings are in `data/validation_pool_gb200/`.

![GB200 replication of paired aggregation effects](plots/validation_aggregation_effects_gb200.png)

### Bounded refinement of rollout-quality weights

After the full-panel F1-weighting result, a small exploratory follow-up fixed
exponents 1, 2, 4 and 8 before computing their pilot outcomes
(`data/weight_power_refinement.json`, `pilot_weight_power.json`). It reuses
identical 128-rollout pools and released reader outputs on all 32 pilot proteins,
with no new GPU inference. The three new rules give each rollout's eligible
unique contacts weight equal to its predicted F1 raised to the stated power.
The original seven-method analysis remains unchanged unless a plan explicitly
requests these additional methods.

| F1 exponent | H100 R-precision | H100 long-range | GB200 R-precision | GB200 long-range |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.60218 | 0.59013 | 0.60378 | 0.58808 |
| 2 | 0.60241 | 0.58890 | 0.60515 | 0.58973 |
| 4 | 0.60059 | 0.59098 | 0.60149 | 0.59175 |
| 8 | 0.60357 | 0.58942 | 0.60353 | 0.59021 |

Every paired R-precision interval for the new exponents versus exponent 1
includes zero, both overall and for long-range contacts. For exponent 2,
the all-contact differences are +0.00024 (95% interval −0.00204–0.00259)
on H100 and +0.00137 (−0.00181–0.00584) on GB200. This pilot does not establish
a stronger weighting rule. All variants reuse the same measured generation and
reader costs: 9.581 H100 or 5.204 GB200 seconds per map, excluding CPU aggregation.
The original frequency and exponent-1 values reproduce exactly for every input.
These are adaptive validation comparisons; no final policy is chosen before
the original guided pilot completes. Data are in `data/pilot_weight_power/`
and `data/pilot_weight_power_gb200/`.

![Stronger rollout-quality weights on the complete pilot](plots/pilot_weight_power.png)

### Larger ordinary pools on H100

All 64 independent 512/2,048-rollout pools are complete on the 32-protein pilot.
Frequency R-precision is 0.60276 and 0.60299 respectively, versus 0.60365 at
128 rollouts. The 2,048-minus-128 paired difference is −0.00066
(95% interval −0.00489–0.00352); long-range difference is −0.00241
(−0.00656–0.00191). Mean warm H100 time rises from 7.165 to 109.962 seconds,
a factor of 15.35. Full-universe AP improves by 0.01334 (0.00847–0.01985).
Candidate coverage reaches 99.21%, but this does not translate into a better
top-R map. These H100 results corroborate the separately timed GB200 frequency
plateau. Data are in `data/pilot_large/` and `data/pilot_pool_scaling/`.

![Larger ordinary pools on H100](plots/pilot_large.png)

### Fixed-length contact selection and overconfidence

All 256 continuations and their 512 nested L/2L prefix maps are complete on the
same 32 pilot proteins. EOS is disabled, and each trajectory contains exactly
L or 2L unique contacts. This adaptive diagnostic was added after observing five
short pilot proteins, before the original 15-policy pilot completed; it is not
a prespecified confirmation. The L prefix is scored before generation continues.

| Budget | Policy | Mean rollout precision | Mean rollout recall | Predicted precision | Four-rollout sum-q R-precision | Mean warm H100 seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L | Ordinary control | 0.43345 | 0.50986 | 0.44090 | 0.57152 | 32.183 |
| L | Top 10% of 16 proposals | 0.49286 | 0.57157 | 0.49471 | 0.57084 | 104.599 |
| 2L | Ordinary control | 0.27788 | 0.63900 | 0.36290 | 0.56116 | 64.270 |
| 2L | Top 10% of 16 proposals | 0.28286 | 0.64964 | 0.52488 | 0.53276 | 267.704 |

At L, filtering improves average individual-rollout precision and recall, but
with four rollouts the R-precision difference is −0.00068 (95% paired interval
−0.01806–0.01717), at 3.25 times the cost. At 2L the reader becomes particularly
overconfident on guided contacts. Four-rollout R-precision loses 0.02839
(−0.04961 to −0.00827) and long-range R-precision loses 0.03341
(−0.05675 to −0.01023), at 4.17 times the cost. With a single 2L rollout,
all-contact R-precision loses 0.10258 (−0.14695 to −0.05601).

These results are consistent with distribution shift and score exploitation
when guidance is forced far beyond normal stopping; they do not identify a
unique causal mechanism. Better per-rollout precision need not produce a better
ensemble map. Neither fixed-length filtering curve improves on the measured
ordinary-pool accuracy/cost frontier. Results, paired comparisons and timings
are in `data/length_probe/`. L and 2L outputs share trajectories and must not be
treated as independent samples; intervals resample the same 32 homology groups.

![Fixed-length rollout quality and reader predictions](plots/length_probe_quality.png)

![Fixed-length map accuracy and measured GPU cost](plots/length_probe_frontier.png)

A retrospective decomposition of these same complete continuations shows that
both contact selection and retrospective scoring degrade beyond L. In guided
rollouts, the newly added second L contacts have actual precision **0.07286**,
while their mean predicted correctness is **0.50726**. For control, the same
quantities are 0.12232 and 0.24450. This is not merely a shift in the total
number of predictions.

The first L contact identities and labels are verified identical across prefix
records. After the guided continuation, their mean predicted correctness rises
from 0.49471 to 0.54250, while their actual precision remains 0.49286. Their
Brier score worsens by 0.02068 (95% paired group interval 0.01511–0.02654), and
mean within-rollout AUROC declines by 0.04665 (−0.06241 to −0.03346). Thus later
forced contacts can also harm judgments of earlier contacts. This exploratory
diagnostic uses all 32 proteins and all four trajectories, with no new model
inference. Data are in `data/prefix_revision/`; AUROC excludes trajectories
without both label classes.

![Earlier contact judgments and new contacts after forced continuation](plots/prefix_revision.png)

## Current scientific conclusion

The completed 32-protein pilot supports larger ordinary rollout pools: 128
frequency-ranked rollouts improve both all-contact and long-range R-precision
relative to eight proofreader-weighted rollouts. A separate 1,375-structure
validation comparison confirms that increasing frequency-ranked pools from eight
to 128 improves R-precision by 0.0463 (95% interval 0.0403–0.0526). Full
proofreader validation confirms that sum q helps small pools; at 128 rollouts,
weighting votes by predicted rollout F1 adds a smaller exploratory gain of
0.00426 (0.00173–0.00763). On GB200, increasing ordinary pools to 2,048
does not establish an accuracy gain over 128 despite 23.68 times the cost.
The fixed-length comparison is complete and does not improve the measured
accuracy/cost frontier. The original 15-policy guided comparison remains
incomplete, so policy selection still waits for that full cohort.
