---
marinfold_experiment:
  issue: 208
  title: 'exp: RL post-training on contacts-v1 with a dense per-contact reward'
  kind: models
  branch: exp/208-rl-dense-contact-reward
---

# exp: RL post-training on contacts-v1 with a dense per-contact reward

**Issue:** [#208](https://github.com/Open-Athena/MarinFold/issues/208) · **Kind:** `models` · **Branch:** `exp/208-rl-dense-contact-reward`

## Question

Does a dense per-contact reward improve **single-shot `<contacts-v1>` prediction**,
measured as the number we actually report — consensus R-precision over 100
resampled rollouts on the 554-protein eval set?

[#200](https://github.com/Open-Athena/MarinFold/issues/200) established that the
reward works and that its headline metric did not move, because the multi-draft
format made best-of-N a product of candidate quality and candidate spread and the
run traded one for the other. Dropping multi-draft deletes the spread-across-
sections axis. What is left is answerable.

## Hypothesis

**The per-contact term will raise per-rollout precision (it already did, +0.0085
at +4.6σ in #200), and that gain will only reach the reported metric if the
document-level term is scored on the consensus rather than on the individual
rollout.**

The mechanism to beat is *vote collapse*. The reported number is a consensus over
100 rollouts, and #82's own README records that "over-sharpening collapses the
vote" — sharpening the sampling distribution *hurts*, and T=1.0/p=0.95 is
near-optimal. A precision-only reward is a sharpening operator by construction: it
pushes every rollout toward the model's single best guess. So it can raise
per-rollout precision and *lower* consensus R-precision. That is #200's failure
arriving through a different door, and it is the specific reason the document term
here is a **leave-one-out marginal contribution to the group's consensus** rather
than the rollout's own F1.

Concretely we expect, ordered by confidence:

1. Per-rollout precision rises in every arm that has a stepwise term (replicating #200).
2. The step-only arm raises precision and moves consensus R-precision by ≤ 0 —
   union coverage shrinks and votes concentrate.
3. The consensus-marginal arm moves consensus R-precision **up**, and does so with
   union coverage flat or up rather than down.

Prediction 2 is a real prediction and it is the one that would most change how we
post-train: if step-only *also* moves the reported metric, the consensus term is
unnecessary machinery and the next experiment is much simpler.

## Background

### What #200 measured, and what it costs to repeat

Paired over 554 eval proteins, #200's single completed arm (lr 1e-6, 150 steps):
per-contact precision **+0.0085 (+4.6σ)**, first-candidate F1 **+0.0128 (+5.1σ)**,
best-of-N F1 **+0.0008 (+0.4σ)** — because sections fell 1.35 (−7.5σ) and
inter-section Jaccard rose 0.0087 (+7.7σ). `first_f1` is the metric that moved
most, and it is generated with no prior candidates in context, so it is the closest
analogue in that experiment to single-shot output. Plain generation is also ~4×
cheaper: 37 s per sampling call against 162 s, 860 tokens against 5,344.

`train/mean_advantages` sat at 0.0028 and `train/ratio_mean` at 1.0024, so the
p̄-centring and the whole logprob path are validated on real hardware. KL k3 was
0.00051 at lr 1e-6 — **the policy barely moved**, which bounds how much of #200's
flat result is attributable to the reward design at all.

### Why this is MarinFold's first RL on the base task

[#98](https://github.com/Open-Athena/MarinFold/issues/98) /
[#100](https://github.com/Open-Athena/MarinFold/issues/100) /
[#120](https://github.com/Open-Athena/MarinFold/issues/120) /
[#175](https://github.com/Open-Athena/MarinFold/issues/175) are all SFT-flavoured.
#120's negative result (regenerated-document fine-tuning does not beat re-epoching)
was imitation of filtered samples, not a gradient on individual contact decisions.

### The measurement

`score_rollout_worker.py` (#82): *n* resampled contacts-v1 rollouts per protein,
each dedup'd to a pair set, voted into a dense `[L, L]` occurrence matrix, ranked
top-R with a pairwise tie-break. `R` is the number of true contacts in the range,
so "R-precision (all)" is precision at top-|GT| over separation ≥ 6, and
"R-precision (long)" the same at separation ≥ 24. exp89's `compute_metrics.py` is
the canonical implementation; #169 reproduces #82's published 0.535 for the #117
final checkpoint. **Four evaluations of one unchanged #117 checkpoint span 0.0023**
(#180) — that is the reproducibility floor of the whole recipe and the number any
claimed effect has to clear.

### The baseline is already in git

`contacts-v1-exp199-1.5B` became the default model on `main` in
[#207](https://github.com/Open-Athena/MarinFold/pull/207) (2026-08-10), so the
blocker named in the issue is cleared. Its per-protein rows are committed at
[`../exp180_evals_contacts_v1_progress_over_time/data/exp199_cw_p06_aug_step145199_rows.csv.gz`](../exp180_evals_contacts_v1_progress_over_time/data/exp199_cw_p06_aug_step145199_rows.csv.gz):

| | R-precision | n |
|---|---|---|
| all (sep ≥ 6) | **0.587348** | 554 |
| long (sep ≥ 24) | **0.542181** | 553 |

Weights: `hf://buckets/open-athena/MarinFold/checkpoints/prot-exp199-cw-cv1-s02-m1-p06-aug/hf/step-145199`.

So the primary comparison is **paired per protein against rows we already have**,
not against a number that has to be regenerated. Phase 1 re-measures it anyway, as
a gate on our own eval invocation rather than as the baseline of record.

## Approach

Mostly configuration on top of #200's code, which lives on
[PR #203](https://github.com/Open-Athena/MarinFold/pull/203) (branch
`claude/marinfold-issue-200-plan-5ba0c8`) and is **not yet merged**. exp208's
directory is a copy of that iris workspace with the multi-draft machinery removed
— see "Dependency on #203" below for why copying rather than importing.

Already true and reusable without change:

- `contact_rewards.dense_rewards(..., mode="plain")` scores section 1 only.
- `contacts_env.ContactsV1RLEnv(mode="plain")` sets the document sentinel to id 2
  (`<contacts-v1>`) by **token id**, never by string.
- The curriculum collapses from two lessons to one, so `minimum_sample_probability`
  and the 50:50 pinning become irrelevant.
- The training pool and the eval targets are **already in `us-central1`**, where
  the v5p capacity is (verified 2026-08-10):
  `gs://marin-us-central1/protein-structure/MarinFold/exp200/train/{targets.parquet,prompts/}`
  and `gs://marin-us-central1/protein-structure/MarinFold/exp169/eval_targets.parquet`.

New work is the consensus-marginal document term, the group-level diagnostics that
plain mode needs, and the eval swap to `score_rollout_worker.py`.

### The reward

Two terms on `<contacts-v1>` prompts only. Per-token advantage:

```
A_t = lam_step · r_step(t)  +  lam_doc · A_consensus
```

**1. Stepwise, per emitted contact — unchanged from #200.** For each
`<contact> <pI> <pJ>` triple, with `x = 1` iff the pair is in ground truth
(sequence-index space, separation ≥ 6, not already emitted in this rollout):

```
r_step = +(1 - p̄)          if x == 1
       = −p̄ · δ^e          if x == 0     (e = earlier errors in the section)
```

spread over the triple's three tokens. `p̄` is the policy's own EMA per-contact
precision, so `E[r_step] ≈ 0` at current performance and the gradient says *beat
yourself*. This matters more than any other constant here: precision is ~0.23, so a
**fixed** penalty makes "emit nothing" optimal and the run collapses to empty
sections. `δ = 0.5` reflects that a wrong contact may be a consequence of an
earlier one. Penalising wrong contacts is also what preserves the incentive to end
a rollout early when the model has nothing good left — wanted, not merely tolerated.

**2. Document-level: leave-one-out marginal contribution to the group consensus — new.**
For a group of `G` rollouts on one protein, let `S_i` be rollout *i*'s dedup'd pair
set, `v(p) = |{i : p ∈ S_i}|` the vote count, and

```
C(T)  =  |topR( v restricted to rollouts in T ) ∩ GT| / R,     R = |GT|
A_i   =  C(all)  −  C(all \ {i})
```

then `A_consensus,i = A_i − mean_j A_j` (RLOO's existing leave-one-out centring,
applied to the marginals).

Three things follow from this definition and are worth stating because they are the
argument for it:

- **It is the deployed metric.** Not a proxy for it, not correlated with it — the
  same function, evaluated at `G` instead of 100.
- **It defends diversity by construction.** A rollout is paid for contributing
  something its siblings did not. A rollout that duplicates the consensus exactly
  has marginal ≈ 0. No hyperparameter is doing that work.
- **It penalises silence.** An empty rollout changes no votes, so its raw marginal
  is exactly 0; after centring against siblings with positive marginals it receives
  a *negative* advantage. The stepwise term's collapse-to-silence failure mode is
  actively opposed rather than merely bounded.

Implementation notes that will otherwise bite:

- **`Rollout` has no spare field.** marin's `Rollout` is a fixed dataclass and
  patching marin is off the table (AGENTS.md: never monkey-patch), so the marginal
  is carried in **`episode_reward`** and `dense_loss.ContactsDenseLoss` applies
  `compute_rloo_advantages` on top as the centring step. This means W&B's `reward`
  panels read ≈ 0 by design; the raw consensus `C(all)` and the per-rollout F1 are
  logged separately as environment metrics so nothing is unobservable.
- **The tie-break needs no approximation** (corrected 2026-08-10, from reading
  the eval path rather than its prose). "Rollout + resample + pairwise tie-break"
  appears in #82/#180 write-ups, but `fetch_cw_scores.py` writes bare integer vote
  counts into the score matrix and `build_rollout_rows.py` ranks them with
  `np.argsort(-s, kind="mergesort")` — a **stable** sort. So ties are settled by
  ascending candidate index and there is no pairwise probability term anywhere on
  this path. The in-loop consensus therefore reproduces the deployed ranking
  exactly, at no cost. What Phase 0 measures instead is `tie_fraction`: how much
  of top-R is decided by that arbitrary index order in the first place.
- **`G` is the estimator's sample size.** At `G = 8` a single rollout moves each of
  its ~100 pairs' vote count by 1 out of 8, which is a large and noisy perturbation;
  at `G = 100` the same rollout is nearly invisible. The *sign structure* — add true
  pairs your siblings missed, do not add false pairs that could enter the top-R — is
  right at both scales, but the magnitudes are not, and that is exactly what Phase 0
  exists to check.

### Phase 0 — is the consensus marginal measurable, and does it say anything precision doesn't? (offline, ~1 v5p-8-hour)

This is the phase that can kill the design cheaply, so it runs first.

`score_rollout_worker.py` writes only the aggregated vote matrix. Add an **opt-in
`--dump-rollouts` flag** (default off, so the eval bytes stay identical and every
published number stays comparable) that additionally writes per-rollout pair sets
for the first *N* proteins. Run it on the exp199 baseline for ~100 proteins at
n=100, pull the dump locally, and answer:

1. **corr(rollout precision, LOO marginal at n=100).** If this is ≈ 1, the
   consensus term carries no information beyond the stepwise term and arms B/D
   below are predicted to behave like arm S. If it is ≈ 0.5, the term is doing
   independent work. *This is the load-bearing measurement of the whole design.*
2. **corr(marginal at G ∈ {8, 16, 32}, marginal at n=100).** Estimability at the
   group sizes we can afford. Picks `G`.
3. **Distribution of the marginal** at each `G` — how many rollouts get exactly 0
   (a degenerate signal would show up here).
4. **How much of top-R is arbitrary** — the share of the selection sitting on a
   vote-count tie, settled by index order. A heavily-tied ranking means the metric
   is reading a coarse signal, which bounds how much any reward can move it.
5. **Sensitivity decomposition of consensus R-precision** into union coverage
   (`|∪S_i|` vs `R`) and vote sharpness. This is what turns "vote collapse" from a
   worry into a monitored quantity with a baseline value.

**Gate 1 is go/no-go; gate 2 is not.** If (1) comes back ≈ 1 the consensus term is
redundant with the stepwise term: drop arms B and D, keep arm S, and say so — the
issue's question is then answered with one arm instead of four.

(2) selects `G` rather than killing the design, and the distinction matters. The
n=100 marginal is not the training target; **consensus-at-G is a legitimate
objective in its own right**, and a noisy unbiased estimate of it is still a valid
policy gradient. A low correlation says the *transfer* from G to 100 is the risky
step, which is a thing to report and to size `G` against — not a reason not to run.

A dry run of the analysis on a synthetic group tuned to the measured rollout
precision (0.234, against #200's real 0.2294) already suggests this is where the
tension will be: corr(precision, marginal) came out **0.17** (gate 1 comfortable),
while single-draw corr against the n=100 marginal rose **−0.01 → 0.14 → 0.29**
across `G` = 8 / 16 / 32. Synthetic decoy structure is not real decoy structure, so
these numbers predict nothing — but they do say the experiment is pointed at the
right question, and that `G` = 8 is likely too small.

### Phase 1 — baseline + publish parity (~1 v5p-8-hour)

- **Publish the starting model as an HF repo.** The rollout worker resolves its
  tokenizer through levanter's `load_tokenizer`, which accepts a local directory, a
  `mirror://` ref, or an HF Hub repo id — and raises `HFValidationError` on a
  `gs://` URL, deep inside a worker after the gang has scheduled. vLLM streaming
  weights from GCS is what makes this trap easy to walk into: the weights path is
  fine and only the tokenizer path is not. Publish the bucket copy (weights **and**
  co-located contacts-v1 tokenizer) unchanged. Repo creation needs the `write2`
  token, not the default `oa-marinfold` one.
- **Preflight it**: top-level `rope_theta` present (levanter writes rope under
  `rope_parameters` and any reader older than transformers 5 then silently falls
  back to a 50× wrong base frequency — this already cost #163 a round of evals) and
  `vocab_size` matching the embedding matrix rather than the tokenizer's nominal
  size.
- **Re-derive the contact vocab from the exp199 tokenizer.**
  `contact_rewards.resolve_contact_vocab` refuses to run on drift from the baked-in
  ids (`<contact>`=5, `<begin_statements>`=9, `<end>`=10, `<p0>`=143). exp200 ran
  against #163's *renamed* tokenizer; exp199's is the standard one and this has
  never been checked against it. Cheap, and a silent drift rewards the wrong token
  positions — a plausible-looking run with no signal.
- **Re-score the published repo** with `score_rollout_worker.py` at n=100 and check
  it lands within 0.0023 of 0.587348 / 0.542181. This gates the publish path *and*
  our eval invocation in one run. Not the baseline of record — the committed exp199
  rows are.

### Phase 2 — nano gate (10 steps, ~15 v5p-8-minutes)

#200's five bring-up failures were each caught by a 10-step nano and none of them
were reachable from a generation-only parity gate. Repeat it, and additionally
assert on a real rollout batch:

- `token_rewards` is aligned (`len == len(response_tokens)`) and **not constant** —
  `dense_loss` already raises on the "uniformly equals `episode_reward`" signature,
  which is what marin's default `create_rollout_from_choice` produces if the
  environment never replaced it.
- `episode_reward` is the consensus marginal, mean ≈ 0 across a group and **not
  all-zero** (the degenerate-marginal failure).
- `train/ratio_mean` ≈ 1 (sampler/trainer logprob agreement) and
  `train/mean_advantages` ≈ 0.
- The new group-level metrics are populated (below).

### Phase 3 — learning-rate probe (3 arms × 30 steps, ~1 v5p-8-hour)

#200's completed arm reached KL k3 = 0.00051 over 150 steps. That is a policy that
did not move, and it means #200's flat headline result cannot distinguish "the
reward design is wrong" from "nothing happened". Do not repeat it.

marin's DAPO normalisation divides by batch size, so a learning rate swept at one
batch size does not transfer to another — and this experiment changes the batch
size (below). Probe 3e-6 / 1e-5 / 3e-5 for 30 steps, read the KL trajectory, and
pick the LR that extrapolates to **KL k3 ≈ 0.005–0.02** at the full step count.
That is a target on *how far the policy moves*, which is the quantity we actually
care about and the one that transfers across configuration changes.

### Phase 4 — the λ arms (4 arms, each 1 trainer + 4 rollout v5p-8)

The issue names the λ ratio as the primary axis. Because the two terms have
different natural scales — a per-contact reward of ~0.23/3 per token on ~300 contact
tokens, against a consensus marginal of order 0.01 broadcast over ~430 response
tokens — **raw λ values are not interpretable**. Define the axis by the *measured*
ratio at step 0:

```
ρ  =  Σ_t |lam_doc · A_consensus|  /  Σ_t |lam_step · r_step(t)|
```

calibrate `lam_doc` on the nano batch to hit the target ρ, and log both magnitudes
every step so ρ is observable rather than assumed.

| arm | ρ | doc term | what it tests |
|---|---|---|---|
| **S** | 0 (`lam_doc = 0`) | — | the issue's literal question with the minimum machinery; the vote-collapse prediction |
| **B** | ≈ 1 | consensus marginal | the main arm |
| **F** | ≈ 1 | **own F1**, RLOO-baselined | #200's document term in plain mode — the ablation that justifies the consensus form |

**Arm D (ρ ≈ 3, consensus) was dropped after Phase 0.** It existed to test whether
the λ axis is monotone, which is only worth an hour if the document term is a
sharp signal. Phase 0 measured the n=100 marginal it targets to be 86% exactly
zero with sd 0.0087, and the G-estimator correlating ~0.2 with it at every group
size — so more weight on that estimator is the least informative arm available.
Arm F gains importance in its place: its document term is a **per-rollout**
quantity with no estimability problem, so S / B / F separate "dense per-contact
reward", "a well-estimated document term", and "a weakly-estimated one that is
literally the deployed metric".

Arm F is the control that makes arm B's result mean something. If B > F, the
consensus form is load-bearing; if B ≈ F, the document term's *form* does not
matter and only its *weight* does, which is a much cheaper thing to tune next time.

Config, with the deltas from #200 called out:

| | #200 | exp208 | why |
|---|---|---|---|
| prompts × generations | 16 × 8 | **8 × 16** | `G = 16` (Phase-0-confirmed) makes the in-loop consensus closer to the deployed n=100 |
| `train_batch_size` | 32 | **64** | 4 groups per batch at G=16, so gradient variance matches #200's; LR is re-picked anyway |
| steps | 150 | **400** | plain rollouts are ~4× cheaper; #200's step budget was set by multi-draft cost |
| checkpoints | every 20 min | **every 25 steps** | #200's stalled arms' rolling checkpoints lagged training by ~30 steps, leaving nothing clean to evaluate |
| launch | one driver, 3 arms | **one driver per arm** | #200 lost two of three arms to a single driver preemption stranding its siblings |
| lessons | plain + multi, pinned 50:50 | **plain only** | the point of this experiment |
| `max_sections` | 8 | 1 (implied by `mode="plain"`) | |

Everything else stays: KL k3 at β 0.01 anchored to the warm start (#163's v1/v2
refiners lost 41–44% of base-task R-precision to a single unanchored fine-tune),
PPO clip 0.2, `do_overlong_filtering=False`, `filter_out_groups_with_no_variance=False`,
`max_samples=1`, `max_rollout_step_delay = sync_interval_steps`, `sync_interval_steps=8`,
T=1.0 / top-p=0.95 / **top-k unset** (#142 traced under-generation to a finite top-k;
`DecodingConfig` rejects a non-positive value, so it is `None` here and translated
to vLLM's −1 in the environment).

### Phase 5 — evaluation

- **Primary — consensus R-precision at n=100**, all and long bands, via
  `score_rollout_worker.py` (`--n-rollouts 100 --top-k -1 --top-p 0.95
  --temperature 1.0 --no-per-request-seed`) on the 554-protein eval set, paired per
  protein against the committed exp199 rows, using #169's `summarize_results.py`
  paired-difference machinery. ~554 × 100 rollouts ≈ 15.7 M tokens, roughly 15 min
  on a v5p-8 without logprobs; sharded ×4 for wall clock.
- **Secondary — single-rollout precision, recall and F1, reported separately.** The
  main risk is a precision/recall trade presented as an improvement, and only
  separate reporting catches it. Recall is not optional here: the stepwise reward
  has an explicit incentive to stop early.
- **Vote-collapse diagnostics, from the vote parquets we already write** — no extra
  generation needed, since `(i, j, votes)` is exactly the right data: union size
  `|{p : v(p) ≥ 1}|` against `R`, mean vote of the top-R pairs, and the vote
  distribution's entropy. If per-rollout precision rises while union size falls, the
  consensus is being sharpened away and we can say so quantitatively.
- **Monitored during training** — policy entropy (#200 logged 2.618), **inter-rollout
  Jaccard across the group**, in-loop consensus `C(all)`, `n_pred`, and the ρ ratio.
- **Output length as an outcome, not a guardrail.** Some reduction is *wanted*.
  Acceptable range stated up front: mean contacts per rollout within **[0.70, 1.15] ×
  baseline**. Below 0.70 is the kill criterion, because collapse-to-silence and the
  wanted stop-incentive are the same mechanism at different strengths, separated only
  by the λ ratio. The metric supplies a principled floor independently: the union of
  100 rollouts must contain at least `R` pairs or top-R cannot be filled.
- **Cheap check — pairwise teacher-forced R-precision** (#163's `rprec_worker_tpu.py`).
  Secondary only: it is computed from the probability matrix, not from rollouts, so
  RL on rollouts may not move it at all. Useful as a "did we damage the base
  distribution" read, not as evidence of improvement.

Export path per arm: `levanter.main.export_lm_to_hf` on a CPU pod (`use_cpu=True`,
so it does not occupy a v5p) → cast f32 → bf16 and translate the rope config → GCS.
marin.rl's train path writes only levanter-native checkpoints; there is no HF export
anywhere in it. **Verify the round-trip is lossless once**: export the step-0
checkpoint and assert bitwise equality against the published bf16 source
(bf16 → f32 → bf16 is exact), so no eval difference can be blamed on the exporter.

### New code

**Built and tested (Phase 0 — needs nothing from #203):**

| file | what | state |
|---|---|---|
| `consensus.py` | vote matrix, top-R consensus, LOO marginals, group diagnostics | done |
| `tests/test_consensus.py` | 15 tests, incl. a per-seed equality check against `build_rollout_rows.metric_rows` (exp89's `compute_metrics`, verbatim) | passing |
| `../exp82_.../score_rollout_worker.py` | opt-in `--dump-rollouts N`; **default 0 = byte-identical** to every published run | done |
| `phase0_marginal_analysis.py` | the Phase 0 measurements + both gates; verifies the dump reconstructs the vote matrix exactly before using it | dry-run on a synthetic fixture |
| `dispatch_phase0.py` | 4 × v5p-8 shards of the baseline eval with dumps; refuses to submit against a worker lacking the flag | dry-run OK |
| `stage_model_gcs.sh` | exp199 HF **bucket** → GCS us-central1, bf16, cloud-side | written, unrun |

The staging script is exp208's own rather than exp169's because exp169's calls
`snapshot_download`, which resolves HF *repos*; exp199 lives in a **bucket**, a
namespace `snapshot_download` cannot see at all and fails on with a repo-not-found
that points nowhere near the real problem. The bucket API is `list_bucket_tree` /
`download_bucket_files`, and the bucket is anonymously readable, so `token=False`
is correct and avoids picking up a workstation token scoped to the wrong org.

**Still to write (needs #200's code from #203):**

| file | what |
|---|---|
| `contacts_env.py` | exp200's, plus the consensus marginal into `episode_reward`, the group-level metrics, and `doc_term` ∈ {`consensus`, `own_f1`, `none`} |
| `dense_loss.py` | exp200's, plus ρ logging |
| `rl_config.py` | exp200's, minus the multi lesson; step-interval checkpointing; exp199 defaults |
| `dispatch_rl.py`, `_submit.py`, `_trace.py`, `read_trace.py`, `reap.py` | exp200's, one driver per arm |
| `export_checkpoint.py`, `publish_checkpoint_hf.py` | exp200's |
| `summarize_results.py` | thin wrapper over #169's paired machinery |

`contact_rewards.py` copies across unchanged apart from dropping the multi-draft
constant, so #200's tests port directly.

### Dependency on #203

#200's code is on an **unmerged PR**. exp208 copies it into its own directory rather
than importing, for two reasons that are not stylistic: iris bundles **one**
directory as the job workspace, so a sibling-experiment import works locally and
fails on the pod (#200 hit this and vendored `parse_doc` for the same reason); and
AGENTS.md forbids libraries importing from experiments, with promotion to a kind
library only once a second use case exists — which is now true, so the alternative
is to promote the RL scaffolding to `models/` as part of #203. **That is a call for
the reviewer of #203, not for this experiment**, and copying does not foreclose it.

Either way exp208 should not land before #203, or the diff will read as if this
experiment invented all of it.

## Success criteria

Pre-registered, on the paired per-protein difference against the committed exp199
rows (n = 554 / 553), for the best λ arm:

- **Primary — consensus R-precision (all) Δ ≥ +0.010 at ≥ 3σ.** For scale: the
  frontier moved 0.534 (#117) → 0.5618 (#166) → 0.5873 (#199) across three full
  training runs, so +0.010 from under an hour of post-training is a real result.
- **Signal floor — Δ ≥ +0.005 at ≥ 3σ**, reported as "a real but sub-threshold
  effect" rather than as success. Pre-registering this is what keeps a small true
  gain from being written up as a failure, or a noise excursion as a win.
- **Hard floor — any claimed Δ must exceed 0.0023**, the measured span of four
  evaluations of one unchanged checkpoint (#180).
- **Long band reported alongside**, not as a separate gate.

Kill criteria, checked from training metrics before spending eval:

- mean contacts per rollout below **0.70 ×** baseline (collapse toward silence);
- union coverage `|∪S_i| / R` at n=100 below **0.90 ×** baseline (vote collapse);
- policy entropy down more than **20%** from the warm start;
- pairwise teacher-forced R-precision down more than **0.01** (the #163
  catastrophic-forgetting failure).

Secondary, reported regardless of the primary outcome:

- single-rollout precision / recall / F1, **separately** — a precision-only gain at
  the cost of recall is a null result, not a positive one;
- the arm ordering S vs B vs D vs F, which is what tells the next experiment whether
  the consensus form or merely the doc weight was load-bearing;
- Phase 0's corr(precision, marginal), which is publishable on its own: it says
  whether "contribute what your siblings missed" is a different objective from "be
  right more often" for this model.

## Compute budget

| phase | shape | wall clock |
|---|---|---|
| 0 — marginal analysis | 1 × v5p-8 + local | ~1 h |
| 1 — publish + baseline parity | 1 × v5p-8 | ~0.5 h |
| 2 — nano gate | 1 + 4 × v5p-8 | ~0.5 h |
| 3 — LR probe | 3 × (1 + 4) v5p-8 | ~1 h |
| 4 — λ arms | 4 × (1 + 4) v5p-8, independent drivers | ~2 h |
| 5 — eval | 5 × v5p-8 (4 arms + parity), ×4 shards | ~1 h |

Peak ~20 v5p-8 slices; us-central1-a has held 100+ ready. Interactive band on the
marin v5p pool — `submit_rl_job` never sets a priority and the default is already
correct, the opposite of the CoreWeave batch rule. Per AGENTS.md, every predictor
run writes a `data/timings.csv`.

## Things that will bite

Carried from #200, where each cost real time:

- **Pin marin at `0.2.76.dev31155643335`.** 0.2.77 (2026-08-08) is the first
  release without `marin.rl`, which is `ContactsDenseLoss`'s base class and the
  whole environment API. The pin cannot be advanced; marin's direction is now
  SkyRL, which is a rewrite. Commit a `uv.lock`.
- **Co-locate data with compute.** us-central1 compute against us-east5 data dies
  with `TransferBudgetExceeded` about an hour in, once rollout spill and checkpoints
  start flowing. Prompt *reads* are trivially small; the spill is not, and reasoning
  about only the reads is how this was justified the first time.
  `check_region_locality` refuses it at config time. Both the pool and the eval
  targets are already mirrored to us-central1 — **verified 2026-08-10**, so no new
  mirror is needed.
- **A run cannot terminate itself.** The trainer's `mark_completed` RPC has no
  timeout and the rollout worker swallows poll errors inside
  `except Exception: pass`, so nothing reaps a finished trainer and
  `train_job.wait()` never returns. Use #200's `reap.py`, which detects completion
  through W&B (a fresh `wandb.Api()` per poll — a cached one keeps returning the
  first summary it saw).
- **`canonical_model_name` must be a registered `MODEL_MAPPINGS` key**
  (`Qwen/Qwen3-1.7B`), not a descriptive string: it feeds both a renderer substring
  match and an exact-key lookup on the weight-transfer path, and only the second one
  fails — which pure-generation gates never touch. All Qwen3 entries share one
  per-architecture mapping, so borrowing the 1.7B key for a 1.5B model is exact.
- **The engine needs an HF repo id**, not a `gs://` path (Phase 1).
- **`WANDB_API_KEY` does not propagate on its own.** `create_environment` forwards
  it from `os.getenv` of the *calling* process, so the chain only works if the
  driver was launched with it; nothing else in marin's RL path carries it, and both
  workers die on `wandb.errors.UsageError` after the gang has scheduled.
- **`prng_key` is a union** of a JAX key and a plain int, selected by
  `use_jax_rng = (inference_type == "levanter")`; a vLLM worker always hands the
  environment an int. marin's own `mock_env` calls `jax.random.randint` unguarded.
- **v5p-16 had 0 ready slices** while v5p-8 had 121. Size the trainer to what
  schedules.
- **TPU vLLM rejects per-request seeds** — engine-level seeding only, in both the
  RL loop and the eval worker (`--no-per-request-seed`).
- **`iris job logs` is empty for a *running* child**, so the environment traces to
  object storage (`_trace.py` / `read_trace.py`).
- **Commit before submitting**: the iris bundle is built from `git ls-files` and
  `_submit.check_clean` enforces it.

New to exp208:

- **Group-level metrics did not exist in plain mode.** `mean_jaccard` in #200 is
  *between sections within a rollout*, which is NaN when there is one section, and
  `n_sections`, `best_f1`, `first_f1`, `last_f1` all degenerate. Diversity collapse
  — the exact failure this experiment is designed to detect — would have been
  **unobservable during training** on a straight port. The inter-rollout metrics are
  not a nicety.
- **`episode_reward` changes meaning** (consensus marginal, not F1). Anything
  reading it as a return — W&B panels, `filter_out_groups_with_no_variance` (already
  off), the replay buffer's `alpha` prioritisation — is reading a centred quantity.
- **Ground truth is AFDB pyconfind, evaluation is PDB-derived.** The stepwise reward
  signs a gradient on individual contact decisions, so label noise is worse than
  merely noisy here — which is why #200 sampled AFDB **round 0** (highest pLDDT,
  mean global pLDDT 89.0) rather than ESM-Atlas. Exact-sequence overlap with the
  eval set is 0; homology-level overlap is **not** addressed (exp41 is the tool if
  it needs closing).

## Results

**Status: Phases 0 and 1 done. No arm has been trained; no accuracy claim.**

### Phase 1 — baseline parity: FAILED, and it mattered

Re-measuring the **unchanged exp199 checkpoint** through exp208's own eval path,
all 554 proteins x 100 rollouts on 4 v5p-8 (11 min/shard, 0/13900 truncated):

| band | exp208 (TPU) | committed #180 row | delta | paired SE |
|---|---|---|---|---|
| all | **0.609926** | 0.587348 | **+0.022578** | 0.001515 |
| long | **0.563922** | 0.542181 | **+0.021741** | 0.002699 |

That is ~15σ and ~10x #180's four-repeat span of 0.0023. The gate exists to catch
exactly this, and had it not run, every arm would have inherited a fabricated
+0.023 "improvement" over the committed baseline.

**It is not the metric.** `n_true`, `n_candidate` and `n_top` are identical on
**100%** of rows, so the GT universe and the top-R cut agree exactly. The
difference is entirely in the score matrix.

**It is not the recipe either.** #199's eval code (branch `exp/199-evals`) was read
end to end against exp82's worker: same `build_document(f"{stem}:r{k}", residues,
[], config=GenerationConfig())` prompt construction, same prefix cut, same
position map, same per-rollout dedup, same `MIN_SEP`, same n=100, same T=1.0 /
top_p=0.95 / top_k=-1, same `6 * L + 128` budget, same bf16. They are the same
measurement written twice.

**The cause is NOT the accelerator** — an earlier version of this section said
it was, and that was wrong. Running exp82's worker on CoreWeave H100, the very
hardware #199 used, reproduces the v5p figure to **+0.0004 (σ +0.3)**; the gap to
#199's published number persists at **+0.0229 (σ +16.1)** on that same hardware.
The difference is in #199's evaluation pipeline. #199's pipeline also evaluated an **exp117 control**
on CoreWeave, and exp117 has an independent v5p measurement from #169: they agree
to **−0.0015**, inside the 0.0023 repeat span. Same two stacks, no gap. So the
discrepancy is specific to the **exp199 checkpoint**, not a property of either
pipeline. The full analysis, including what else was ruled out (rope, weights,
metric, recipe) and the leading hypothesis (exp199's unusually large weights make
it numerically sensitive in bf16), is in
[`RPRECISION_PIPELINE_DISCREPANCY.md`](RPRECISION_PIPELINE_DISCREPANCY.md).

Two consequences:

1. **exp208's baseline of record is its own parity run (0.6099 / 0.5639)**, not
   the committed rows. Every arm is scored through the identical path, so the
   paired comparison stays valid; comparing an arm to 0.5873 would not be.
2. **The published exp199 R-precision is understated by ~0.023.** Scored the way
   every other frontier row was scored, #199 reads 0.6103 rather than 0.5873, so
   #180's #166 → #199 step is ~0.048 rather than ~0.026. The cause is localised to
   #199's pipeline; the mechanism is not yet identified. **That belongs to #180
   and #204, not to #208**, but it is filed here because this is where it was
   measured — see [`RPRECISION_PIPELINE_DISCREPANCY.md`](RPRECISION_PIPELINE_DISCREPANCY.md).

### Phase 0 — the consensus marginal: gate 1 passes, gate 2 does not

100 proteins x 100 rollouts, dumped per rollout and verified to reconstruct the
vote matrix exactly. Sources: [`data/phase0_summary.csv`](data/phase0_summary.csv),
[`data/phase0_per_protein.csv`](data/phase0_per_protein.csv),
[`data/phase0_per_rollout.csv.gz`](data/phase0_per_rollout.csv.gz).

**Gate 1 — is the marginal just precision in disguise? No.** Within-protein
corr(rollout precision, LOO marginal) = **0.236** (Spearman 0.237, p10-p90
0.01-0.43). The document term carries information the stepwise term does not, so
the design premise holds and the arms are worth distinguishing.

**Gate 2 — is it estimable at an affordable group size? Not against the n=100
marginal.** corr(single-draw marginal at G, n=100 marginal) = **0.198 / 0.185 /
0.214** at G = 8 / 16 / 32 — and, unlike the synthetic dry run, it does **not**
improve with G.

The reason is visible in the same table, and it is more interesting than the
correlation:

| | |
|---|---|
| n=100 marginals that are **exactly zero** | **85.8%** (median protein: 95.5%) |
| sd of the n=100 marginal | 0.00867 |
| mean votes on a top-R pair | 57.7 of 100 |
| union of predictions / R | 10.8x |

**The deployed metric is nearly insensitive to any individual rollout.** Top-R is
decided by pairs that ~58 of 100 rollouts agree on, so removing one rollout moves
nothing 86% of the time. Coverage is nowhere near binding (union is 10.8x R), and
only 2.5% of top-R is settled by index-order ties.

That reframes gate 2 rather than simply failing it. The n=100 leave-one-out
marginal is itself a **near-degenerate target** — 86% zeros, sd 0.0087 — so a
correlation measured against it is attenuated by construction, and "does the
G-marginal predict the n=100 marginal" is close to the wrong question. The
question that matters is whether optimizing consensus-at-G shifts the whole
rollout distribution enough to move consensus-at-100, which is a distributional
effect no leave-one-out statistic can predict and only a trained arm can answer.

It also cuts in favour of the step-only arm on the merits: if no single rollout
matters to the consensus, the only route to the reported metric is making *every*
rollout better, which is precisely what the dense per-contact term does.

### A stale constant Phase 0 caught

Single-rollout per-contact precision for this model is **0.482** over 10,000
plain rollouts. exp200's `INITIAL_PRECISION = 0.23` was exp163 arm F in
multi-draft mode — stale by a factor of two. Starting `p_bar` far below the truth
makes every correct contact look like a large win and every error nearly free,
biasing the first steps toward over-emission. Now 0.45 (not 0.482: the training
pool is AFDB round-0 with pyconfind labels, and only the PDB-derived eval set has
been measured). The environment EMA-tracks it from there, so this shapes only the
opening steps.

### Phase 2 — nano gate: BLOCKED on `RuntimeError: Loss is NaN`

Six RL gangs. **No arm has trained a step.** Every configuration dies at the
first training step (`last_update_step=0`, W&B history empty), while generation
is demonstrably healthy throughout.

| run | doc term | lam_doc | live rho | lr | KL beta | outcome |
|---|---|---|---|---|---|---|
| nano | consensus | 30 (guess) | 6.7 | 1e-5 | 0.01 | preempted at 37 min, then NaN |
| nano3 B | consensus | 4.5 (calibrated) | 2.2 | 1e-5 | 0.01 | **NaN** |
| nano3 F | own F1 | 0.59 (calibrated) | 3.6 | 1e-5 | 0.01 | **NaN** |
| nanolr | consensus | 4.5 | — | **1e-6** | 0.01 | **NaN** |
| minimal | **none** | 0 | 0 | **1e-6** | **0** | **NaN** |

The `minimal` row is the important one: arm S carries no document term at all and
the KL anchor is off, so the advantage is nothing but the dense per-contact
stepwise reward — and 1e-6 is the learning rate exp200 completed 150 steps at.
That exonerates everything #208 added: **the document term (two structurally
different ones), the lambda scale, the learning rate, and the KL**.

**Generation is not the problem.** Across the failing runs the environment
produced 35-39 sampling calls each, 64 rollouts per call, **0 empty and 0 ragged
groups**, `n_pred` ~130, per-contact precision ~0.22, group consensus ~0.43,
`union_over_r` ~12 and inter-rollout Jaccard ~0.10. Nothing is collapsing and
nothing is malformed.

**What was ruled out by reading marin rather than by launching jobs:**

* `compute_rloo_advantages` — plain leave-one-out, no division by a standard
  deviation that could be zero for a group of identical rewards;
* `compute_dapo_loss` — normalises by the **global** token count, so a short or
  fully-masked row cannot produce 0/0;
* `compute_ppo_loss_objective`'s `per_batch_loss` **does** divide per row and
  would go 0/0 on an all-masked row, but it only feeds metadata metrics;
* the dense-advantage broadcast, which `tests/test_dense_advantage_broadcast.py`
  pins and which passes.

**The surviving hypothesis is the levanter-side model load.** Generation and
training load the model twice, independently: vLLM from the HF repo, levanter
from `initial_checkpoint`. A levanter load that disagrees with vLLM leaves
rollouts looking perfect while `exp(policy_logp - vllm_logp)` overflows on the
first step. exp200 measured `train/ratio_mean` at 1.0024 and called it the check
that validates the whole logprob path; exp208 crashes before it can be logged.

exp208 is the first run to warm-start from an **exp199** export — exp200 used
exp163 arm F. Their configs were diffed field by field and are semantically
identical, including rope (`rope_theta` 500000 and the same llama3
`rope_scaling`); exp199 additionally carries a redundant transformers-5
`rope_parameters` block with the *same* values, so rope is not the difference.

**The control settled it: the harness is sound, and the exp199 export is at
fault.** exp208's code, unchanged, warm-started from exp163 arm F (exp200's own
checkpoint) instead of exp199 — same arm S, same lr 1e-6, same `kl_beta=0`, same
everything else — trained **10 clean steps with no NaN**:

| step | `train/ratio_mean` | `train/max_advantages` | `train/policy_entropy` |
|---|---|---|---|
| 0 | 0.999794 | 0.1833 | 2.599 |
| 4 | 0.999862 | 0.2047 | 2.381 |
| 9 | 1.000513 | 0.2376 | 2.324 |

`ratio_mean` holds at 1.0000 ± 0.0005 for every step — sampler and trainer agree
on logprobs to within 0.05%, which is the check exp200 used (1.0024) to validate
the whole policy-gradient path. Advantages are non-zero and reach the loss, and
entropy sits where exp200 measured it (2.618). So the environment, the dense
reward, `ContactsDenseLoss`, the config assembly and the weight-transfer path are
all correct; **only the choice of warm start separates a clean run from NaN.**

That run then died at step 9 with `ValueError: weight transfer hook ran at step
9, which is not aligned with sync_interval_steps=8` — unrelated to the NaN, and
an exp208 config bug: the final transfer fires at `num_train_steps - 1`, so a
10-step nano with sync 8 cannot align. `build_rl_job_config` now rejects a
misaligned pair at config time (the real arms use 400 steps, which is aligned).

**Open: why does the exp199 export NaN in levanter?** Its config is semantically
identical to exp163 arm F's, rope included, and vLLM loads the same files and
generates well (precision 0.22, consensus 0.43), so the weights are finite and
correctly mapped for inference. What differs is the model itself. The next
diagnostic is a weight-statistics comparison between the two exports — exp199 is
a much stronger model trained under marin's newer loss scale, and larger
activations overflowing levanter's bf16 compute (`mp="p=f32,c=bfloat16"`) in the
backward pass would produce exactly this: healthy inference, NaN on the first
training step.


### SkyRL port: arm S trains on 8×A100, and `err_decay` looks exploitable

The marin.rl path stayed blocked on TPU capacity, so the harness was ported to
SkyRL (`skyrl/`, [PORT_DESIGN.md](skyrl/PORT_DESIGN.md)) and arm S now trains
end-to-end on the private 8×A100 box: 8-way FSDP2 policy + ref, 4 vLLM engines at
TP 2, 512 trajectories (32 prompts × 16 samples) per step, **~53 s/step** and
~3900 tok/s/GPU. Getting real placement took three config knobs, each of which
failed loudly and separately: `policy_num_gpus_per_node`, `ref_num_gpus_per_node`
(colocated models must match), and `num_engines × tensor_parallel_size` = policy
GPUs. Left at defaults, SkyRL runs the whole job on **one** GPU at 136 s/step.

The document term (arms B and F) is now implemented and tested. It maps rewards
via the **per-row `trajectory_ids`** on `GeneratorOutput`, not row position: SkyRL
does document that `generate` preserves input order, but a marginal attributed to
the wrong rollout is still a plausible number, so the test
([`test_document_term_mapping.py`](skyrl/tests/test_document_term_mapping.py))
hands rows over shuffled and asserts the result differs from what a positional
implementation would produce.

### The blocker: SkyRL destroys the policy on its first weight sync

Arm S does not train. It collapses after exactly one step, and the collapse is
**not** in the reward, the gradient, the export, or the rope config — all four are
excluded by measurement below.

On first look the run seemed to be length-gaming: reward rising (0.0010 → 0.0453)
alongside response length (488 → 985 against a 1024 cap), which `err_decay` gives a
real incentive for. **The contact tallies refuted that**, and in the opposite
direction — the policy does not emit *more* contacts, it stops emitting them:

| step | contacts/rollout | precision | pred/gt | resp. length | reward |
|------|-----------------|-----------|---------|--------------|--------|
| 0    | 160.2           | 0.267     | 1.11    | 483          | 29.56  |
| 1    | 0.86            | 0.039     | 0.006   | 982          | 0.017  |
| 2    | 0.94            | 0.044     | 0.007   | 975          | 0.033  |
| 3    | 0.71            | 0.064     | 0.005   | 987          | 0.036  |

Step 0 is the model behaving correctly (160 contacts/rollout at precision 0.267,
matching its known quality). After one step it emits **0.9 contacts per rollout**
and runs to the token cap — length rises because the document never completes, not
because contacts are being spammed. Reward "rising" from 0.017 was noise near zero
*after* a fall from 29.56.

Four candidate causes, each excluded by a measurement rather than by argument:

* **The gradient.** A zero-learning-rate control collapses *identically*
  (precision 0.259 → 0.022, pred/gt 1.13 → 0.005, length 489 → 996). With `lr=0`
  no gradient can move a weight, so nothing about the reward, the advantage, or
  `err_decay` can be responsible. This single control retires the whole reward-shape
  hypothesis.
* **The rope config.** The export declares llama3 scaling at `factor: 8.0` in both
  `rope_scaling` and `rope_parameters` — exp199's known export hazard. Scoring real
  prompts under each interpretation gives 3.6804 vs 3.6861 nats/token
  ([`probe_rope_config.py`](skyrl/probe_rope_config.py)). A 0.006 difference; a wrong
  rope costs whole nats. Not it.
* **The export / loaders.** transformers and vLLM score the same 1870 tokens at
  3.7273 vs 3.7267 nats, mean |diff| 0.017
  ([`probe_hf_vs_vllm.py`](skyrl/probe_hf_vs_vllm.py)). The two stacks read this
  checkpoint identically, so the export is sound.
* **Therefore: SkyRL's own trainer-side copy.** SkyRL reports
  `rollout_train_logprobs_abs_diff_mean` = **1.33 nats at step 0**, dropping to
  0.08 from step 1 on. Since the loaders agree to 0.017 nats outside SkyRL, that
  1.33 is manufactured inside it: at step 0 the engines hold the real model and the
  FSDP2 policy holds something else. `sync_weights` then pushes the policy's copy
  into the engines — after which the two "agree" at 0.08 and generation is dead.
  The ordering (disagree → sync → agree → collapse) is the signature.

**Confirmed: it is the multi-GPU FSDP2 sharding.** Removing sharding and changing
nothing else fixes it completely:

| | step-0 logprob gap | precision | pred/gt | response length |
|---|---|---|---|---|
| 8×A100, `policy_num_gpus_per_node=8` | **1.33 nats** | 0.267 → 0.04 | 1.11 → 0.006 | 483 → 985 (cap) |
| 1×A100, `policy_num_gpus_per_node=1` | **0.017 nats** | 0.15–0.44, no trend | 0.99–1.31 | 419–573, stable |

The single-GPU gap of 0.0174 nats reproduces the standalone HF-vs-vLLM probe
(0.0173) to three decimals — with one GPU the trainer and the engines run the same
model, the sync is harmless, and the policy survives. With eight they do not.
Precision on the 1-GPU run fluctuates between 0.15 and 0.44 with no trend; at
8 prompts × 8 samples that is 64 rollouts/step, far too noisy to read as learning.
The claim here is only that it does not collapse.

So arm S is runnable today at `policy_num_gpus_per_node=1`, and the 8-GPU
configuration must not be used until the sharding path is fixed — a sharded run
looks superficially fine (it trains, it logs, it reports a falling reward) while
generating from a destroyed policy. Throughput can be recovered without sharding
by setting `colocate_all=False` and giving the spare GPUs to inference engines.

Worth noting what made this findable. Reward and response length alone were
consistent with a plausible and completely wrong story (length-gaming);
`contacts/pred_per_gt` is what separated "emits more" from "emits none", and
`rollout_train_logprobs_abs_diff_mean` is what pointed at the sync. Neither was in
the original metric set, and the zero-LR control is what turned a hypothesis into
an exclusion.

`err_decay` remains theoretically dubious for the reason first suspected — the k-th
error in a section costs `p̄·δ^k`, a convergent series bounded near `p̄/(1−δ)`, while
each correct contact pays a full `1−p̄`, so the marginal contact is worth
`p − p̄·δ^k` and the p̄-centring's zero baseline does not hold. But that is now an
untested concern, not an observed one, and it cannot be tested until the policy
survives a weight sync.

### Artifacts

- Baseline of record: `gs://marin-us-central1/protein-structure/MarinFold/exp208/phase0/scores/exp199_cw_p06_aug_step145199`
- Warm start: [`timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199`](https://huggingface.co/timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199) (bf16, tokenizer co-located, `rope_theta` 500000, vocab 2845)
- bf16 weights on GCS: `.../exp208/models/exp199` (5.48 GiB fp32 -> 2.74 GiB)

## Conclusion

_(Pending.)_
