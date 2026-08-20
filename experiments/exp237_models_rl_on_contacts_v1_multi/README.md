---
marinfold_experiment:
  issue: 237
  title: 'exp: RL on the <contacts-v1.multi> model from #230 — consensus, final-section, and oracle rewards'
  kind: models
  branch: claude/marinfold-237-rl-experiments-e0615a
---

# exp: RL on the `<contacts-v1.multi>` model from #230 — consensus, final-section, and oracle rewards

**Issue:** [#237](https://github.com/Open-Athena/MarinFold/issues/237) · **Kind:** `models` · **Branch:** `claude/marinfold-237-rl-experiments-e0615a`

## Question

Apply RL to the `<contacts-v1.multi>` model from
[#230](https://github.com/Open-Athena/MarinFold/issues/230) to improve the contact
sets it generates. Three targets, tested separately: the **consensus** across a
rollout's sections, the **final** section it commits to, and the **oracle best**
section it produces.

## Hypothesis

[#208](https://github.com/Open-Athena/MarinFold/issues/208) ran eleven scored runs
across five reward designs and **none improved consensus R-precision**. The
finding was structural rather than a matter of reward tuning:

> the eval is a vote over 100 rollouts, and a reward that makes each rollout
> individually better makes the hundred redundant. Consensus scoring cannot rank
> a pair that no rollout emits, so a policy that becomes more selective — the
> natural response to almost any per-contact reward — destroys the ranking
> underneath its own improving precision.

**That is a unit mismatch.** The reward acted on one rollout; the metric scored a
vote over 100 *independent* rollouts, an object no single rollout can see. Under
`<contacts-v1.multi>` the candidate set lives **inside a single rollout** — #230
measured a mean of 22.0 contact sets per rollout, in one sequence — so a reward
computed on the aggregate of one rollout's sections is computed on the same kind
of object the metric scores, and its credit assignment is *within* the sequence,
where the policy gradient can reach it.

That is the one hypothesis this experiment tests. It is falsifiable, and it is the
only reason to expect a different outcome from #208.

## Background

| Prior work | What this run takes from it |
|---|---|
| [#230](https://github.com/Open-Athena/MarinFold/issues/230) | The warm start (`plm-exp230-cv1-multi-1_5b-lr1e-4-e1-cos-a100/hf/step-1988`), the eval targets, `eval_agg_worker.py` and `score_agg_modes.py` — invoked unchanged |
| [#208](https://github.com/Open-Athena/MarinFold/issues/208) | The SkyRL port (env, generator shape, advantage registry, exporter), the training prompt pool, the reward-design invariant, and the negative result this experiment is the control for |
| [#209](https://github.com/Open-Athena/MarinFold/issues/209) | exp82's `score_rollout_worker.py` is the **reference scorer**; the metric is not re-derived anywhere here |
| [#226](https://github.com/Open-Athena/MarinFold/issues/226) | The 577-unit target universe (554 legacy + 23) and the eval2 cuts |
| [#163](https://github.com/Open-Athena/MarinFold/issues/163) | The `<contacts-v1.multi>` format itself, and the id-7 rename |

### What has to be beaten

R-precision (all), legacy 554, from #230:

| what | R-prec | note |
|---|---:|---|
| plain `<contacts-v1>`, 100 rollouts voted | 0.6058 | the base task's ceiling |
| **plain, 22 rollouts voted** | **0.5896** | **the bar** — budget-matched to one multi rollout |
| multi, consensus over ~22 sections | 0.5673 | target 1 |
| multi, oracle best section | 0.5342 | target 3 (a ceiling; uses ground truth) |
| multi, last section | 0.4566 | target 2 |
| multi, second-to-last section | 0.4284 | |
| plain, single rollout | 0.4454 | |

Two facts shape the design. **Consensus beats the ORACLE best single candidate**
in both regimes (0.5896 > 0.5680; 0.5673 > 0.5342), so candidates carry
complementary information rather than being noisy copies — a reward that can
exploit *combinations* has more to work with than best-of-N over individuals. And
**at matched budget independent sampling still wins** (0.5896 vs 0.5673): the
union of 22 independent rollouts covers 1,065 distinct pairs against 658 for 22
sections of one rollout, so the multi format starts ~62 % behind on explored
space. Closing that gap is what success looks like; RL that closes it by
sharpening will instead widen it.

## Approach

### Phase 0 — is the reward measurable at all? (offline, no GPU)

#208's lesson is that *a null result at a learning rate that does not move the
policy is not a result*. The cheaper version of that lesson is to ask, before
booking any compute, whether arm M-C's reward exists. Two ways it could be
identically useless:

1. **`m_k` is discrete.** Consensus R-precision is computed over integer vote
   counts with a stable positional tie-break, so removing one section out of ~22
   very often changes nothing. A rollout whose marginals are all equal
   contributes **zero** advantage after centring.
2. **`m_k` might be a restatement of section F1**, in which case arm M-C is arm
   M-B with extra steps and #230's oracle number already bounds it.

Both are answerable on generations that already exist — #230's `eval/agg_sections`
parquets, 577 proteins × 8 multi rollouts × ~22 sections, from the very checkpoint
this experiment warm-starts from. [`phase0_marginals.py`](phase0_marginals.py).

### The reward

Everything operates on **sections of one rollout**, never on rollouts of a group.
`consensus.py` is vendored from #208 unchanged — the leave-one-out machinery is
identical; only the population changes.

| arm | reward | shape | estimator |
|---|---|---|---|
| **M-C** | `m_k = C(all) − C(all \ {k})`, section *k*'s marginal contribution to its **own rollout's** consensus | **per-section**, dense — each section's tokens carry their own advantage | `contacts_section` |
| **M-F** | `F1(last section)` | **whole-rollout scalar** — one number for the entire generation, GRPO-centred against its 8 siblings and broadcast to every token | `grpo` |
| **M-B** | `max_k F1(section k)` — **ORACLE** | **whole-rollout scalar**, same shape as M-F: score the whole generation by the best contact set anywhere in it, with no per-section credit assignment at all | `grpo` |
| **M-BC** | `GRPO(max_k F1) + lam·GRPO(C_i(all))` — M-B's scalar blended with the rollout's own consensus, each standardised **separately** | **whole-rollout scalar** | `contacts_rollout` |
| **M-FC** | `GRPO(F1(last)) + lam·GRPO(C_i(all))` — synthesis rather than selection: reward the last section, with the consensus as a restoring force | **whole-rollout scalar** | `contacts_rollout` |
| **M-K** | `C_i(all)` — **the rollout's own consensus R-precision**, i.e. the deployed metric computed on the object the model emits | **whole-rollout scalar** | `grpo` |
| **M-BP** | `max_k F1 + beta·min(0, K − floor)` — M-B with a one-sided floor on the candidate count, added **raw** so the deadband survives standardisation | **whole-rollout scalar** | `grpo` |
| **M-0** | M-C's reward at **lr = 0** | — | `contacts_section` |

The last three did not exist when the experiment started. M-BC and M-FC were
added to test whether a blend could hold two objectives at once; **M-K was
designed from the diagnosis of M-C's failure** — it is the rollout-level,
scale-correct base that M-C's per-section marginal should have been standing on,
and it produced the best consensus in the experiment.

The M-C / (M-F, M-B) split is the axis this experiment varies: M-C decides *which
section* earned the reward and shapes those tokens specifically; M-F and M-B do
not look inside the rollout at all — they reduce it to one number and let GRPO's
group baseline do the rest.

#### The expectation calculation, done on paper first

#237 carries #208's reward-design invariant verbatim: three separate
modifications there broke `E[r] = p − p̄` by weighting **one side** of a centred
reward, each costing a full training run and each catchable by a five-line
calculation beforehand. So, explicitly, for M-C:

Let the group `g` be **every section of every rollout sampled from this prompt**
(G rollouts × ~25 sections). Section *k*'s advantage is

```
A_k = (m_k − mean_g(m)) / (std_g(m) + eps)
```

so `E_g[A] = 0` **exactly**, per prompt, by construction. Consequences:

* **No first-order pressure on section count.** An extra section is worth
  emitting exactly when its marginal beats the group's mean marginal. A section
  that duplicates its siblings changes no vote, scores `m_k = 0`, and is
  therefore **below** the mean and net-negative — which is the pressure this
  model needs at #230's measured Jaccard of 0.304. A section carrying a true pair
  its siblings missed scores positive. Neither direction is free.
* **The normalisation is a division, not a re-weighting of one side.** `std_g` is
  computed over the same population as `mean_g`, so it scales both signs
  identically and cannot tilt the zero point — which is exactly how `err_decay`,
  the unweighted `p̄` and unnormalised novelty each broke in #208.
* **A prompt with zero marginal spread contributes zero**, not `0/eps`.

`test_section_rewards.py` pins the identity (`pooled.mean() == 0` to 1e-12) rather
than asserting it in prose.

#### Assigning a per-section advantage to tokens

`A_k` lands on **every response token of section k, unscaled** — not spread as
`A_k / n_tokens`. GRPO gives one sequence-level scalar to every token, so a
per-token advantage of magnitude ~1 is the scale the learning rate is calibrated
for; spreading `A_k` over a section's ~300 tokens would make M-C's gradient ~300×
smaller than M-F's at the same lr. #208 paid a full run for this mistake in the
other direction — `lam_doc = 4.5` carried **0.42 %** of the stepwise term's spread
("it was not a weak signal, it was no signal").

A section owns the `<begin_statements>` token that **opens** it, so the decision
to start another candidate is shaped by whether that candidate turned out to be
worth starting. The final section owns `<end>`. Tokens past the first `<end>` —
a rollout that runs on into a second document — carry zero.

**No arm here is per-contact-only**, per #237's rule. #208 established that a
`p̄`-centred per-contact reward is a sharpening operator to first order and that
novelty weighting is a second-order redistribution which cannot overcome it; that
ladder is explicitly out of scope.

### What "scale-correct" means, precisely

The term is used throughout this write-up and it is **not** what it sounds like.
It does not mean the reward is invariant to the number of candidates — the reward
this document calls scale-correct is emphatically *not* invariant. It means the
reward's dependence on the candidate count has **the same sign as the deployed
metric's**.

**The setup.** Take a real rollout's ordered sections `S = (s₁ … s_K)`. For
`n ≤ K` let `R(n)` be the reward recomputed on the truncated rollout
`(s₁ … s_n)` — same candidates, fewer of them. The deployed metric behaves like
this, measured on #230's own generations and **monotone throughout**, not merely
at the endpoints:

| n | 1 | 2 | 4 | 8 | 12 | 16 | 22 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C(n)`, the rollout's consensus | 0.341 | 0.405 | 0.458 | 0.505 | 0.525 | 0.534 | 0.543 |

So **more candidates is genuinely better**, and any reward that pays *more* for
fewer is pointing away from the thing being measured.

**The definition.** What the gradient actually sees is not `R(n)` but the
group-centred advantage a rollout receives as a function of *its own* count when
its siblings differ — which is what the second table of
[`analyze_section_count_incentive.py`](analyze_section_count_incentive.py)
measures, on synthetic groups differing in nothing else:

```
A(n) = mean over groups of  ( R_i(n) − mean_g R ) / sd_g R
```

* **count-aligned** (what this document calls *scale-correct*): `A(n)` increases
  in `n`, so emitting fewer candidates lowers the advantage — the same direction
  as `C(n)`.
* **count-adverse** (what it calls *pathological*): `A(n)` decreases in `n`, so
  the policy is paid to emit fewer candidates and thereby produce a worse answer.

| A(n) | 1 | 4 | 22 | |
|---|---:|---:|---:|---|
| M-C, per-section marginal | **+4.79** | +0.21 | −0.22 | count-adverse |
| causal prefix marginal | +2.03 | +0.48 | −0.22 | count-adverse |
| **`C_i(all)`, arm M-K** | **−1.37** | −0.14 | **+0.79** | **count-aligned** |

**Two things the term does not claim, both of which matter.**

1. **It is not scale-*invariance*.** `C_i(all)` rises with the count, steeply. The
   claim is alignment of sign, not independence. A genuinely invariant reward
   would be a third thing and is not what any arm here used.
2. **It is not un-gameability.** Count-alignment is a statement about **one**
   nuisance direction. Arm M-K is count-aligned and still raised its reward
   partly through a different route — making its sections *more alike* (mean
   pairwise Jaccard 0.255 → 0.328, votes per pair 2.64 → 3.28, medians of the
   first 13 training batches against the last 5), which lifts a vote count
   without improving any individual candidate. Its **section count held flat at
   ~22** across all 48 steps, which is the count-alignment working; the drift is
   on the other axis. **Scale-correct means "not gameable along the
   candidate-count axis", nothing more.**

A separate axis is **boundedness**. M-C's advantage is not merely adverse but
*divergent* — 366× larger at one section than at 22 — which is why no fixed `lam`
could have balanced it inside a blend, and why it ran away rather than drifting.
An adverse but bounded reward would be a milder problem.

Finally, `C(n)` rising is an **empirical** property of this model's sections, not
a theorem: it holds because the drafts are complementary. If a model emitted
redundant drafts, `C(n)` would be flat and the whole distinction would collapse.

### The RL algorithm

**GRPO — PPO's clipped surrogate with a group-relative baseline instead of a
learned critic, plus a KL penalty to the frozen reference.** There is no critic
and no GAE anywhere in this experiment; every advantage estimator used
(`grpo`, `contacts_section`, `contacts_rollout`) is critic-free. Resolved
configuration, read from the run logs rather than from intent:

| | |
|---|---|
| policy loss | `regular` — PPO clipped surrogate (not CISPO or GSPO) |
| clip range | `eps_clip_low = eps_clip_high = 0.2` |
| KL | **`use_kl_loss = true`, `kl_loss_coef = 0.001`, estimator `k3`** — a loss term, *not* folded into the reward (`use_kl_in_reward = false`) |
| loss reduction | `token_mean` — a rollout's contribution is proportional to its own token count |
| advantage | per-arm; `advantage_batch_normalize = false`, so the only normalisation is the per-group one described below |
| optimiser | AdamW, `max_grad_norm = 1.0`, constant-with-warmup |
| inner loop | `update_epochs_per_batch = 1`, `policy_mini_batch_size = train_batch_size = 8` |

#### In effect this is REINFORCE with a group baseline, and the clipping is inert

That last row has a consequence worth stating plainly, because "GRPO" invites the
assumption that the trust region is doing work. It is not. With **one inner epoch
and one minibatch**, and `recompute_old_logprobs_per_minibatch = true`, the
"old" policy is recomputed and **equals the current policy at the moment of the
update** — so the importance ratio is exactly 1 and can never leave [0.8, 1.2].

Measured, not inferred: **`loss_metrics/clip_ratio` is 0.0 in every arm, at every
one of 468 steps.**

So the update actually applied is

```
grad  =  −E[ A · ∇ log π ]  +  0.001 · ∇ KL_k3( π ‖ π_ref )
```

— a vanilla policy gradient with a group-relative baseline and a weak KL pull.
The clipping machinery is present, correct, and never activates.

**This is load-bearing for how the results read.** There is no trust region
limiting the step, and `kl_loss_coef = 0.001` is far too weak to act as one
(terminal KLs of 0.09, 0.49 and 3.26 were all reached). Distance from the warm
start is therefore governed only by learning rate × steps and by gradient
clipping — which is exactly why the dose-response is so clean, why two runs at a
3.3× different learning rate land on the same numbers at matched KL, and why
"how far the policy moved" turned out to be the variable that orders every result
in this experiment. **The runs are not being held anywhere; they walk until
something breaks.** That is also why the diversity gates had to do the stopping.

`policy_kl`, quoted throughout as "distance moved", is the k3 estimator of
KL(π ‖ π_ref) against the frozen #230 checkpoint.

### What `GRPO(·)` means here, precisely

`GRPO(·)` appears throughout this write-up as shorthand. It is not a
paraphrase — it is exactly SkyRL's `compute_grpo_outcome_advantage`, reproduced
in `section_rewards.grpo_standardise` and pinned by test. For a prompt **group**
`g` of `G` rollouts (here `G = n_samples_per_prompt = 8`, all sampled from the
same prompt), with one scalar reward `R_i` per rollout:

```
GRPO(R)_i  =  ( R_i − mean_g(R) ) / ( std_g(R) + 1e-6 )
```

and that single number is then assigned to **every response token** of rollout
`i` (padding zeroed). Four details that the name does not carry, each of which
changes a number:

| detail | value | why it matters |
|---|---|---|
| `std` | `torch.std`, the **unbiased sample** sd (`ddof = 1`) | numpy's default is `ddof = 0`; on a group of 8 that is a 7 % difference in every denominator |
| `epsilon` | added to the **standard deviation**, not the variance | `1e-6` |
| singleton group | `mean = 0`, `std = 1` | the raw reward passes through **uncentred** |
| the scalar reward | recovered as `token_level_rewards.sum(dim=-1)` | a reward placed on one token is summed back out |

One consequence worth stating because it interacts with section count: SkyRL's
`loss_reduction` is `token_mean`, so a rollout's contribution to the loss is
proportional to **its own token count**. Longer rollouts carry more gradient.

**Arm M-BC then is:**

```
A_i  =  GRPO( max_k F1(section k) )_i  +  lam_consensus * GRPO( C_i(all) )_i
```

with each term standardised **separately** over the same group. That is a
deliberate choice, not a convenience: because both terms are divided by their own
within-group spread, `lam_consensus` is a ratio of *standardised* quantities, so
`1.0` means "these two objectives get equal weight, in units of within-group
standard deviations". Standardising the sum instead — `GRPO(best + lam·C)` —
would let the raw scales decide, and on a typical group the best-section F1
spreads ~4x wider than the rollout consensus, so the consensus term would
contribute ~4 % of the variance while appearing to be weighted equally. That is
the calibration #208 got wrong twice with `lam_doc`, in both directions.

**Neither term can be gamed by section count**, which is the whole reason this is
the blend rather than M-B + M-C's per-section marginal: `max_k F1` does not depend
on how many sections exist, and `C_i(all)` *falls* when sections are dropped
(0.543 at 22 sections, 0.341 at one) — against M-C's marginal, which is **+4.79**
at one section and **−0.22** at 22.

### The diversity gates, as kill criteria

#230's checkpoint reads Jaccard **0.304**, already past exp200's 0.30
diversity-collapse criterion *before any RL*. #208's dominant failure mode was RL
collapsing diversity. So the gates are checked **every batch**, from the run's own
opening measurement, and tripping one three batches running **stops the run** —
that is the result, and continuing only spends GPU hours confirming it:

- **kill** if union coverage per rollout falls > 20 % from the opening batch;
- **kill** if mean pairwise Jaccard exceeds 0.45;
- **kill** if mean sections per rollout falls below 12.

`union pairs`, `total votes` and `votes/pair` are reported every batch, because
#208 showed these separate the two failure modes (volume collapse vs diversity
collapse) where reward and accuracy alone cannot.

### Infrastructure — the traps #208 paid for, and how each is handled here

| # | trap | what this run does |
|---|---|---|
| 1 | SkyRL FSDP policy sharding silently destroys the policy via a weight sync that pushes a divergent copy into the engines (logprob gap 1.33 nats sharded vs 0.017 unsharded) | **unsharded**: `policy_num_gpus_per_node=1`, `colocate_all=false`, six cards given to engines. `minibatch_rollout_logprobs_abs_diff_mean` is reported per step as the tripwire |
| 2 | vLLM pads 2845 → 2848 with zero rows that emit logit 0.0; #208 measured them taking 12.4 % of sampled tokens and NaN-ing the trainer on step 1 | `vocab_size=2845`, enforced in the generator with a hard raise |
| 3 | `custom_chat_template` disables per-token rewards in the stock generator, silently, degrading to one scalar per trajectory | `MultiSectionGenerator` overrides `_build_per_token_rewards` whole |
| 4 | the config key is `trainer.algorithm.advantage_estimator`, not `adv_estimator`; custom estimators work only on `fsdp`/`megatron` | both encoded in `run_arm.sh` |
| 5 | terminal KL is the most useful column — several #208 arms "did nothing" because they never moved | reported per arm; below ~0.0015 an arm is **untested**, not negative |
| 6 | guard assertions must be checked for reachability — #208's constant-advantage guard took `std` across the padded row, so it could never fire | the std is over **response tokens only**, inherited with the fix |

Two more this run paid for itself:

7. **The tokenizer must carry a pass-through chat template**, because SkyRL's
   `PromptDataset` templates through the tokenizer rather than through
   `generator.chat_template` — and the max-prompt-length filter *passes* an empty
   render precisely because it tokenizes to zero. `prepare_model.py` bakes it in
   and asserts the render is token-identical to the raw string.
8. **Ray's raylet dies with "Too many open files"** at the login shell's default
   1,024 descriptors — six vLLM engines plus a policy and a ref worker open more
   sockets than that between them — and it surfaces three minutes in as
   `LocalRayletDiedError`, saying nothing about descriptors. `run_arm.sh` raises
   the soft limit to 65,536 (the hard limit is 1,048,576).

### Deliberate deviations from the issue, and why

| the issue says | this run does | why |
|---|---|---|
| group size 16 | **8**, with 8 prompts per step (64 rollouts/step either way) | a multi rollout is ~4,000 generated tokens against plain's ~500, so the per-step budget is *rollouts*, not prompts. Halving the group doubles the number of distinct proteins the run sees, and M-C's centring population is *sections* (8 × ~26 ≈ 200 per prompt), not rollouts, so it loses nothing |
| — | **lr 1e-5**, not #208's 1e-6 | every arm here hands the optimiser an advantage normalised to unit spread. #208's 1e-6 runs on normalised rewards never moved (arm C v1 KL 0.0004, D v1 KL 0.0014). 1e-5 is where its normalised arm reached KL 0.084, an order of magnitude below the 4e-5 that diverged to 3.96 |
| one epoch | **72 steps** (4,608 rollouts, ~576 proteins) | wall-clock. One epoch of the 10k pool is 1,250 steps at ~103 s/step = 36 h/arm; three arms plus evaluation had to fit one night |

The training pool is #208's, unchanged apart from the mode marker — see
`build_multi_dataset.py` for why holding the data fixed is what makes #208 the
control.

## Success criteria

**Primary.** Multi-mode consensus R-precision (legacy 554) **> 0.5896**, the
budget-matched plain baseline. Beating #230's own 0.5673 is necessary but not
sufficient.

**Secondary.** Final-section R-precision > 0.4566 (M-F), ideally toward the 0.5342
oracle; oracle-best > 0.5342 (M-B); AUC ≥ the #230 checkpoint's on every arm.

**Kill criteria.** The three diversity gates above; Gate A regression worse than
−0.005; Gate B failure; terminal KL > 1.0.

## Parameters, in full

**Warm start and data**

| | |
|---|---|
| checkpoint | `hf://buckets/open-athena/MarinFold/checkpoints/plm-exp230-cv1-multi-1_5b-lr1e-4-e1-cos-a100/hf/step-1988` |
| prepared as | bf16 (2.94 GB), top-level `rope_theta` 500000 + `rope_scaling` restored beside `rope_parameters`, pass-through chat template baked into the tokenizer and asserted token-identical |
| verified | vocab 2,845; token id 7 = `<contacts-v1.multi>`; per-contact precision 0.4136 in SkyRL's vLLM against #230's 0.4095 over 8.3 M rollouts |
| training prompts | #208's `skyrl_train_10k.parquet` — 10,000 AFDB proteins, one realization each, L median 159 / max 512 — with token 0 swapped to `<contacts-v1.multi>` and **nothing else changed** |

**RL**

| | |
|---|---|
| framework | SkyRL, `trainer.strategy=fsdp`, vLLM 0.23 / transformers 5.8 / torch 2.11 |
| placement | **unsharded policy**: `policy_num_gpus_per_node=1`, `ref_num_gpus_per_node=1`, `colocate_all=false`, **6 inference engines** at `tensor_parallel_size=1` |
| group / batch | `n_samples_per_prompt=8`, `train_batch_size=8` prompts → **64 rollouts/step**; `micro_train_batch_size_per_gpu=1` |
| sampler | T 1.0, top-p 0.95, top-k −1, `max_generate_length=7000`, `max_prompt_length=2048`, ctx 8,192 |
| optimizer | AdamW **lr 1e-5** constant-with-warmup, wd 0.01, `max_grad_norm` 1.0, `kl_loss_coef` 0.001 (k3), `loss_reduction=token_mean` |
| estimator | `contacts_section` (M-C, M-0) / `grpo` (M-F, M-B, **M-K**) / `contacts_rollout` (M-BC, M-FC) |
| cost | ~101 s/step; 8–120 steps per arm on 8 × A100-80GB; ~90 GPU-h training, ~200 GPU-h evaluation |

**Evaluation**

| | |
|---|---|
| targets | the **577-unit universe** (#89's 554 + #226's 23), `eval577_targets.parquet`, both of #89's filters (`MIN_SEP 6` **and** `MIN_DEG 0.001`) |
| generation | #230's `eval_agg_worker.py --mode multi`, **8 rollouts/protein**, full context, T 1.0 / top-p 0.95 / top-k −1, 8 GPUs |
| scoring | #230's `score_agg_modes.py`, which calls exp89's `compute_metrics` via `score_gate_a.metrics_for` — **imported, not re-derived** |
| cuts | legacy554 (554), eval2 (307), **eval2-natural (78)**, eval2 <30 % (275) |
| significance | paired per-protein bootstrap, 10,000 resamples, seed 237 (`compare_arms.py`) |
| validation | the pipeline reproduces #230's published table exactly (0.5673 / 0.5342 / 0.4566 / 0.4284), and the lr-0 control returns within ±0.003 |

## Compute

Eight A100-80GB. **1 policy (unsharded, mandatory — #208's sharding bug) + 1
reference + 6 vLLM engines at `tensor_parallel_size=1`.** Measured over 468
training steps:

| phase | median | GPUs busy |
|---|---:|---:|
| `generate` | 36.3 s | 6 |
| `fwd_logprobs` (old + ref) | 21.8 s | 2 |
| `policy_train` | 40.1 s | **1** |
| `sync_weights` | 2.3 s | 8 |
| **step** | **102.4 s** | |

The phases **do not overlap** (they sum to 100.5 s of a 102.4 s step), so node
utilisation is **39 %** — 320 of 819 GPU-seconds. **The trainer is the
bottleneck**: 61 % of the step runs on one or two cards. MFU is **33 % during
`policy_train`** and **1.6 % node-wide** — a scheduling result, not a kernel one.
Generation is bandwidth-bound and under-batched at ~11 concurrent sequences per
engine. Full detail, including the memory figures and three infrastructure traps,
in [RESULTS.md](RESULTS.md#compute-how-the-eight-gpus-are-used-and-what-limits-the-step).

~39 h wall clock on the node across seven runs — ~90 GPU-h of training against
~200 GPU-h of evaluation — and 728 GB of checkpoints.

## Run book

```bash
# 0. Phase 0 -- does arm M-C's reward exist? (CPU, ~10 min, no GPU)
python phase0_marginals.py --sections ~/exp230_data/eval/agg_sections \
    --targets ~/exp230_data/eval577_targets.parquet --out data/

# 1. push the port to the GPU host (--host is required and has no default)
./skyrl/run_on_host.sh --host <user@host> --smoke

# 2. prepare the warm start and the prompts, then run every arm and its eval
./skyrl/run_on_host.sh --host <user@host> -- bash ~/exp237/skyrl/run_pipeline.sh

#    or one arm at a time, on the host:
ARM=M-C LR=1e-5 STEPS=72 CKPT_EVERY=18 bash run_arm.sh
ARM=M-C bash run_eval.sh

# 3. reduce
python summarize_runs.py --logs ~/exp237_logs --out data/
python build_results.py --eval ~/exp237_data/eval --out data/

# tests
python -m pytest skyrl/tests -q
```

## Results

**Full detail in [RESULTS.md](RESULTS.md).** Seven reward designs, nine runs,
**47 scored checkpoints**, every number from #230's scorer unchanged.
R-precision (all), legacy 554, ordered by how far the policy moved:

| checkpoint | KL | consensus | best *ORACLE* | last |
|---|---:|---:|---:|---:|
| **plain, 22 rollouts — the bar** | — | **0.5896** | 0.5680 | — |
| #230 warm start | 0 | 0.5673 | 0.5342 | 0.4566 |
| M-0, lr 0 *(control)* | 0 | 0.5678 | 0.5364 | 0.4594 |
| **M-C step-18** | 0.0072 | 0.5750 | 0.5578 | **0.5267** |
| M-B lr3e-6 step-90 | 0.0087 | 0.5775 | 0.5646 | 0.5091 |
| M-B lr1e-5 step-18 | 0.0088 | 0.5763 | **0.5663** | 0.5108 |
| **M-K step-36** | 0.0162 | **0.5806** | 0.5602 | 0.5178 |
| M-K step-30 | 0.0287 | 0.5803 | 0.5530 | 0.5112 |
| M-FC step-24 | 0.0368 | 0.5717 | 0.5464 | 0.5201 |
| M-F step-36 | 0.0306 | 0.5529 | 0.5189 | 0.5075 |
| M-B step-80 | 0.4863 | 0.3969 | 0.3440 | 0.1905 |

At a **larger** budget — all sections of 8 rollouts pooled, ~54k tokens against
plain-100's ~50k — **M-K reads 0.6098** against plain's **0.6058** (paired
Δ +0.0041, CI [−0.0010, +0.0090] — level, not a win), M-B 0.6054, warm start
0.5992.

- **Primary criterion: NOT met.** Nothing beats 0.5896. The best is **M-K's
  0.5806**, **0.0090** short.
- **Secondary criteria: met** — oracle-best 0.5663 > 0.5342 (M-B), final section
  0.5267 > 0.4566 (M-C) — but **no arm owns more than one**, and the three go to
  three different arms.
- **M-K is the only arm that improves all four aggregation modes with every CI
  excluding zero, on every cut**, including `second_last` (+0.0676) — it improved
  the whole rollout rather than moving quality toward its end.
- Five arms peak at **KL ≈ 0.009** and turn over. **M-K peaks at step 36 on a
  broad plateau** and is the only arm neither killed by a gate nor diverged — it
  ran out of scheduled steps.

## Conclusion

**The hypothesis is half right, and right for a different reason than proposed.**
Moving the reward's unit from *rollouts of a group* to *one rollout's candidate
set* produces what #208 could not — RL checkpoints that improve consensus
R-precision (+0.0133, and +0.0106 after pooling). But the unit that worked is the
**whole rollout**, not the **section**: the arm the issue predicted (M-C,
per-section credit assignment) is beaten on consensus by the arm that just scores
the rollout's own consensus and broadcasts it (M-K). Within-sequence credit
assignment was the hypothesis's mechanism, and it was not needed. What was needed
was for the reward to be *computable on the object the metric scores* — which the
multi format is what makes possible.

It still does not clearly close the gap to independent sampling: **RL brought the
multi format level with plain sampling rather than ahead of it** (M-K pooled
0.6098 vs plain-100's 0.6058, a paired CI that includes zero, where the warm start
was 0.5992).

**#208's negative result is the far end of a dose-response, not a verdict.**
Consensus against distance reads 0.5673 → **0.5806** (KL 0.016) → 0.5529 (0.031)
→ 0.3969 (0.486). Every reward here helps at small KL and damages at large. #208
ran its arms at KL 0.06–0.10 and to 3.96 — past the peak on every one — and its
two arms under KL 0.0015 never moved. **The window it needed lay between the two
learning rates it tried.** Two runs at a 3.3× different rate agree to 0.002 at
matched KL, so this is a property of distance, not of the schedule.

**The most portable finding is about reward design, and it is three-part.**

1. **`E[r] = 0` is necessary and not sufficient.** It constrains the reward's mean
   over the candidates the policy emitted, and says nothing about whether the
   reward's *scale* depends on how many candidates that was. M-C's per-section
   marginal is **366× larger at 1 section than at 22**, so it paid the policy to
   emit a worse answer; centring could not see it, because centring is computed
   *inside* the quantity being gamed. Checkable from a histogram before any run.
2. **Every arm moved its candidate count in the direction its own reward pointed,
   and the gradient's magnitude decided whether it stopped.** M-C's is strongly
   negative (collapsed to 1.1 sections), M-B's strongly positive and *self-paying*
   (grew to ~26 and held), M-F's nearly flat — and **a weak gradient is not a safe
   one, it is an unconstrained one**: M-F ran to 259 sections carrying 1.4 contacts
   each.
3. **Gates specified against the failure you have already seen will miss the next
   one.** All three original criteria are one-sided; M-F failed in a third
   direction that pushed every one of them *away* from its threshold, and none
   fired. The instruments that saw it — per-contact precision and contacts per
   section — were already being reported and simply were not gated on.

**Two results worth carrying beyond this experiment.** *Selection is dominated by
aggregation:* an ORACLE selector of one draft reads 0.5646 where voting the same
drafts reads 0.5750, so a final section should synthesise its predecessors, not
pick among them. And *the spread is the resource:* making candidates more uniform
raises the mean section F1 from 0.432 to 0.532 and **lowers** best-of-22 — the
same trade that defeats every sharpening reward here, showing up in the corpus's
own section-size law.

**What would be worth doing next**, in order:

- **Train M-K further, on a KL leash.** Training longer at a fixed learning rate
  was tested directly and **failed**: M-B at lr 3e-6, resumed from step 120, was
  killed at step 180 with its section count collapsed to 11.0 — its four flat
  evaluations at steps 60–120 were a plateau in *steps* while the policy kept
  travelling in KL (0.0087 → 0.0397). More steps is more distance, and past
  KL ≈ 0.02 there is nothing further along. The version that is not pre-answered
  is to make the KL penalty bind (`kl_loss_coef` 0.001 → 0.05) and ask whether more
  optimisation at a *fixed* distance buys anything. **That was run too, and the
  answer is also no**: at matched KL the leashed arm is uniformly below the
  unleashed one (0.5712 against 0.5806 at KL ~0.016, having taken 3.3× the steps)
  and it never improves on its own first checkpoint. **The path to a distance is
  not interchangeable**, which is a stronger statement than "outcome tracks
  distance". See [RESULTS.md](RESULTS.md#do-long-trajectories-beat-short-ones).
- **Then turn on `beta` and `lam`.** M-K is the `beta = lam = 0` corner of the arm
  derived in [RESULTS.md](RESULTS.md#the-arm-this-implies): a scale-correct
  rollout-level base, plus a **zero-sum** within-rollout shaping term for credit
  assignment, plus M-B's term. The base alone already beat every shaped and blended
  arm, so the shaping term is now a hypothesis to test rather than a fix to apply.
  Note the obvious form of that term, scoring against the causal prefix, was
  **tested and refuted** as a standalone reward: it telescopes, but `token_mean`
  reads the mean, not the sum — it is safe only *inside* a zero-sum shaping slot.
- **Nothing further on M-B at lr 1e-5.** Two learning rates, 10 checkpoints, a
  smooth unimodal peak, and agreement to 0.002 at matched distance.
- **The diversity gap is a corpus question, not an RL one.** One rollout's sections
  cover 658 distinct pairs against 1,065 for 22 independent rollouts, and no reward
  here closed that — the arms that improved diversity did so by emitting less, and
  M-K, the best arm, *lost* union coverage (0.89×) on its way up.
