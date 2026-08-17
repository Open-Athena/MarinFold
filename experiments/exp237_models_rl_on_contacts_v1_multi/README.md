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
| **M-C** | `m_k = C(all) − C(all \ {k})`, section *k*'s marginal contribution to its **own rollout's** consensus | per-section, dense | `contacts_section` |
| **M-F** | `F1(last section)` | one scalar per rollout | `grpo` |
| **M-B** | `max_k F1(section k)` — **ORACLE** | one scalar per rollout | `grpo` |
| **M-0** | M-C's reward at **lr = 0** | — | `contacts_section` |

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
| estimator | `contacts_section` (M-C, M-0) / `grpo` (M-F, M-B) |
| cost | ~103 s/step; 26–80 steps per arm on 8 × A100-80GB |

**Evaluation**

| | |
|---|---|
| targets | the **577-unit universe** (#89's 554 + #226's 23), `eval577_targets.parquet`, both of #89's filters (`MIN_SEP 6` **and** `MIN_DEG 0.001`) |
| generation | #230's `eval_agg_worker.py --mode multi`, **8 rollouts/protein**, full context, T 1.0 / top-p 0.95 / top-k −1, 8 GPUs |
| scoring | #230's `score_agg_modes.py`, which calls exp89's `compute_metrics` via `score_gate_a.metrics_for` — **imported, not re-derived** |
| cuts | legacy554 (554), eval2 (307), **eval2-natural (78)**, eval2 <30 % (275) |
| significance | paired per-protein bootstrap, 10,000 resamples, seed 237 (`compare_arms.py`) |
| validation | the pipeline reproduces #230's published table exactly (0.5673 / 0.5342 / 0.4566 / 0.4284), and the lr-0 control returns within ±0.003 |

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

**Full detail in [RESULTS.md](RESULTS.md).** R-precision (all), legacy 554,
ordered by how far the policy moved:

| checkpoint | KL | consensus | best *ORACLE* | last |
|---|---:|---:|---:|---:|
| **plain, 22 rollouts — the bar** | — | **0.5896** | 0.5680 | — |
| #230 warm start | 0 | 0.5673 | 0.5342 | 0.4566 |
| M-0, lr 0 | 0 | 0.5678 | 0.5364 | 0.4594 |
| **M-C step-18** | 0.0072 | **0.5750** | **0.5578** | **0.5267** |
| M-F step-18 | 0.0136 | 0.5647 | 0.5283 | 0.4949 |
| M-B step-36 | 0.0163 | 0.5741 | 0.5574 | 0.4908 |
| M-F step-36 | 0.0306 | 0.5529 | 0.5189 | 0.5075 |
| M-B step-80 | 0.4863 | 0.3969 | 0.3440 | 0.1905 |

- **Primary criterion: NOT met.** Nothing beats 0.5896. M-C step-18 comes within
  **0.0146**, the closest a multi-mode number has come, and is still short.
- **Both secondary criteria: met** — final-section 0.5267 > 0.4566, oracle-best
  0.5578 > 0.5342 — but by **arm M-C**, not by the arms designed for them.
- M-C step-18 improves every mode on every cut, all paired CIs excluding zero:
  consensus +0.0077, oracle-best +0.0235, last **+0.0701** (480 wins / 69 losses).
  On eval2-natural, last goes 0.1696 → **0.2421**.
- Gates: all three arms tripped the preregistered coverage criterion. It was in
  the wrong units — see below.

## Conclusion

**The hypothesis is half right, and the half that is right is the interesting
half.** Moving the reward's unit from *rollouts of a group* to *sections of one
rollout* does produce what #208 could not: **an RL checkpoint that improves
consensus R-precision** (+0.0077, and +0.0109 on eval2-natural). Arm M-C, the arm
#237 singled out, is the best checkpoint on every aggregation mode. But it does
not close the gap to independent sampling: 22 plain rollouts still beat one
rollout's 22 sections, 0.5896 against 0.5750.

**#208's negative result is the far end of a dose-response, not a verdict.**
Consensus against distance moved reads 0.5673 → **0.5750** (KL 0.007) → 0.5741
(0.016) → 0.5529 (0.031) → 0.3969 (0.486). Every reward here helps at small KL and
damages at large. #208 ran its arms at KL 0.06–0.10 and to 3.96 — past the peak on
every one — and its two arms under KL 0.0015 were too small to move at all. **The
window it needed lay between the two learning rates it tried.** That is the most
portable thing this experiment produced, and it is a statement about *how far*
rather than about *what reward*.

**Reward shape decides how a run fails, not whether.** M-C and M-F halve the
contacts emitted while becoming *more* diverse; M-B holds the volume and emits the
same contacts 1.7× more often. Both routes end at the same place. #208 found these
two modes across different reward families; here they are produced deliberately,
by reward shape, on one model and one data order.

**Two things for the next reward design, both cheap and both learned the
expensive way:**

1. **A per-candidate reward must be scale-free in the number of candidates, and
   centring cannot enforce that.** Arm M-C pays each section its leave-one-out
   contribution to its own rollout's consensus — a quantity that grows as the
   rollout emits fewer sections, because each survivor is then more load-bearing.
   Measured by truncating real rollouts: the reward per section is **366× larger
   at 1 section than at 22**, while the rollout's own consensus *falls* from
   0.5431 to 0.3413. In groups differing in nothing but section count, a
   one-section rollout receives **+4.80** advantage and a 22-section rollout
   **−0.22**. `E[A] = 0` holds exactly throughout — centring is computed *within*
   the quantity being gamed, so it cannot see the problem. This explains
   everything M-C did, including why the collapse accelerates (the payoff for
   shortening grows as sections disappear) and why M-F and M-B, which have no
   such term, kept or increased their section counts instead.

   The check is cheap and belongs before any run: truncate a few real rollouts,
   plot the reward against the number of candidates, look for a slope.
2. **Gate on `union/R`, not on union relative to the run's own start** — and gate
   on precision too. The preregistered coverage criterion stopped all three arms;
   union/R never left 2.8–4.6 in any of them, including the one that collapsed to
   0.3969, whose union/R was *higher* than the warm start's. Coverage was never the
   binding constraint. Per-contact precision (0.50 → 0.14) was.

**What would be worth doing next**, in order:

- **An LR/step sweep around the peak.** Every arm's best checkpoint was its
  earliest. The interesting region is KL 0.005–0.02, which this run sampled with
  three points by accident. A proper sweep is cheap and is the only obvious route
  to the remaining 0.0146.
- **Fix M-C's reward before re-running it.** The scale pathology has a clean fix:
  compute each section's marginal against a **fixed-size** reference — subsample
  the rollout's sections to a constant *G* before the leave-one-out — so the
  reward no longer depends on how many the rollout emitted. Equivalently,
  decompose a rollout-level `C(all)` advantage across its sections rather than
  scoring each section independently. Arm M-C's 0.5750 was obtained *despite* this
  term, in the 18 steps before it took hold; the fixed version is the obvious
  thing to test next.
- **A rank-transformed marginal**, separately: 55 % of section marginals are an
  atom at exactly zero, which is why 7.6 % of rollouts contribute no gradient at
  all. A second-order fix, not the main one.
- **The diversity gap is a corpus question, not an RL one.** #230 said this and
  this run does not contradict it: one rollout's sections cover 658 pairs against
  1,065 for 22 independent rollouts, and no reward here closed that — the two that
  improved diversity did so by emitting less.
