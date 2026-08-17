# exp237 results — RL on `<contacts-v1.multi>`, section-level rewards

**Issue [#237](https://github.com/Open-Athena/MarinFold/issues/237) · warm start
`plm-exp230-cv1-multi-1_5b-...-a100/hf/step-1988` · 8×A100-80GB · every number
scored by #230's `eval_agg_worker.py` + `score_agg_modes.py`, unchanged**

> **Status: in progress.** Sections are filled in as arms land; anything not yet
> measured says so rather than being left to look measured.

## The bar

R-precision (all), legacy 554, from #230:

| what | R-prec |
|---|---:|
| **plain, 22 rollouts voted** — the budget-matched bar | **0.5896** |
| multi, consensus over ~22 sections — the warm start | 0.5673 |
| multi, oracle best section (ORACLE) | 0.5342 |
| multi, last section | 0.4566 |

**Primary criterion: multi consensus > 0.5896.** Beating the warm start's own
0.5673 is necessary but not sufficient.

## Phase 0 — arm M-C's reward exists, and is not a restatement of F1

Measured offline on #230's own generations — 577 proteins × 8 multi rollouts ×
~22 sections, 4,614 rollouts, no GPU. [`phase0_marginals.py`](phase0_marginals.py).

| quantity | value | what it decides |
|---|---:|---|
| rollouts with a non-degenerate marginal spread | **92.4 %** | **Gate 1.** Below ~50 % and M-C trains mostly on zeros |
| ρ(section marginal, section F1) | **0.337** | **Gate 2.** Near 1.0 and M-C is M-B with extra steps |
| individual section marginals that are exactly 0 | 45.2 % | the honest cost: within a rollout about half the sections change no vote |
| ρ(marginal, section novelty) | −0.087 | it is not novelty either |
| ρ(marginal, section size) | 0.045 | and it is not rewarding volume |
| pooled sd of section marginals | 0.0119 | the scale the unit normalisation removes |

Both gates pass. For context, the same pass reproduces #230's published multi
behaviour on the eval set: 22.21 sections per rollout, mean pairwise Jaccard
0.327, 680.8 union pairs, within-rollout consensus 0.5589, best-section F1 0.5541
against last-section 0.4780.

## The warm start, re-measured inside SkyRL's own vLLM

#230 measured its checkpoint under vLLM 0.27 / transformers 5.15; SkyRL's venv is
vLLM 0.23 / transformers 5.8, a different reader of the same rope config, and a
reader that misses the llama3 scaling loses 0.76 nats/token **silently** (#163's
retraction). [`probe_multi_generation.py`](skyrl/probe_multi_generation.py), 64
rollouts on the actual training prompts:

| | probe | #230, at eval |
|---|---:|---:|
| per-contact precision | **0.4136** | 0.4095 *(over 8.3M rollouts)* |
| sections per rollout | 25.0 | 22.0 |
| union pairs per rollout | 623 | 658 |
| mean pairwise Jaccard | 0.200 | 0.304 |
| generated tokens | 5,149 | ~6,760 |
| sampled ids outside the 2,845 vocabulary | **0** | — |

Precision lands within 0.004 of a number measured over 8.3 million rollouts, so
rope, the tokenizer and the mode marker all survived the port.

## Arm M-0 — the zero-LR control, and the noise floor it exposed

8 steps, lr 0, 64 rollouts a step. `policy_kl` is **0.0000** at every step and
`minibatch_rollout_logprobs_abs_diff_mean` sits at **0.0117 ± 0.0003**, against
#208's healthy unsharded 0.017 and its sharded 1.33. So the unsharded placement
holds and the weight sync is not corrupting the engines.

The control's real value was not the confirmation. **With a policy that did not
change at all**, the per-batch diversity statistics moved like this over eight
batches:

| quantity | min | median | max | max/min |
|---|---:|---:|---:|---:|
| sections per rollout | 17.95 | 23.48 | 29.64 | 1.65× |
| union pairs per rollout | 481.6 | 628.4 | 906.4 | 1.88× |
| votes per pair | 2.06 | 2.43 | 3.49 | 1.69× |
| **mean pairwise Jaccard** | **0.079** | 0.194 | **0.285** | **3.62×** |
| within-rollout consensus R-prec | 0.406 | 0.467 | 0.570 | 1.40× |
| per-contact precision | 0.234 | 0.375 | 0.558 | 2.39× |

A batch is 8 proteins, and these statistics are dominated by *which* 8. This is
the measurement that forced the gates onto rolling medians: the first version
compared a single batch against a single baseline batch, and it had already
recorded a strike against arm M-C at step 6 — a healthy run, KL 0.0012, about to
be killed and reported as #237's preregistered diversity collapse. **The most
expensive wrong answer this experiment could have produced was one the control
caught for 14 minutes of compute.**

It also sets the resolution of everything below: a Jaccard difference smaller
than ~0.1, or a coverage difference smaller than ~25 %, measured on training
batches, is not a finding.

### And the control was scored too — the harness is a no-op to ±0.003

M-0's step-8 checkpoint went through the whole downstream path — FSDP shard to
HF directory, rope repair, bf16, re-generation on 577 proteins, #230's scorer —
and came out where its warm start went in. R-precision (all), legacy 554:

| mode | M-0 step-8 (lr 0) | #230 step-1988 | Δ |
|---|---:|---:|---:|
| consensus | 0.5678 | 0.5673 | +0.0005 |
| best *ORACLE* | 0.5364 | 0.5342 | +0.0022 |
| last | 0.4594 | 0.4566 | +0.0028 |
| second_last | 0.4300 | 0.4284 | +0.0016 |

Everything within 0.003, i.e. at #204's 0.0023 four-replicate noise span. **Every
number in the arms below is therefore attributable to the reward and not to the
export, the cast, the sampler or the scorer** — which is the entire reason a
zero-LR arm is worth a GPU-hour.

## Arm M-C — the arm the hypothesis predicted, and what it actually did

**Stopped at step 26 of 72 on #237's preregistered coverage kill criterion**:
union pairs per rollout fell to 80 % of the warmup median on three consecutive
batches. Terminal KL **0.0173** — an order of magnitude above the ~0.0015 below
which #208 calls an arm untested, so this is a result and not a non-event. The
checkpoint at `global_step_18` is what gets evaluated.

Medians over the first 13 batches against the last 5, so the batch noise measured
on M-0 above is averaged out on both sides:

| quantity | steps 1–13 | steps 22–26 | ratio |
|---|---:|---:|---:|
| **mean pairwise Jaccard** | 0.261 | **0.126** | **0.48×** |
| **votes per pair** | 2.62 | **1.85** | **0.71×** |
| total votes per rollout | 1,887 | 902 | 0.48× |
| generated tokens | 5,687 | 2,734 | 0.48× |
| contacts emitted / true contacts | 11.9 | 6.8 | 0.57× |
| sections per rollout | 20.3 | 15.4 | 0.76× |
| **union pairs per rollout** | 655 | 488 | **0.74×** |
| rollouts emitting `<end>` | 0.58 | **1.00** | 1.73× |
| per-contact precision | 0.424 | 0.299 | 0.71× |

**The reward did the thing it was designed to do, and lost anyway.** Jaccard
halved and votes-per-pair fell 29 %: the sections became genuinely *more*
complementary, which is exactly what a leave-one-out consensus marginal is for
and is the opposite of #208's diversity-collapse mode. Coverage still fell 26 %,
because **total volume collapsed by half**. #208's two failure modes were arm S
(fewer contacts) and arm D v2 (the same contacts every time); M-C is the first,
reached from the opposite direction — and the pair `union pairs` / `votes per
pair`, which #237 mandates reporting, is the only thing that distinguishes them.

### Why volume collapsed, when `E[A] = 0` holds exactly

This is the finding worth carrying forward, and Phase 0 had already measured the
cause without knowing it:

> **45.2 % of section marginals are exactly zero.**

`A_k = (m_k − mean_g) / std_g` centres the **mean**. It says nothing about the
**median**. With a 45 % atom at exactly zero and the remaining mass skewed
positive, `mean_g > 0`, so *every one of those 45 % of sections receives a
negative advantage* — as does every section below the mean among the rest. The
majority of `<begin_statements>` markers and the majority of contact tokens in a
batch are therefore being pushed down, while a minority are pushed up hard enough
to keep the average at zero. Gradient ascent on that shape shrinks what is
emitted, and it does so **without violating the invariant** the reward was
designed around.

So #208's reward-design rule needs a clause:

> `E[r] = p − p̄` is necessary and **not sufficient**. A centred reward with an
> atom at zero holding most of its mass is a shrinking operator on whatever the
> atom is attached to, because the policy gradient follows the median as much as
> the mean.

That is checkable on paper, from a histogram, before a run — the same class of
five-line calculation #208 says would have saved it three training runs, applied
to a distribution's shape rather than to its mean.

### The one thing that improved

`finished` went from 0.58 to **1.00**: by step 22 every rollout closed itself with
`<end>` instead of running into the context limit, and last-section F1 rose 11 %
against a falling best-section F1. The model learned to commit — which is arm
M-F's objective, obtained here as a side effect of shorter sections leaving room
to terminate.

## Arm M-F — the model learns to commit, and it is worth +0.051

**Stopped at step 42 of 72** on the same coverage kill criterion (union pairs to
60 % of the warmup median). Terminal KL **0.0306**.

### What it did during training

Medians, first 13 batches against the last 5:

| quantity | steps 1–13 | steps 38–42 | ratio |
|---|---:|---:|---:|
| **last-section F1** | 0.333 | **0.479** | **1.44×** |
| best-section F1 | 0.475 | 0.497 | 1.05× |
| **best − last gap** | **0.142** | **0.018** | 0.13× |
| sections per rollout | 24.1 | **27.0** | 1.12× |
| mean pairwise Jaccard | 0.170 | **0.065** | 0.38× |
| rollouts emitting `<end>` | 0.52 | **1.00** | 1.94× |
| per-contact precision | 0.403 | 0.491 | 1.22× |
| contacts emitted / true contacts | 11.4 | 6.3 | 0.55× |
| union pairs per rollout | 645 | 360 | 0.56× |

The best-minus-last gap went from 0.142 to **0.018**. Best-section F1 barely
moved: the model did not get better at *producing* a good candidate, it got
better at **ending on one**. That is precisely what arm M-F was chartered to ask.

### Evaluation — 577-unit universe, #230's scorer, unchanged

R-precision (all), paired per protein against the #230 warm start, 10,000
bootstrap resamples:

| cut | mode | M-F step-36 | #230 | Δ | 95 % CI | win/loss |
|---|---|---:|---:|---:|---|---|
| legacy554 | **last** | **0.5075** | 0.4566 | **+0.0509** | [+0.0433, +0.0584] | 379/170 |
| legacy554 | consensus | 0.5529 | 0.5673 | −0.0144 | [−0.0176, −0.0111] | 182/363 |
| legacy554 | best *ORACLE* | 0.5189 | 0.5342 | −0.0154 | [−0.0199, −0.0110] | 196/355 |
| legacy554 | second_last | 0.2649 | 0.4281 | −0.1631 | [−0.1784, −0.1479] | 78/467 |
| **eval2-natural** | **last** | **0.2232** | 0.1696 | **+0.0535** | [+0.0379, +0.0690] | 63/15 |
| eval2-natural | consensus | 0.2742 | 0.2889 | −0.0147 | [−0.0254, −0.0054] | 29/47 |
| eval2 | last | 0.4419 | 0.3781 | +0.0637 | [+0.0530, +0.0750] | 219/84 |
| eval2 | consensus | 0.4874 | 0.5029 | −0.0155 | [−0.0203, −0.0107] | 106/197 |

Every one of these excludes zero. **Final-section R-precision beats #237's
secondary criterion (> 0.4566) by 0.051** on the legacy 554 and by 0.054 on
eval2-natural — a 32 % relative gain on the low-homology cut, winning on 63 of 78
proteins. It closes **66 %** of the 0.078 selection headroom #230 identified, and
M-F's own last is now within **0.011** of its own oracle best.

`second_last` falling to 0.2649 is the same finding from the other side: the model
now treats every non-final section as scratch and the final one as the answer.

### And it is cheaper

Measured on the eval generations themselves, per rollout:

| | #230 step-1988 | M-F step-36 |
|---|---:|---:|
| generated tokens | 6,825 | **3,697** (0.54×) |
| sections | 22.20 | **23.97** (1.08×) |
| contacts emitted | 2,264 | 1,199 (0.53×) |
| rollouts emitting `<end>` | 0.677 | **0.966** |

**46 % fewer tokens, 8 % more candidates, and a final section 0.051 better.** The
format did not collapse — sections went *up* — and the candidates became more
complementary, not less (Jaccard −62 % in training).

### The trade, stated plainly

The vote lost 53 % of its mass and 44 % of its coverage, and consensus paid
0.014 for it. Every individual candidate improved and the deployable
single-candidate number gained 0.051. **Both arms bought the same thing with the
same currency**: selectivity, priced in vote coverage.

## Arm M-B

_Re-running: its first attempt died 80 s after the previous stage's vLLM engines
exited, on a teardown race that reports itself as "Engine core initialization
failed"._

## Evaluation

_Pending._
