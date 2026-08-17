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

## Arms M-C / M-F / M-B

_Pending._

## Evaluation

_Pending._
