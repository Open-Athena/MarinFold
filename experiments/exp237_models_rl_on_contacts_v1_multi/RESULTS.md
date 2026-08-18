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
| individual section marginals that are exactly 0 | 45.2 % per rollout, **54.9 % pooled** | the honest cost: more than half of all sections change no vote |
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

## The whole result, ordered by how far the policy moved

R-precision (all), legacy 554, every checkpoint scored by #230's
`eval_agg_worker.py` + `score_agg_modes.py` unchanged, 8 rollouts × 577 proteins:

| checkpoint | KL | consensus | best *ORACLE* | last | second_last | union/R |
|---|---:|---:|---:|---:|---:|---:|
| **plain, 22 rollouts — the bar** | — | **0.5896** | 0.5680 | — | — | — |
| #230 warm start | 0 | 0.5673 | 0.5342 | 0.4566 | 0.4284 | 3.98 |
| M-0, lr 0 | 0 | 0.5678 | 0.5364 | 0.4594 | 0.4300 | 3.98 |
| **M-B lr3e-6 step-90** | 0.0087 | **0.5775** | 0.5646 | 0.5091 | — | — |
| M-B step-18 | 0.0088 | 0.5763 | **0.5663** | 0.5108 | 0.5027 | — |
| **M-C step-18** | 0.0072 | 0.5750 | 0.5578 | **0.5267** | 0.4795 | 3.17 |
| M-F step-18 | 0.0136 | 0.5647 | 0.5283 | 0.4949 | 0.3464 | 3.69 |
| M-B step-36 | 0.0163 | 0.5741 | 0.5574 | 0.4908 | **0.4933** | 2.76 |
| M-F step-36 | 0.0306 | 0.5529 | 0.5189 | 0.5075 | 0.2649 | 2.80 |
| M-B step-80 | 0.4863 | 0.3969 | 0.3440 | 0.1905 | 0.2469 | 4.63 |

eval2-natural (78 proteins, the honest low-homology readout):

| checkpoint | consensus | last |
|---|---:|---:|
| #230 warm start | 0.2889 | 0.1696 |
| **M-C step-18** | 0.2998 | **0.2421** |
| **M-B step-36** | **0.3040** | 0.2329 |
| M-F step-36 | 0.2742 | 0.2232 |

### Every checkpoint, paired against the warm start

**legacy 554** — Δ against the #230 warm start, paired per protein:

| checkpoint | consensus | best *ORACLE* | last | second_last |
|---|---|---|---|---|
| M-C step-18 | +0.0077 \*<br><sub>327/213</sub> | +0.0235 \*<br><sub>425/120</sub> | +0.0701 \*<br><sub>480/69</sub> | +0.0511 \*<br><sub>486/61</sub> |
| M-B step-36 | +0.0068 \*<br><sub>301/246</sub> | +0.0232 \*<br><sub>409/142</sub> | +0.0341 \*<br><sub>389/161</sub> | +0.0652 \*<br><sub>464/82</sub> |
| M-F step-36 | -0.0144 \*<br><sub>182/363</sub> | -0.0154 \*<br><sub>196/355</sub> | +0.0509 \*<br><sub>379/170</sub> | -0.1631 \*<br><sub>78/467</sub> |
| M-F step-18 | -0.0026 \*<br><sub>231/306</sub> | -0.0059 \*<br><sub>238/308</sub> | +0.0383 \*<br><sub>378/165</sub> | -0.0817 \*<br><sub>114/432</sub> |
| M-B step-80 | -0.1704 \*<br><sub>49/503</sub> | -0.1903 \*<br><sub>39/514</sub> | -0.2661 \*<br><sub>76/474</sub> | -0.1797 \*<br><sub>85/460</sub> |
| M-0 step-8 (lr 0) | +0.0006<br><sub>260/259</sub> | +0.0021 \*<br><sub>297/225</sub> | +0.0028<br><sub>273/260</sub> | +0.0016<br><sub>272/257</sub> |

**eval2-natural (78)** — Δ against the #230 warm start, paired per protein:

| checkpoint | consensus | best *ORACLE* | last | second_last |
|---|---|---|---|---|
| M-C step-18 | +0.0109 \*<br><sub>49/27</sub> | +0.0273 \*<br><sub>64/12</sub> | +0.0725 \*<br><sub>71/7</sub> | +0.0355 \*<br><sub>70/8</sub> |
| M-B step-36 | +0.0151 \*<br><sub>56/22</sub> | +0.0319 \*<br><sub>62/15</sub> | +0.0633 \*<br><sub>65/13</sub> | +0.0493 \*<br><sub>69/9</sub> |
| M-F step-36 | -0.0147 \*<br><sub>29/47</sub> | -0.0156 \*<br><sub>28/50</sub> | +0.0535 \*<br><sub>63/15</sub> | -0.0375 \*<br><sub>21/56</sub> |
| M-F step-18 | -0.0035<br><sub>27/45</sub> | -0.0063<br><sub>32/45</sub> | +0.0263 \*<br><sub>52/23</sub> | -0.0255 \*<br><sub>23/55</sub> |
| M-B step-80 | -0.0664 \*<br><sub>12/65</sub> | -0.0716 \*<br><sub>9/69</sub> | -0.0767 \*<br><sub>26/51</sub> | -0.0403 \*<br><sub>15/61</sub> |
| M-0 step-8 (lr 0) | +0.0002<br><sub>41/32</sub> | +0.0026<br><sub>46/29</sub> | +0.0046<br><sub>40/34</sub> | +0.0010<br><sub>40/34</sub> |

\* the 95 % paired bootstrap CI excludes zero (10,000 resamples, seed 237).
Cell subscript is wins/losses over proteins. **M-0's consensus CI includes zero
and its record is 260/259** — the harness is a coin flip against its own input,
which is what makes every other row readable.

### Four things this says

**1. The primary criterion is not met.** Nothing beats 0.5896, the budget-matched
plain baseline. M-C step-18 gets within **0.0146** of it, which is the closest any
multi-mode number has come, and it is still short. *At matched sampling budget,
22 independent rollouts remain better than one rollout's 22 sections.*

**2. Both secondary criteria are met, and by the wrong arm.** M-F's target was
final-section R-precision > 0.4566: met at 0.5075 — but **M-C reaches 0.5267**,
higher than the arm designed for it. M-B's target was oracle-best > 0.5342: met at
0.5574 — but **M-C reaches 0.5578**. The arm #237's hypothesis singled out is the
best checkpoint on every mode, which is the one prediction in the issue that came
out right.

**3. #208's result is the far end of a dose-response, not a verdict.** Consensus
against distance moved: 0.5673 at KL 0, **0.5750 at 0.007**, 0.5741 at 0.016,
0.5529 at 0.031, 0.3969 at 0.486. Every arm improves consensus at small KL and
damages it at large. #208 ran its arms at KL 0.06–0.10 and to 3.96, i.e. past the
peak on every one — and its two arms that stayed under 0.015 (C v1 at 0.0004, D v1
at 0.0014) were too small to move at all. **The window it needed was between the
two learning rates it tried.**

**4. Reward shape decides *how* a run fails, not *whether*.** Training medians,
first 13 batches against the last 5 before each arm was stopped:

| arm | total votes | votes/pair | Jaccard | what it did |
|---|---:|---:|---:|---|
| M-C | **0.52×** | 0.75× | 0.60× | halve the contacts, become **more** diverse |
| M-F | **0.51×** | 0.89× | 0.44× | halve the contacts, become **more** diverse |
| M-B | 0.97× | **1.73×** | **1.48×** | hold the contacts, emit the **same** ones |

That is exactly what each reward asks for. M-B pays for the *best* section, so the
optimal policy finds its best mode and repeats it — nothing pays for being
different. M-C pays for marginal contribution, so being different pays. M-F pays
for the last section, so the earlier ones become scratch (its `second_last` falls
to 0.2649). #208 found these two modes across different reward *families*; here
they are produced deliberately by reward *shape*, on one model and one data order.

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

### Why M-C collapsed: the reward is not scale-free in the section count

**This is the mechanism, and it is a design flaw in arm M-C's reward rather than
a property of RL.** `m_k = C(all) − C(all \ {k})` measures what section *k*
contributes to its own rollout's consensus — and that quantity depends on **how
many sections there are to contribute against**. With 22 sections, removing one
barely moves an integer vote count and `m_k ≈ 0`. With two, removing one halves
the vote. With one, `C(all \ {k})` is the consensus of nothing.

So a rollout can raise the reward on *every one of its sections* simply by
**emitting fewer of them** — by making its own consensus worse, so each surviving
section is more load-bearing. Group centring does not remove this, because the
group is the prompt's rollouts and the *shorter* rollout is the one above the
mean.

Measured directly, by truncating #230's real rollouts to a controlled number of
sections and re-running the reward
([`analyze_section_count_incentive.py`](analyze_section_count_incentive.py),
544 rollouts / 128 synthetic groups):

| sections emitted | mean `m_k` | × vs 22 | the rollout's own consensus | **group-centred advantage** |
|---:|---:|---:|---:|---:|
| 1 | 0.33547 | **366×** | 0.3413 | **+4.80** |
| 2 | 0.07053 | 77× | 0.4048 | +1.08 |
| 4 | 0.01916 | 21× | 0.4577 | +0.21 |
| 8 | 0.00668 | 7.3× | 0.5049 | −0.02 |
| 16 | 0.00122 | 1.3× | 0.5336 | −0.18 |
| 22 | 0.00092 | 1.0× | 0.5431 | **−0.22** |

![the section-count incentive](plots/section_count_incentive.png)

The right-hand column is the decisive one. Those are groups constructed to differ
in **nothing but section count**, centred exactly as `centred_section_advantages`
does it. A rollout that emits one section receives **+4.80** on it; a rollout that
emits 22 receives **−0.22** on each of them. The reward pays, enormously, for
producing a worse answer.

**Everything M-C did follows from that one table:**

- Section count falls (0.89× by step 26) rather than holding.
- The fall **accelerates**: the payoff for shortening *grows* as sections
  disappear — 7× at 8 sections, 21× at 4, 366× at 1 — so it is a positive
  feedback loop, not a drift. Measured, over global steps 28→33: 13.7 → 11.4 →
  4.5 → 2.2 → 2.3 → **1.1**.
- The terminal state is ~1 section, which is the pathology's global optimum and
  also the destruction of the format the experiment exists to use.
- **M-F does not do this** (sections 1.14×) and **M-B does the opposite**
  (1.31×), because neither reward has a term whose magnitude depends on the
  section count. The divergence between the arms is explained by the term only
  M-C has.

**Why the observational test missed it.** Correlating section count against
marginal across #230's own generations gives ρ = **−0.04**: the base model fills
the context every time, so 95 % of rollouts sit within ±5 % of their group's
median section count and there is almost nothing to correlate. The incentive is
**latent** — invisible in the base distribution, and precisely what gradient
ascent goes looking for. It had to be measured by intervention, not observation.

#### The clause this adds to #208's reward-design rule

> `E[r] = 0` constrains the reward's **mean over the candidates the policy
> emitted**. It says nothing about whether the reward's *scale* depends on **how
> many candidates that was**. A per-candidate reward whose magnitude grows as the
> candidate set shrinks is an instruction to shrink the candidate set, and
> centring cannot see it because centring is computed *within* the very quantity
> being gamed.

Checkable before any run, and cheaply: truncate a handful of real rollouts, plot
the reward against the number of candidates, and look for a slope. The table
above took ten minutes of CPU on generations that already existed.

#### What was written here before, and why it was wrong

An earlier version of this document blamed the collapse on the reward's **atom at
zero** — 54.9 % of sections change no vote, and those sections average −0.062
after centring ([`plots/reward_shape.png`](plots/reward_shape.png)). That
measurement is real and is kept, because it explains why 7.6 % of rollouts
contribute no gradient at all. But it was **refuted as the cause** by arm M-F,
whose reward is a continuous scalar with no atom and which reduced *volume* by
the same factor — and it never explained the fact that most needed explaining,
which is that only M-C lost **sections**. The scale pathology explains both.

### The peak, resolved — and it is the same phenomenon

M-C was resumed from its own step-18 checkpoint under the corrected gate, with a
checkpoint every 4 steps, to find where the dose-response turns over. R-precision
(all), legacy 554:

| global step | KL | consensus | best *ORACLE* | last | **sections/rollout** |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.0072 | **0.5750** | 0.5578 | 0.5267 | 17.9 |
| 20 | ~0.008 | 0.5739 | 0.5547 | 0.5261 | 16.5 |
| 24 | ~0.011 | 0.5484 | 0.5325 | 0.5146 | 11.0 |
| 32 | ~0.023 | 0.4576 | 0.4585 | 0.4578 | **1.1** |

**The peak is step 18, and the decline is the section count.** Consensus tracks
sections almost exactly, and by step 32 the three aggregation modes have
*converged on the same number* — 0.4576 — because with 1.09 sections per rollout
there is nothing left to aggregate. The "dose-response" and the scale pathology
are not two findings; the dose is how long the pathology has had to act.

Training stopped on the sections gate (`median 3.39 < 12`) — #237's own
preregistered criterion, firing on a genuine collapse this time, which is the
clearest available evidence that the gate correction was the right one.

### Where is the headroom in oracle-best? Two levers, one of them refuted

Arm M-B's target is `max_k F1(section k)`, and two arms have already moved it —
M-B step-36 to 0.5574 and M-C step-18 to 0.5578, from #230's 0.5342. A maximum
over a sample has exactly two levers, **how many draws** and **what distribution
they come from**, so both are priced offline on #230's own generations
([`analyze_oracle_headroom.py`](analyze_oracle_headroom.py), 2,416 rollouts).

**Lever 1 — more candidates. Not saturating.**

| candidates | 1 | 2 | 4 | 8 | 16 | 22 |
|---|---:|---:|---:|---:|---:|---:|
| E[max F1] | 0.4269 | 0.4770 | 0.5132 | 0.5411 | 0.5643 | 0.5739 |

About **+0.022 F1 per doubling**, still climbing at 22 — which is where the
8,192-token context runs out. This is a real lever and it is *not* an RL lever:
the cheapest way to get more draws is more rollouts, which is exactly #230's
finding that 22 independent rollouts beat one rollout's 22 sections.

**Lever 2 — a better distribution. Tested and refuted.** Per-section F1 against
section size is strongly **non-monotone**: it peaks at ~80 contacts and collapses
for small sections, of which #230's power-law size draw makes a great many.

| section size (median) | 8 | 46 | 67 | **80** | 96 | 117 | 149 | 225 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean F1 | 0.157 | 0.342 | 0.479 | **0.552** | 0.506 | 0.480 | 0.410 | 0.476 |

**36.6 % of sections are under 73 contacts, with mean F1 0.319.** The obvious
conclusion is that uniform ~80-contact sections would raise the ceiling. Paired
on the same 271 proteins, they do the opposite for anything but a single draw:

| candidates | 1 | 2 | 4 | 8 | 16 | 22 |
|---|---:|---:|---:|---:|---:|---:|
| all sections | 0.4226 | 0.4652 | 0.4983 | 0.5243 | 0.5453 | 0.5536 |
| in-band only | 0.4388 | 0.4720 | 0.4980 | 0.5206 | 0.5375 | 0.5434 |
| gain | **+0.0162** | +0.0068 | −0.0002 | −0.0037 | −0.0078 | **−0.0102** |

Restricting to the good size band raises the **mean** section F1 from 0.432 to
0.532 and **lowers** E[max of 22]. The variance that makes the average candidate
worse is precisely what best-of-N feeds on — the same trade this experiment keeps
finding, now in the corpus's section-size law rather than in a reward. It also
says something useful about *deployment*: size-uniformity is worth +0.016 if you
are going to read one candidate, and a loss if you are going to aggregate.

*Caveat:* this re-samples sections that were generated **under** the power law.
It shows that selecting for size does not help; it does not prove that a model
retrained to emit uniform sections would fail.

### Continuing M-F: it diverges, and all three gates miss it

M-F was the one arm whose own reward was still climbing when its first run
stopped, so it was resumed from step 36 and given 84 more steps. It reached
terminal KL **3.2568** — the scale of #208's diverged arm C v4 (3.96) — and
**tripped no gate at any point**.

Training, `last_f1` being M-F's own reward (global step = local batch + 36):

| step | 42 | **48** | 54 | 66 | 72 | 78 | 96 | 120 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sections/rollout | 25.6 | 24.8 | 36.5 | 49.3 | 93.4 | 166 | 191 | 146 |
| **last_f1 (the reward)** | 0.538 | **0.608** | 0.442 | 0.361 | 0.418 | 0.120 | 0.029 | 0.006 |
| per-contact precision | 0.495 | 0.551 | 0.396 | 0.346 | 0.265 | 0.113 | 0.069 | 0.169 |
| total votes | 829 | 429 | 658 | 382 | 519 | 736 | 653 | 371 |

The reward peaks at step 48 and then the policy discovers it can emit
**unboundedly many nearly-empty sections** — 259 of them at the worst, carrying
~370 contacts between them, i.e. **1.4 contacts per section**. Evaluated:

| M-F step | consensus | best *ORACLE* | last | sections |
|---:|---:|---:|---:|---:|
| 36 *(first run)* | 0.5529 | 0.5189 | **0.5075** | 24.2 |
| 60 | 0.5157 | 0.4989 | 0.4696 | 50.4 |
| 84 | 0.3974 | 0.1993 | 0.0928 | 181.5 |
| 120 | 0.3758 | 0.0695 | 0.0064 | 147.2 |

#### Why *this* reward produced *that* behaviour

The section count is a direction in policy space, and each reward has a gradient
along it. Measured the same way M-C's scale bug was — truncate real rollouts to a
controlled section count and re-run each reward
([`analyze_reward_vs_count.py`](analyze_reward_vs_count.py), 120 rollouts):

| the arm's own reward | 1 | 2 | 4 | 8 | 16 | 22 | direction |
|---|---:|---:|---:|---:|---:|---:|---|
| **M-B** `max_k F1` | 0.3762 | 0.4434 | 0.4712 | 0.5135 | 0.5483 | 0.5582 | **rises**, +48 % |
| **M-F** `F1(last)` | 0.3762 | 0.3914 | 0.3797 | 0.4177 | 0.4194 | 0.4267 | **~flat**, +13 % |
| **M-C** `mean m_k` | 0.3382 | 0.0740 | 0.0195 | 0.0075 | 0.0016 | 0.0003 | **falls**, 1000x |

As group-centred advantage, in groups differing in nothing but section count:

| | 1 | 2 | 4 | 8 | 16 | 22 | what the arm did |
|---|---:|---:|---:|---:|---:|---:|---|
| M-B | −1.09 | −0.49 | −0.22 | +0.31 | +0.69 | **+0.81** | grew to ~26, **stable** |
| M-F | −0.28 | −0.16 | −0.21 | +0.12 | +0.21 | **+0.31** | **ran away to 259** |
| M-C | **+1.59** | +0.17 | −0.26 | −0.42 | −0.54 | −0.54 | collapsed to **1.1** |

![each reward against section count](plots/reward_vs_section_count.png)

**Every arm moved its section count in the direction its own reward pointed, and
the magnitude of the gradient set whether it stopped.** M-C's gradient is huge and
negative, so it ran to the floor. M-B's is large and positive, so it grew — and
kept being *paid* for growing, which is a restoring force: the reward keeps
tracking the thing the policy is doing. M-F's is the weakest of the three, and
that is exactly the problem.

**A weak gradient is not a safe one; it is an unconstrained one.** Two properties
combine in M-F and in neither of the others:

1. *There is a mild upward pull.* `F1(last)` rises slightly with section count —
   not because more sections are better, but because the model's later sections
   are genuinely better than its earlier ones (#230's "the model treats its final
   section as a commitment"). So emitting one more section is weakly rewarded.
2. *Nothing pays for the quality of any section except the last.* Under `max_k`,
   every section is a lottery ticket for the maximum, so improving **any** of them
   can pay and degrading them all must cost. Under `F1(last)`, sections 1..K−1 are
   invisible to the reward. They are free space.

Together: adding sections is weakly encouraged, and keeping them good is not
encouraged at all. The policy takes the free direction — more sections, each
cheaper — until they carry **1.4 contacts each**, at which point the final section
has degraded with everything else and the reward collapses with it. `second_last`
falling to 0.2649 in M-F's *first* run was this mechanism already visible at step
36; the continuation just let it run.

That is also why M-F looked like the most promising arm from its reward curve. A
rising reward with a weak gradient in an unconstrained direction is exactly what a
slow, healthy improvement looks like right up until it isn't.

#### The gates were built against the failures we had already seen

This is the methodological finding, and it is the third time in this experiment
that a criterion turned out to be specified against the *known* failure rather
than the failure *space*:

| gate | what it catches | M-F's value | fired? |
|---|---|---:|---|
| `min_sections` ≥ 12 | too **few** sections — arm M-C's collapse | 146–259 | no |
| `max_jaccard` ≤ 0.45 | too **similar** sections — arm M-B's collapse | 0.003 | no |
| `min_union_over_r` ≥ 1.25 | lost vote coverage | ~1.6 | no |

Every one is one-sided in the direction of a failure already observed. M-F failed
in a **third** direction — many, tiny, mutually disjoint sections — which pushes
each gated quantity *away* from its threshold. Jaccard near zero and 259 sections
look, to these gates, like a maximally healthy run.

The instruments that did see it were already being reported and were not gated
on: **per-contact precision** (0.55 → 0.07) and **contacts per section**
(33 → 1.4). Both belong in the gate set, and a `max_sections` ceiling is the
cheap direct fix.

### Arm M-BC — the blend is worse than M-B alone, at every distance

`A_i = GRPO(max_k F1)_i + 1.0 * GRPO(C_i(all))_i`, 48 steps at lr 1e-5, **zero
gate strikes**. R-precision (all), legacy 554:

| step | KL | consensus | best *ORACLE* | last | sections |
|---:|---:|---:|---:|---:|---:|
| 12 | 0.0024 | 0.5735 | 0.5543 | 0.4946 | 20.6 |
| 24 | 0.0107 | 0.5646 | 0.5538 | 0.5086 | 17.3 |
| 36 | 0.0246 | 0.5616 | 0.5473 | 0.4918 | 16.5 |
| 48 | 0.0626 | 0.5504 | 0.5258 | 0.3775 | 14.5 |

Head to head against **M-B alone** — which is the same reward at `lam = 0` — at
each arm's own best checkpoint, paired per protein:

| M-BC step-12 vs M-B lr3e-6 step-90 | Δ | 95 % CI | win/loss |
|---|---:|---|---|
| consensus | **−0.0040** | [−0.0070, −0.0011] | 234/304 |
| best *ORACLE* | **−0.0103** | [−0.0133, −0.0074] | 182/363 |
| last | **−0.0145** | [−0.0187, −0.0102] | 191/354 |

**Adding the consensus term made every number worse, and all three CIs exclude
zero.** The comparison is not confounded by distance: M-BC's step-12 sits at KL
0.0024 and step-24 at 0.0107, bracketing M-B's optimum of 0.0087, and *both* are
below M-B's 0.5775. Nor is it the scale pathology — `C_i(all)` used as a
rollout-level scalar is scale-correct by construction, and the run never tripped
a gate or lost its section count the way arm M-C did.

**The two objectives compete rather than compose.** `max_k F1` is maximised by
concentrating quality into one excellent section; `C(all)` is maximised by
spreading coverage across many complementary ones. At `lam = 1` the blend lands
between them and is beaten by either endpoint — M-B on every mode here, and M-C
on final-section (0.5267) elsewhere.

**What this does and does not license.** One `lam`, one learning rate,
checkpoints every 12 steps: a smaller `lam` would sit closer to M-B and would
presumably recover most of the difference. What the run rules out is the
*motivating* claim — that the two terms counteract each other's failure modes and
so beat either alone. They do not, at the one weight where "equal" is
well-defined. Chasing `lam` downward to approach an endpoint we have already
measured is not worth a night of GPU time.

### The slow walk: the peak is a function of distance, not of learning rate

M-B re-run at **lr 3e-6** — a third the step size, 120 steps, eight checkpoints
across the window the 1e-5 run crossed in about a dozen. It is the only arm that
**never tripped a gate**. R-precision (all), legacy 554:

| step | KL | consensus | best *ORACLE* | last | sections |
|---:|---:|---:|---:|---:|---:|
| 30 | 0.0018 | 0.5713 | 0.5511 | 0.4719 | 21.3 |
| 60 | 0.0047 | 0.5754 | 0.5613 | 0.4989 | 20.5 |
| 75 | 0.0079 | 0.5760 | 0.5627 | 0.5044 | 20.4 |
| **90** | **0.0087** | **0.5775** | **0.5646** | **0.5091** | 20.2 |
| 120 | 0.0259 | 0.5739 | 0.5568 | 0.5072 | 19.3 |

Smooth, unimodal, and peaked at **KL 0.0087**. Paired against the warm start,
step-90 reads consensus **+0.0102** [+0.0072, +0.0133] (353/187), oracle-best
**+0.0304**, last **+0.0524** — every CI excluding zero, and the **best consensus
measured anywhere in this experiment**.

**The result that makes this worth the GPU-hours** is not the +0.0012 over the
1e-5 run. It is that the two runs *agree*:

| at matched KL ~0.0087 | consensus | best | last |
|---|---:|---:|---:|
| M-B, lr 1e-5, step 18 (KL 0.0088) | 0.5763 | 0.5663 | 0.5108 |
| M-B, lr 3e-6, step 90 (KL 0.0087) | 0.5775 | 0.5646 | 0.5091 |
| difference | 0.0012 | 0.0017 | 0.0017 |

Two runs at a **3.3x different learning rate**, reaching the same distance by
different paths, land within 0.002 of each other on all three modes — at or below
#204's 0.0023 noise floor. **The outcome is a function of how far the policy
moved, not of the rate it moved at or of where the checkpoints happened to fall.**
That retires the last alternative explanation for the dose-response: it is not a
checkpoint-spacing artifact.

It also settles a question this experiment kept asking. The 0.5763 that arm M-B
hit at step 18 was a single checkpoint from a coarsely-sampled run, and could
have been luck. It was not.

#### One multi rollout's best section has caught the 22-rollout oracle

| oracle-best, legacy 554 | |
|---|---:|
| plain, **22 rollouts** | 0.5680 |
| M-B lr 1e-5 step-18, **one rollout** | 0.5663 |
| M-B lr 3e-6 step-90, **one rollout** | 0.5646 |

Within **0.002–0.003**. Whatever a perfect selector could extract from 22
independent plain rollouts, it can now extract from the ~20 sections of a single
one. The gap that remains is entirely in the *selector*, not in the candidates —
which is precisely what arm M-F is being continued to attack.

### The reward curves, and the one arm still climbing when it stopped

![reward over training](plots/reward_curves.png)

There is no single reward curve for this experiment — the arms do not share a
reward — so each is drawn against its own objective. Three things read off it:

1. **M-F is the only clean "reward goes up" curve.** `last_f1` climbs 0.33 →
   0.48 across 42 batches with no plateau, and the run was stopped by the
   coverage criterion later shown to be in the wrong units (its union/R was
   **2.80**, nowhere near the 1.25 floor where #208's mechanism binds). *M-F was
   never trained to exhaustion.*
2. **M-C and M-B both peak at step 15–20 and turn over** — the same peak the
   evaluations found, visible from the training batches alone.
3. **M-B at lr 3e-6 oscillates without trend from step ~40 onward.** If its
   evaluations agree, KL ~0.009 is a genuine ceiling for that reward rather than
   an artifact of how coarsely the 1e-5 run was checkpointed.

M-F rising while *everything else about it degrades* (consensus −0.0144,
oracle-best −0.0154, `second_last` −0.1631) is the sharpest illustration in this
experiment that a rising reward is not a result.

### The single-rollout case: what a final section costs, and what it buys

A consensus prediction is a **ranking over pairs** that must be cut at `R` to
become a contact set — and `R` is the number of true contacts, which comes from
ground truth. A rollout's final section is already a set: one generation, one
self-consistent answer, no vote and no cutoff. That difference is worth pricing.

Generated tokens per protein, measured on the eval runs themselves:

| prediction | tokens / protein | R-precision (legacy 554) |
|---|---:|---:|
| plain, one rollout | 500 | 0.4454 |
| **M-F step-36, final section** | **3,697** | **0.5075** |
| **M-C step-18, final section** | 5,503 | **0.5267** |
| plain, 22 rollouts, consensus | 11,005 | 0.5896 |

**Stated against itself rather than flattered:** at matched *tokens* a single
multi rollout is probably not yet the better buy — 3,697 tokens would fund ~7
plain rollouts, and consensus over 7 plain rollouts is untested but sits
somewhere between 0.4454 and 0.5896. What the single rollout wins on is not
tokens; it is that the output is a committed contact set rather than a ranking
needing a ground-truth-derived cutoff, and that it needs no aggregation step at
all. Whether it can also win on accuracy is what continuing M-F is for.

### M-F is dominated by M-C, at M-F's own objective

Worth stating separately, because it inverts the obvious plan for improving the
deployable single-candidate number:

| | last | best *ORACLE* | consensus |
|---|---:|---:|---:|
| M-F step-36 (rewards the last section) | 0.5075 | 0.5189 | 0.5529 |
| **M-C step-18** (rewards section marginals) | **0.5267** | **0.5578** | **0.5750** |

**M-C beats M-F on M-F's own target**, and on everything else. The likely reason
is visible in the training metrics: M-F's reward touches one section out of ~24,
so the other 23 receive no signal and decay (`second_last` falls to 0.2649, from
0.4284) — and the final section is written in the context of those decayed
predecessors. M-C rewards every section, so the whole rollout stays useful.

So "make M-F better" is probably not the route to a better final section. Giving
*every* section a signal appears to be, and the corrected M-C base is that.

### Designing the next arm: which reward definitions are scale-free?

Three candidate rewards, measured on the same 120 synthetic groups that differ in
**nothing but section count** — mean group-centred advantage per section:

| sections emitted | `loo`, what M-C ran | `prefix` | **`rollout_grpo`** |
|---:|---:|---:|---:|
| 1 | +4.79 | +2.03 | **−1.37** |
| 2 | +1.08 | +1.23 | −0.52 |
| 4 | +0.20 | +0.48 | −0.14 |
| 8 | −0.02 | +0.10 | +0.43 |
| 16 | −0.18 | −0.15 | +0.81 |
| 22 | −0.22 | −0.22 | **+0.79** |

- **`loo`** = `C(all) − C(all \ {k})`, centred over the group. The bug.
- **`prefix`** = `C(1..k) − C(1..k−1)`, i.e. the *causal* marginal against exactly
  what was in context when the section was written. It has an attractive property
  — it telescopes, so `Σ_k m_k = C(all) − C(∅)` — and **it does not fix the
  pathology**: still +2.03 at one section. Telescoping constrains the **sum**,
  but `loss_reduction=token_mean` reads the **mean**, and a short rollout's early
  sections are scored against a near-empty prefix, so its mean is large. Worth
  recording as a refuted fix: it is the obvious repair and it does not work.
- **`rollout_grpo`** = the rollout's own `C(all)`, GRPO-centred across the group.
  Correct sign, monotone, and it saturates near 16–22 exactly as `C(K)` does
  (0.341 → 0.543). Scale-correct **by construction**: emitting fewer sections
  lowers your own consensus and therefore your advantage.

#### The arm this implies

```
A_i  =  GRPO_group( C_i(all) )                      # base: rollout-level, scale-correct
      + beta * ( m_k - mean_k m )                   # zero-sum within-rollout shaping
      + lam  * GRPO_group( max_k F1(section k) )    # arm M-B's term
```

Three properties worth stating, because each is the reason a previous version
failed:

1. **The base cannot be gamed by section count.** It is the deployed metric,
   computed on the object the model emits, and it *falls* when sections are
   dropped.
2. **The shaping term sums to zero within a rollout**, so it cannot reintroduce
   any section-count pressure — it only says *which* section earned the rollout's
   advantage. Using the **prefix** form here is what answers "condition on what
   you have already written": a section is credited for what it added *given its
   predecessors*, so duplicating an earlier section earns nothing while covering
   something they missed earns a lot. The property that made `prefix` attractive
   is kept exactly where it is safe.
3. **`lam` is now a real trade-off.** Blending M-B against the *current* M-C could
   not have worked: M-C's term runs from −0.22 at 22 sections to **+4.79** at one,
   so a fixed `lam` balanced at 22 sections is overwhelmed by the time the count
   reaches 4 — which is the runaway measured above. Once the base is
   rollout-level, both terms are O(1) and `lam` trades "a good best candidate"
   against "a good consensus" rather than racing a divergent term.

#### On simply penalising short rollouts

A direct penalty that grows as sections disappear is the other obvious repair,
and the measurement prices it: it would have to cancel a term that reaches +4.79,
so it is a tuned counterweight to one specific measured curve rather than a fix
to the quantity being measured. Under the corrected base it is also **redundant** —
`C(all)` already supplies −1.37 at one section. It remains worth keeping as a
cheap guardrail, with one thing to watch: a size penalty creates an incentive to
pad with junk sections, and `multi/empty_sections` (currently 0.000) is the
instrument for that.

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

## Arm M-B — one number for the whole rollout, and the only arm that improved everything

**What it is, stated plainly because the shorthand hides it:** M-B scores the
**entire rollout** by `max_k F1(section k)` — the oracle-best contact set
anywhere in the generation — as a *single scalar*, centred by GRPO against the
prompt's other 7 rollouts and broadcast to every token. It never looks inside the
rollout; there is no per-section credit assignment. That is the whole of it, and
it is the opposite end of this experiment's axis from M-C, which shapes each
section's tokens individually.

It is also, on the evidence so far, the best-performing design tried here.

**Stopped at step 36** on the preregistered coverage criterion, scored, then
**resumed from its own step-36 checkpoint to step 80** under the corrected gate
(see below). Terminal KL **0.4863**.

| | step 36 (KL 0.016) | step 80 (KL 0.486) |
|---|---:|---:|
| consensus | **0.5741** | 0.3969 |
| best *ORACLE* | **0.5574** | 0.3440 |
| last | 0.4908 | 0.1905 |
| per-contact precision (train) | 0.50 | **0.14** |
| union/R | 2.76 | 4.63 |

**The extension destroyed the model, and neither coverage gate saw it coming.**
union/R at step 80 is 4.63 — *higher* than the warm start's 3.98 — because the
policy emitted more pairs, not fewer. They were simply wrong: per-contact
precision fell from 0.50 to 0.14. Coverage was never the binding constraint; the
votes were.

So the corrected gate is necessary and still not sufficient. `contacts/precision`
is the metric that catches this, it was already being reported, and it is the one
to gate on next time.

## The gate that was measuring the wrong thing

All three arms were stopped by #237's preregistered coverage criterion — union
pairs per rollout below 80 % of the run's own warmup. Every one of them was
stopped **past its own optimum but for the wrong reason**.

#208's coverage mechanism is that R-precision cuts a ranking at R = |gt|, so
zero-vote pairs begin padding the top-R **only once the union falls below R**.
Measured at eval, union/R was 3.98 for the warm start and **never left 2.8–4.6 in
any arm, including the collapsed one**. The coverage that triggered the gate was
headroom nothing was using.

`min_union_over_r` (default 1.25) is the same criterion in the units the
mechanism is in. `min_union_ratio` is retained and defaults to 0, with this
measurement recorded where the default lives.

**The honest reading:** the preregistration named a real failure mode and
specified it in the wrong units, and the cost was three arms stopped early — which
turned out not to matter, because every arm's best checkpoint was *earlier* than
where the gate fired anyway.

## Reproducing

```bash
python phase0_marginals.py --sections <#230 agg_sections> --targets <eval577>
./skyrl/run_on_host.sh --host <user@host> --smoke
./skyrl/run_on_host.sh --host <user@host> -- bash ~/exp237/skyrl/run_pipeline.sh
python summarize_runs.py --logs ~/exp237_logs --out data/
python build_results.py --eval ~/exp237_data/eval --out data/
python compare_arms.py --arm <arm per-rollout parquet> --ref <#230 per-rollout parquet>
python make_plots.py --steps data/training_steps.csv.gz --out plots/
```

![the diversity gates over training](plots/gates_over_training.png)

![coverage against distance moved](plots/coverage_vs_kl.png)
