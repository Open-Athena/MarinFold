# `err_decay` breaks the reward it was added to protect

**Issue #208 · analysis on 9,900 Phase 0 rollouts · no GPU required**

**Summary.** exp208's per-contact reward is built around a self-referential
baseline: a contact is worth emitting exactly when its chance of being correct
beats the policy's own recent precision. `err_decay = 0.5` discounts repeat
errors, and in doing so removes 97.7% of the penalty term — the marginal wrong
contact ends up costing a **median of exactly zero** against `+0.745` for a
correct one. The baseline the design rests on is not there. Setting
`err_decay = 1.0` restores it and, on real rollouts, ranks them better by every
quality measure. A second bug surfaced alongside it: `p̄`'s starting value was
0.45 while the training pool measures 0.26.

Reproduce with [`analyze_err_decay.py`](analyze_err_decay.py).

---

## 1. What the reward is supposed to do

For each emitted contact, with `x = 1` if the (i, j) pair is in ground truth:

```
r = (1 - p̄)              if x == 1        (correct)
    -p̄ · δ^k             if x == 0        (the k-th error in this section)
```

`p̄` is an EMA of the policy's own recent per-contact precision. With `δ = 1` the
expected value of emitting a contact whose correctness probability is `p` is

```
E[r] = p(1 - p̄) - (1 - p)p̄ = p - p̄
```

This is the entire point of the design. It is a **zero baseline**: emitting is
worth it precisely when the contact beats the policy's current precision, so the
gradient says "beat yourself" rather than "emit more" or "emit less". Summed over
a rollout it comes out to exactly

```
total reward = n_scored · (precision - p̄)
```

which is the intended signal written plainly — volume multiplied by *quality
relative to the baseline*.

## 2. What `δ < 1` does to it

The k-th error costs `p̄·δ^k`, so the marginal contact is worth `p - p̄·δ^k`
rather than `p - p̄`. That is positive for **any** contact — however unlikely —
once enough errors have accumulated. The total penalty a rollout can ever pay is
a convergent geometric series:

```
penalty ≤ p̄ · (1 + δ + δ² + …) = p̄/(1 - δ)  =  2p̄ ≈ 0.51   at δ = 0.5
```

while the positive term grows without bound at `1 - p̄ ≈ 0.745` per correct
contact. Errors are not so much discounted as **capped**.

Whether that matters is an empirical question about how many errors real
rollouts make. It is not close.

## 3. Measurement

Scored on the 9,900 Phase 0 rollouts (10,000 less those emitting nothing) using
the production `dense_rewards`. The penalty is **order-independent** — the k-th
error costs `p̄·δ^k` wherever it falls, so a rollout's total is
`p̄·(1-δ^n_wrong)/(1-δ)` regardless of how errors interleave with correct
contacts. That closed form was checked against `dense_rewards` on 200 synthetic
token streams × 2 values of `p̄` × 3 of `δ` and reproduces it exactly
(max |diff| = 0.0), so what follows is the real reward function and not a
re-derivation of it.

At `p̄ = 0.2547`, the value the EMA actually settles to on the training pool:

| δ | penalty as % of positive term | cost of the NEXT wrong contact | ρ(reward, F1) | ρ(reward, precision) | ρ(reward, n_pred) |
|---|---|---|---|---|---|
| 0.5 | 2.3% | **median 0.000000** | 0.8675 | 0.8269 | 0.6771 |
| 1.0 | 32.7% | 0.2547 | **0.9215** | **0.9057** | **0.4711** |

Three readings of the same table:

**Errors are free.** A correct contact pays `+0.745`. At `δ = 0.5` the next wrong
one costs a median of `0.000000`, and for **92.8%** of rollouts it costs under 1%
of what a correct contact earns. The median rollout makes 25 errors, and
`0.5²⁵ = 3×10⁻⁸`.

**The penalty is a rounding error.** It accounts for 2.3% of the reward's
magnitude at `δ = 0.5`, against 32.7% at `δ = 1.0`. A signal that is 97.7% "count
of correct contacts" is a recall signal wearing a precision-shaped hat.

**`δ = 1` ranks rollouts better.** Higher rank agreement with F1 (0.9215 vs
0.8675) and with precision (0.9057 vs 0.8269), and *lower* agreement with raw
contact count (0.4711 vs 0.6771) — it is less easily satisfied by emitting more.

### There is no intermediate setting

The obvious hope is that some `δ` keeps a little forgiveness without giving away
the baseline. It does not exist:

| δ | 0.0 | 0.25 | 0.5 | 0.75 | 0.9 | 0.99 | 1.0 |
|---|---|---|---|---|---|---|---|
| penalty % of positive | 1.2 | 1.6 | 2.3 | 4.5 | 9.9 | 27.4 | 32.7 |
| ρ(reward, F1) | 0.8614 | 0.8668 | 0.8675 | 0.8689 | 0.8784 | 0.9197 | 0.9215 |
| ρ(reward, precision) | 0.8193 | 0.8259 | 0.8269 | 0.8288 | 0.8411 | 0.8995 | 0.9057 |

Ranking quality is **flat** from 0.0 to 0.9 and only recovers at 0.99–1.0.
Anything below ~0.9 is indistinguishable from having no penalty at all. The
decay's motivating observation — that later errors in a spoiled section may be
consequences of the first rather than independent mistakes — is a real one, but
it cannot be bought at any `δ` that leaves the baseline standing.

## 4. The second bug: `p̄` started in the wrong place

At `δ = 1` the reward is `n_scored·(precision - p̄)`, so **`p̄` must track the
pool's true precision**. Set above it, every contact has negative expected value
and the policy is paid to say nothing — which is the collapse `err_decay` was
introduced to prevent, arriving through the front door instead.

`INITIAL_PRECISION` was **0.45**. That was a defensible compromise when written:
Phase 0 measured 0.482 for this model, but on the PDB-derived *eval* set, while
the RL training pool is AFDB round-0 with pyconfind labels — and nothing had
measured the pool directly. The comment in `rl_config.py` said as much.

The SkyRL runs measure it directly: step-0 per-contact precision, before any
gradient update, across five configurations —

```
0.267   0.259   0.250   0.263   0.281
```

The pool sits at ~0.26, so 0.45 was **~0.19 too high**: a silence pressure of
`0.26 - 0.45 = -0.19` per contact applied to every rollout in the opening steps.
The EMA converges within a single batch, but batch 0's gradient is computed on
the way there. Now **0.26**.

Does `δ = 1` reintroduce the silence failure at a correct `p̄`? No:

| p̄ | 0.20 | 0.2547 | 0.30 | 0.45 | 0.50 |
|---|---|---|---|---|---|
| rollouts with reward > 0 | 78.2% | 70.6% | 65.6% | 53.9% | 49.5% |
| mean reward | +17.59 | +14.49 | +11.93 | +3.44 | +0.61 |

At a correctly-tracked `p̄` most rollouts still earn positive reward, and at the
pool's own precision the per-contact expectation is ≈ +0.01 — near zero and
correctly centred, exactly as intended.

## 5. A wrong argument, corrected

`contact_rewards.py` justified the decay this way:

> precision is ~0.3, so a *fixed* penalty would make "emit nothing" the optimal
> policy and the run would collapse to empty sections.

That is true of a penalty of fixed *size* — a flat `-1` per error would indeed
make silence optimal at 30% precision. But the penalty is already `-p̄`, which
scales with the policy, giving `E[r] = p - p̄ ≈ 0`. **The p̄-centring already
solves the problem the decay was added to solve.** The decay was a second fix for
an addressed problem, paid for with the baseline. Docstring corrected.

## 6. What changed

| | before | after |
|---|---|---|
| `err_decay` | 0.5 | **1.0** (off) |
| `INITIAL_PRECISION` / `p_bar` | 0.45 | **0.26** |

Applied to both the SkyRL path (`skyrl/main_exp208.py`) and the marin.rl path
(`rl_config.py`), and to `contact_rewards.py`'s own function default.

## 7. Why this was worth doing before spending GPU-hours

Arm S at `δ = 0.5` would have run, trained, logged, and produced a number. The
number would have described a policy optimizing something close to recall under a
baseline that did not function — and it would have looked like a clean test of
"does dense per-contact RL help contacts-v1", because nothing in the run reports
that its own reward has stopped discriminating. The analysis above costs no GPU
and a few minutes.

It is also worth recording how the concern arose, because the route was not
clean: from watching arm S's response length climb toward its cap and
*incorrectly* diagnosing it as reward-gaming. The length was actually a symptom
of policy collapse caused by SkyRL's FSDP2 sharding (see the run notes in
[README.md](README.md)) and had nothing to do with `err_decay`. The hypothesis
was right about the reward and wrong about the evidence, and only separating the
two — a zero-LR control for the collapse, this offline scoring for the reward —
settled either.
