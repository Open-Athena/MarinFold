# Porting exp208's RL to SkyRL on local A100s — design

**Why:** `marin.rl` was deleted upstream at 0.2.77 (2026-08-07), so the 0.2.76 pin
can never advance and never receives a fix. exp208 alone hit six of its defects —
runs that cannot self-terminate, the weight-transfer step-alignment assertion,
`canonical_model_name` doing double duty, the `prng_key` union, a per-rollout
dense-advantage guard that would have crashed arm S, and the vocab-padding
interaction that produced the NaN. `RunConfig` is also TPU-only (no GPU field at
all), and the shared v5p pool has cost this experiment a ~25 min vLLM source build
per gang, two preemptions by other users, and a whole-node RAM request we
inherited from exp200 without noticing. marin's own direction is now SkyRL.

**Target:** SkyRL (`github.com/novasky-ai/SkyRL`) on the private 8x A100-80GB box.
CUDA 12.8 and Python 3.12 are the documented requirements and the box satisfies
both. The host is **never hardcoded** — every script here takes `--host` with no
default (see the private infra note; it is not referenced in this repo).

## What survives the port unchanged

`contact_rewards.py` and `consensus.py` are pure numpy and framework-independent:
the dense per-contact stepwise reward, the leave-one-out consensus marginal, the
group diagnostics, and the exp89-verified R-precision all carry over as-is. So
does the Phase 0 result that motivated the design, and the exp199 baseline of
record (0.6099 / 0.6114 on v5p, 0.6103 on CoreWeave).

What gets rewritten is the harness: environment, advantage plumbing, launcher.

## Where the dense reward goes — verified against source, not docs

This is the load-bearing question, and the documentation is misleading about it.
`BaseTextEnv.step()` returns a **float** per turn, and the tutorial says rewards
are "allocated to the final token of each turn's response". Taken at face value
that would be fatal: exp208's whole mechanism is a distinct reward on each
`<contact> <pI> <pJ>` triple, not one scalar per turn.

Reading `skyrl/train/generators/skyrl_gym_generator.py` shows the real shape:

```python
token_level_rewards: List[float] = [0.0] * len(response_ids)
for i, (step_reward, idx) in enumerate(per_step_rewards):
    token_level_rewards[idx] += step_reward
reward_out = token_level_rewards
```

The generator already builds a **dense vector the length of the response** and
hands it downstream; per-turn scalars are merely the sparse case of it. And the
custom-advantage registry receives exactly that:

```python
def compute_advantage(token_level_rewards: torch.Tensor,   # [batch, response_len]
                      response_mask: torch.Tensor, index: np.ndarray, **kwargs)
```

So there are two clean extension points, both documented API rather than an
exploit:

1. **Generator override** — subclass the generator and populate
   `token_level_rewards` from `contact_rewards.dense_rewards`. This is the natural
   home for the **stepwise** term.
2. **Custom advantage estimator** — registered via SkyRL's registry, returning
   per-token advantages. This is the natural home for the **consensus-marginal
   document term**, which needs the whole rollout group (`index` supplies the
   grouping) exactly as RLOO's leave-one-out baseline does.

Contrast with marin.rl, where the same capability existed only because
`np.full` happens to broadcast an array `fill_value` — undocumented, unpromised,
and pinned by `test_dense_advantage_broadcast.py` precisely because a silent
regression would degrade to constant advantages and read as "RL didn't help".

Constraint noted: custom algorithms are supported **only on the `fsdp` and
`megatron` backends**. FSDP on 8x A100 is the intended configuration anyway.

## Correction to an earlier claim

An earlier summary of this port said SkyRL has "token-level rewards as a
first-class feature", citing a search snippet. That overstated the *environment*
API, which is one float per turn. The dense path is real and is better than
marin.rl's, but it lives in the generator and the advantage registry, not in
`BaseTextEnvStepOutput`.

## Open questions before committing

1. Does the custom advantage estimator's `**kwargs` carry the response **token
   ids** and per-protein ground truth? The consensus term needs both. If not, the
   document term computes in the generator override alongside the stepwise term,
   and the estimator only combines them.
2. Rollout/training colocation on 8x A100-80GB for a 1.5B policy — expected
   comfortable, but unmeasured.
3. Whether SkyRL's sampler constrains token ids to the real vocabulary. exp208's
   NaN came from vLLM sampling its own vocab padding (2845 -> 2848); the same
   trap exists on any engine that pads, so the `allowed_token_ids` guard ports
   across regardless of framework.
