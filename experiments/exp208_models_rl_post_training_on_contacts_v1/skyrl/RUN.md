# Running exp208 on SkyRL

Verified on 8x A100-80GB: SkyRL installed (py3.12.13, torch 2.11.0+cu128, vLLM
0.23.0), env and advantage estimator register, and the config below parses.

The GPU host is a **required** argument everywhere and has no default; it is not
named in this repository.

```bash
# 1. build the dataset from the exp200 prompt pool (same data as the marin.rl path)
uv run python build_dataset.py --out data/skyrl_train.parquet --n 2000

# 2. sanity: registration + GPU visibility on the host
./run_on_host.sh --host <user@host> --smoke

# 3. train
./run_on_host.sh --host <user@host> -- python main_exp208.py \
    trainer.policy.model.path=timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199 \
    trainer.algorithm.advantage_estimator=contacts_dense \
    trainer.strategy=fsdp2 \
    generator.n_samples_per_prompt=16 \
    vocab_size=2845 lam_doc=4.5
```

## Config keys that are easy to get wrong

| what | key | note |
|---|---|---|
| advantage estimator | `trainer.algorithm.advantage_estimator` | **not** `adv_estimator` — that name only exists inside `compute_advantages_and_returns`. The config validator rejects the wrong key with the full valid-field list, which is how this was caught. |
| rollout group size | `generator.n_samples_per_prompt` | This is what supplies the consensus term's group. 16 on the marin.rl path. |
| backend | `trainer.strategy=fsdp2` | Custom advantage estimators are supported **only** on `fsdp`/`megatron`. |
| loss reduction | `trainer.algorithm.loss_reduction` | Defaults to `token_mean`. SkyRL pre-applies reduction to advantages before the policy loss, so a custom loss should do a masked **sum**. Relevant if we add one. |
| vocabulary guard | `vocab_size=2845` | Constrains sampling to real ids. Not optional for exp199 — see below. |

## Two traps carried over from the marin.rl path

1. **vLLM samples its own vocab padding.** 2845 pads to 2848 with zero rows that
   emit logit 0.0. exp199's logits sit low (top-logit median 1.16, min -4.03), so
   those rows took 12.4% of sampled tokens across 256 of 256 rollouts and NaN'd
   the marin.rl trainer on step 1. The trap belongs to the inference engine, not
   the framework, so `vocab_size` must be set here too.

2. **`custom_chat_template` disables per-token rewards in the stock generator.**
   contacts-v1 has no chat format, so a custom template is required — and
   `_build_per_token_rewards` returns a float rather than a vector when one is
   set. `DenseContactsGenerator` overrides the whole method for exactly this
   reason. If a SkyRL bump reorganises it, check this first: the failure is
   silent and degrades to one scalar per trajectory.
