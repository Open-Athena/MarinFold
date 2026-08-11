# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Do levanter's logprobs agree with vLLM's on the recorded rollouts? — issue #208.

THE LAST UNTESTED LINK. exp208's RL loop NaNs at the first training step from the
exp199 warm start and trains cleanly from exp163 arm F. Everything else has been
eliminated by measurement: the learning rate, the KL anchor, all three document
terms, and — via `probe_levanter_forward.py` / `probe_length_sweep.py` — levanter's
conversion, forward and backward, which are finite in bf16 for both checkpoints at
15, 256 and 512 tokens with gradients that shrink rather than grow with length.

What remains is the importance ratio, `exp(policy_logp - sampler_logp)`. It is the
one quantity that couples the two engines: the sampler logprobs come from vLLM and
the policy logprobs from levanter. If they disagree badly the ratio overflows, and
`inf * 0` is NaN — which arm S makes likely, because its advantage is exactly zero
on every token outside a `<contact> <pI> <pJ>` triple. It also explains the one
fact no other hypothesis does: dropping the learning rate tenfold changed nothing,
because this is not a divergence.

exp200 measured `train/ratio_mean` at 1.0024 and called it the check that validates
the whole policy-gradient path. exp208's crash happens before that metric is ever
logged — so this reconstructs it offline from the spilled rollouts, which carry
vLLM's `response_logprobs` alongside the tokens.

**The exp163 arm F control is the method's own validation**: its run logged
ratio_mean 1.0000 +/- 0.0005, so if this probe does not reproduce ~1.0 there, the
probe is wrong rather than the checkpoint.

    uv run python probe_importance_ratio.py
"""

import argparse
import pickle

import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from levanter.layers.attention import AttentionMask

from rl_config import MODEL_CONFIG

SPILL = "gs://marin-us-central1/protein-structure/MarinFold/exp208/rollouts"
CASES = [
    # label, spill run name, warm-start repo, what its run did
    ("exp163F-control", "plm-exp208-rl-cv1-1_5b-armS-lr1em06-s10-exp163ctl",
     "timodonnell/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF-step404", "trained 10 steps"),
    ("exp199-minimal", "plm-exp208-rl-cv1-1_5b-armS-lr1em06-s10-minimal",
     "timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199", "NaN at step 1"),
]


def load_rollouts(run: str, n_rollouts: int):
    """Pull a handful of rollouts out of the first spilled batch."""
    fs, root = fsspec.core.url_to_fs(f"{SPILL}/{run}")
    files = sorted(fs.ls(root, detail=False))
    if not files:
        raise SystemExit(f"no spill under {run}")
    with fs.open(files[0], "rb") as fh:
        batch = pickle.load(fh)
    groups = getattr(batch, "groups", None) or []
    out = []
    for g in groups:
        for r in getattr(g, "rollouts", []):
            out.append(r)
            if len(out) >= n_rollouts:
                return out, files[0]
    return out, files[0]


def ratios_for(model, cfg, rollout):
    """exp(levanter_logp - vllm_logp) for every response token of one rollout."""
    prompt = np.asarray(rollout.prompt_tokens, dtype=np.int32)
    response = np.asarray(rollout.response_tokens, dtype=np.int32)
    sampler = np.asarray(rollout.response_logprobs, dtype=np.float32)
    ids = np.concatenate([prompt, response])

    # levanter's flash attention requires the query axis to be a multiple of its
    # 1024 block ("q axis size 1723 is not a multiple of 1024"). The trainer never
    # trips this because train_batch pads to curriculum.max_seq_len (8192). Pad
    # right to the next multiple: attention is causal, so tokens appended after
    # the real sequence cannot influence any position we read.
    BLOCK = 1024
    n_real = len(ids)
    if n_real % BLOCK:
        ids = np.concatenate([ids, np.zeros(BLOCK - n_real % BLOCK, dtype=np.int32)])

    Pos = cfg.max_Pos.resize(len(ids))
    tokens = hax.named(jnp.array(ids, dtype=jnp.int32)[None, :], ("batch", Pos))
    # CAUSAL MASK, EXPLICITLY. `attn_mask=None` is not "default causal" -- it is
    # no mask at all, i.e. bidirectional attention over the whole sequence. With
    # it the reconstructed logprobs are garbage: the exp163 arm F control, whose
    # own run logged train/ratio_mean 1.0000, reconstructed to 0.0011. That
    # control is the reason this probe is trustworthy at all.
    out = model(tokens, attn_mask=AttentionMask.causal(), key=None)
    logits = jnp.asarray(out.array if hasattr(out, "array") else out).astype(jnp.float32)[0]
    logp = jax.nn.log_softmax(logits, axis=-1)

    # marin's convention (contact_rewards docstring): logprobs[t] = log pi(token_t |
    # tokens_<t>), i.e. the weight lands ON that token with no shift. So response
    # token at absolute index p is predicted by logits at p-1.
    start = len(prompt)
    idx = np.arange(start, start + len(response))
    policy = np.asarray(logp[idx - 1, ids[idx]])
    return policy, sampler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rollouts", type=int, default=3)
    args = ap.parse_args()

    mesh = Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "model"))
    setter = getattr(jax, "set_mesh", None) or jax.sharding.use_mesh
    cfg = MODEL_CONFIG
    conv = cfg.hf_checkpoint_converter()

    with setter(mesh):
        for label, run, repo, note in CASES:
            print(f"=== {label} ({note}) ===", flush=True)
            rollouts, src = load_rollouts(run, args.n_rollouts)
            print(f"  spill {src.split('/')[-1]}  {len(rollouts)} rollout(s)", flush=True)
            model = conv.load_pretrained(cfg.model_type, ref=repo, config=cfg)
            for k, r in enumerate(rollouts):
                policy, sampler = ratios_for(model, cfg, r)
                d = policy - sampler
                # SIGN MATTERS AND ABS HIDES IT. The trainer computes
                # exp(policy - sampler) in float32, which overflows to +inf above
                # log(3.4e38) = 88.7. A large NEGATIVE difference underflows to 0
                # and is harmless; a large POSITIVE one is +inf, and inf * 0 is
                # NaN -- and arm S gives exactly-zero advantage to every token
                # outside a <contact> <pI> <pJ> triple.
                F32_OVERFLOW = float(np.log(np.finfo(np.float32).max))
                over = int((d > F32_OVERFLOW).sum())
                under = int((d < -F32_OVERFLOW).sum())
                ratio32 = np.exp(d.astype(np.float32))
                # WHERE are the outliers? Position within the response, and which
                # token id. A structural cause (all at the truncation boundary, all
                # one token id, all after <end>) looks completely different from
                # numerical drift, and the two demand different fixes.
                bad = np.where(d < -20)[0]
                if len(bad):
                    resp = np.asarray(r.response_tokens, dtype=np.int32)
                    ids_bad = resp[bad]
                    uniq, cnt = np.unique(ids_bad, return_counts=True)
                    top = sorted(zip(cnt.tolist(), uniq.tolist()), reverse=True)[:6]
                    print(f"      outliers(d<-20): {len(bad)}  "
                          f"pos_frac[min/med/max]={bad.min()/len(d):.2f}/"
                          f"{np.median(bad)/len(d):.2f}/{bad.max()/len(d):.2f}  "
                          f"top_token_ids={[(int(i), int(c)) for c, i in top]}", flush=True)
                print(f"    rollout {k}: n={len(d):5d}  mean_ratio={np.mean(np.exp(d)):8.4f}  "
                      f"dlogp max={np.max(d):+8.3f} min={np.min(d):+8.3f}  "
                      f"| >+{F32_OVERFLOW:.1f}: {over}  <-{F32_OVERFLOW:.1f}: {under}  "
                      f"| f32 ratio inf={int(np.isinf(ratio32).sum())} "
                      f"nan={int(np.isnan(ratio32).sum())}", flush=True)
            del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
