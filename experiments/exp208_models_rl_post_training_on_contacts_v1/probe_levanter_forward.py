# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Is levanter's forward pass finite on a given warm start? — issue #208.

WHY. exp208's RL loop NaNs on the FIRST training step from the exp199 warm start,
in every configuration tried (lr 1e-5 and 1e-6, KL on and off, all three document
terms). The identical code trains 10 clean steps from exp163 arm F with
`train/ratio_mean` 1.0000 +/- 0.0005. So the checkpoint is the variable.

Every healthy signal observed so far comes from **vLLM** — rollouts, precision,
consensus — and vLLM is not what NaNs. The trainer crashes before it logs a single
metric, so there is no evidence either way about levanter's own forward. This
probe supplies it, on CPU, without a TPU gang.

It reports three things per checkpoint, which between them localise the failure:

* whether levanter's HF conversion produces any non-finite parameter;
* the largest logit magnitude from a forward in **f32**;
* the same in **bf16**, which is what the trainer actually computes in
  (`mp="p=f32,c=bfloat16"`).

A checkpoint that is finite in f32 and non-finite in bf16 is an overflow in the
compute dtype, and the fix is a mixed-precision change. One that is non-finite in
both has a conversion problem. One that is finite in both puts the fault
downstream of the forward — in the loss, the importance ratio, or the backward.

    uv run python probe_levanter_forward.py
"""

import argparse

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from rl_config import MODEL_CONFIG

DEFAULT = [
    "exp163F=timodonnell/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF-step404",
    "exp199=timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199",
]
# A plausible contacts-v1 prompt fragment: doc sentinel, an n-term, a couple of
# residues, then contact triples. Content barely matters — this is a finiteness
# probe, not an accuracy one — but real ids keep it in distribution.
IDS = [2, 11, 160, 160, 30, 161, 45, 9, 5, 160, 168, 5, 161, 175, 10]


def probe(name: str, repo: str, cfg, conv, ids: list[int]) -> None:
    print(f"=== {name} ({repo}) ===", flush=True)
    model = conv.load_pretrained(cfg.model_type, ref=repo, config=cfg, dtype=jnp.float32)
    leaves = [jnp.asarray(l) for l in jax.tree_util.tree_leaves(model)
              if hasattr(l, "dtype") and jnp.issubdtype(l.dtype, jnp.floating)]
    bad = sum(int(jnp.count_nonzero(~jnp.isfinite(l))) for l in leaves)
    print(f"  converted : {len(leaves)} float arrays  non-finite={bad}  "
          f"max|w|={max(float(jnp.max(jnp.abs(l))) for l in leaves):.3f}", flush=True)

    # `max_Pos`, not `Pos`: Qwen3Config exposes the position axis under that name.
    Pos = cfg.max_Pos.resize(len(ids))
    tokens = hax.named(jnp.array(ids, dtype=jnp.int32)[None, :], ("batch", Pos))
    for policy, dt in (("f32", jnp.float32), ("bf16", jnp.bfloat16)):
        m = model if dt == jnp.float32 else jax.tree_util.tree_map(
            lambda x: x.astype(dt) if hasattr(x, "dtype")
            and jnp.issubdtype(x.dtype, jnp.floating) else x, model)
        try:
            out = m(tokens, attn_mask=None, key=None)
            arr = jnp.asarray(out.array if hasattr(out, "array") else out).astype(jnp.float32)
            n_bad = int(jnp.count_nonzero(~jnp.isfinite(arr)))
            print(f"  forward {policy:4s}: max|logit|={float(jnp.nanmax(jnp.abs(arr))):10.2f}  "
                  f"non-finite={n_bad}/{arr.size}"
                  + ("   <-- FAILS HERE" if n_bad else ""), flush=True)
        except Exception as exc:
            print(f"  forward {policy:4s}: EXC {type(exc).__name__}: {str(exc)[:120]}", flush=True)

    # THE BACKWARD IS THE POINT. A finite forward says nothing about the gradient,
    # and exp199's RMSNorm gains (input_layernorm 19.75 against exp163 arm F's
    # 3.80) act on the backward path too. If the gradient overflows to inf in the
    # compute dtype, optax's clip_by_global_norm scales by max_norm/inf = 0 and
    # turns inf into NaN -- exactly the trainer's symptom, and immune to lowering
    # the learning rate because it is not a divergence.
    for policy, dt in (("f32", jnp.float32), ("bf16", jnp.bfloat16)):
        m = model if dt == jnp.float32 else jax.tree_util.tree_map(
            lambda x: x.astype(dt) if hasattr(x, "dtype")
            and jnp.issubdtype(x.dtype, jnp.floating) else x, model)

        def loss_fn(mm):
            out = mm(tokens, attn_mask=None, key=None)
            logits = (out.array if hasattr(out, "array") else out).astype(jnp.float32)
            logp = jax.nn.log_softmax(logits, axis=-1)
            tgt = jnp.array(ids, dtype=jnp.int32)[None, :]
            picked = jnp.take_along_axis(logp[:, :-1], tgt[:, 1:, None], axis=-1)
            return -jnp.mean(picked)

        try:
            loss, grads = jax.value_and_grad(loss_fn)(m)
            gl = [jnp.asarray(g).astype(jnp.float32) for g in jax.tree_util.tree_leaves(grads)
                  if hasattr(g, "dtype") and jnp.issubdtype(g.dtype, jnp.floating)]
            n_bad = sum(int(jnp.count_nonzero(~jnp.isfinite(g))) for g in gl)
            gnorm = float(jnp.sqrt(sum(float(jnp.sum(jnp.where(jnp.isfinite(g), g, 0) ** 2)) for g in gl)))
            print(f"  BACKWARD {policy:4s}: loss={float(loss):8.4f} "
                  f"grad_norm(finite)={gnorm:12.4f} non-finite={n_bad}"
                  + ("   <-- FAILS HERE" if n_bad else ""), flush=True)
        except Exception as exc:
            print(f"  BACKWARD {policy:4s}: EXC {type(exc).__name__}: {str(exc)[:120]}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], metavar="NAME=REPO")
    args = ap.parse_args()

    # levanter's sharded safetensors reader asks the mesh for a 'data' axis, so a
    # mesh must be active even on one CPU device — and under jax 0.10 the old
    # `with mesh:` form no longer reaches haliax's partitioning lookup.
    mesh = Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "model"))
    setter = getattr(jax, "set_mesh", None) or jax.sharding.use_mesh

    cfg = MODEL_CONFIG
    conv = cfg.hf_checkpoint_converter()
    with setter(mesh):
        for spec in (args.model or DEFAULT):
            name, _, repo = spec.partition("=")
            probe(name, repo, cfg, conv, IDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
