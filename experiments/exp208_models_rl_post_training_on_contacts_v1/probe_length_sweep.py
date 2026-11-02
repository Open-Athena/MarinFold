# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Does the exp199 gradient blow up with SEQUENCE LENGTH? — issue #208.

`probe_levanter_forward.py` showed both warm starts clean through conversion,
forward and backward — but on a **15-token** input, while the trainer runs
sequences of ~400-2500 tokens. Length is the largest unexamined difference
between the probe that passes and the run that NaNs, and it is the axis on which
attention actually compounds: RMSNorm is per token and cannot accumulate with
length, but attention over many keys can.

It also reports the realised parameter dtype, because the earlier probe's "bf16"
row returned a loss and gradient norm identical to f32 to four decimals — which
is the signature of a cast that silently did not take, not of a model that is
numerically identical in two precisions.

    uv run python probe_length_sweep.py --lengths 15,256,1024,2048
"""

import argparse

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from rl_config import MODEL_CONFIG

MODELS = [
    ("exp163F", "timodonnell/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF-step404"),
    ("exp199", "timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199"),
]
# Repeating contact triples: the real response is overwhelmingly
# `<contact> <pI> <pJ>`, so this keeps the probe in distribution as it lengthens.
def make_ids(n: int) -> list[int]:
    head = [2, 11, 160, 160, 30, 161, 45, 9]
    body = []
    rng = np.random.default_rng(0)
    while len(head) + len(body) < n:
        i, j = rng.integers(143, 2142, size=2)
        body += [5, int(i), int(j)]
    return (head + body)[:n]


def run(name, repo, cfg, conv, lengths):
    print(f"=== {name} ===", flush=True)
    model = conv.load_pretrained(cfg.model_type, ref=repo, config=cfg, dtype=jnp.float32)
    dtypes = {str(jnp.asarray(l).dtype) for l in jax.tree_util.tree_leaves(model)
              if hasattr(l, "dtype")}
    print(f"  param dtypes: {sorted(dtypes)}", flush=True)

    for n in lengths:
        ids = make_ids(n)
        Pos = cfg.max_Pos.resize(len(ids))
        tokens = hax.named(jnp.array(ids, dtype=jnp.int32)[None, :], ("batch", Pos))

        def loss_fn(mm):
            out = mm(tokens, attn_mask=None, key=None)
            logits = (out.array if hasattr(out, "array") else out).astype(jnp.float32)
            logp = jax.nn.log_softmax(logits, axis=-1)
            tgt = jnp.array(ids, dtype=jnp.int32)[None, :]
            return -jnp.mean(jnp.take_along_axis(logp[:, :-1], tgt[:, 1:, None], axis=-1))

        try:
            loss, grads = jax.value_and_grad(loss_fn)(model)
            gl = [jnp.asarray(g).astype(jnp.float32) for g in jax.tree_util.tree_leaves(grads)
                  if hasattr(g, "dtype") and jnp.issubdtype(g.dtype, jnp.floating)]
            n_bad = sum(int(jnp.count_nonzero(~jnp.isfinite(g))) for g in gl)
            gnorm = float(jnp.sqrt(sum(float(jnp.sum(jnp.where(jnp.isfinite(g), g, 0) ** 2))
                                       for g in gl)))
            gmax = max(float(jnp.nanmax(jnp.abs(g))) for g in gl)
            print(f"  len={n:5d}  loss={float(loss):8.4f}  grad_norm={gnorm:14.2f}  "
                  f"max|g|={gmax:12.2f}  non-finite={n_bad}"
                  + ("   <-- FAILS HERE" if n_bad else ""), flush=True)
        except Exception as exc:
            print(f"  len={n:5d}  EXC {type(exc).__name__}: {str(exc)[:110]}", flush=True)
    del model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="15,256,1024,2048")
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    mesh = Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "model"))
    setter = getattr(jax, "set_mesh", None) or jax.sharding.use_mesh
    cfg = MODEL_CONFIG
    conv = cfg.hf_checkpoint_converter()
    with setter(mesh):
        for name, repo in MODELS:
            run(name, repo, cfg, conv, lengths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
