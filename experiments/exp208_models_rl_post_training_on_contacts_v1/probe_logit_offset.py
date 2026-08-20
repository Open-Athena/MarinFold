# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Where does a ZERO logit rank? — issue #208 root cause.

vLLM pads the vocabulary to a hardware-friendly multiple: 2845 -> 2848, adding
three rows. Those rows are zero, so they produce a logit of exactly **0.0**.
Softmax is shift-invariant, so a model is free to sit anywhere on the logit axis
without changing its distribution — but whether a 0.0 logit is competitive
depends entirely on where the real logits sit.

Measured consequence (`probe_importance_ratio.py`, and a direct scan of the
spilled rollouts): the exp199 runs emit token ids 2845/2846/2847 in **12.4% of
all tokens and in 256 of 256 rollouts**, while the exp163 arm F control emits a
maximum id of 2142 and **zero** out-of-range tokens. Those ids do not exist in a
2845-row embedding, so the trainer scores them out of bounds.

This reports, per position, the top real logit and the rank/probability a 0.0
logit would take. If exp199's real logits sit low and exp163F's sit high, the
padding rows are sampleable for one and not the other, and that is the whole bug.
"""

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from levanter.layers.attention import AttentionMask

from rl_config import MODEL_CONFIG

IDS = [2, 11, 160, 160, 30, 161, 45, 9] + [5, 200, 400] * 40


def main() -> int:
    mesh = Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "model"))
    setter = getattr(jax, "set_mesh", None) or jax.sharding.use_mesh
    cfg = MODEL_CONFIG
    conv = cfg.hf_checkpoint_converter()
    ids = (IDS + [0] * 1024)[:1024]

    with setter(mesh):
        for name, repo in (("exp163F", "timodonnell/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos-tpuF-step404"),
                           ("exp199", "timodonnell/marinfold-contacts-v1-exp199-1_5b-step145199")):
            model = conv.load_pretrained(cfg.model_type, ref=repo, config=cfg)
            Pos = cfg.max_Pos.resize(len(ids))
            tok = hax.named(jnp.array(ids, dtype=jnp.int32)[None, :], ("batch", Pos))
            out = model(tok, attn_mask=AttentionMask.causal(), key=None)
            lg = np.asarray(jnp.asarray(out.array if hasattr(out, "array") else out)
                            .astype(jnp.float32))[0][:len(IDS)]

            top = lg.max(axis=-1)
            # A padding row contributes logit 0.0. Its softmax probability against
            # the real logits, and how many real tokens it would outrank.
            shifted = lg - top[:, None]
            denom = np.exp(shifted).sum(axis=-1)
            p_zero = np.exp(0.0 - top) / (denom + np.exp(0.0 - top))
            rank = (lg > 0.0).sum(axis=-1)
            print(f"  {name:8s} top_logit[min/med/max]={top.min():7.2f}/{np.median(top):7.2f}/"
                  f"{top.max():7.2f}   P(a zero-logit pad row)[med]={np.median(p_zero):.4f}   "
                  f"real tokens above 0.0 [med]={int(np.median(rank))} of {lg.shape[-1]}",
                  flush=True)
            del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
