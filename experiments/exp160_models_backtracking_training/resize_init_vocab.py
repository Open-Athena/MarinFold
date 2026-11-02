# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Grow exp120's checkpoint from the 2845-token vocab to the 3849-token superset (#160).

Why this step exists
--------------------
The plan called the superset tokenizer "a +1004 embedding resize, not a remap",
and that is true of the *vocabulary* — every one of exp120's 2845 ids keeps its
meaning, because the contacts-v1 vocab is append-only and ``<retract>`` was
appended last. But Levanter does not do the resize for you. Its warm-start path
builds the model at the tokenizer's vocab size and then loads the checkpoint
**strictly**::

    model = config.model.build(Vocab, key=model_key)          # Vocab = 3849
    model = load_checkpoint(model, checkpoint_path, subpath="model")

so pointing the run at exp120 directly dies with::

    ValueError: Axis vocab has different sizes in ('vocab', 'embed')
    and (2845, 2048): 3849 != 2845

This script does the resize once, offline, and writes a new checkpoint the
training job can load strictly.

What the resize does to the weights
-----------------------------------
``LmHeadModel.resize_vocab`` grows the token embedding matrix and the untied
lm_head along ``Vocab``, keeping rows 0..2844 exactly as trained and drawing
the 1004 new rows from the layer's own initializer. So the model starts as
*exactly* exp120 on every token it has seen before, and only ``<retract>`` (id
3848) plus the unused crops/ccoord ids begin untrained. The unused ids never
appear in the corpus, so they simply stay near their init.

Where to run it
---------------
On a v5p-8 in **us-east5-a**, with ``JAX_PLATFORMS=cpu``: we want the VM's 448
GB of host RAM and its in-region GCS access, not its chips (the whole job is a
load, a concatenate and a store). The marin CPU pool is n2-highmem-2 — 16 GiB,
too small to hold two copies of a 1.5B model in f32.

    iris --cluster=marin job run --no-wait --tpu v5p-8 --zone us-east5-a \\
        --enable-extra-resources --priority interactive --extra tpu \\
        --cpu 200 --memory 400GB --disk 100GB \\
        -- bash -lc 'JAX_PLATFORMS=cpu uv run python resize_init_vocab.py'

Idempotent: exits early if the destination already holds a checkpoint.
"""

from __future__ import annotations

import argparse

import contextlib

import fsspec
import haliax as hax
import jax
import numpy as np
from levanter.checkpoint import load_checkpoint, save_checkpoint

from train_backtracking import MODEL_CONFIG


@contextlib.contextmanager
def single_host_mesh():
    """Minimal device mesh.

    ``load_checkpoint`` / ``save_checkpoint`` go through haliax's partitioning
    layer, which raises ``ValueError: No mesh found`` outside a mesh context —
    building the model works fine without one, so the failure lands on the
    load, not on the build. Under ``JAX_PLATFORMS=cpu`` this is a 1x1x1 mesh
    over the single CPU device; the axis names match the marin training mesh.
    """
    devices = np.array(jax.devices()).reshape(1, -1, 1)
    with jax.sharding.Mesh(devices, ("replica", "data", "model")) as mesh:
        yield mesh

SRC = (
    "gs://marin-us-east5/protein-structure/MarinFold/"
    "exp120_regen_vs_reepoch_contacts_v1/checkpoints/"
    "exp120-cv1-1_5b-orig-lr3e-4-e1-cos-fb79f7/checkpoints/step-1005"
)
DST = (
    "gs://marin-us-east5/protein-structure/MarinFold/"
    "exp160_backtracking_training/init/exp120-step-1005-vocab3849"
)
# exp120 / Eric's contacts-v1 tokenizer.
SRC_VOCAB = 2845
# crops/ccoord superset incl. <retract> at 3848.
DST_VOCAB = 3849


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--src-vocab", type=int, default=SRC_VOCAB)
    ap.add_argument("--dst-vocab", type=int, default=DST_VOCAB)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    fs, _ = fsspec.core.url_to_fs(args.dst)
    if not args.force and fs.exists(f"{args.dst.rstrip('/')}/metadata.json"):
        print(f"already resized -> {args.dst}", flush=True)
        return

    print(f"jax devices: {jax.devices()}", flush=True)

    with single_host_mesh():
        src_vocab = hax.Axis("vocab", args.src_vocab)
        model = MODEL_CONFIG.build(src_vocab, key=jax.random.PRNGKey(0))
        print(f"built base model at vocab={args.src_vocab}", flush=True)

        model = load_checkpoint(model, args.src, subpath="model")
        print(f"loaded {args.src}", flush=True)

        before = np.asarray(model.embeddings.token_embeddings.weight.array)
        model = model.resize_vocab(args.dst_vocab, key=jax.random.PRNGKey(1))
        after = np.asarray(model.embeddings.token_embeddings.weight.array)
        print(f"resized embeddings {before.shape} -> {after.shape}", flush=True)

        # The rows we claim are untouched really are: this is the whole safety
        # argument for an append-only vocab, so assert it rather than assume it.
        assert (after[: args.src_vocab] == before).all(), "resize perturbed existing token rows"
        print(f"verified rows 0..{args.src_vocab - 1} are bit-identical", flush=True)

        # Saved under the `model` subpath so the training job's
        # `load_checkpoint(model, path, subpath="model")` finds it.
        save_checkpoint({"model": model}, step=1005, checkpoint_path=args.dst, is_temporary=False)
    print(f"saved -> {args.dst}", flush=True)


if __name__ == "__main__":
    main()
