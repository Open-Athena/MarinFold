# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare ordinary CE, exact soft CE, target entropy and KL on fixed val packs.

Run on the checkpoint's compute/storage region. This complements the stock
ordinary-CE validation callback; contact accuracy still requires exp245 eval-val.
Both arms use identical input indices and the same scored-token denominator.
"""

import argparse
import asyncio
import csv
import json
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jmp
import numpy as np
from levanter.checkpoint import load_checkpoint
from levanter.data.text.datasets import PackedTokenDataset
from levanter.models.loss import maybe_fused_next_token_loss, next_token_loss_weight
from levanter.store.cache import TreeCache
from levanter.tokenizers import load_tokenizer

from .data import ContactPackedDataset
from .inputs import verify_manifest
from .model import contact_loss_terms, reference_model_config
from .recipe import TOKENIZER
from .targets import Vocabulary


@eqx.filter_jit
def diagnostic_sums(model, example):
    """Scored-token sums, so variable padding cannot change aggregation weights."""
    model = jmp.get_policy("p=f32,c=bfloat16").cast_to_compute(model)
    activations = model.activations(example.tokens, example.attn_mask)
    ce = maybe_fused_next_token_loss(
        model.Pos,
        model.Embed,
        model.Vocab,
        activations,
        model.get_lm_head(),
        example.tokens,
        loss_weight=example.loss_weight,
        reduction=None,
    )
    soft, entropy = contact_loss_terms(activations, model.get_lm_head(), example)
    weight = next_token_loss_weight(model.Pos, example.loss_weight)
    return {
        "ce_sum": hax.sum(ce).array,
        "soft_ce_sum": hax.sum(soft * weight).array,
        "entropy_sum": hax.sum(entropy * weight).array,
        "kl_sum": hax.sum((soft - entropy) * weight).array,
        "scored_tokens": hax.sum(weight).array,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="Exact native step-N checkpoint"
    )
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_examples <= 0:
        raise ValueError("max-examples must be positive")
    manifest = json.loads(args.manifest.read_text())
    verify_manifest(manifest)
    cache = TreeCache.load(
        manifest["inputs"]["val"]["cache_dir"], {"input_ids": np.zeros(0, np.int32)}
    )
    vocab = Vocabulary.from_tokenizer(load_tokenizer(TOKENIZER))
    dataset = ContactPackedDataset(
        PackedTokenDataset(cache, hax.Axis("position", 8192)), cache, vocab, 2731
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    started = time.perf_counter()
    config = reference_model_config(soft_targets=True)
    template = eqx.filter_eval_shape(
        lambda: config.build(hax.Axis("vocab", 2845), key=jax.random.PRNGKey(0))
    )
    model = load_checkpoint(
        template, args.checkpoint, subpath="model", mesh=mesh, axis_mapping={}
    )
    model_load_seconds = time.perf_counter() - started
    count = min(args.max_examples, asyncio.run(dataset.async_len()))
    if count == 0:
        raise ValueError("Validation cache contains no examples")
    rows = []
    for index in range(count):
        example = asyncio.run(dataset.get_batch([index]))[0]
        start = time.perf_counter()
        sums = {
            key: float(value) for key, value in diagnostic_sums(model, example).items()
        }
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "stem": f"val-pack-{index}",
                "mode": "teacher_forced_document_diagnostics",
                "model_nickname": args.checkpoint,
                "runner_tag": "local",
                **sums,
                "n_tokens": int(np.count_nonzero(example.tokens.array)),
                "elapsed_seconds": elapsed,
                "model_load_seconds": model_load_seconds,
                "total_seconds": model_load_seconds + elapsed,
                "gpu_name": jax.devices()[0].device_kind,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    denominator = sum(float(row["scored_tokens"]) for row in rows)
    result = {
        name.removesuffix("_sum"): sum(float(row[name]) for row in rows) / denominator
        for name in ("ce_sum", "soft_ce_sum", "entropy_sum", "kl_sum")
    }
    result.update(checkpoint=args.checkpoint, examples=count, scored_tokens=denominator)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
