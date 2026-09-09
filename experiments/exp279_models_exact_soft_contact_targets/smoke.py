# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Small accelerator training through the actual stock entry point and exporter.

Run once per arm. Outputs are local synthetic-test artifacts, not scientific
training results. The separate inference_smoke.py uses MarinFold's own env.
"""

import argparse
import json
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from huggingface_hub import snapshot_download
from levanter.checkpoint import CheckpointerConfig, load_checkpoint
from levanter.data.text.datasets import DatasetComponent
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.main.train_lm import main as train_main
from levanter.optim.config import AdamConfig
from levanter.store.cache import CacheLedger
from levanter.store.tree_store import TreeStore
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import TrainerConfig

from .data import ContactDataConfig
from .model import ContactQwen3Config
from .recipe import TOKENIZER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("ce", "soft"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if jax.default_backend() == "cpu":
        raise RuntimeError("This smoke explicitly requires an accelerator")
    args.output.mkdir(parents=True, exist_ok=True)
    repo, revision = TOKENIZER.split("@")
    tokenizer_path = snapshot_download(
        repo, revision=revision, allow_patterns=["*token*", "special_tokens_map.json"]
    )
    doc = np.asarray(
        [
            2,
            8,
            143,
            86,
            144,
            86,
            145,
            86,
            3,
            150,
            4,
            151,
            9,
            5,
            143,
            144,
            5,
            143,
            145,
            10,
            1,
        ],
        np.int32,
    )
    cache_path = args.output / "cache"
    store = TreeStore.open(
        {"input_ids": np.zeros(0, np.int32)}, str(cache_path), mode="w"
    )
    store.extend([{"input_ids": doc} for _ in range(128)])
    ledger = CacheLedger(
        total_num_rows=128,
        shard_rows={"synthetic": 128},
        is_finished=True,
        finished_shards=["synthetic"],
        field_counts={"input_ids": 128 * len(doc)},
    )
    (cache_path / "shard_ledger.json").write_text(ledger.to_json())
    model_config = ContactQwen3Config(
        max_seq_len=64,
        hidden_dim=32,
        intermediate_dim=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=1,
        use_qk_norm=True,
        rope=Llama3RotaryEmbeddingsConfig(),
        attn_backend=AttentionBackend.VANILLA,
        tokenizer=tokenizer_path,
        soft_targets=args.arm == "soft",
        loss_block_size=16,
    )
    data = ContactDataConfig(
        tokenizer=tokenizer_path,
        shuffle=False,
        auto_build_caches=False,
        edge_capacity=8,
        components={
            name: DatasetComponent(
                cache_dir=str(cache_path),
                flat_cache=True,
                split=split,
                pack=True,
                format=TextLmDatasetFormat(text_key="document"),
            )
            for name, split in (("synthetic", "train"), ("validation", "validation"))
        },
        train_weights={"synthetic": 1.0},
    )
    run_name = f"exp279-{args.arm}-smoke"
    trainer = TrainerConfig(
        id=run_name,
        tracker=NoopConfig(),
        require_accelerator=True,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=4,
        per_device_parallelism=2,
        per_device_eval_parallelism=2,
        num_train_steps=3,
        steps_per_eval=2,
        max_eval_batches=1,
        log_dir=args.output / "logs",
        log_jaxprs=False,
        log_xla_hlo=False,
        load_checkpoint=False,
        checkpointer=CheckpointerConfig(
            base_path=str(args.output / "checkpoints" / run_name),
            append_run_id_to_base_path=False,
            save_interval=None,
            keep=[{"every": 1}],
        ),
    )
    config = TrainLmConfig(
        data=data,
        model=model_config,
        trainer=trainer,
        optimizer=AdamConfig(learning_rate=1e-3, warmup=0, decay=0),
        hf_save_path=str(args.output / "checkpoints" / run_name / "hf"),
        hf_save_steps=1,
        hf_save_dtype="bfloat16",
        hf_generation_eos_token_ids=[1, 10],
    )
    train_main(config)
    checkpoint = args.output / "checkpoints" / run_name / "step-2"
    template = model_config.build(hax.Axis("vocab", 2845), key=jax.random.PRNGKey(0))
    trained = load_checkpoint(
        template,
        str(checkpoint),
        subpath="model",
        mesh=trainer.device_mesh,
        axis_mapping={},
    )
    # Match the exported weight rounding, then compare forward passes in FP32.
    rounded = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16).astype(jnp.float32), trained
    )
    tokens = hax.named(doc, hax.Axis("position", len(doc)))
    # Match the ordinary CPU inference check's FP32 arithmetic; GPU defaults
    # otherwise permit TF32 matmuls even with FP32 operands.
    with jax.default_matmul_precision("highest"):
        logits = rounded(tokens, AttentionMask.causal()).array
    np.savez(args.output / "expected.npz", tokens=doc, logits=np.asarray(logits))
    report = {
        "arm": args.arm,
        "device": str(jax.devices()[0]),
        "steps": 3,
        "native_checkpoint": str(checkpoint),
        "hf_export": str(checkpoint.parent / "hf" / "step-2"),
    }
    for filename in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        if not (Path(report["hf_export"]) / filename).is_file():
            raise FileNotFoundError(f"Missing HF export artifact: {filename}")
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
