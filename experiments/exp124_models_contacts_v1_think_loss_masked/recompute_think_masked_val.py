# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a HF checkpoint on exp124's think-masked validation cache."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import fsspec
import numpy as np


THINK_CACHE_ROOT = "gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/cache/think-masked/2026.07.29.2"
CACHE_EXEMPLAR = {
    "input_ids": np.zeros((0,), dtype=np.int32),
    "loss_weights": np.zeros((0,), dtype=np.float32),
}


def stage_directory(src: str, dst: Path, *, flatten: bool) -> Path:
    """Copy a remote directory to local disk for HF checkpoint loading."""
    if "://" not in src:
        return Path(src)

    dst.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(src)
    files = [entry for entry in fs.find(root, detail=True).values() if entry["type"] == "file"]
    if not files:
        raise FileNotFoundError(f"No files under path {src}")
    start = time.time()
    for entry in files:
        rel = os.path.basename(entry["name"]) if flatten else os.path.relpath(entry["name"], root)
        dest = dst / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(entry["name"], str(dest))
    size = sum(entry["size"] for entry in files)
    print(f"[exp124-eval] staged {src} -> {dst} ({len(files)} files, {size / 2**30:.2f} GiB, {time.time() - start:.0f}s)", flush=True)
    return dst


def write_json(record: dict[str, Any], path: str) -> None:
    with fsspec.open(path, "w") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model directory, local or remote.")
    parser.add_argument("--label", required=True, help="Short model label for the output record.")
    parser.add_argument("--cache-dir", default=THINK_CACHE_ROOT, help="Prebuilt think-masked cache root.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--per-device-eval-parallelism", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=2845)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--intermediate-dim", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=24)
    args = parser.parse_args()

    import equinox as eqx
    import jmp
    import optax
    from levanter.callbacks import eval_loss_loop
    from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
    from collections.abc import Sequence

    from haliax import Axis
    from levanter.data.dataset import AsyncDataset
    from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig, PackedTokenDataset
    from levanter.data.text.examples import GrugLmExample
    from levanter.store.cache import TreeCache
    from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
    from levanter.main.eval_lm import EvalLmConfig
    from levanter.models.qwen import Qwen3Config
    from levanter.tracker.tracker import NoopConfig
    from levanter.trainer import Trainer, TrainerConfig
    from levanter.utils.tree_utils import inference_mode
    import levanter.trainer

    model_dir = stage_directory(args.model, Path("/tmp/exp124_think_eval_model"), flatten=True)
    model_config = Qwen3Config(
        max_seq_len=args.max_seq_len,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_layers=args.num_layers,
        rope=Llama3RotaryEmbeddingsConfig(),
        tokenizer=str(model_dir),
    )
    class LazyThinkPackedDataset(AsyncDataset[GrugLmExample]):
        """Lazily construct the packed think dataset after JAX distributed init."""

        def __init__(self, split: str):
            self.split = split
            self._dataset: PackedTokenDataset | None = None

        def _inner(self) -> PackedTokenDataset:
            if self._dataset is None:
                self._dataset = PackedTokenDataset(
                    TreeCache.load(f"{args.cache_dir.rstrip('/')}/{self.split}", CACHE_EXEMPLAR),
                    Axis("position", args.max_seq_len),
                    max_segments_per_example=64,
                    slice_strategy="left",
                    loss_weights_key="loss_weights",
                    block_cross_document_attention=True,
                )
            return self._dataset

        async def async_len(self) -> int:
            return await self._inner().async_len()

        def is_finite(self) -> bool:
            return True

        async def get_batch(self, indices: Sequence[int]) -> Sequence[GrugLmExample]:
            return await self._inner().get_batch(indices)

    data = LmDataConfig(
        components={
            "contacts-v1-think-masked": DirectDatasetComponent(
                datasets={"validation": LazyThinkPackedDataset("validation")}
            )
        },
        tokenizer=str(model_dir),
        cache_dir=None,
        shuffle=False,
        block_cross_document_attention=True,
    )
    trainer_config = TrainerConfig(
        tracker=NoopConfig(),
        require_accelerator=True,
        per_device_eval_parallelism=args.per_device_eval_parallelism,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        max_eval_batches=args.max_eval_batches,
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    config = EvalLmConfig(
        hf_checkpoint=RepoRef(str(model_dir)),
        trainer=trainer_config,
        data=data,
        max_eval_length=args.max_seq_len,
        model=model_config,
    )

    levanter.trainer.initialize(config)
    batch_axis = config.trainer.EvalBatch
    pos = config.model.max_Pos.resize(config.max_eval_length)
    converter: HFCheckpointConverter = config.model.hf_checkpoint_converter()
    converter = converter.replaced(reference_checkpoint=config.hf_checkpoint, tokenizer=config.data.the_tokenizer)

    def train_loss_fn(model, example, *, key=None):
        return model.compute_next_token_loss(example, key=key, logsumexp_weight=None)

    with config.trainer.use_device_mesh():
        model = converter.load_pretrained(
            config.model.model_type,
            ref=config.hf_checkpoint,
            axis_mapping=config.trainer.parameter_axis_mapping,
            dtype=config.trainer.mp.compute_dtype,
        )
        with Trainer(config.trainer, optax.identity(), train_loss_fn, add_default_hooks=False) as trainer:
            log_dict: dict[str, Any] = {}
            for name, dataset in config.data.validation_sets(pos).items():
                if config.trainer.max_eval_batches is not None:
                    dataset = dataset.take(config.trainer.max_eval_batches * config.trainer.eval_batch_size)
                loader = trainer.data_loader(dataset, batch_axis)

                @eqx.filter_jit
                def eval_loss(model, batch):
                    eval_model = trainer.mp.cast_to_compute(model)
                    return trainer.loss_fn(eval_model, batch, key=None)

                loss, metrics = eval_loss_loop(
                    eval_loss,
                    inference_mode(model, True),
                    loader,
                    max_batches=config.trainer.max_eval_batches,
                    name=name,
                )
                log_dict[f"eval/{name}/loss"] = loss
                log_dict.update({f"eval/{name}/{key.removeprefix('eval/')}": value for key, value in metrics.items()})
            if len(config.data.validation_sets(pos)) == 1:
                (loss_key,) = [key for key in log_dict if key.endswith("/loss")]
                log_dict["eval/loss"] = log_dict[loss_key]

    record = {
        "label": args.label,
        "model": args.model,
        "cache_dir": args.cache_dir,
        "metric": "think_augmented_validation_loss_with_think_targets_masked",
        "max_eval_batches": args.max_eval_batches,
        "eval_batch_size": config.trainer.eval_batch_size,
        "vocab_size": args.vocab_size,
        **{key: float(value) for key, value in log_dict.items() if isinstance(value, (int, float))},
    }
    write_json(record, args.output)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
