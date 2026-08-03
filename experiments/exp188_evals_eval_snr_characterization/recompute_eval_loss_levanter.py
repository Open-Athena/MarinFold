# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Recompute exp117 contacts-v1 validation loss with Levanter's eval path."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import fsspec


def configure_s3() -> None:
    """Normalize CoreWeave S3 fsspec config across fsspec/s3fs versions."""
    raw = os.environ.get("FSSPEC_S3_CONFIG_KWARGS")
    if not raw:
        return
    parsed = json.loads(raw)
    # botocore Config expects service-specific options under the "s3" key.
    fsspec.config.conf.setdefault("s3", {})["config_kwargs"] = parsed


def stage_directory(src: str, dst: Path, *, flatten: bool) -> Path:
    """Copy a remote directory to local disk."""
    if "://" not in src:
        return Path(src)

    dst.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(src)
    files = [entry for entry in fs.find(root, detail=True).values() if entry["type"] == "file"]
    if not files:
        raise FileNotFoundError(f"No files under path {src}")
    start = time.time()
    for entry in files:
        if flatten:
            rel = os.path.basename(entry["name"])
        else:
            rel = os.path.relpath(entry["name"], root)
        dest = dst / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        fs.get_file(entry["name"], str(dest))
    size = sum(entry["size"] for entry in files)
    print(
        f"[levanter-loss] staged {src} -> {dst} "
        f"({len(files)} files, {size / 2**30:.2f} GiB, {time.time() - start:.0f}s)",
        flush=True,
    )
    return dst


def stage_model(src: str, dst: Path) -> Path:
    return stage_directory(src, dst, flatten=True)


def stage_checkpoint(src: str | None, dst: Path) -> str | None:
    if src is None:
        return None
    if src.startswith("hf://"):
        return str(stage_directory(src, dst, flatten=False))
    return src


def write_json(record: dict[str, Any], path: str) -> None:
    with fsspec.open(path, "w") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model directory used for tokenizer, and for weights unless --checkpoint-path is set.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer-only directory. Use with --checkpoint-path to avoid staging HF model weights.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional native Levanter checkpoint path to load instead of HF weights.")
    parser.add_argument(
        "--cache-dir",
        default="gs://marin-us-east5/tokenized/contacts-v1-val/2026.07.13.1",
        help="Canonical contacts-v1-val Levanter cache root, or a cache output root when --raw-validation-url is set.",
    )
    parser.add_argument(
        "--raw-validation-url",
        action="append",
        default=[],
        help="Raw validation shard URL to tokenize into --cache-dir before eval. May be repeated.",
    )
    parser.add_argument("--text-key", default="document", help="Text column for --raw-validation-url shards.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--per-device-eval-parallelism", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=None, help="Optional model vocab axis size; defaults to tokenizer length.")
    parser.add_argument("--model-arch", choices=("qwen3", "llama"), default="qwen3")
    parser.add_argument("--rope-type", choices=("llama3", "default"), default="llama3")
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--rope-factor", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--intermediate-dim", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--use-qk-norm", action="store_true")
    parser.add_argument("--attn-backend", default=None, choices=("default", "nvte", "splash", "jax_flash", "vanilla"))
    parser.add_argument(
        "--loss-weight-mode",
        choices=("example", "legacy-uniform"),
        default="example",
        help="`example` uses dataset-provided loss weights; `legacy-uniform` reproduces the pre-padding-successor-mask packed eval objective.",
    )
    parser.add_argument("--padding-target-loss", choices=("mask", "include"), default="mask")
    parser.add_argument(
        "--eval-mode",
        choices=("tagged", "train-hook"),
        default="tagged",
        help="Use exp188 tagged eval_model path or Levanter's train-time validation callback path.",
    )
    parser.add_argument(
        "--allow-cross-document-attention",
        action="store_true",
        help="Disable Levanter document-boundary attention masks for packed eval examples.",
    )
    args = parser.parse_args()
    configure_s3()

    import jmp
    from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
    from levanter.data.text.formats import TextLmDatasetFormat
    import equinox as eqx
    import optax
    from levanter.eval import eval_model
    from levanter.main.eval_lm import EvalLmConfig
    from levanter.layers.attention import AttentionBackend
    from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, Llama3RotaryEmbeddingsConfig
    from levanter.models.llama import LlamaConfig
    from levanter.models.loss import maybe_fused_next_token_loss, next_token_loss_weight
    from levanter.models.qwen import Qwen3Config
    from levanter.tracker.tracker import NoopConfig
    from levanter.trainer import Trainer, TrainerConfig
    from levanter.checkpoint import load_checkpoint
    from levanter.compat.hf_checkpoints import RepoRef
    import levanter.trainer

    tokenizer_src = args.tokenizer if args.checkpoint_path and args.tokenizer else args.model
    model_dir = stage_model(tokenizer_src, Path("/tmp/marinfold_exp188_levanter_model"))
    checkpoint_path = stage_checkpoint(args.checkpoint_path, Path("/tmp/marinfold_exp188_native_checkpoint"))

    rope_config = (
        Llama3RotaryEmbeddingsConfig()
        if args.rope_type == "llama3"
        else DefaultRotaryEmbeddingsConfig(theta=args.rope_theta, factor=args.rope_factor)
    )
    config_cls = Qwen3Config if args.model_arch == "qwen3" else LlamaConfig
    model_config = config_cls(
        max_seq_len=args.max_seq_len,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_layers=args.num_layers,
        head_dim=args.head_dim,
        use_qk_norm=args.use_qk_norm,
        attn_backend=AttentionBackend(args.attn_backend) if args.attn_backend is not None else None,
        rope=rope_config,
        tokenizer=str(model_dir),
    )
    if args.raw_validation_url:
        validation_component = DatasetComponent(
            source=UrlDatasetSourceConfig(validation_urls=args.raw_validation_url),
            cache_dir=args.cache_dir,
            format=TextLmDatasetFormat(text_key=args.text_key),
            pack=True,
            split="validation",
        )
    else:
        validation_component = DatasetComponent(cache_dir=args.cache_dir, pack=True)

    data = LmDataConfig(
        components={"tokenized/contacts-v1-val": validation_component},
        tokenizer=str(model_dir),
        cache_dir=None,
        shuffle=False,
        block_cross_document_attention=not args.allow_cross_document_attention,
        padding_target_loss=args.padding_target_loss,
    )
    trainer = TrainerConfig(
        tracker=NoopConfig(),
        require_accelerator=True,
        per_device_eval_parallelism=args.per_device_eval_parallelism,
        mp=jmp.get_policy("compute=bfloat16,params=bfloat16,output=f32"),
        max_eval_batches=args.max_eval_batches,
        log_jaxprs=False,
        log_xla_hlo=False,
    )

    config = EvalLmConfig(
        checkpoint_path=checkpoint_path,
        hf_checkpoint=None if checkpoint_path is not None else RepoRef(str(model_dir)),
        trainer=trainer,
        data=data,
        max_eval_length=args.max_seq_len,
        model=model_config,
    )

    levanter.trainer.initialize(config)
    tokenizer = config.data.the_tokenizer
    batch = config.trainer.EvalBatch
    pos = config.model.max_Pos.resize(config.max_eval_length)
    datasets = config.data.tagged_eval_sets(pos)
    if config.trainer.max_eval_batches is not None:
        max_examples = config.trainer.max_eval_batches * config.trainer.eval_batch_size
        datasets = [(ds.take(max_examples), tags) for ds, tags in datasets]
    else:
        max_examples = None

    from levanter.main.eval_lm import TaggedEvaluator
    from levanter.utils.tree_utils import inference_mode
    from levanter.compat.hf_checkpoints import HFCheckpointConverter

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    def load_model():
        if checkpoint_path is not None:
            import jax
            from haliax.partitioning import round_axis_for_partitioning
            from haliax import Axis
            from levanter.utils.jax_utils import use_cpu_device
            vocab_size = args.vocab_size or len(tokenizer)
            vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
            with use_cpu_device():
                loaded = eqx.filter_eval_shape(config.model.build, vocab, key=jax.random.PRNGKey(0))
                loaded = load_checkpoint(loaded, checkpoint_path, subpath="model")
            import haliax as hax
            return hax.shard_with_axis_mapping(loaded, parameter_axis_mapping)

        converter: HFCheckpointConverter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(reference_checkpoint=config.hf_checkpoint, tokenizer=tokenizer)
        return converter.load_pretrained(
            config.model.model_type,
            ref=config.hf_checkpoint,
            axis_mapping=parameter_axis_mapping,
            dtype=config.trainer.mp.compute_dtype,
        )

    with config.trainer.use_device_mesh():
        model = load_model()

        if args.eval_mode == "train-hook":
            from levanter.callbacks import eval_loss_loop

            def train_loss_fn(model, example, *, key=None):
                return model.compute_next_token_loss(example, key=key, logsumexp_weight=None)

            with Trainer(config.trainer, optax.identity(), train_loss_fn, add_default_hooks=False) as trainer:
                log_dict = {}
                for name, dataset in config.data.validation_sets(pos).items():
                    if config.trainer.max_eval_batches is not None:
                        dataset = dataset.take(config.trainer.max_eval_batches * config.trainer.eval_batch_size)
                    loader = trainer.data_loader(dataset, batch)

                    @eqx.filter_jit
                    def eval_loss(model, batch):
                        model = trainer.mp.cast_to_compute(model)
                        return trainer.loss_fn(model, batch, key=None)

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
        else:
            def loss_fn(model, example):
                import jax.numpy as jnp
                model = inference_mode(model, True)
                model = config.trainer.mp.cast_to_compute(model)
                if args.loss_weight_mode == "legacy-uniform":
                    activations = model.activations(example.tokens, example.attn_mask, key=None)
                    if isinstance(activations, tuple):
                        activations = activations[0]
                    per_pos_loss = maybe_fused_next_token_loss(
                        model.Pos,
                        model.Embed,
                        model.Vocab,
                        activations,
                        model.get_lm_head(),
                        example.tokens,
                        loss_weight=None,
                        reduction=None,
                        reduction_axis=(),
                    ).array
                    legacy_weight = example.tokens.astype(jnp.float32) * 0.0 + 1.0
                    per_pos_weight = next_token_loss_weight(model.Pos, legacy_weight).array
                else:
                    per_pos_loss = model.compute_next_token_loss(example, reduction=None, reduction_axis=()).array
                    per_pos_weight = example.loss_weight.array
                per_pos_token_id = jnp.roll(example.tokens.array, -1, axis=-1)
                return per_pos_loss, per_pos_weight, per_pos_token_id

            evaluator = TaggedEvaluator(
                EvalBatch=batch,
                tagged_eval_sets=datasets,
                loss_fn=loss_fn,
                tokenizer=tokenizer,
                axis_mapping=compute_axis_mapping,
                max_examples_per_dataset=max_examples,
            )
            log_dict = eval_model(evaluator, model, prefix="eval")

    record = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "checkpoint_path": args.checkpoint_path,
        "resolved_checkpoint_path": checkpoint_path,
        "cache_dir": args.cache_dir,
        "raw_validation_url": args.raw_validation_url,
        "text_key": args.text_key,
        "max_eval_batches": args.max_eval_batches,
        "eval_batch_size": config.trainer.eval_batch_size,
        "eval_mode": args.eval_mode,
        "loss_weight_mode": args.loss_weight_mode,
        "padding_target_loss": args.padding_target_loss,
        "block_cross_document_attention": not args.allow_cross_document_attention,
        "vocab_size": args.vocab_size or len(tokenizer),
        "model_config": {
            "model_arch": args.model_arch,
            "rope_type": args.rope_type,
            "rope_theta": args.rope_theta,
            "rope_factor": args.rope_factor,
            "max_seq_len": args.max_seq_len,
            "hidden_dim": args.hidden_dim,
            "intermediate_dim": args.intermediate_dim,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "num_layers": args.num_layers,
            "head_dim": args.head_dim,
            "use_qk_norm": args.use_qk_norm,
            "attn_backend": args.attn_backend,
        },
        **{key: float(value) for key, value in log_dict.items() if isinstance(value, (int, float))},
    }
    write_json(record, args.output)
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
