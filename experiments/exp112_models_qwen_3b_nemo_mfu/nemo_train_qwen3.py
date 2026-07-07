# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""In-container NeMo 2.0 training/benchmark entrypoint for exp112 (issue #112).

Runs **inside** the ``nvcr.io/nvidia/nemo`` container under
``torchrun --standalone --nproc-per-node=<G>``. It is SELF-CONTAINED (imports
only NeMo/Megatron/torch, never the launcher's ``common.py``) because
``dispatch_nemo.py`` base64-inlines this file into the pod's bootstrap — there is
no repo checkout in the container.

Builds the exp112 Qwen3 ~2.9B model (geometry identical to exp108, so the MFU
number is apples-to-apples), trains a few hundred steps on **mock/synthetic data**
by default (MFU is compute-bound and data-content-independent — this is how
NVIDIA publishes its perf tables), and writes a throughput/MFU summary.

Primary metric is **throughput** (tokens/s, s/step) — formula-free and directly
comparable to exp108's Levanter numbers (~52k tok/s, ~20 s/step, ~15% MFU).
Framework/analytical **MFU** is reported as a cross-check.

Example (real benchmark, on the pod)::

    torchrun --standalone --nproc-per-node=8 nemo_train_qwen3.py \
        --max-steps 300 --micro-batch-size 1 --global-batch-size 128

Example (local API smoke on 1 small GPU)::

    torchrun --standalone --nproc-per-node=1 nemo_train_qwen3.py \
        --tiny --max-steps 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp112 NeMo Qwen3-3B MFU benchmark")
    # Parallelism / trainer
    p.add_argument("--devices", type=int, default=int(os.environ.get("EXP112_DEVICES", "8")))
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--pipeline-parallel", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("EXP112_MAX_STEPS", "300")))
    p.add_argument("--precision", default=os.environ.get("EXP112_PRECISION", "bf16-mixed"))
    # Batch / sequence
    p.add_argument("--micro-batch-size", type=int, default=int(os.environ.get("EXP112_MICRO_BATCH", "1")))
    p.add_argument("--global-batch-size", type=int, default=int(os.environ.get("EXP112_GLOBAL_BATCH", "128")))
    p.add_argument("--seq-length", type=int, default=8192)
    # Model geometry (defaults = exp112/exp108 Qwen3 ~2.9B)
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--ffn-hidden-size", type=int, default=8192)
    p.add_argument("--num-attention-heads", type=int, default=32)
    p.add_argument("--num-query-groups", type=int, default=8)
    p.add_argument("--kv-channels", type=int, default=64)
    p.add_argument("--rotary-base", type=float, default=1_000_000.0)
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="disable activation checkpointing (exp108 needed it ON to fit)")
    # Optim (only meaningful for a real run; harmless for the benchmark)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.2)
    p.add_argument("--warmup-fraction", type=float, default=0.1)
    # Data
    p.add_argument("--data", default=os.environ.get("EXP112_DATA", "mock"),
                   help="'mock' (benchmark) or 'real' (Megatron .bin/.idx via --train/--val-data-prefix)")
    p.add_argument("--train-data-prefix", default=os.environ.get("EXP112_TRAIN_DATA", "/tmp/exp112/data/train_document"),
                   help="local Megatron .bin/.idx prefix for training (bootstrap downloads it from S3)")
    p.add_argument("--val-data-prefix", default=os.environ.get("EXP112_VAL_DATA", "/tmp/exp112/data/val_document"),
                   help="local Megatron .bin/.idx prefix for validation")
    p.add_argument("--tokenizer", default=os.environ.get("EXP112_TOKENIZER", "timodonnell/contacts-v1-tokenizer"))
    # Checkpointing + resume (real run; batch priority is preemptible). No shared
    # FS on cw-rno2a -> checkpoint to node-local disk + sync to S3; resume by
    # downloading the latest from S3 before AutoResume.
    p.add_argument("--checkpoint-dir", default=os.environ.get("EXP112_CKPT_DIR", "/tmp/exp112/ckpt"))
    p.add_argument("--checkpoint-s3", default=os.environ.get("EXP112_CKPT_S3", ""),
                   help="s3://…/checkpoints/<run>  — where to sync sharded checkpoints (empty = no sync)")
    p.add_argument("--checkpoint-every", type=int, default=int(os.environ.get("EXP112_CKPT_EVERY", "500")))
    p.add_argument("--keep-top-k", type=int, default=int(os.environ.get("EXP112_KEEP_TOPK", "1")))
    p.add_argument("--resume", action="store_true", default=bool(os.environ.get("EXP112_RESUME")),
                   help="download the latest S3 checkpoint (if any) and resume from it")
    # Validation
    p.add_argument("--val-check-interval", type=int, default=int(os.environ.get("EXP112_VAL_INTERVAL", "1000")))
    p.add_argument("--limit-val-batches", type=int, default=int(os.environ.get("EXP112_VAL_BATCHES", "50")))
    # Logging / output
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "MarinFold"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "open-athena"))
    p.add_argument("--wandb-name", default=os.environ.get("EXP112_RUN_NAME", "plm-exp112-cv1-3b-nemo-bench"))
    p.add_argument("--results-json", default=os.environ.get("EXP112_RESULTS_JSON", "/tmp/exp112_mfu.json"))
    p.add_argument("--measure-warmup-steps", type=int, default=int(os.environ.get("EXP112_WARMUP_MEASURE", "10")),
                   help="steps to skip before timing steady-state (compile/cache warmup)")
    # Local API smoke: shrink everything so it fits a tiny GPU
    p.add_argument("--tiny", action="store_true", help="tiny model+seq for a local API smoke test")
    return p.parse_args()


def apply_tiny(args: argparse.Namespace) -> None:
    args.num_layers = 4
    args.hidden_size = 256
    args.ffn_hidden_size = 1024
    args.num_attention_heads = 8
    args.num_query_groups = 4
    args.kv_channels = 32
    args.seq_length = 256
    args.micro_batch_size = 1
    args.global_batch_size = max(1, args.devices)
    args.measure_warmup_steps = 1


# --------------------------------------------------------------------------- #
# Throughput / MFU
# --------------------------------------------------------------------------- #
def model_flops_per_token(args: argparse.Namespace, vocab_size: int) -> float:
    """Analytical **model** FLOPs per token (fwd+bwd, causal), Megatron/PaLM-style.

    'Model' FLOPs = the useful work, EXCLUDING activation-recompute — so
    gradient checkpointing lowers MFU by lengthening step time, not by inflating
    this count (the standard MFU convention, matching Levanter's).

    Per token (multiply by tokens for per-step): the 6*N term is folded into the
    explicit per-layer GEMMs so GQA (kv heads < q heads) and SwiGLU (3 FFN
    matmuls) are counted correctly.
    """
    h = args.hidden_size
    L = args.num_layers
    s = args.seq_length
    f = args.ffn_hidden_size
    hq = args.num_attention_heads * args.kv_channels          # q projection out-dim
    hkv = args.num_query_groups * args.kv_channels            # k,v projection out-dim
    # Per layer, per token, forward matmul MACs -> *2 for FLOPs, *3 for fwd+bwd.
    # Attention projections: q(h*hq) + k(h*hkv) + v(h*hkv) + out(hq*h)
    attn_proj = h * hq + 2 * (h * hkv) + hq * h
    # Attention scores+context are seq-dependent: ~2 * s * hq per token (QK^T + AV),
    # causal halves it -> s * hq.
    attn_sdp = s * hq
    # SwiGLU MLP: gate(h*f) + up(h*f) + down(f*h)
    mlp = 3 * (h * f)
    per_layer = 2 * (attn_proj + attn_sdp + mlp)  # 2 = MAC->FLOP
    # Embedding + LM head (tied? we keep untied -> lm head): 2 * h * V (fwd)
    head = 2 * h * vocab_size
    fwd = L * per_layer + head
    return 3.0 * fwd  # fwd + bwd ~= 3x forward


class ThroughputState:
    """Plain timing state (NOT a Lightning Callback — a thin Callback subclass
    delegates to it inside main()). Kept separate so multiple-inheritance MRO
    can't let Lightning's base-``Callback`` no-op hooks shadow our timers.

    Records median steady-state per-global-step wall time -> tokens/s + MFU.
    """

    def __init__(self, measure_warmup, tokens_per_step, flops_per_step, world_size, peak_tflops=989.0):
        self.measure_warmup = measure_warmup
        self.tokens_per_step = tokens_per_step
        self.flops_per_step = flops_per_step
        self.world_size = world_size
        self.peak = peak_tflops * 1e12
        self._t = None
        self._durations = []
        self._step = 0

    def start(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t = time.perf_counter()

    def stop(self):
        import torch
        if self._t is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t
        self._step += 1
        if self._step > self.measure_warmup:
            self._durations.append(dt)

    def summary(self) -> dict:
        if not self._durations:
            return {"measured_steps": 0}
        step_time = statistics.median(self._durations)
        toks = self.tokens_per_step / step_time
        achieved_flops = self.flops_per_step / step_time
        return {
            "measured_steps": len(self._durations),
            "median_step_time_s": round(step_time, 4),
            "tokens_per_step": self.tokens_per_step,
            "tokens_per_s": round(toks, 1),
            "tokens_per_s_per_gpu": round(toks / self.world_size, 1),
            "model_tflops_per_gpu": round(achieved_flops / self.world_size / 1e12, 2),
            "mfu": round(achieved_flops / (self.peak * self.world_size), 4),
        }


# --------------------------------------------------------------------------- #
# S3 checkpoint persistence (no shared/persistent FS on cw-rno2a; batch priority
# is preemptible). Megatron writes a SHARDED (DCP) checkpoint dir; each node
# holds its own ranks' shards on local disk. We sync per-node to a shared S3
# prefix (union = complete checkpoint) and, on resume, mirror the latest back to
# every node. Coordination uses torchrun's RANK/LOCAL_RANK env (no torch.dist
# needed for the file I/O) + a torch.distributed.barrier where a group exists.
# --------------------------------------------------------------------------- #
def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _s3():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),  # in-cluster: http://cwlota.com
        config=Config(s3={"addressing_style": "virtual"}, retries={"max_attempts": 10}),
    )


def _split_s3(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), uri
    b, _, k = uri[5:].partition("/")
    return b, k.rstrip("/")


def _barrier():
    import torch
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _upload_tree(local_dir, s3_uri):
    """Upload every file under local_dir to s3_uri/<relpath> (this node's shards)."""
    import concurrent.futures as cf
    from pathlib import Path
    s3 = _s3()
    bucket, prefix = _split_s3(s3_uri)
    files = [p for p in Path(local_dir).rglob("*") if p.is_file()]
    def _put(p):
        rel = p.relative_to(local_dir).as_posix()
        s3.upload_file(str(p), bucket, f"{prefix}/{rel}")
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_put, files))
    return len(files)


def _download_tree(s3_uri, local_dir):
    """Mirror s3_uri/* down into local_dir (full checkpoint on every node)."""
    import concurrent.futures as cf
    from pathlib import Path
    s3 = _s3()
    bucket, prefix = _split_s3(s3_uri)
    keys = []
    for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        keys += [o["Key"] for o in pg.get("Contents", [])]
    def _get(key):
        rel = key[len(prefix) + 1:]
        dst = Path(local_dir) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dst))
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_get, keys))
    return len(keys)


def _s3_read_text(s3_uri) -> str | None:
    s3 = _s3()
    bucket, key = _split_s3(s3_uri)
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    except Exception:  # noqa: BLE001 — missing = None
        return None


def _s3_write_text(s3_uri, text: str):
    s3 = _s3()
    bucket, key = _split_s3(s3_uri)
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode())


def _list_s3_subdirs(s3_uri) -> list[str]:
    s3 = _s3()
    bucket, prefix = _split_s3(s3_uri)
    out = set()
    for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/", Delimiter="/"):
        for cp in pg.get("CommonPrefixes", []):
            out.add(cp["Prefix"][len(prefix) + 1:].rstrip("/"))
    return sorted(out)


def _delete_s3_subdir(s3_uri):
    s3 = _s3()
    bucket, prefix = _split_s3(s3_uri)
    keys = []
    for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        keys += [{"Key": o["Key"]} for o in pg.get("Contents", [])]
    for i in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=bucket, Delete={"Objects": keys[i:i + 1000]})


def restore_latest_checkpoint_from_s3(ckpt_dir: str, s3_uri: str) -> str | None:
    """Before training: node-level (LOCAL_RANK 0) mirror the latest S3 checkpoint
    into ckpt_dir so AutoResume finds it. Peers wait on a local marker. Returns
    the restored subdir name, or None if there is no checkpoint yet."""
    import time
    from pathlib import Path
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    marker = Path(ckpt_dir) / ".restore_done"
    local_rank = _env_int("LOCAL_RANK", 0)
    latest_uri = f"{s3_uri}/latest.txt"
    if local_rank == 0:
        latest = _s3_read_text(latest_uri)
        if latest:
            latest = latest.strip()
            n = _download_tree(f"{s3_uri}/{latest}", str(Path(ckpt_dir) / latest))
            print(f"[exp112] restored checkpoint '{latest}' from S3 ({n} files) -> {ckpt_dir}", flush=True)
            marker.write_text(latest)
        else:
            marker.write_text("")
    else:
        for _ in range(1200):  # up to ~20 min
            if marker.exists():
                break
            time.sleep(1)
    val = marker.read_text().strip() if marker.exists() else ""
    return val or None


# --------------------------------------------------------------------------- #
# Callbacks — MODULE LEVEL (not in main()). NeMo's ModelCheckpoint io-dumps the
# TrainerContext (incl. callbacks) via fiddle, which pyref's each callback CLASS;
# a class defined inside main() is `main.<locals>.X` and can't be pyref'd (fiddle
# AttributeError). Module-level classes with PRIMITIVE-only init args serialize
# cleanly. `_Callback` is imported here (this file only ever runs in-container);
# fall back to `object` so `--help` works in the launcher env without lightning.
# --------------------------------------------------------------------------- #
try:
    from lightning.pytorch import Callback as _Callback
except Exception:  # noqa: BLE001
    try:
        from pytorch_lightning import Callback as _Callback
    except Exception:  # noqa: BLE001
        _Callback = object


def _pick_latest(subdirs):
    """Highest-step checkpoint subdir (deterministic across nodes)."""
    def step_of(d):
        m = re.search(r"step=?(\d+)", d.name)
        return int(m.group(1)) if m else -1
    return max(subdirs, key=lambda d: (step_of(d), d.stat().st_mtime))


class ThroughputCallback(_Callback):
    """Times steady-state global steps → tokens/s + MFU (via a ThroughputState)."""

    def __init__(self, measure_warmup: int, tokens_per_step: int, flops_per_step: float,
                 world_size: int, peak_tflops: float = 989.0):
        super().__init__()
        self.state = ThroughputState(measure_warmup, tokens_per_step, flops_per_step,
                                     world_size, peak_tflops)

    def on_train_batch_start(self, *a, **k):
        self.state.start()

    def on_train_batch_end(self, *a, **k):
        self.state.stop()


class S3CheckpointSync(_Callback):
    """After each local checkpoint, sync the newest sharded dir to S3 (each node
    uploads its own shards; union = complete). Prune S3 to just the latest +
    commit latest.txt. Non-fatal on error (checkpoint still on local disk)."""

    def __init__(self, checkpoint_s3: str, checkpoint_dir: str, checkpoint_every: int):
        super().__init__()
        self.checkpoint_s3 = checkpoint_s3
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every

    def on_train_batch_end(self, trainer, *a, **k):
        if not self.checkpoint_s3:
            return
        step = int(trainer.global_step)
        if step <= 0 or step % self.checkpoint_every != 0:
            return
        try:
            from pathlib import Path
            _barrier()  # all ranks finished the (sync) save
            ck = Path(self.checkpoint_dir)
            subdirs = [d for d in ck.iterdir() if d.is_dir() and not d.name.startswith(".")] if ck.exists() else []
            if not subdirs:
                return
            latest = _pick_latest(subdirs)
            n = _upload_tree(str(latest), f"{self.checkpoint_s3}/{latest.name}")
            _barrier()  # all nodes finished uploading their shards
            if _env_int("RANK", 0) == 0:
                for name in _list_s3_subdirs(self.checkpoint_s3):
                    if name != latest.name:
                        _delete_s3_subdir(f"{self.checkpoint_s3}/{name}")
                _s3_write_text(f"{self.checkpoint_s3}/latest.txt", latest.name)
                print(f"[exp112] checkpoint synced to S3: {latest.name} (node shards: {n} files)", flush=True)
        except Exception as e:  # noqa: BLE001 — never kill training over a sync
            print(f"[exp112] S3 checkpoint sync error (non-fatal): {e}", flush=True)


def main() -> None:
    args = parse_args()
    if args.tiny:
        apply_tiny(args)

    import torch
    import torch.nn.functional as F
    from nemo import lightning as nl
    from nemo.collections import llm
    from nemo.collections.llm.gpt.data.mock import MockDataModule

    # Tokenizer (tiny, from HF) -> real vocab size for faithful head FLOPs.
    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    tokenizer = AutoTokenizer(pretrained_model_name=args.tokenizer)
    vocab_size = tokenizer.vocab_size

    # --- Model: Qwen3 geometry via the generic Megatron-backed GPTConfig ------
    config = llm.GPTConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,   # GQA
        kv_channels=args.kv_channels,
        seq_length=args.seq_length,
        # Qwen3 specifics
        normalization="RMSNorm",
        qk_layernorm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        position_embedding_type="rope",
        rotary_base=args.rotary_base,
        rotary_percent=1.0,
        share_embeddings_and_output_weights=False,
        make_vocab_size_divisible_by=128,
        bf16=True,
        # Activation checkpointing (exp108 needed it ON to fit 48L*seq8192).
        recompute_granularity=None if args.no_grad_ckpt else "full",
        recompute_method=None if args.no_grad_ckpt else "uniform",
        recompute_num_layers=None if args.no_grad_ckpt else 1,
    )
    model = llm.GPTModel(config, tokenizer=tokenizer)

    # --- Data -----------------------------------------------------------------
    if args.data == "mock":
        data = MockDataModule(
            seq_length=args.seq_length,
            tokenizer=tokenizer,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
        )
    else:
        # Real run: Megatron .bin/.idx (local prefixes; bootstrap downloaded them
        # from S3). Explicit train/validation/test splits. reset_*: pack docs but
        # DON'T attend/count position across the <eod> boundary (block cross-doc
        # attention — matches exp108's block_cross_document_attention).
        from pathlib import Path

        from nemo.collections.llm.gpt.data import PreTrainingDataModule
        # Megatron writes .npy sample-index maps here — MUST be writable (keep it
        # off the data dir, which may be a read-only mount).
        idx_dir = os.environ.get("EXP112_INDEX_DIR", "/tmp/exp112/index")
        Path(idx_dir).mkdir(parents=True, exist_ok=True)
        data = PreTrainingDataModule(
            paths={
                "train": [args.train_data_prefix],
                "validation": [args.val_data_prefix],
                "test": [args.val_data_prefix],
            },
            seq_length=args.seq_length,
            tokenizer=tokenizer,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            reset_attention_mask=True,
            reset_position_ids=True,
            eod_mask_loss=False,
            index_mapping_dir=idx_dir,
            num_workers=4,
        )

    # --- Optimizer / schedule (tracks #75; irrelevant to MFU) -----------------
    from megatron.core.optimizer import OptimizerConfig
    from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
    opt = MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=args.lr,
            weight_decay=args.weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.95,
            bf16=True,
            use_distributed_optimizer=True,
            clip_grad=1.0,
        ),
        lr_scheduler=CosineAnnealingScheduler(
            warmup_steps=max(1, int(args.warmup_fraction * args.max_steps)),
            constant_steps=0,
            min_lr=args.lr * 0.1,
        ),
    )

    # --- Trainer --------------------------------------------------------------
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tensor_parallel,
        pipeline_model_parallel_size=args.pipeline_parallel,
        sequence_parallel=False,
        ckpt_async_save=False,
    )
    world_size = args.devices * args.num_nodes
    tokens_per_step = args.global_batch_size * args.seq_length
    flops_per_step = model_flops_per_token(args, vocab_size) * tokens_per_step

    is_real = args.data != "mock"
    # NeMo writes checkpoints under <log_dir>/<name>/checkpoints; align our S3
    # sync + restore to that exact dir (compute BEFORE building callbacks).
    log_dir = os.environ.get("EXP112_LOG_DIR", "/tmp/exp112/nemo_logs")
    ckpt_dir = os.path.join(log_dir, args.wandb_name, "checkpoints")
    args.checkpoint_dir = ckpt_dir

    callbacks = [ThroughputCallback(args.measure_warmup_steps, tokens_per_step, flops_per_step, world_size)]
    tp_state = callbacks[0].state  # for the end-of-run summary
    if is_real and args.checkpoint_s3:
        callbacks.append(S3CheckpointSync(args.checkpoint_s3, ckpt_dir, args.checkpoint_every))

    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        accelerator="gpu",
        strategy=strategy,
        max_steps=args.max_steps,
        limit_val_batches=(args.limit_val_batches if is_real else 0),
        val_check_interval=(args.val_check_interval if is_real else args.max_steps),
        num_sanity_val_steps=0,
        log_every_n_steps=10 if is_real else 1,
        enable_checkpointing=is_real,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(precision=args.precision),
    )

    # --- W&B (optional; only if a key was forwarded) --------------------------
    wandb_logger = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            try:
                from lightning.pytorch.loggers import WandbLogger
            except ImportError:
                from pytorch_lightning.loggers import WandbLogger
            wandb_logger = WandbLogger(
                project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name,
                tags=["exp112", "qwen3", "3b", "nemo", "mfu", "contacts-v1", "coreweave"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"[exp112] W&B logger unavailable: {e}")

    # log_dir / ckpt_dir were computed above (before the callbacks).
    ckpt_cb = None
    resume = None
    if is_real:
        from nemo.lightning import AutoResume
        from nemo.lightning.pytorch.callbacks import ModelCheckpoint
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            every_n_train_steps=args.checkpoint_every,
            save_last=True,
            save_top_k=args.keep_top_k,
            save_on_train_epoch_end=False,
            save_optim_on_train_end=True,
        )
        # Preemption resume: pull the latest S3 checkpoint into ckpt_dir first
        # (node-level), then let AutoResume pick it up.
        if args.resume and args.checkpoint_s3:
            restored = restore_latest_checkpoint_from_s3(ckpt_dir, args.checkpoint_s3)
            print(f"[exp112] resume: restored={restored!r}", flush=True)
        resume = AutoResume(resume_if_exists=True, resume_ignore_no_checkpoint=True)

    nemo_logger = nl.NeMoLogger(name=args.wandb_name, wandb=wandb_logger, log_dir=log_dir, ckpt=ckpt_cb)

    print(f"[exp112] Qwen3 {args.num_layers}L h{args.hidden_size} ffn{args.ffn_hidden_size} "
          f"{args.num_attention_heads}h/{args.num_query_groups}kv seq{args.seq_length} vocab{vocab_size} | "
          f"gbs{args.global_batch_size} mbs{args.micro_batch_size} | {world_size} GPU | "
          f"data={args.data} max_steps={args.max_steps} | est {flops_per_step/1e12:.1f} TFLOP/step | "
          f"ckpt_every={args.checkpoint_every} s3={args.checkpoint_s3 or 'none'}")

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=opt,
        tokenizer="data",
        resume=resume,
    )

    # --- Report (rank 0) ------------------------------------------------------
    is_rank0 = int(os.environ.get("RANK", "0")) == 0
    if is_rank0:
        summary = tp_state.summary()
        summary.update(
            run_name=args.wandb_name,
            world_size=world_size,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            seq_length=args.seq_length,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
            vocab_size=vocab_size,
            grad_ckpt=(not args.no_grad_ckpt),
        )
        print("[exp112] ===== MFU SUMMARY =====")
        print(json.dumps(summary, indent=2))
        try:
            with open(args.results_json, "w") as fh:
                json.dump(summary, fh, indent=2)
            print(f"[exp112] wrote {args.results_json}")
        except Exception as e:  # noqa: BLE001
            print(f"[exp112] could not write results json: {e}")


if __name__ == "__main__":
    main()
