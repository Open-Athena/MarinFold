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
                   help="'mock' (default) or a Megatron .bin/.idx dataset path prefix")
    p.add_argument("--tokenizer", default=os.environ.get("EXP112_TOKENIZER", "timodonnell/contacts-v1-tokenizer"))
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

    def __init__(self, args, tokens_per_step, flops_per_step, world_size, peak_tflops=989.0):
        self.args = args
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
        if self._step > self.args.measure_warmup_steps:
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
        # Deferred real-run path: Megatron .bin/.idx via PreTrainingDataModule.
        from nemo.collections.llm.gpt.data import PreTrainingDataModule
        data = PreTrainingDataModule(
            paths=[args.data],
            seq_length=args.seq_length,
            tokenizer=tokenizer,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            reset_attention_mask=True,     # block cross-document attention
            reset_position_ids=True,
            eod_mask_loss=False,
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

    # Build the throughput callback as a real PTL Callback subclass that DELEGATES
    # to the plain state object (no MRO trap). NeMo 2.0 moved to the
    # `lightning.pytorch` namespace; fall back to `pytorch_lightning`.
    try:
        from lightning.pytorch import Callback
    except ImportError:
        from pytorch_lightning import Callback

    tp_state = ThroughputState(args, tokens_per_step, flops_per_step, world_size, peak_tflops=989.0)

    class ThroughputCallback(Callback):
        def on_train_batch_start(self, *a, **k):
            tp_state.start()

        def on_train_batch_end(self, *a, **k):
            tp_state.stop()

    cb = ThroughputCallback()

    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        accelerator="gpu",
        strategy=strategy,
        max_steps=args.max_steps,
        limit_val_batches=0,
        val_check_interval=args.max_steps,  # effectively no mid-run val
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_checkpointing=False,
        callbacks=[cb],
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

    nemo_logger = nl.NeMoLogger(name=args.wandb_name, wandb=wandb_logger, log_dir="/tmp/exp112_nemo_logs")

    print(f"[exp112] Qwen3 {args.num_layers}L h{args.hidden_size} ffn{args.ffn_hidden_size} "
          f"{args.num_attention_heads}h/{args.num_query_groups}kv seq{args.seq_length} vocab{vocab_size} | "
          f"gbs{args.global_batch_size} mbs{args.micro_batch_size} | {world_size} GPU | "
          f"data={args.data} max_steps={args.max_steps} | est {flops_per_step/1e12:.1f} TFLOP/step")

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=opt,
        tokenizer="data",
        resume=None,
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
