"""Diagnostics for the matched LR continuations, without changing train loss."""

from dataclasses import dataclass

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from levanter.callbacks.watch import WatchCallback, WatchConfig
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import PackedTokenDataset
from levanter.eval import cb_tagged_lm_evaluate
from levanter.models.lm_model import LmExample
from levanter.store.cache import TreeCache
from levanter.tokenizers import load_tokenizer
from levanter.tracker import log
from levanter.tracker.histogram import SummaryStats
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from .data import ContactPackedDataset
from .targets import ContactExample, Vocabulary


def endpoint_example(example: ContactExample) -> LmExample:
    """Score only predictions of contact endpoints, retaining stock input/masks."""
    return LmExample(
        example.tokens,
        example.loss_weight * (example.targets.kind != 0),
        example.attn_mask,
    )


class EndpointDataset(AsyncDataset[LmExample]):
    """Expose full-document-derived contact endpoint masks to ordinary CE eval."""

    def __init__(self, source: ContactPackedDataset):
        self.source = source

    async def async_len(self):
        return await self.source.async_len()

    def is_finite(self):
        return self.source.is_finite()

    async def get_batch(self, indices):
        return [
            endpoint_example(example)
            for example in await self.source.get_batch(indices)
        ]


def gradient_metrics(
    grads, updates, *, completed, start: int, stop: int
) -> dict[str, jax.Array | SummaryStats]:
    """Record pre-clip global norm and the unchanged clip-at-one decision."""
    norm = jnp.asarray(optax.global_norm(grads))
    return {
        "grad/norm/total": norm,
        "grad/clipped": (norm > 1.0).astype(jnp.float32),
        "grad/clip_scale": 1.0 / jnp.maximum(norm, 1.0),
        "updates/norm/total": jnp.asarray(optax.global_norm(updates)),
        "lr_trial/progress": jnp.asarray(completed - start) / (stop - start),
        "lr_trial/completed_updates": jnp.asarray(completed - start),
    }


@dataclass(frozen=True)
class LrWatchConfig(WatchConfig):
    """Use the stock callback extension point for scalar and endpoint metrics."""

    start: int = 0
    stop: int = 1
    validation_cache: str = ""
    tokenizer: str = ""
    sequence_length: int = 8192
    endpoint_examples: int = 128
    endpoint_interval: int = 1000
    eval_per_device: int = 1
    eval_mesh: MeshConfig = MeshConfig()
    checkpoint_root: str | None = None
    final_update: int | None = None
    interval: int = 1

    def build(self):
        return LrWatchCallback(self)


class LrWatchCallback(WatchCallback):
    """Evaluate fixed endpoint packs separately, preserving ordinary eval/loss."""

    def __init__(self, config: LrWatchConfig):
        self.config = config
        self.endpoint_callback = None
        self.last_endpoint_step = None

    def inside_step(self, state, inside_info):
        return gradient_metrics(
            inside_info.grads,
            inside_info.updates,
            # Levanter passes the pre-update state to inside-step callbacks.
            completed=state.step + 1,
            start=self.config.start,
            stop=self.config.stop,
        )

    def on_step(self, step_info, cb_info):
        step = int(step_info.step)
        log(cb_info, step=step)
        completed = step + 1 - self.config.start
        if completed % self.config.endpoint_interval and step + 1 != (
            self.config.final_update or self.config.stop
        ):
            return
        if self.last_endpoint_step == step:
            return
        if self.endpoint_callback is None:
            cfg = self.config
            tokenizer = load_tokenizer(cfg.tokenizer)
            cache = TreeCache.load(
                cfg.validation_cache, {"input_ids": np.zeros(0, np.int32)}
            )
            source = ContactPackedDataset(
                PackedTokenDataset(cache, hax.Axis("position", cfg.sequence_length)),
                cache,
                Vocabulary.from_tokenizer(tokenizer),
                edge_capacity=2731,
            )
            # Select before the evaluator's shuffle so every checkpoint/arm
            # scores the same fixed packs and the same endpoint denominator.
            dataset = EndpointDataset(source).take(cfg.endpoint_examples)
            geometry = TrainerConfig(
                mesh=cfg.eval_mesh, per_device_eval_parallelism=cfg.eval_per_device
            )
            self.endpoint_callback = cb_tagged_lm_evaluate(
                geometry.EvalBatch,
                [(dataset, ["endpoints"])],
                tokenizer,
                geometry.device_mesh,
                geometry.compute_axis_mapping,
                eval_ema=False,
                prefix="contact_eval",
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                checkpoint_path=cfg.checkpoint_root,
            )
        self.endpoint_callback(step_info)
        self.last_endpoint_step = step
