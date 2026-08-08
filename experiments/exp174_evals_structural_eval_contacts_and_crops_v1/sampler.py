# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Batched sampling for the contacts-and-crops-v1 Qwen3, with two temperatures.

Two things this needs that a stock ``model.generate()`` call does not give:

**Two temperatures.** The vocabulary splits cleanly into a *coordinate* block
(the 1000 ``<xyz-DDD>`` tokens) and everything else — which residue to mention,
which atom, which box to crop, when to stop. Those are different decisions and
there is no reason one temperature should serve both. ``coord_temperature``
controls how tightly each sampled position concentrates; ``struct_temperature``
controls the structural choices, where low values invite degenerate repetition.
:class:`Sampler` applies them per token class at every step.

**Explicit prefix reuse.** Plans C and F evaluate many short continuations that
share a long prompt. Recomputing that prompt per continuation is the difference
between a two-hour run and a twelve-hour one, so :meth:`prefill` returns a KV
cache that :meth:`sample_from_cache` expands across a batch and decodes from.
The cache is the whole cost model; see ``PLANS.md`` §7.

Everything here is deliberately plain ``transformers``: the workloads are short
continuations off a controlled prompt, which is exactly the case where vLLM's
scheduler buys little and its opaque prefix cache costs control.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marinfold.document_structures.contacts_and_crops_v1.vocab import NUM_XYZ_TOKENS


@dataclass(frozen=True)
class SamplingConfig:
    """Decoding controls.

    Attributes:
        coord_temperature: temperature for the ``<xyz-DDD>`` coordinate tokens.
        struct_temperature: temperature for every other token (which residue,
            which atom, which box, ``<end>``).
        top_k: optional top-k truncation applied after temperature; ``0``
            disables. Applied to both classes.
        max_new_tokens: hard cap per continuation.
    """

    coord_temperature: float = 1.0
    struct_temperature: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 8192


class Sampler:
    """One loaded checkpoint, plus batched sampling with a reusable prefix."""

    def __init__(
        self,
        model_dir: Path,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=dtype
        ).to(device)
        self.model.eval()

        # The coordinate block is contiguous in the vocabulary (SPEC →
        # Additional tokens: the doc type, then <xyz-000>..<xyz-999>, then
        # <crop>), but derive it by name rather than trusting the layout.
        coord_ids = [
            self.tokenizer.convert_tokens_to_ids(f"<xyz-{i:03d}>")
            for i in range(NUM_XYZ_TOKENS)
        ]
        if any(i is None or i < 0 for i in coord_ids):
            raise ValueError(f"{model_dir}: tokenizer is missing <xyz-DDD> tokens")
        mask = torch.zeros(self.model.config.vocab_size, dtype=torch.bool)
        mask[torch.tensor(coord_ids)] = True
        self.coord_token_mask = mask.to(device)
        self.eos_id = self.tokenizer.convert_tokens_to_ids("<end>")
        # A crop ends when the *next* `<crop>` header starts — `<end>`
        # terminates the whole document, not a crop. Callers that want one crop
        # body must therefore stop on `<crop>` too, or they free-run the rest of
        # Pass 2 and the "forced tiling" is not forced at all.
        self.crop_id = self.tokenizer.convert_tokens_to_ids("<crop>")

    def encode(self, tokens: list[str]) -> list[int]:
        """Token strings → ids. The tokenizer is WordLevel, so this is 1:1."""
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def decode(self, ids) -> list[str]:
        """Ids → token strings."""
        return self.tokenizer.convert_ids_to_tokens(list(ids))

    def _apply_temperatures(
        self, logits: torch.Tensor, config: SamplingConfig
    ) -> torch.Tensor:
        """Scale coordinate and structural logits by their own temperatures."""
        scaled = torch.where(
            self.coord_token_mask,
            logits / config.coord_temperature,
            logits / config.struct_temperature,
        )
        if config.top_k > 0:
            kth = torch.topk(scaled, config.top_k, dim=-1).values[..., -1, None]
            scaled = scaled.masked_fill(scaled < kth, float("-inf"))
        return scaled

    @torch.inference_mode()
    def prefill(self, prompt_ids: list[int]):
        """Run the prompt once and return ``(cache_tuples, last_logits)``.

        The cache comes back in the legacy ``((keys, values), ...)`` per-layer
        form rather than as a live ``DynamicCache``, because
        :meth:`sample_from_cache` has to build a *fresh* batched cache for every
        call — decoding mutates the cache in place, and the whole point of the
        prefix is that many calls reuse it untouched.
        """
        input_ids = torch.tensor([prompt_ids], device=self.device)
        out = self.model(input_ids=input_ids, use_cache=True)
        return out.past_key_values.to_legacy_cache(), out.logits[:, -1, :].float()

    @torch.inference_mode()
    def sample_from_cache(
        self,
        cache,
        last_logits: torch.Tensor,
        prompt_length: int,
        *,
        n_samples: int,
        config: SamplingConfig,
        max_new_tokens: int | None = None,
        forced_ids: list[int] | None = None,
        stop_token_ids: list[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> list[list[int]]:
        """Sample ``n_samples`` continuations from a prefilled prompt.

        The cache is expanded across the batch (a view-level repeat, so the
        prompt is not recomputed), then decoded token by token until every row
        has emitted ``<end>`` or hit the cap.

        Args:
            forced_ids: tokens appended to the prompt *before* free sampling
                starts, in one batched forward pass. This is how Plans C and F
                pin a crop header (and, for F, the neighbouring crops) onto a
                cached prefix without recomputing that prefix. They are not
                included in the returned continuations.
            stop_token_ids: extra tokens that terminate a row, on top of
                ``<end>``. Pass ``[crop_id]`` to get exactly one crop body:
                without it the model simply opens the next crop and keeps going.

        Returns one list of generated ids per row, stop tokens excluded.
        """
        from transformers import DynamicCache

        cap = max_new_tokens if max_new_tokens is not None else config.max_new_tokens
        batch_cache = DynamicCache.from_legacy_cache(
            tuple(
                (
                    keys.expand(n_samples, -1, -1, -1).contiguous(),
                    values.expand(n_samples, -1, -1, -1).contiguous(),
                )
                for keys, values in cache
            )
        )

        logits = last_logits.expand(n_samples, -1).contiguous()
        if forced_ids:
            forced = torch.tensor(forced_ids, device=self.device)
            out = self.model(
                input_ids=forced.expand(n_samples, -1),
                past_key_values=batch_cache,
                use_cache=True,
                cache_position=torch.arange(
                    prompt_length, prompt_length + len(forced_ids), device=self.device
                ),
            )
            batch_cache = out.past_key_values
            logits = out.logits[:, -1, :].float()
            prompt_length += len(forced_ids)
        outputs: list[list[int]] = [[] for _ in range(n_samples)]
        finished = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
        position = prompt_length
        stops = torch.tensor(
            [self.eos_id] + list(stop_token_ids or []), device=self.device
        )

        for _ in range(cap):
            probs = torch.softmax(self._apply_temperatures(logits, config), dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=generator)
            flat = next_ids.squeeze(-1)
            newly_done = (flat[:, None] == stops[None, :]).any(dim=-1)
            for row in range(n_samples):
                if not finished[row] and not newly_done[row]:
                    outputs[row].append(int(flat[row]))
            finished |= newly_done
            if bool(finished.all()):
                break
            out = self.model(
                input_ids=next_ids,
                past_key_values=batch_cache,
                use_cache=True,
                cache_position=torch.tensor([position], device=self.device),
            )
            batch_cache = out.past_key_values
            logits = out.logits[:, -1, :].float()
            position += 1
        return outputs

    @torch.inference_mode()
    def sample(
        self,
        prompt_ids: list[int],
        *,
        n_samples: int = 1,
        config: SamplingConfig = SamplingConfig(),
        max_new_tokens: int | None = None,
        forced_ids: list[int] | None = None,
        stop_token_ids: list[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> list[list[int]]:
        """Prefill ``prompt_ids`` and sample from it — the one-shot convenience."""
        cache, logits = self.prefill(prompt_ids)
        return self.sample_from_cache(
            cache,
            logits,
            len(prompt_ids),
            n_samples=n_samples,
            config=config,
            max_new_tokens=max_new_tokens,
            forced_ids=forced_ids,
            stop_token_ids=stop_token_ids,
            generator=generator,
        )

    @torch.inference_mode()
    def token_logprobs(self, prompt_ids: list[int], continuation_ids: list[int]) -> np.ndarray:
        """Per-token log-probabilities of a continuation under the model.

        Used by the E3 probe, which needs the model's belief about a *given*
        crop rather than a sample of one.
        """
        ids = torch.tensor([prompt_ids + continuation_ids], device=self.device)
        logits = self.model(input_ids=ids).logits.float()
        logprobs = torch.log_softmax(logits[0, len(prompt_ids) - 1 : -1, :], dim=-1)
        targets = torch.tensor(continuation_ids, device=self.device)
        return logprobs.gather(-1, targets[:, None]).squeeze(-1).cpu().numpy()

    @torch.inference_mode()
    def next_token_distribution(self, prompt_ids: list[int]) -> np.ndarray:
        """Full next-token probability vector after ``prompt_ids``."""
        ids = torch.tensor([prompt_ids], device=self.device)
        logits = self.model(input_ids=ids).logits[0, -1, :].float()
        return torch.softmax(logits, dim=-1).cpu().numpy()


def load_sampler(model_dir: Path, **kwargs) -> Sampler:
    """Load a checkpoint, reporting how long it took (for the timings CSV)."""
    started = time.time()
    sampler = Sampler(model_dir, **kwargs)
    sampler.model_load_seconds = time.time() - started
    return sampler
