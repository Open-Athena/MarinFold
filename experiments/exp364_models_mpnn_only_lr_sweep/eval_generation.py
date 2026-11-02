"""Continue capped samples without redrawing their prompts or generated prefixes.

The initial call retains the exp277 6L+128 recipe. Only length-capped samples
receive a second call, conditioned on their exact prompt and generated tokens.
Continuation uses an independently derived deterministic seed and the remaining
8192-token context. Completed samples are untouched. Context exhaustion remains
an incomplete result, never a reason to discard or replace a sample.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Completion:
    """Immutable completion fields consumed by the original vote parser."""

    token_ids: tuple[int, ...]
    text: str
    finish_reason: str | None


@dataclass(frozen=True)
class Rollout:
    """A single sampled sequence and its original prompt tokens."""

    prompt_token_ids: tuple[int, ...]
    outputs: tuple[Completion, ...]


def token_digest(tokens: tuple[int, ...]) -> str:
    """Hash token IDs with an unambiguous encoding for prefix provenance."""
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def continuation_seed(
    seed: int, prompt: tuple[int, ...], prefix: tuple[int, ...]
) -> int:
    """Derive a separate reproducible random stream for conditional continuation."""
    identity = (
        f"exp364-continuation-v1:{seed}:{token_digest(prompt)}:{token_digest(prefix)}"
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % (
        2**63 - 1
    )


def as_rollout(output: Any) -> Rollout:
    """Copy the documented vLLM output fields without mutating engine objects."""
    if len(output.outputs) != 1 or output.prompt_token_ids is None:
        raise ValueError("Expected exactly one completion with prompt token IDs")
    completion = output.outputs[0]
    return Rollout(
        tuple(output.prompt_token_ids),
        (
            Completion(
                tuple(completion.token_ids), completion.text, completion.finish_reason
            ),
        ),
    )


def extend_capped_rollouts(
    model: Any,
    outputs: list[Any],
    parameters: list[Any],
    tokenizer: Any,
    *,
    max_context: int = 8192,
) -> tuple[list[Rollout], list[dict]]:
    """Continue only capped outputs, preserving every sampled prefix exactly."""
    if len(outputs) != len(parameters):
        raise ValueError("Output and sampling-parameter counts differ")
    results = [as_rollout(output) for output in outputs]
    indices, prompts, params, records = [], [], [], []
    for index, (output, original) in enumerate(zip(results, parameters, strict=True)):
        completion = output.outputs[0]
        if completion.finish_reason == "stop":
            continue
        if completion.finish_reason != "length":
            raise ValueError(f"Unexpected finish reason: {completion.finish_reason}")
        prefix = output.prompt_token_ids + completion.token_ids
        remaining = max_context - len(prefix)
        seed = continuation_seed(
            original.seed, output.prompt_token_ids, completion.token_ids
        )
        record = {
            "sample_index": index,
            "original_seed": original.seed,
            "continuation_seed": seed,
            "original_budget": original.max_tokens,
            "prompt_tokens": len(output.prompt_token_ids),
            "initial_tokens": len(completion.token_ids),
            "prefix_sha256": token_digest(completion.token_ids),
            "additional_budget": max(remaining, 0),
            "added_tokens": 0,
            "total_tokens": len(completion.token_ids),
            "finish_reason": completion.finish_reason,
            "context_limit": max_context,
        }
        records.append(record)
        if remaining <= 0:
            continue
        copied = original.clone()
        copied.max_tokens = remaining
        copied.seed = seed
        indices.append((index, record))
        prompts.append({"prompt_token_ids": list(prefix)})
        params.append(copied)
    if not indices:
        return results, records
    continued = model.generate(prompts, params, use_tqdm=False)
    if len(continued) != len(indices):
        raise ValueError("Continuation changed the number of samples")
    for (index, record), response, prompt in zip(
        indices, continued, prompts, strict=True
    ):
        suffix = as_rollout(response)
        if suffix.prompt_token_ids != tuple(prompt["prompt_token_ids"]):
            raise ValueError("The engine changed the preserved sample prefix")
        previous = results[index]
        generated = suffix.outputs[0]
        if len(generated.token_ids) > record["additional_budget"]:
            raise ValueError("Continuation exceeded the model context")
        combined = previous.outputs[0].token_ids + generated.token_ids
        # Decode the complete token sequence together: a cap can fall inside a
        # contact statement, so concatenating separately decoded text is unsafe.
        text = tokenizer.decode(combined, skip_special_tokens=False)
        results[index] = Rollout(
            previous.prompt_token_ids,
            (Completion(combined, text, generated.finish_reason),),
        )
        record.update(
            added_tokens=len(generated.token_ids),
            total_tokens=len(combined),
            finish_reason=generated.finish_reason,
        )
    return results, records
