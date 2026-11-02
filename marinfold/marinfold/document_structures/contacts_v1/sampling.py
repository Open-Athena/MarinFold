# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Contact-budget sampling using the backends' native EOS suppression.

Decode in segments whose length is the minimum number of tokens still
needed to complete the contact budget. EOS is suppressed throughout each
segment. Recount at its boundary, then resume ordinary sampling as soon
as the budget is met. This never suppresses EOS past the requested minimum,
even with think tokens, malformed statements, or partial contact triples.
"""

from collections import defaultdict
from typing import Any

from marinfold import Backend

from .vocab import CONTACT_TOKEN, END_TOKEN, NUM_POSITION_INDICES, position_token


def _token_id(tokenizer: Any, token: str) -> int:
    """Resolve one domain token to its id; fail loudly on an UNK collapse.

    A wrong / missing contacts-v1 tokenizer maps every position or contact
    token to UNK, which would silently corrupt contact scoring and counting.
    """
    tid = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk_id is not None and tid == unk_id):
        raise ValueError(
            f"Tokenizer has no dedicated id for {token!r} (got {tid}). The "
            f"tokenizer is missing the contacts-v1 vocabulary — make sure it "
            f"is co-located with the model."
        )
    return int(tid)


def _contact_progress(
    tokens: list[int], contact_id: int, position_ids: set[int]
) -> tuple[int, int]:
    """Return complete emitted triples and the length of a trailing partial."""
    count = sum(
        head == contact_id and a in position_ids and b in position_ids
        for head, a, b in zip(tokens, tokens[1:], tokens[2:])
    )
    if tokens and tokens[-1] == contact_id:
        return count, 1
    if len(tokens) >= 2 and tokens[-2] == contact_id and tokens[-1] in position_ids:
        return count, 2
    return count, 0


def sample_contacts(
    backend: Backend,
    prefix_token_ids_batch: list[list[int]],
    *,
    max_new_tokens: int,
    min_new_contacts: int | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
    seed: int | None = None,
    batch_size: int | None = None,
) -> list[list[int]]:
    """Sample contacts-v1 completions, optionally requiring new contacts.

    Args:
        backend: A sampling backend (transformers or vLLM).
        prefix_token_ids_batch: Equal-length prompts, which may already
            contain contacts. Prompts should end at a statement boundary.
        max_new_tokens: Total generated-token cap per completion, across
            all segments, including any final end token.
        min_new_contacts: Minimum complete ``<contact> <p_i> <p_j>``
            statements in each completion. Prompt contacts do not count.
            Duplicates count as emitted statements; retractions do not undo
            this count. This is not a minimum of unique or live pairs.
            None or zero leaves the backend's ordinary sampling unchanged.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling cutoff; zero disables filtering.
        seed: Optional reproducibility seed. Segments use successive seeds.
        batch_size: Backend generation batch-size hint.

    Returns:
        Generated token IDs for each prompt, excluding the prompt and end
        token. After reaching the minimum, generation continues until the
        model samples end or exhausts max_new_tokens.

    Raises:
        ValueError: Invalid budget or unequal/empty prompts.
        RuntimeError: A completion exhausts its token cap before meeting
            the requested minimum, or the backend stops while constrained.
    """
    if min_new_contacts is not None and (
        not isinstance(min_new_contacts, int) or min_new_contacts < 0
    ):
        raise ValueError("min_new_contacts must be a non-negative integer or None.")
    tokenizer = backend.tokenizer
    stop_id = _token_id(tokenizer, END_TOKEN)
    sampling: dict[str, Any] = dict(
        temperature=temperature, top_p=top_p, top_k=top_k,
        stop_token_id=stop_id, batch_size=batch_size,
    )
    if not min_new_contacts:
        return backend.sample_completions(
            prefix_token_ids_batch, max_new_tokens=max_new_tokens, seed=seed,
            **sampling,
        )
    if max_new_tokens < 3 * min_new_contacts:
        raise ValueError(
            f"min_new_contacts={min_new_contacts} needs at least "
            f"{3 * min_new_contacts} new tokens; max_new_tokens={max_new_tokens}."
        )
    if not prefix_token_ids_batch:
        return []
    lengths = {len(prefix) for prefix in prefix_token_ids_batch}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("sample_contacts requires non-empty, equal-length prompts.")
    contact_id = _token_id(tokenizer, CONTACT_TOKEN)
    position_ids = {
        _token_id(tokenizer, position_token(p))
        for p in range(NUM_POSITION_INDICES)
    }
    completions: list[list[int]] = [[] for _ in prefix_token_ids_batch]
    pending = list(range(len(completions)))
    call_index = 0
    while pending:
        groups: dict[tuple[int, int, bool], list[int]] = defaultdict(list)
        for row in pending:
            tokens = completions[row]
            count, partial = _contact_progress(tokens, contact_id, position_ids)
            remaining = max_new_tokens - len(tokens)
            constrained = count < min_new_contacts
            needed = 3 * (min_new_contacts - count) - partial if constrained else 0
            if constrained and remaining < needed:
                raise RuntimeError(
                    f"Completion {row} emitted {count}/{min_new_contacts} new "
                    f"contacts within max_new_tokens={max_new_tokens}; increase "
                    "the token budget."
                )
            if remaining:
                chunk = needed if constrained else remaining
                groups[(len(tokens), chunk, constrained)].append(row)
        pending = []
        for (_, chunk, constrained), rows in groups.items():
            prompts = [prefix_token_ids_batch[row] + completions[row] for row in rows]
            outputs = backend.sample_completions(
                prompts, max_new_tokens=chunk,
                min_new_tokens=chunk if constrained else 0,
                seed=None if seed is None else seed + call_index,
                **sampling,
            )
            call_index += 1
            for row, output in zip(rows, outputs, strict=True):
                if constrained and (len(output) != chunk or stop_id in output):
                    raise RuntimeError(
                        f"Backend stopped completion {row} before its contact "
                        "minimum despite EOS suppression."
                    )
                completions[row].extend(output)
                if constrained:
                    pending.append(row)
    return completions
