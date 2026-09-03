# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 position-intervention probe for issue #262.

The NoPE half of the proposal claims that contacts-v1 needs relative position
*inside a statement* and nothing beyond it: statements are emitted in uniformly
random order, so the distance between two statements carries no information. If
that is true, the trained model should already be insensitive to what RoPE says
about cross-statement distances.

We test it directly, with no training, by rewriting ``position_ids`` at
inference and re-scoring the same tokens:

``baseline``
    True positions. The reference NLL.
``gapN``
    Positions still increment by 1 *inside* each statement, but a random gap of
    1..N is inserted *between* statements. Order and intra-statement geometry
    are exactly preserved; every cross-statement distance is perturbed. Loss
    that does not move means cross-statement distance was not being used.
    ``gap1`` is the degenerate case and must reproduce ``baseline`` exactly — it
    is carried as a self-test of the statement parser and the position builder.
``randgapN`` / ``fixedgapN``
    **The matched pair, and the sharpest comparison in the probe.** Both give the
    inter-statement gap a mean of exactly N, so both stretch the position range
    by the same expected factor and neither ever repeats a position.
    ``fixedgapN`` uses the gap N every time — distances are *rescaled* but stay
    perfectly predictable. ``randgapN`` draws uniformly from ``1..2N-1`` — same
    mean, but every exact cross-statement distance is destroyed. Their
    difference isolates *randomising* cross-statement distance from *rescaling*
    it, which is the question the NoPE half of the proposal turns on.
``gapN``
    Gaps uniform on ``1..N``. Kept from the first pass; superseded by
    ``randgapN`` for the matched comparison because its mean stretch does not
    line up with any ``fixedgapN``.
``jitterN``
    Gaps averaging 1, drawn so that some are **zero** — two tokens then share a
    position id. Stretch-neutral, but the shared ids are their own severe
    distribution shift, so this measures "does the model need positions to be
    distinct?" and not "does it read exact distances". Retained because that
    turns out to be the more interesting question of the two.
    The same inter-statement gap N everywhere — deterministic, so cross-statement
    distances are *rescaled* but not *randomized*. This is the control that
    decides how to read ``gapN``: both stretch the position range by a similar
    factor, but only ``gapN`` destroys the exact distance between two
    statements. If the two sit on the same damage-versus-stretch curve, the
    model is reacting to the range and not reading cross-statement distances,
    and the proposal's premise survives. If ``gapN`` is worse at matched
    stretch, the model is reading them.
``shiftN``
    Every position translated by N. RoPE is relative, so this must be a no-op up
    to numerical error; carried as a second control.
``flat``
    Every statement is given the same base position, so all cross-statement
    distances collapse to ~0 while intra-statement rope survives. This is the
    proposal's inductive bias imposed post-hoc on a RoPE-trained model — the
    sharpest zero-training test available, and badly out of distribution, so a
    large number here is an upper bound on the cost rather than a measurement
    of it.
``rope_off``
    Rotary replaced by the identity: cos=1, sin=0. Measures how load-bearing
    RoPE is at all for a model that was trained with it. Also out of
    distribution; same caveat.

The interventions are only informative in one direction. ``gapN`` leaving loss
unchanged is strong evidence the information is unused; ``flat`` or ``rope_off``
raising loss is weak evidence it is used, because a model trained *without* rope
would not have to pay the distribution shift. Read the small numbers, not the
big ones.

Writes ``data/phase0_position_interventions.csv``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

from probe_common import build_probe_document, load_model, load_targets

VERB_TOKENS = {"<n-term>", "<c-term>"}
HEADER_TOKENS = {"<contacts-v1>", "<begin_sequence>", "<begin_statements>", "<end>"}


def statement_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """Split a contacts-v1 token stream into ``[start, end)`` statement spans.

    The grammar is unambiguous given a 2-token lookback (see
    ``grammar_lookback.py``), so a left-to-right walk on token identity is
    exact: headers stand alone, ``<contact>`` takes two arguments, ``<n-term>``
    / ``<c-term>`` take one, and a bare position token heads a
    ``<pX> <AA>`` residue statement.
    """
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in HEADER_TOKENS:
            width = 1
        elif token == "<contact>":
            width = 3
        elif token in VERB_TOKENS:
            width = 2
        elif token.startswith("<p") and token[2:-1].isdigit():
            width = 2
        else:
            raise ValueError(f"unexpected token {token!r} at index {index}")
        spans.append((index, min(index + width, len(tokens))))
        index += width
    return spans


def build_position_ids(spans: list[tuple[int, int]], length: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Positions for one intervention. Intra-statement offsets always survive."""
    positions = np.zeros(length, dtype=np.int64)
    if mode == "baseline":
        return np.arange(length, dtype=np.int64)
    if mode == "flat":
        for start, end in spans:
            positions[start:end] = np.arange(end - start)
        return positions
    if mode.startswith("shift"):
        return np.arange(length, dtype=np.int64) + int(mode[5:])
    if mode.startswith("randgap"):
        # Uniform on 1..2N-1: mean N, matching fixedgapN's stretch exactly,
        # and never zero so positions stay strictly increasing.
        mean_gap = int(mode[7:])
        cursor = 0
        for start, end in spans:
            positions[start:end] = cursor + np.arange(end - start)
            cursor += (end - start) - 1 + int(rng.integers(1, 2 * mean_gap))
        return positions
    if mode.startswith("jitter"):
        # Gaps average to 1 so the range matches baseline: either uniform on
        # {0..2N} (N=1) or a sparse spike, gap 0 mostly and gap N occasionally.
        span = int(mode[6:])
        cursor = 0
        for start, end in spans:
            positions[start:end] = cursor + np.arange(end - start)
            if span == 1:
                gap = int(rng.integers(0, 3))
            else:
                gap = span if rng.random() < 1.0 / span else 0
            cursor += (end - start) - 1 + gap
        return positions
    if mode.startswith("fixedgap"):
        gap = int(mode[8:])
        cursor = 0
        for start, end in spans:
            positions[start:end] = cursor + np.arange(end - start)
            cursor += (end - start) - 1 + gap
        return positions
    if not mode.startswith("gap"):
        raise ValueError(f"unknown mode {mode!r}")
    max_gap = int(mode[3:])
    cursor = 0
    for start, end in spans:
        positions[start:end] = cursor + np.arange(end - start)
        cursor += (end - start) - 1 + int(rng.integers(1, max_gap + 1))
    return positions


class RopeOff:
    """Context manager replacing the rotary embedding with the identity."""

    def __init__(self, model):
        self.rotary = model.model.rotary_emb
        self.original = self.rotary.forward

    def __enter__(self):
        original = self.original

        def identity(x, position_ids):
            cos, sin = original(x, position_ids)
            return torch.ones_like(cos), torch.zeros_like(sin)

        self.rotary.forward = identity
        return self

    def __exit__(self, *exc):
        self.rotary.forward = self.original
        return False


def section_nll(model, token_ids: torch.Tensor, position_ids: torch.Tensor, statements_start: int) -> tuple[float, float]:
    """Mean NLL (nats/token) over sequence-section and structure-section targets."""
    with torch.no_grad():
        logits = model(token_ids, position_ids=position_ids).logits.float()
    targets = token_ids[0, 1:]
    losses = functional.cross_entropy(logits[0, :-1], targets, reduction="none")
    target_index = torch.arange(1, token_ids.shape[1], device=losses.device)
    sequence = losses[target_index < statements_start]
    structure = losses[target_index >= statements_start]
    return sequence.mean().item(), structure.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--documents", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=262)
    parser.add_argument("--out", type=Path, default=Path("data/phase0_position_interventions.csv"))
    arguments = parser.parse_args()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)

    directory, tokenizer, model = load_model(arguments.model, attn_implementation="sdpa")
    print(f"[probe] model={directory}")
    max_positions = model.config.max_position_embeddings

    targets = load_targets()
    ordered = targets.sort_values("L").reset_index(drop=True)
    rng = np.random.default_rng(arguments.seed)
    picks = np.unique(np.linspace(0, len(ordered) - 1, num=min(arguments.documents * 3, len(ordered))).astype(int))
    rng.shuffle(picks)

    modes = [
        "baseline", "gap1", "shift1024",
        "fixedgap2", "randgap2",
        "fixedgap3", "randgap3",
        "fixedgap5", "randgap5",
        "gap2", "gap4", "gap8",
        "jitter1", "jitter4",
        "flat", "rope_off",
    ]
    rows = []
    used = 0
    for index in picks:
        if used >= arguments.documents:
            break
        row = ordered.iloc[index]
        document = build_probe_document(row.stem, row.input_seq, row.contacts, tokenizer)
        length = len(document.token_ids)
        if length > arguments.max_tokens or document.contact_statements < 8:
            continue
        spans = statement_spans(document.tokens)
        token_ids = torch.tensor([document.token_ids], device="cuda")

        for mode in modes:
            rope_off = mode == "rope_off"
            position_mode = "baseline" if rope_off else mode
            positions = build_position_ids(spans, length, position_mode, np.random.default_rng(arguments.seed + index))
            if positions.max() >= max_positions:
                print(f"[probe] skipping {document.stem} {mode}: max position {positions.max()} >= {max_positions}")
                continue
            position_ids = torch.tensor(positions, device="cuda").unsqueeze(0)
            if rope_off:
                with RopeOff(model):
                    sequence_nll, structure_nll = section_nll(model, token_ids, position_ids, document.statements_start)
            else:
                sequence_nll, structure_nll = section_nll(model, token_ids, position_ids, document.statements_start)
            rows.append(
                {
                    "stem": document.stem,
                    "residues": document.residue_count,
                    "tokens": length,
                    "statements": len(spans),
                    "mode": mode,
                    "max_position": int(positions.max()),
                    "sequence_nll": sequence_nll,
                    "structure_nll": structure_nll,
                }
            )
        used += 1
        base = [r for r in rows if r["stem"] == document.stem and r["mode"] == "baseline"][0]
        print(f"[probe] {document.stem}: {length} tokens, baseline structure NLL {base['structure_nll']:.4f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(arguments.out, index=False)

    pivot = frame.pivot_table(index="stem", columns="mode", values="structure_nll")
    summary = pd.DataFrame(
        {
            "structure_nll": pivot.mean(),
            "delta_vs_baseline": pivot.mean() - pivot["baseline"].mean(),
            "paired_delta_mean": (pivot.sub(pivot["baseline"], axis=0)).mean(),
            "paired_delta_std": (pivot.sub(pivot["baseline"], axis=0)).std(),
        }
    ).loc[modes]
    print("\n[probe] structure-section NLL (nats/token), n =", len(pivot))
    print(summary.to_string())
    sequence_pivot = frame.pivot_table(index="stem", columns="mode", values="sequence_nll")
    print("\n[probe] sequence-section NLL, paired delta vs baseline:")
    print((sequence_pivot.sub(sequence_pivot["baseline"], axis=0)).mean().loc[modes].to_string())

    control = abs(summary.loc["gap1", "paired_delta_mean"])
    if control > 1e-4:
        raise ValueError(f"gap1 must reproduce baseline exactly; paired delta was {control:.6f}")


if __name__ == "__main__":
    main()
