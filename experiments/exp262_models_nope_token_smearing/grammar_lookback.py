# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""How wide a causal window does the contacts-v1 grammar actually need?

The proposal in issue #262 is to delete RoPE and replace it with a width-3
causal *smear* — mix the previous two tokens' embeddings into the current one.
That is only the right width if the parse state of a contacts-v1 document is
recoverable from a 2-token lookback, so this script measures it directly.

contacts-v1 statements are 2 tokens (``<pX> <AA>``, ``<n-term> <pX>``,
``<c-term> <pX>``) or 3 tokens (``<contact> <pX> <pY>``) with **no separators**,
so a reader has to infer statement boundaries from token identity alone. We
enumerate the grammar, label every token with its true ``(statement form, slot)``
role, and ask whether that role is a deterministic function of the token
*classes* in the window ``[t-W, t]``. A window is sufficient iff no context maps
to two different roles.

Token identities (which residue index, which amino acid) are deliberately
collapsed to classes: the smear mixes whole embeddings, so what a downstream
layer can read off it is the *type* pattern, and a bound proved over types is
the conservative one.

Writes ``data/grammar_lookback.csv``.
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

# Stand-ins for the real vocab. Only the class partition matters here, and
# using synthetic names keeps this script free of a marinfold import.
AMINO_ACIDS = [f"<AA{index}>" for index in range(20)]
NUM_POSITION_INDICES = 2000


def position_token(index: int) -> str:
    return f"<p{index}>"


def token_class(token: str) -> str:
    """Collapse a token to the class a smeared embedding would expose."""
    if token.startswith("<p") and token[2:-1].isdigit():
        return "POS"
    if token.startswith("<AA"):
        return "AA"
    return token


def synthesize_document(rng: random.Random) -> tuple[list[str], list[tuple[str, int]]]:
    """Emit one grammatical contacts-v1 document plus a per-token role label.

    Mirrors ``marinfold.document_structures.contacts_v1``: a shuffled sequence
    section of ``<pX> <AA>`` statements plus one ``<n-term>`` / ``<c-term>``
    pair, then a shuffled structure section of ``<contact> <pX> <pY>``.
    """
    residue_count = rng.randint(30, 120)
    start = rng.randrange(NUM_POSITION_INDICES)
    indices = [(start + offset) % NUM_POSITION_INDICES for offset in range(residue_count)]

    statements = [("res", [position_token(index), rng.choice(AMINO_ACIDS)]) for index in indices]
    statements.append(("nterm", ["<n-term>", position_token(indices[0])]))
    statements.append(("cterm", ["<c-term>", position_token(indices[-1])]))
    rng.shuffle(statements)

    contacts = []
    for _ in range(rng.randint(10, 200)):
        first, second = rng.sample(indices, 2)
        contacts.append(("contact", ["<contact>", position_token(first), position_token(second)]))
    rng.shuffle(contacts)

    tokens = ["<contacts-v1>", "<begin_sequence>"]
    roles: list[tuple[str, int]] = [("doc", 0), ("hdr", 0)]
    for form, statement in statements:
        for slot, token in enumerate(statement):
            tokens.append(token)
            roles.append((form, slot))
    tokens.append("<begin_statements>")
    roles.append(("hdr", 1))
    for form, statement in contacts:
        for slot, token in enumerate(statement):
            tokens.append(token)
            roles.append((form, slot))
    tokens.append("<end>")
    roles.append(("end", 0))
    return tokens, roles


def ambiguity_table(lookback: int, documents: int, seed: int) -> dict[tuple[str, ...], set]:
    """Map each observed class-context to the set of roles it can denote."""
    table: dict[tuple[str, ...], set] = defaultdict(set)
    rng = random.Random(seed)
    for _ in range(documents):
        tokens, roles = synthesize_document(rng)
        for position in range(len(tokens)):
            context = tuple(
                token_class(tokens[index]) if index >= 0 else "BOS"
                for index in range(position - lookback, position + 1)
            )
            table[context].add(roles[position])
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-lookback", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("data/grammar_lookback.csv"))
    arguments = parser.parse_args()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for lookback in range(0, arguments.max_lookback + 1):
        table = ambiguity_table(lookback, arguments.documents, arguments.seed)
        ambiguous = {context: roles for context, roles in table.items() if len(roles) > 1}
        rows.append(
            {
                "lookback": lookback,
                "contexts": len(table),
                "ambiguous_contexts": len(ambiguous),
                "example_ambiguity": (
                    " ".join(next(iter(ambiguous))) if ambiguous else ""
                ),
                "example_roles": (
                    "|".join(f"{form}:{slot}" for form, slot in sorted(next(iter(ambiguous.values()))))
                    if ambiguous
                    else ""
                ),
            }
        )
        print(
            f"lookback {lookback}: {len(table):4d} contexts, {len(ambiguous)} ambiguous"
            + (f"   e.g. {rows[-1]['example_ambiguity']} -> {rows[-1]['example_roles']}" if ambiguous else "")
        )

    with arguments.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
