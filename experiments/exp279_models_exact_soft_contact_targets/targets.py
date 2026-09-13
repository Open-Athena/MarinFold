# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact contact targets from complete serialized documents.

An edge's index is its position in the sampled serialization. At either
endpoint prediction, the remaining edges are a suffix of that document's edge
list. This is sufficient metadata for the exact generator distribution. It
also survives slicing: unseen edges beyond a packed fragment remain targets.
No metadata is passed to the transformer.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.models.lm_model import LmExample


@dataclass(frozen=True)
class Vocabulary:
    """Token IDs of the one supported, unmodified contacts-v1 format."""

    size: int
    position_ids: tuple[int, ...]
    amino_acid_ids: tuple[int, ...] = tuple(range(86, 106)) + (2844,)
    document: int = 2
    contact: int = 5
    sequence: int = 8
    structure: int = 9
    end: int = 10
    eos: int = 1
    pad: int = 0

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "Vocabulary":
        vocab = tokenizer.get_vocab()
        positions = tuple(vocab[f"<p{i}>"] for i in range(2000))
        special = {
            name: vocab[token]
            for name, token in (
                ("document", "<contacts-v1>"),
                ("contact", "<contact>"),
                ("sequence", "<begin_sequence>"),
                ("structure", "<begin_statements>"),
                ("end", "<end>"),
                ("eos", "<eos>"),
                ("pad", "<pad>"),
            )
        }
        if len(vocab) != 2845 or special != dict(
            document=2, contact=5, sequence=8, structure=9, end=10, eos=1, pad=0
        ):
            raise ValueError("The pinned contacts-v1 tokenizer contract changed")
        return cls(size=len(vocab), position_ids=positions, **special)


class ContactTargets(eqx.Module):
    """Fixed-shape metadata; ranges and kinds index prediction positions."""

    first: hax.NamedArray
    second: hax.NamedArray
    start: hax.NamedArray
    stop: hax.NamedArray
    kind: hax.NamedArray  # 0 = ordinary CE, 1 = first endpoint, 2 = second


class ContactExample(LmExample):
    targets: ContactTargets = eqx.field(kw_only=True)


@dataclass(frozen=True)
class DocumentSlice:
    """A complete source document and the interval retained by the packer."""

    tokens: np.ndarray
    start: int = 0
    stop: int | None = None


def document_edges(
    tokens: np.ndarray, vocab: Vocabulary
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a complete document and return edge tokens and marker offsets."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("Expected a one-dimensional integer token document")
    if np.any((tokens < 0) | (tokens >= vocab.size)):
        raise ValueError("Token outside vocabulary")
    body = tokens[:-1] if len(tokens) and tokens[-1] == vocab.eos else tokens
    if (
        len(body) < 4
        or list(body[:2]) != [vocab.document, vocab.sequence]
        or body[-1] != vocab.end
    ):
        raise ValueError(
            "Expected a complete contacts-v1 document ending in <end> [<eos>]"
        )
    markers = np.flatnonzero(body == vocab.structure)
    if len(markers) != 1:
        raise ValueError("Expected exactly one <begin_statements>")
    structure = int(markers[0])
    seq = body[2:structure]
    if len(seq) % 2:
        raise ValueError("Malformed two-token sequence statements")
    pairs = seq.reshape(-1, 2)
    residue = np.isin(pairs[:, 0], vocab.position_ids) & np.isin(
        pairs[:, 1], vocab.amino_acid_ids
    )
    termini = np.isin(pairs[:, 0], (3, 4)) & np.isin(pairs[:, 1], vocab.position_ids)
    if not np.all(residue | termini):
        raise ValueError("Malformed residue/terminus statement")
    if len(np.unique(pairs[residue, 0])) != np.count_nonzero(residue):
        raise ValueError("Repeated residue assignment")
    statements = body[structure + 1 : -1]
    if len(statements) % 3:
        raise ValueError("Incomplete contact statement")
    triples = statements.reshape(-1, 3)
    if np.any(triples[:, 0] != vocab.contact):
        raise ValueError("Unsupported structure statement")
    edges = triples[:, 1:].astype(np.int32)
    if not np.all(np.isin(edges, pairs[residue, 0])):
        raise ValueError("Contact endpoint absent from sequence")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("Self contact")
    if len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges):
        raise ValueError("Duplicate undirected contact")
    return edges, np.arange(structure + 1, len(body) - 1, 3)


def pack_targets(
    documents: Sequence[DocumentSlice],
    packed_tokens: np.ndarray,
    vocab: Vocabulary,
    *,
    edge_capacity: int,
    position_axis: str = "position",
) -> ContactTargets:
    """Build from complete documents, then slice exactly like the stock packer.

    Checks the resulting tokens against the actual stock example. Truncation,
    reordered reads, or an incorrect pack mapping therefore cannot silently
    change the conditioning set. Padding carries ordinary, masked CE targets.
    """
    size = len(packed_tokens)
    first = np.zeros(edge_capacity, dtype=np.int32)
    second = np.zeros_like(first)
    starts, stops, kinds = (np.zeros(size, dtype=np.int32) for _ in range(3))
    expected = np.full(size, vocab.pad, dtype=np.int32)
    offset = edge_offset = 0
    for document in documents:
        tokens = np.asarray(document.tokens)
        stop = len(tokens) if document.stop is None else document.stop
        if not 0 <= document.start < stop <= len(tokens):
            raise ValueError("Invalid document slice")
        count = stop - document.start
        if offset + count > size:
            raise ValueError("Documents exceed packed length")
        edges, markers = document_edges(tokens, vocab)
        edge_end = edge_offset + len(edges)
        if edge_end > edge_capacity:
            raise ValueError(
                f"Contact metadata capacity exceeded: {edge_end} > {edge_capacity}"
            )
        first[edge_offset:edge_end] = edges[:, 0]
        second[edge_offset:edge_end] = edges[:, 1]
        expected[offset : offset + count] = tokens[document.start : stop]
        for i, marker in enumerate(markers):
            for kind, position in ((1, marker), (2, marker + 1)):
                if document.start <= position < stop:
                    packed_position = offset + position - document.start
                    starts[packed_position] = edge_offset + i
                    stops[packed_position] = edge_end
                    kinds[packed_position] = kind
        offset += count
        edge_offset = edge_end
    if not np.array_equal(expected, packed_tokens):
        raise ValueError("Metadata documents do not reproduce stock packed tokens")
    Pos, Edge = hax.Axis(position_axis, size), hax.Axis("contact_edge", edge_capacity)
    # Host arrays are intentional: the stock loader handles device placement.
    return ContactTargets(
        hax.named(first, Edge),
        hax.named(second, Edge),
        hax.named(starts, Pos),
        hax.named(stops, Pos),
        hax.named(kinds, Pos),
    )


def distributions(
    first: jax.Array,
    second: jax.Array,
    start: jax.Array,
    stop: jax.Array,
    kind: jax.Array,
    observed: jax.Array,
    hard_target: jax.Array,
    vocab_size: int,
) -> jax.Array:
    """Return q for a block of prediction positions, accumulating shared residues."""
    edge = jnp.arange(first.shape[0])[None, :]
    remaining = (edge >= start[:, None]) & (edge < stop[:, None])
    # Each remaining directed orientation has equal mass. Conditioning on the
    # first endpoint retains only the matching orientations, then renormalizes.
    a_weight = remaining & (
        (kind[:, None] == 1) | ((kind[:, None] == 2) & (second == observed[:, None]))
    )
    b_weight = remaining & (
        (kind[:, None] == 1) | ((kind[:, None] == 2) & (first == observed[:, None]))
    )
    rows = jnp.arange(len(kind))[:, None]
    mass = jnp.zeros((len(kind), vocab_size), jnp.float32)
    mass = mass.at[rows, first[None, :]].add(a_weight.astype(jnp.float32))
    mass = mass.at[rows, second[None, :]].add(b_weight.astype(jnp.float32))
    mass /= jnp.maximum(mass.sum(-1, keepdims=True), 1)
    return jnp.where(kind[:, None] != 0, mass, jax.nn.one_hot(hard_target, vocab_size))
