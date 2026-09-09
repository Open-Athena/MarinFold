# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-aligned documents shared by construction and model execution.

Adapted from ``experiments/probabilistic_dataflow/documents.py`` at Marin
commit ``58c97325b``. A document owns one token axis, typed model-input
coordinates, one attention layout, and immutable scoring ranges. Scoring is
metadata consumed after the model forward pass; it is never a model input.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from marinfold.document_structures.core import Token, VocabularyIdentity


class AttentionLayout(StrEnum):
    """Attention behavior within one packed document segment."""

    FULL = "full_segment"
    CAUSAL = "causal_segment"
    BLOCK_CAUSAL = "block_causal_segment"


@dataclass(frozen=True, eq=False)
class Coordinate:
    """Definition of one typed, token-aligned document coordinate."""

    name: str
    dtype: Any = np.int32
    missing: Any = -1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A coordinate requires a name")
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        missing = np.asarray(self.missing, dtype=self.dtype)
        if missing.ndim != 0:
            raise ValueError("A coordinate missing value must be scalar")
        object.__setattr__(self, "missing", missing.item())


POSITION_IDS = Coordinate("position_ids", missing=0)
QUERY = Coordinate("query", dtype=np.bool_, missing=False)
ATTENTION_BLOCK = Coordinate("attention_block", missing=0)

RUNTIME_COORDINATES = (POSITION_IDS, QUERY, ATTENTION_BLOCK)


TokenLike = int | Token
_MISSING = object()


@dataclass(frozen=True, eq=False)
class ScoreRange:
    """One half-open range with optional sparse categorical targets.

    A scored range without explicit targets uses ordinary shifted next-token
    targets from the document tokens. Explicit targets use one shared sparse
    ``target_ids`` vector and a dense
    ``target_weights[position_within_range, target]`` matrix. Each weight row
    is normalized independently. An unscored range carries no targets.
    """

    start: int
    stop: int
    scored: bool = True
    target_ids: tuple[int, ...] | None = None
    target_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(
                f"Score ranges must be non-empty and ordered, got [{self.start}, {self.stop})"
            )
        if (self.target_ids is None) != (self.target_weights is None):
            raise ValueError(
                "Explicit target_ids and target_weights must be supplied together"
            )
        if not self.scored and self.target_ids is not None:
            raise ValueError("An unscored range cannot carry explicit targets")
        if self.target_ids is None:
            return

        target_ids = tuple(int(target) for target in self.target_ids)
        if not target_ids:
            raise ValueError("Explicit target_ids cannot be empty")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("Explicit target_ids must be unique within a range")
        weights = np.asarray(self.target_weights, dtype=np.float32)
        expected_shape = (self.stop - self.start, len(target_ids))
        if weights.shape != expected_shape:
            raise ValueError(
                f"Target weights have shape {weights.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Target weights must be finite and non-negative")
        row_sums = weights.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("Every target-weight row must have positive mass")
        normalized = weights / row_sums
        normalized.setflags(write=False)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "target_weights", normalized)

    @property
    def has_explicit_targets(self) -> bool:
        return self.target_ids is not None

    @property
    def target_count(self) -> int:
        if not self.scored:
            return 0
        if self.has_explicit_targets:
            return self.stop - self.start
        return self.stop - self.start - 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScoreRange):
            return NotImplemented
        return (
            self.start == other.start
            and self.stop == other.stop
            and self.scored == other.scored
            and self.target_ids == other.target_ids
            and (
                (self.target_weights is None and other.target_weights is None)
                or (
                    self.target_weights is not None
                    and other.target_weights is not None
                    and np.array_equal(self.target_weights, other.target_weights)
                )
            )
        )


class Document:
    """One attention domain with model inputs and post-forward scoring rules."""

    __slots__ = (
        "_coordinates",
        "_score_ranges",
        "attention",
        "token_ids",
        "vocabulary",
    )

    def __init__(
        self,
        token_ids: np.ndarray | Sequence[TokenLike],
        coordinates: Mapping[Coordinate, np.ndarray | tuple[Any, ...] | list[Any]]
        | None = None,
        *,
        attention: AttentionLayout,
        score_ranges: tuple[ScoreRange, ...] | None = None,
        vocabulary: VocabularyIdentity | None = None,
    ) -> None:
        self.token_ids, self.vocabulary = _token_vector(token_ids, vocabulary)
        if len(self.token_ids) == 0:
            raise ValueError("A document requires at least one token")
        self.attention = attention

        provided = coordinates or {}
        values = {
            coordinate: _aligned_vector(
                coordinate, coordinate_values, len(self.token_ids)
            )
            for coordinate, coordinate_values in provided.items()
        }
        for coordinate in RUNTIME_COORDINATES:
            if coordinate not in values:
                values[coordinate] = _filled_coordinate(coordinate, len(self.token_ids))
        if self.attention == AttentionLayout.BLOCK_CAUSAL:
            _validate_attention_blocks(values[ATTENTION_BLOCK])
        self._coordinates = MappingProxyType(values)

        ranges = (ScoreRange(0, len(self)),) if score_ranges is None else score_ranges
        self._score_ranges = _validate_and_coalesce_score_ranges(ranges, len(self))
        if self.vocabulary is not None:
            for score_range in self._score_ranges:
                if score_range.target_ids is not None:
                    _validate_token_ids(score_range.target_ids, self.vocabulary)

    def __len__(self) -> int:
        return len(self.token_ids)

    def __add__(self, other: "Document") -> "Document":
        return concatenate(self, other)

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        return tuple(self._coordinates)

    @property
    def query_positions(self) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self[QUERY]))

    @property
    def score_ranges(self) -> tuple[ScoreRange, ...]:
        """Immutable post-forward scoring annotations in token-axis order."""
        return self._score_ranges

    def __getitem__(self, coordinate: Coordinate) -> np.ndarray:
        values = self._coordinates.get(coordinate)
        if values is None:
            return _filled_coordinate(coordinate, len(self))
        return values

    def __getattr__(self, name: str) -> np.ndarray:
        coordinates = object.__getattribute__(self, "_coordinates")
        matches = [
            values
            for coordinate, values in coordinates.items()
            if coordinate.name == name
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AttributeError(f"Coordinate name {name!r} is ambiguous")
        raise AttributeError(name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return NotImplemented
        if (
            self.attention != other.attention
            or self.vocabulary != other.vocabulary
            or self.score_ranges != other.score_ranges
            or not np.array_equal(self.token_ids, other.token_ids)
        ):
            return False
        if self._coordinates.keys() != other._coordinates.keys():
            return False
        return all(
            np.array_equal(values, other[coordinate])
            for coordinate, values in self._coordinates.items()
        )

    def __repr__(self) -> str:
        coordinate_names = ", ".join(coordinate.name for coordinate in self.coordinates)
        return (
            f"Document(tokens={len(self)}, coordinates=[{coordinate_names}], "
            f"score_ranges={len(self.score_ranges)}, attention={self.attention.value!r}, "
            f"vocabulary={None if self.vocabulary is None else self.vocabulary.name!r})"
        )

    def unscored(self, *, start: int = 0, stop: int | None = None) -> "Document":
        """Return a copy whose selected range contributes no training score."""
        normalized_start, normalized_stop = _normalize_range(
            start, len(self) if stop is None else stop, len(self)
        )
        return self._replace_score_range(
            normalized_start,
            normalized_stop,
            scored=False,
            target_ids=None,
            target_weights=None,
        )

    def with_targets(
        self,
        target_ids: TokenLike | Sequence[TokenLike],
        *,
        start: int | None = None,
        stop: int | None = None,
    ) -> "Document":
        """Attach one-hot token targets to a range.

        With no range arguments, targets align to the final ``len(target_ids)``
        positions, so ``document.with_targets(token_id)`` naturally targets the
        last position. Supplying ``start`` anchors the targets there; supplying
        only ``stop`` makes them end at that boundary.
        """
        targets = _normalize_targets(target_ids, self.vocabulary)
        candidates = tuple(dict.fromkeys(targets))
        candidate_indices = {
            target_id: index for index, target_id in enumerate(candidates)
        }
        weights = np.zeros((len(targets), len(candidates)), dtype=np.float32)
        for position, target_id in enumerate(targets):
            weights[position, candidate_indices[target_id]] = 1.0
        return self.with_target_distribution(
            candidates,
            weights,
            start=start,
            stop=stop,
        )

    def with_target_distribution(
        self,
        target_ids: Sequence[TokenLike],
        target_weights: np.ndarray | Sequence[Sequence[float]],
        *,
        start: int | None = None,
        stop: int | None = None,
    ) -> "Document":
        """Attach sparse categorical target distributions to a position range.

        ``target_ids`` is the shared sparse candidate vocabulary for the range.
        ``target_weights`` has shape ``(positions, len(target_ids))``. Weight
        rows may be unnormalized counts; they are normalized when stored.
        """
        targets = _normalize_targets(target_ids, self.vocabulary)
        weights = np.asarray(target_weights, dtype=np.float32)
        if weights.ndim != 2:
            raise ValueError(
                f"Target weights must be a matrix, got shape {weights.shape}"
            )
        if weights.shape[1] != len(targets):
            raise ValueError(
                f"Target weights have {weights.shape[1]} columns for "
                f"{len(targets)} target ids"
            )
        target_positions = weights.shape[0]
        if target_positions == 0:
            raise ValueError("Target weights must contain at least one position")
        if start is None and stop is None:
            normalized_stop = len(self)
            normalized_start = normalized_stop - target_positions
        elif start is None:
            normalized_stop = _normalize_boundary(stop, len(self))
            normalized_start = normalized_stop - target_positions
        else:
            normalized_start = _normalize_boundary(start, len(self))
            normalized_stop = (
                normalized_start + target_positions
                if stop is None
                else _normalize_boundary(stop, len(self))
            )
        normalized_start, normalized_stop = _normalize_range(
            normalized_start, normalized_stop, len(self)
        )
        if normalized_stop - normalized_start != target_positions:
            raise ValueError(
                f"Target range [{normalized_start}, {normalized_stop}) has length "
                f"{normalized_stop - normalized_start}, but received "
                f"{target_positions} weight rows"
            )
        return self._replace_score_range(
            normalized_start,
            normalized_stop,
            scored=True,
            target_ids=targets,
            target_weights=weights,
        )

    def take(self, indices: np.ndarray | tuple[int, ...] | list[int]) -> "Document":
        """Select or reorder token positions along the implicit axis."""
        indices_array = np.asarray(indices, dtype=np.intp)
        if indices_array.ndim != 1:
            raise ValueError(
                f"Document indices must be one-dimensional, got shape {indices_array.shape}"
            )
        normalized_indices = np.where(
            indices_array < 0, indices_array + len(self), indices_array
        )
        if np.any((normalized_indices < 0) | (normalized_indices >= len(self))):
            raise IndexError("Document index is outside the token axis")
        ranges = tuple(
            _score_range_at(self.score_ranges, int(old_index), new_index)
            for new_index, old_index in enumerate(normalized_indices)
        )
        return Document(
            self.token_ids[indices_array],
            {
                coordinate: values[indices_array]
                for coordinate, values in self._coordinates.items()
            },
            attention=self.attention,
            score_ranges=ranges,
            vocabulary=self.vocabulary,
        )

    def _replace_score_range(
        self,
        start: int,
        stop: int,
        *,
        scored: bool | object = _MISSING,
        target_ids: tuple[int, ...] | None | object = _MISSING,
        target_weights: np.ndarray | None | object = _MISSING,
    ) -> "Document":
        ranges: list[ScoreRange] = []
        for score_range in self.score_ranges:
            if score_range.stop <= start or score_range.start >= stop:
                ranges.append(score_range)
                continue
            overlap_start = max(start, score_range.start)
            overlap_stop = min(stop, score_range.stop)
            if score_range.start < overlap_start:
                ranges.append(
                    _slice_score_range(score_range, score_range.start, overlap_start)
                )
            overlap = _slice_score_range(score_range, overlap_start, overlap_stop)
            replacement_scored = (
                overlap.scored if scored is _MISSING else cast(bool, scored)
            )
            if target_ids is _MISSING:
                replacement_targets = overlap.target_ids
                replacement_weights = overlap.target_weights
            else:
                replacement_targets = cast(tuple[int, ...] | None, target_ids)
                weights = cast(np.ndarray | None, target_weights)
                replacement_weights = (
                    None
                    if weights is None
                    else weights[overlap_start - start : overlap_stop - start]
                )
            ranges.append(
                ScoreRange(
                    overlap_start,
                    overlap_stop,
                    replacement_scored,
                    replacement_targets,
                    replacement_weights,
                )
            )
            if overlap_stop < score_range.stop:
                ranges.append(
                    _slice_score_range(score_range, overlap_stop, score_range.stop)
                )
        return Document(
            self.token_ids,
            self._coordinates,
            attention=self.attention,
            score_ranges=tuple(ranges),
            vocabulary=self.vocabulary,
        )


def concatenate(*documents: Document) -> Document:
    """Concatenate documents and shift their scoring ranges along the token axis."""
    if not documents:
        raise ValueError("concatenate requires at least one document")
    attentions = {document.attention for document in documents}
    if len(attentions) != 1:
        raise ValueError(
            f"Concatenated documents must share one attention layout, got {sorted(attentions)}"
        )
    vocabularies = {document.vocabulary for document in documents}
    if len(vocabularies) != 1:
        names = sorted(
            "<raw>" if vocabulary is None else vocabulary.name
            for vocabulary in vocabularies
        )
        raise ValueError(
            f"Concatenated documents must share one vocabulary, got {names}"
        )
    vocabulary = vocabularies.pop()
    coordinates = tuple(
        dict.fromkeys(
            coordinate for document in documents for coordinate in document.coordinates
        )
    )
    shifted_ranges: list[ScoreRange] = []
    offset = 0
    for document in documents:
        shifted_ranges.extend(
            ScoreRange(
                score_range.start + offset,
                score_range.stop + offset,
                score_range.scored,
                score_range.target_ids,
                score_range.target_weights,
            )
            for score_range in document.score_ranges
        )
        offset += len(document)
    concatenated_coordinates = {
        coordinate: np.concatenate(
            tuple(document[coordinate] for document in documents)
        )
        for coordinate in coordinates
    }
    if documents[0].attention == AttentionLayout.BLOCK_CAUSAL:
        shifted_blocks: list[np.ndarray] = []
        block_offset = 0
        for document in documents:
            blocks = document[ATTENTION_BLOCK]
            shifted_blocks.append(blocks + block_offset)
            block_offset += int(blocks[-1]) + 1
        concatenated_coordinates[ATTENTION_BLOCK] = np.concatenate(shifted_blocks)

    return Document(
        np.concatenate(tuple(document.token_ids for document in documents)),
        concatenated_coordinates,
        attention=documents[0].attention,
        score_ranges=tuple(shifted_ranges),
        vocabulary=vocabulary,
    )


def causal_training_document(token_ids: Sequence[TokenLike]) -> Document:
    """Build a causal document using ordinary shifted next-token targets."""
    if len(token_ids) < 2:
        raise ValueError("Causal training documents require at least two tokens")
    length = len(token_ids)
    return Document(
        token_ids,
        {
            POSITION_IDS: np.arange(length, dtype=POSITION_IDS.dtype),
            QUERY: np.arange(length) + 1 < length,
        },
        attention=AttentionLayout.CAUSAL,
    )


def _validate_attention_blocks(blocks: np.ndarray) -> None:
    """Validate an ordered partition used by block-causal attention."""
    if blocks[0] != 0:
        raise ValueError("Block-causal attention blocks must start at 0")
    differences = np.diff(blocks)
    if np.any((differences < 0) | (differences > 1)):
        raise ValueError(
            "Block-causal attention blocks must be nondecreasing contiguous integers"
        )


@dataclass(frozen=True)
class PackedScoreRange:
    """One document target range shifted into a packed row."""

    row: int
    start: int
    stop: int
    document_index: int
    scored: bool
    target_ids: tuple[int, ...] | None = None
    target_weights: np.ndarray | None = None


@dataclass(frozen=True)
class PackedBatch:
    """Dense model inputs plus document-local scoring annotations."""

    token_ids: np.ndarray
    coordinates: dict[Coordinate, np.ndarray]
    segment_ids: np.ndarray
    document_indices: np.ndarray
    score_ranges: tuple[PackedScoreRange, ...]
    attention: AttentionLayout
    vocabulary: VocabularyIdentity | None

    def __getitem__(self, coordinate: Coordinate) -> np.ndarray:
        values = self.coordinates.get(coordinate)
        if values is None:
            return np.full(
                self.token_ids.shape, coordinate.missing, dtype=coordinate.dtype
            )
        return values

    def __getattr__(self, name: str) -> np.ndarray:
        matches = [
            values
            for coordinate, values in self.coordinates.items()
            if coordinate.name == name
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AttributeError(f"Coordinate name {name!r} is ambiguous")
        raise AttributeError(name)


def pack(documents: tuple[Document, ...], *, max_seq_len: int) -> PackedBatch:
    """Greedily pack documents while preserving attention and scoring bounds."""
    if not documents:
        raise ValueError("Cannot pack an empty document collection")
    attentions = {document.attention for document in documents}
    if len(attentions) != 1:
        raise ValueError(
            f"Packed documents must share one attention layout, got {sorted(attentions)}"
        )
    attention = attentions.pop()
    vocabularies = {document.vocabulary for document in documents}
    if len(vocabularies) != 1:
        names = sorted(
            "<raw>" if vocabulary is None else vocabulary.name
            for vocabulary in vocabularies
        )
        raise ValueError(f"Packed documents must share one vocabulary, got {names}")
    vocabulary = vocabularies.pop()
    if any(len(document) > max_seq_len for document in documents):
        longest = max(len(document) for document in documents)
        raise ValueError(f"Document length {longest} exceeds max_seq_len={max_seq_len}")

    rows: list[list[tuple[int, Document]]] = [[]]
    row_lengths = [0]
    for document_index, document in enumerate(documents):
        if row_lengths[-1] + len(document) > max_seq_len:
            rows.append([])
            row_lengths.append(0)
        rows[-1].append((document_index, document))
        row_lengths[-1] += len(document)

    shape = (len(rows), max_seq_len)
    token_ids = np.zeros(shape, dtype=np.int32)
    segment_ids = np.full(shape, -1, dtype=np.int32)
    document_indices = np.full(shape, -1, dtype=np.int32)
    coordinate_definitions = tuple(
        dict.fromkeys(
            coordinate for document in documents for coordinate in document.coordinates
        )
    )
    coordinates = {
        coordinate: np.full(shape, coordinate.missing, dtype=coordinate.dtype)
        for coordinate in coordinate_definitions
    }
    packed_score_ranges: list[PackedScoreRange] = []

    for row_index, row in enumerate(rows):
        offset = 0
        for segment_id, (document_index, document) in enumerate(row):
            end = offset + len(document)
            token_ids[row_index, offset:end] = document.token_ids
            segment_ids[row_index, offset:end] = segment_id
            document_indices[row_index, offset:end] = document_index
            for coordinate in coordinate_definitions:
                coordinates[coordinate][row_index, offset:end] = document[coordinate]
            packed_score_ranges.extend(
                PackedScoreRange(
                    row=row_index,
                    start=score_range.start + offset,
                    stop=score_range.stop + offset,
                    document_index=document_index,
                    scored=score_range.scored,
                    target_ids=score_range.target_ids,
                    target_weights=score_range.target_weights,
                )
                for score_range in document.score_ranges
            )
            offset = end

    return PackedBatch(
        token_ids=token_ids,
        coordinates=coordinates,
        segment_ids=segment_ids,
        document_indices=document_indices,
        score_ranges=tuple(packed_score_ranges),
        attention=attention,
        vocabulary=vocabulary,
    )


def _validate_and_coalesce_score_ranges(
    ranges: tuple[ScoreRange, ...], length: int
) -> tuple[ScoreRange, ...]:
    if not ranges:
        raise ValueError(
            "A document requires scoring coverage, including unscored ranges"
        )
    ordered = tuple(sorted(ranges, key=lambda score_range: score_range.start))
    expected_start = 0
    for score_range in ordered:
        if score_range.start != expected_start:
            raise ValueError(
                f"Scoring ranges must partition the document; expected range at "
                f"{expected_start}, got {score_range.start}"
            )
        expected_start = score_range.stop
    if expected_start != length:
        raise ValueError(
            f"Scoring ranges cover [0, {expected_start}), expected [0, {length})"
        )
    coalesced: list[ScoreRange] = []
    for score_range in ordered:
        if coalesced and _can_merge_score_ranges(coalesced[-1], score_range):
            previous = coalesced.pop()
            targets = None
            weights = None
            if previous.target_ids is not None:
                if score_range.target_ids is None:
                    raise AssertionError("Mergeable scoring ranges disagree on targets")
                targets = previous.target_ids
                if (
                    previous.target_weights is None
                    or score_range.target_weights is None
                ):
                    raise AssertionError("Explicit target range is missing weights")
                weights = np.concatenate(
                    (previous.target_weights, score_range.target_weights), axis=0
                )
            coalesced.append(
                ScoreRange(
                    previous.start,
                    score_range.stop,
                    previous.scored,
                    targets,
                    weights,
                )
            )
        else:
            coalesced.append(score_range)
    return tuple(coalesced)


def _can_merge_score_ranges(left: ScoreRange, right: ScoreRange) -> bool:
    return (
        left.stop == right.start
        and left.scored == right.scored
        and (
            (left.target_ids is None and right.target_ids is None)
            or left.target_ids == right.target_ids
        )
    )


def _slice_score_range(score_range: ScoreRange, start: int, stop: int) -> ScoreRange:
    weights = None
    if score_range.target_weights is not None:
        weights = score_range.target_weights[
            start - score_range.start : stop - score_range.start
        ]
    return ScoreRange(
        start,
        stop,
        score_range.scored,
        score_range.target_ids,
        weights,
    )


def _score_range_at(
    score_ranges: tuple[ScoreRange, ...], old_index: int, new_index: int
) -> ScoreRange:
    for score_range in score_ranges:
        if score_range.start <= old_index < score_range.stop:
            weights = None
            if score_range.target_weights is not None:
                weights = score_range.target_weights[
                    old_index - score_range.start : old_index - score_range.start + 1
                ]
            return ScoreRange(
                new_index,
                new_index + 1,
                score_range.scored,
                score_range.target_ids,
                weights,
            )
    raise IndexError(f"Token index {old_index} is outside scoring coverage")


def _normalize_targets(
    target_ids: TokenLike | Sequence[TokenLike],
    vocabulary: VocabularyIdentity | None,
) -> tuple[int, ...]:
    if isinstance(target_ids, (int, np.integer, Token)):
        raw_targets = (target_ids,)
    else:
        raw_targets = tuple(target_ids)
    token_handles = tuple(target for target in raw_targets if isinstance(target, Token))
    if token_handles:
        identities = {target.vocabulary for target in token_handles}
        if len(identities) != 1 or len(token_handles) != len(raw_targets):
            raise ValueError(
                "Explicit targets must not mix vocabularies or raw token IDs"
            )
        target_vocabulary = identities.pop()
        if vocabulary != target_vocabulary:
            raise ValueError(
                "Explicit targets must use the document vocabulary; got "
                f"{target_vocabulary.name!r} for "
                f"{None if vocabulary is None else vocabulary.name!r}"
            )
    targets = tuple(int(target_id) for target_id in raw_targets)
    if not targets:
        raise ValueError("Explicit targets cannot be empty")
    if vocabulary is not None:
        _validate_token_ids(targets, vocabulary)
    return targets


def _normalize_boundary(boundary: int | None, length: int) -> int:
    if boundary is None:
        return length
    normalized = boundary + length if boundary < 0 else boundary
    if normalized < 0 or normalized > length:
        raise IndexError(f"Document boundary {boundary} is outside [0, {length}]")
    return normalized


def _normalize_range(start: int, stop: int, length: int) -> tuple[int, int]:
    normalized_start = _normalize_boundary(start, length)
    normalized_stop = _normalize_boundary(stop, length)
    if normalized_start >= normalized_stop:
        raise ValueError(
            f"Document range must be non-empty, got [{normalized_start}, {normalized_stop})"
        )
    return normalized_start, normalized_stop


def _aligned_vector(coordinate: Coordinate, values: Any, length: int) -> np.ndarray:
    array = _vector(values, coordinate.dtype, name=coordinate.name)
    if len(array) != length:
        raise ValueError(
            f"Coordinate {coordinate.name!r} has length {len(array)}, expected {length}"
        )
    return array


def _filled_coordinate(coordinate: Coordinate, length: int) -> np.ndarray:
    return _readonly_array(np.full(length, coordinate.missing, dtype=coordinate.dtype))


def _vector(values: Any, dtype: np.dtype[Any], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return _readonly_array(array)


def _token_vector(
    values: np.ndarray | Sequence[TokenLike],
    vocabulary: VocabularyIdentity | None,
) -> tuple[np.ndarray, VocabularyIdentity | None]:
    if isinstance(values, np.ndarray):
        array = _vector(values, np.dtype(np.int32), name="token_ids")
        if vocabulary is not None:
            _validate_token_ids(array, vocabulary)
        return array, vocabulary

    raw_values = tuple(values)
    token_handles = tuple(value for value in raw_values if isinstance(value, Token))
    if token_handles:
        identities = {token.vocabulary for token in token_handles}
        if len(identities) != 1 or len(token_handles) != len(raw_values):
            raise ValueError(
                "Document tokens must not mix vocabularies or raw token IDs"
            )
        inferred_vocabulary = identities.pop()
        if vocabulary is not None and vocabulary != inferred_vocabulary:
            raise ValueError(
                f"Document tokens use {inferred_vocabulary.name!r}, not "
                f"{vocabulary.name!r}"
            )
        vocabulary = inferred_vocabulary
    array = _vector(raw_values, np.dtype(np.int32), name="token_ids")
    if vocabulary is not None:
        _validate_token_ids(array, vocabulary)
    return array, vocabulary


def _validate_token_ids(
    token_ids: Sequence[int] | np.ndarray, vocabulary: VocabularyIdentity
) -> None:
    values = np.asarray(token_ids)
    if values.size and (np.any(values < 0) or np.any(values >= vocabulary.size)):
        raise ValueError(
            f"Token IDs must lie in [0, {vocabulary.size}) for vocabulary "
            f"{vocabulary.name!r}"
        )


def _readonly_array(values: np.ndarray) -> np.ndarray:
    array = np.array(values, copy=True)
    array.flags.writeable = False
    return array
