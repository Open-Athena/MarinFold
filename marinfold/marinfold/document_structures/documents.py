# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-aligned documents shared by construction and model execution.

Adapted from ``experiments/probabilistic_dataflow/documents.py`` at Marin
commit ``58c97325b``. A document owns one token axis, typed model-input
coordinates, one attention layout, and immutable scoring ranges. Scoring is
metadata consumed after the model forward pass; it is never a model input.
"""

from collections.abc import Callable, Mapping, Sequence
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

RUNTIME_COORDINATES = (POSITION_IDS, QUERY)


@dataclass(frozen=True)
class ScoreContext:
    """Document-local inputs supplied to a range scorer after model execution."""

    token_ids: Any
    target_ids: Any | None = None


Scorer = Callable[[Any, ScoreContext], Any]
TokenLike = int | Token
_MISSING = object()


def _array_namespace(array: Any) -> Any:
    namespace = getattr(array, "__array_namespace__", None)
    return namespace() if namespace is not None else np


def _token_cross_entropy(logits: Any, target_ids: Any) -> Any:
    xp = _array_namespace(logits)
    targets = xp.asarray(target_ids)
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Scorer received {logits.shape[0]} logit positions for "
            f"{targets.shape[0]} targets"
        )
    maxima = xp.max(logits, axis=-1, keepdims=True)
    log_normalizers = xp.log(xp.sum(xp.exp(logits - maxima), axis=-1))
    log_normalizers = log_normalizers + maxima[..., 0]
    target_logits = xp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return log_normalizers - target_logits


def next_token_score(logits: Any, context: ScoreContext) -> Any:
    """Mean token cross-entropy, shifted unless explicit targets are supplied.

    Scorers return a mean over their scored token positions. The Levanter
    adapter weights range means by their target counts when combining them.
    This implementation uses the logits' array namespace, so it works with
    NumPy for construction tests and JAX arrays inside the training step.
    """
    xp = _array_namespace(logits)
    token_ids = xp.asarray(context.token_ids)
    if context.target_ids is None:
        if token_ids.shape[0] < 2:
            return xp.sum(logits) * 0.0
        score_logits = logits[:-1]
        target_ids = token_ids[1:]
    else:
        score_logits = logits
        target_ids = xp.asarray(context.target_ids)
    return xp.mean(_token_cross_entropy(score_logits, target_ids))


@dataclass(frozen=True)
class ScoreRange:
    """One half-open document range and the callback that scores it.

    ``target_ids=None`` asks the scorer to infer ordinary next-token targets
    from the range's input tokens. Explicit targets align one-to-one with the
    positions in ``[start, stop)``. ``scorer=None`` marks an unscored range.
    """

    start: int
    stop: int
    scorer: Scorer | None = next_token_score
    target_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.scorer is not None and not callable(self.scorer):
            raise TypeError("A score range scorer must be callable or None")
        if self.target_ids is not None:
            object.__setattr__(
                self, "target_ids", tuple(int(target) for target in self.target_ids)
            )
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(
                f"Score ranges must be non-empty and ordered, got [{self.start}, {self.stop})"
            )
        if (
            self.target_ids is not None
            and len(self.target_ids) != self.stop - self.start
        ):
            raise ValueError(
                f"Score range [{self.start}, {self.stop}) has {len(self.target_ids)} "
                f"targets, expected {self.stop - self.start}"
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

    def scored_by(
        self,
        scorer: Scorer | None,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> "Document":
        """Return a copy scored by ``scorer`` over ``[start, stop)``.

        Passing ``None`` makes the selected range input-only. Existing explicit
        targets remain attached, allowing ``with_targets`` and ``scored_by`` to
        compose in either order.
        """
        normalized_start, normalized_stop = _normalize_range(
            start, len(self) if stop is None else stop, len(self)
        )
        return self._replace_score_range(
            normalized_start,
            normalized_stop,
            scorer=scorer,
        )

    def unscored(self, *, start: int = 0, stop: int | None = None) -> "Document":
        """Return a copy whose selected range contributes no training score."""
        return self.scored_by(None, start=start, stop=stop)

    def with_targets(
        self,
        target_ids: TokenLike | Sequence[TokenLike],
        *,
        start: int | None = None,
        stop: int | None = None,
    ) -> "Document":
        """Attach explicit token targets to a range without changing its scorer.

        With no range arguments, targets align to the final ``len(target_ids)``
        positions, so ``document.with_targets(token_id)`` naturally targets the
        last position. Supplying ``start`` anchors the targets there; supplying
        only ``stop`` makes them end at that boundary.
        """
        targets = _normalize_targets(target_ids, self.vocabulary)
        if start is None and stop is None:
            normalized_stop = len(self)
            normalized_start = normalized_stop - len(targets)
        elif start is None:
            normalized_stop = _normalize_boundary(stop, len(self))
            normalized_start = normalized_stop - len(targets)
        else:
            normalized_start = _normalize_boundary(start, len(self))
            normalized_stop = (
                normalized_start + len(targets)
                if stop is None
                else _normalize_boundary(stop, len(self))
            )
        normalized_start, normalized_stop = _normalize_range(
            normalized_start, normalized_stop, len(self)
        )
        if normalized_stop - normalized_start != len(targets):
            raise ValueError(
                f"Target range [{normalized_start}, {normalized_stop}) has length "
                f"{normalized_stop - normalized_start}, but received {len(targets)} targets"
            )
        return self._replace_score_range(
            normalized_start,
            normalized_stop,
            target_ids=targets,
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
        scorer: Scorer | None | object = _MISSING,
        target_ids: tuple[int, ...] | object = _MISSING,
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
            replacement_scorer = (
                overlap.scorer if scorer is _MISSING else cast(Scorer | None, scorer)
            )
            if target_ids is _MISSING:
                replacement_targets = overlap.target_ids
            else:
                targets = cast(tuple[int, ...], target_ids)
                replacement_targets = targets[
                    overlap_start - start : overlap_stop - start
                ]
            ranges.append(
                ScoreRange(
                    overlap_start,
                    overlap_stop,
                    replacement_scorer,
                    replacement_targets,
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
                score_range.scorer,
                score_range.target_ids,
            )
            for score_range in document.score_ranges
        )
        offset += len(document)
    return Document(
        np.concatenate(tuple(document.token_ids for document in documents)),
        {
            coordinate: np.concatenate(
                tuple(document[coordinate] for document in documents)
            )
            for coordinate in coordinates
        },
        attention=documents[0].attention,
        score_ranges=tuple(shifted_ranges),
        vocabulary=vocabulary,
    )


def causal_training_document(token_ids: Sequence[TokenLike]) -> Document:
    """Build a causal document using the default shifted-token scorer."""
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


@dataclass(frozen=True)
class PackedScoreRange:
    """One document score range shifted into a packed row."""

    row: int
    start: int
    stop: int
    document_index: int
    scorer: Scorer | None
    target_ids: tuple[int, ...] | None = None


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
                    scorer=score_range.scorer,
                    target_ids=score_range.target_ids,
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
            if previous.target_ids is not None:
                if score_range.target_ids is None:
                    raise AssertionError("Mergeable scoring ranges disagree on targets")
                targets = previous.target_ids + score_range.target_ids
            coalesced.append(
                ScoreRange(
                    previous.start,
                    score_range.stop,
                    previous.scorer,
                    targets,
                )
            )
        else:
            coalesced.append(score_range)
    return tuple(coalesced)


def _can_merge_score_ranges(left: ScoreRange, right: ScoreRange) -> bool:
    return (
        left.stop == right.start
        and left.scorer == right.scorer
        and (left.target_ids is None) == (right.target_ids is None)
    )


def _slice_score_range(score_range: ScoreRange, start: int, stop: int) -> ScoreRange:
    targets = None
    if score_range.target_ids is not None:
        targets = score_range.target_ids[
            start - score_range.start : stop - score_range.start
        ]
    return ScoreRange(start, stop, score_range.scorer, targets)


def _score_range_at(
    score_ranges: tuple[ScoreRange, ...], old_index: int, new_index: int
) -> ScoreRange:
    for score_range in score_ranges:
        if score_range.start <= old_index < score_range.stop:
            targets = None
            if score_range.target_ids is not None:
                targets = (score_range.target_ids[old_index - score_range.start],)
            return ScoreRange(new_index, new_index + 1, score_range.scorer, targets)
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
