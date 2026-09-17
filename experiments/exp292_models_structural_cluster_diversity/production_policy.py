"""Frozen selection policy for the exp292 production supplement."""

from collections.abc import Callable, Mapping, Sequence

DIVERSITY_TM_THRESHOLD = 0.8
MIN_COMPARABLE_COVERAGE = 0.8
MIN_LENGTH_RATIO = 0.8
ADDITIONS_PER_CLUSTER = 3


def is_anchor(row: Mapping) -> bool:
    """Interpret the CSV/parquet forms of the anchor flag."""
    value = row["is_anchor"]
    return value is True or str(value).lower() == "true"


def pair_index(pairs: Sequence[Mapping]) -> dict[frozenset[str], Mapping]:
    """Index symmetric pair metrics and reject duplicates."""
    result = {}
    for pair in pairs:
        key = frozenset((pair["entry_a"], pair["entry_b"]))
        if len(key) != 2 or key in result:
            raise ValueError("Pair metrics must contain one unique row per protein pair")
        result[key] = pair
    return result


def length_ratio(a: Mapping, b: Mapping) -> float:
    """Return the shorter-to-longer sequence-length ratio."""
    shorter = min(int(a["seq_len"]), int(b["seq_len"]))
    longer = max(int(a["seq_len"]), int(b["seq_len"]))
    return shorter / longer


def metrics_to_selected(
    candidate: Mapping,
    selected: Sequence[Mapping],
    pairs: Mapping[frozenset[str], Mapping],
) -> dict:
    """Aggregate structural comparability and similarity to a selected set."""
    comparisons = []
    for row in selected:
        key = frozenset((candidate["entry_id"], row["entry_id"]))
        if key not in pairs:
            raise ValueError(
                f"Missing structural metrics for {candidate['entry_id']} and {row['entry_id']}"
            )
        comparisons.append(pairs[key])
    core_values = [pair["core_tm_max"] for pair in comparisons]
    whole_tm = max(float(pair["tm_max"]) for pair in comparisons)
    core_tm = (
        max(float(value) for value in core_values)
        if all(value is not None for value in core_values)
        else None
    )
    minimum_coverage = min(
        min(float(pair["coverage_a"]), float(pair["coverage_b"]))
        for pair in comparisons
    )
    minimum_length_ratio = min(length_ratio(candidate, row) for row in selected)
    maximum_sequence_identity = max(
        float(pair["aligned_sequence_identity"]) for pair in comparisons
    )
    comparable = (
        minimum_coverage >= MIN_COMPARABLE_COVERAGE
        and minimum_length_ratio >= MIN_LENGTH_RATIO
        and core_tm is not None
    )
    strict = (
        comparable
        and whole_tm <= DIVERSITY_TM_THRESHOLD
        and core_tm <= DIVERSITY_TM_THRESHOLD
    )
    combined_tm = max(whole_tm, core_tm) if core_tm is not None else whole_tm
    return {
        "max_selected_tm": whole_tm,
        "max_selected_core_tm": core_tm,
        "min_selected_coverage": minimum_coverage,
        "min_selected_length_ratio": minimum_length_ratio,
        "max_selected_sequence_identity": maximum_sequence_identity,
        "structural_novelty": 1.0 - combined_tm,
        "structurally_comparable": comparable,
        "strict_structural_diversity": strict,
    }


def confidence_key(row: Mapping, anchor_length: int) -> tuple:
    """Sort quality fillers by confidence, length match, identity, then ID."""
    identity = row.get("max_anchor_sequence_identity")
    return (
        -float(row["global_plddt"]),
        -float(row.get("ptm") or 0.0),
        abs(int(row["seq_len"]) - anchor_length),
        float(identity) if identity is not None else 1.0,
        str(row["entry_id"]),
    )


def select_three(
    rows: Sequence[Mapping],
    pair_rows: Sequence[Mapping],
    additions: int = ADDITIONS_PER_CLUSTER,
) -> list[dict]:
    """Select structural alternatives first, then fill every remaining slot.

    All input candidates have already passed source integrity, confidence,
    length, canonical-residue, and held-out sequence filters. Clusters with no
    choice (at most ``additions`` candidates) require no structural comparison.
    Larger clusters must supply complete all-pairs metrics so a proposed mode is
    checked against every current anchor and every earlier addition.
    """
    selected, _ = select_three_dynamic(rows, pair_rows, additions=additions)
    return selected


def select_three_dynamic(
    rows: Sequence[Mapping],
    pair_rows: Sequence[Mapping] = (),
    *,
    compare_pair: Callable[[Mapping, Mapping], Mapping] | None = None,
    additions: int = ADDITIONS_PER_CLUSTER,
) -> tuple[list[dict], list[dict]]:
    """Select three while computing only candidate-to-selected comparisons.

    ``compare_pair`` is invoked lazily for a missing candidate/anchor or
    candidate/addition pair. This avoids the unused candidate-to-candidate
    comparisons in a full all-pairs matrix.
    """
    anchors = [dict(row) for row in rows if is_anchor(row)]
    candidates = [dict(row) for row in rows if not is_anchor(row)]
    if not anchors:
        raise ValueError("Every production cluster needs a retained training anchor")
    if not candidates:
        return [], []
    cluster_ids = {str(row["struct_cluster_id"]) for row in rows}
    if len(cluster_ids) != 1:
        raise ValueError("Selection input spans multiple original clusters")
    slots = min(additions, len(candidates))
    anchor_length = round(sum(int(row["seq_len"]) for row in anchors) / len(anchors))
    if len(candidates) <= additions:
        return [
            {
                **candidate,
                "selection_rank": rank,
                "selection_tier": "quality_fill",
                "max_selected_tm": None,
                "max_selected_core_tm": None,
                "min_selected_coverage": None,
                "min_selected_length_ratio": None,
                "max_selected_sequence_identity": None,
                "structural_novelty": None,
                "structurally_comparable": None,
                "strict_structural_diversity": False,
            }
            for rank, candidate in enumerate(
                sorted(candidates, key=lambda row: confidence_key(row, anchor_length)),
                1,
            )
        ], []

    pairs = pair_index(pair_rows)
    measured = [dict(pair) for pair in pair_rows]

    def ensure(candidate: Mapping, selected: Sequence[Mapping]) -> None:
        for row in selected:
            key = frozenset((candidate["entry_id"], row["entry_id"]))
            if key in pairs:
                continue
            if compare_pair is None:
                raise ValueError(
                    f"Missing structural metrics for {candidate['entry_id']} and {row['entry_id']}"
                )
            pair = {
                "struct_cluster_id": candidate["struct_cluster_id"],
                "entry_a": candidate["entry_id"],
                "entry_b": row["entry_id"],
                **compare_pair(candidate, row),
            }
            pairs[key] = pair
            measured.append(pair)

    selected_set = list(anchors)
    pending = candidates[:]
    chosen_rows = []
    while pending and len(chosen_rows) < slots:
        assessed = []
        for candidate in pending:
            ensure(candidate, selected_set)
            assessed.append(
                {**candidate, **metrics_to_selected(candidate, selected_set, pairs)}
            )
        strict = [row for row in assessed if row["strict_structural_diversity"]]
        if not strict:
            break
        chosen = min(
            strict,
            key=lambda row: (
                -float(row["structural_novelty"]),
                *confidence_key(row, anchor_length),
            ),
        )
        chosen["selection_rank"] = len(chosen_rows) + 1
        chosen["selection_tier"] = "structural_diversity"
        chosen_rows.append(chosen)
        selected_set.append(chosen)
        pending = [row for row in pending if row["entry_id"] != chosen["entry_id"]]

    anchor_metrics = {
        candidate["entry_id"]: metrics_to_selected(candidate, anchors, pairs)
        for candidate in pending
    }
    fillers = []
    for candidate in pending:
        assessed = {**candidate, **anchor_metrics[candidate["entry_id"]]}
        assessed["max_anchor_sequence_identity"] = assessed[
            "max_selected_sequence_identity"
        ]
        fillers.append(assessed)
    fillers.sort(key=lambda row: confidence_key(row, anchor_length))
    for filler in fillers[: slots - len(chosen_rows)]:
        filler["selection_rank"] = len(chosen_rows) + 1
        filler["selection_tier"] = "quality_fill"
        chosen_rows.append(filler)
    return chosen_rows, measured
