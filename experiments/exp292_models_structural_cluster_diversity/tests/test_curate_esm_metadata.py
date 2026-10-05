"""Pure filtering tests for the ESM metadata stage."""

from curate_esm_metadata import partition_after_screen, quality_filter


def row(
    entry_id: str,
    sequence: str,
    *,
    cluster: str = "cluster",
    anchor: bool = False,
    plddt: float = 90,
    ptm: float = 0.8,
) -> dict:
    """Make one decoded Atlas metadata row."""
    return {
        "entry_id": entry_id,
        "protein_hash": entry_id,
        "struct_cluster_id": cluster,
        "cluster_id": cluster,
        "is_anchor": anchor,
        "sequence": sequence,
        "seq_len": len(sequence),
        "anchor_seq_len": len(sequence) if anchor else 80,
        "global_plddt": plddt,
        "ptm": ptm,
        "reservoir_rank": 0 if anchor else 1,
    }


def test_noncanonical_anchor_removes_whole_cluster() -> None:
    rows = [
        row("anchor", "A" * 79 + "X", anchor=True),
        row("candidate", "A" * 80),
    ]
    kept, rejected = quality_filter(rows)
    assert kept == []
    assert len(rejected) == 2
    assert {item["rejection_reason"] for item in rejected} == {
        "noncanonical_anchor_cluster"
    }


def test_sequence_filter_fills_three_and_queues_clusters_with_choice() -> None:
    anchor = row("anchor", "A" * 80, anchor=True)
    candidates = [row(f"candidate-{index}", "A" * 80, plddt=81 + index) for index in range(5)]
    quality, rejected = quality_filter([anchor, *candidates])
    assert not rejected
    selected, queue, sequence_rejections = partition_after_screen(
        quality,
        {"candidate-4": {"identity": 0.4, "shorter_coverage": 0.8}},
    )
    assert selected == []
    assert len(queue) == 5
    assert len(sequence_rejections) == 1

    selected, queue, sequence_rejections = partition_after_screen(
        quality,
        {
            "candidate-3": {"identity": 0.4, "shorter_coverage": 0.8},
            "candidate-4": {"identity": 0.4, "shorter_coverage": 0.8},
        },
    )
    assert len(selected) == 3
    assert queue == []
    assert len(sequence_rejections) == 2
    assert all(item["selection_tier"] == "quality_fill" for item in selected)
