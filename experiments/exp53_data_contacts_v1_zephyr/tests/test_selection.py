# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Stage A (``selection.py``) on the synthetic manifest."""

from pathlib import Path

import pyarrow.parquet as pq

import selection as sel


def _split_shards(out_dir: Path, split: str) -> list[Path]:
    """Manifest shards for ``split`` in filename (== round-descending) order."""
    return sorted((out_dir / split).glob("shard_*.parquet"))


def _entry_to_round(out_dir: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for split_dir in out_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for shard in split_dir.glob("shard_*.parquet"):
            t = pq.read_table(shard)
            for e, r in zip(t.column("entry_id").to_pylist(), t.column("round").to_pylist()):
                mapping[e] = r
    return mapping


def test_cluster_drops_and_counts(synthetic_afdb, tmp_path):
    stats = sel.run(str(synthetic_afdb), tmp_path / "m", shard_size=2)
    # Clusters A,B,D,F kept; C (size 2) and E (only 2 usable after seq_len) dropped.
    assert stats["clusters_kept"] == 4
    assert stats["clusters_dropped"] == 2
    assert stats["clusters_post_seqlen"] == 6
    # A(5) + B(3) + D(4) + F(3) = 15 selected documents.
    assert stats["selected_docs_total"] == 15


def test_round_sizes_monotone_and_equal_early(synthetic_afdb, tmp_path):
    stats = sel.run(str(synthetic_afdb), tmp_path / "m", shard_size=2)
    psr = {(d["split"], d["round"]): d["docs"] for d in stats["per_split_round"]}
    # Issue #53: round-0 == round-1 == round-2 (no cluster stops before round 2),
    # then non-increasing.
    assert psr[("train", 0)] == psr[("train", 1)] == psr[("train", 2)] == 3
    assert psr[("train", 3)] == 1
    assert psr[("train", 4)] == 1
    assert psr[("test", 0)] == psr[("test", 1)] == psr[("test", 2)] == psr[("test", 3)] == 1
    assert ("test", 4) not in psr        # D has only 4 members
    assert not any(k[0] == "val" for k in psr)  # C dropped -> no val survivors


def test_round_assignment_by_plddt_with_entry_tiebreak(synthetic_afdb, tmp_path):
    out = tmp_path / "m"
    sel.run(str(synthetic_afdb), out, shard_size=2)
    e2r = _entry_to_round(out)
    expected = {
        "a1": 0, "a2": 1, "a3": 2, "a4": 3, "a5": 4,   # pLDDT desc; a6 dropped (round 5)
        "b2": 0, "b3": 1, "b1": 2,                       # by pLDDT, not insertion order
        "d4": 0, "d2": 1, "d1": 2, "d3": 3,
        "f1": 0, "f2": 1, "f3": 2,                       # equal pLDDT -> entry_id ascending
    }
    assert e2r == expected
    for dropped in ("a6", "c1", "c2", "e1", "e2", "e3"):
        assert dropped not in e2r


def test_physical_order_is_round_descending(synthetic_afdb, tmp_path):
    out = tmp_path / "m"
    sel.run(str(synthetic_afdb), out, shard_size=2)
    for split in ("train", "test"):
        rounds_in_order: list[int] = []
        for shard in _split_shards(out, split):
            vals = pq.read_table(shard).column("round").to_pylist()
            assert len(set(vals)) == 1, "each shard must hold a single round"
            rounds_in_order.extend(vals)
        # Highest round first, round-0 last; never increases.
        assert rounds_in_order == sorted(rounds_in_order, reverse=True)
        assert rounds_in_order[0] == max(rounds_in_order)
        assert rounds_in_order[-1] == 0


def test_seqlen_filter_applied(synthetic_afdb, tmp_path):
    out = tmp_path / "m"
    sel.run(str(synthetic_afdb), out, shard_size=2)
    for split_dir in out.iterdir():
        if not split_dir.is_dir():
            continue
        for shard in split_dir.glob("shard_*.parquet"):
            lens = pq.read_table(shard).column("seq_len").to_pylist()
            assert all(2 <= n <= 2000 for n in lens)


def test_deterministic(synthetic_afdb, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    sel.run(str(synthetic_afdb), a, shard_size=2)
    sel.run(str(synthetic_afdb), b, shard_size=2)
    a_shards = sorted(p.relative_to(a) for p in a.rglob("*.parquet"))
    b_shards = sorted(p.relative_to(b) for p in b.rglob("*.parquet"))
    assert a_shards == b_shards
    for rel in a_shards:
        assert pq.read_table(a / rel).equals(pq.read_table(b / rel))
