# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidation must shard evenly, account for every drop, and rank honestly."""

import json
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from consolidate import (
    MIN_INTERFACE_CONTACTS,
    PINDER_EXCLUDED_SPLITS,
    build,
    flatten,
)


def _afcdb_row(i: int, *, pair: tuple[str, str], ctype: str = "homodimer",
               inter_chain: int = 10) -> dict:
    return {
        "source_model_key": f"{ctype}|AF-{i:010d}",
        "model_id": f"AF-{i:010d}",
        "complex_type": ctype,
        "document": f"doc-afcdb-{i}",
        "sha1": f"{i:040x}",
        "seq_len": 200,
        "num_tokens": 900,
        "num_chains": 2,
        "chain_ids": ["A", "B"],
        "chain_lengths": [100, 100],
        "contacts_pre_filter": 50,
        "contacts_emitted": 40,
        "contacts_emitted_inter_chain": inter_chain,
        "contacts_pre_filter_inter_chain": 12,
        "truncated": False,
        "accession_a": pair[0],
        "accession_b": pair[1],
        "total_residues_manifest": 200,
        "quality_ratio": 0.8,
        "ipsae_score": 0.7,
        "pdockq2_score": 0.3,
        "confidence_tier": "A",
        "source_tar_uri": "https://example/t.tar",
        "member_bytes": 1,
        "cif_bytes": 2,
    }


def _pinder_row(i: int, *, cluster: str, split: str = "train",
                inter_chain: int = 9) -> dict:
    return {
        "system_id": f"sys{i:05d}",
        "source_arm": "pinder",
        "document": f"doc-pinder-{i}",
        "sha1": f"{i + 10**6:040x}",
        "seq_len": 180,
        "num_tokens": 800,
        "num_chains": 2,
        "chain_ids": ["R", "L"],
        "chain_lengths": [90, 90],
        "contacts_pre_filter": 40,
        "contacts_emitted": 30,
        "contacts_emitted_inter_chain": inter_chain,
        "contacts_pre_filter_inter_chain": 11,
        "truncated": False,
        "uniprot_R": f"U{i:05d}",
        "uniprot_L": f"V{i:05d}",
        "uniprot_pair": f"U{i:05d}--V{i:05d}",
        "pdb_id": f"{i:04x}",
        "cluster_id": cluster,
        "split": split,
        "resolution": 2.0,
        "method": "X-RAY",
        "release_date": "2020-01-01",
        "total_residues_manifest": 180,
        "intermolecular_contacts": 20,
        "buried_sasa": 900.0,
        "pair_also_in_afcdb": False,
        "sequence_R": "AAAA",
        "sequence_L": "CCCC",
    }


def _arm(root: Path, name: str, rows: list[dict]) -> Path:
    directory = root / name / "documents"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), directory / "part.parquet")
    return root / name


def _build(tmp_path: Path, *, cluster_cap: float) -> dict:
    """One AFCDB pair repeated 8x (a redundant cluster) plus 20 singletons, and a
    PINDER arm carrying a 5-member cluster, a val system, a test system, and one
    on the eval drop list. Each arm also contributes one model whose chains never
    touch."""
    afcdb = [_afcdb_row(i, pair=("P1", "P2")) for i in range(8)]
    afcdb += [_afcdb_row(100 + i, pair=(f"Q{i}", f"R{i}")) for i in range(20)]
    afcdb += [_afcdb_row(200, pair=("Z1", "Z2"), inter_chain=0)]
    pinder = [_pinder_row(i, cluster="c-big") for i in range(5)]
    pinder += [_pinder_row(50 + i, cluster=f"c{i}") for i in range(10)]
    pinder += [_pinder_row(90, cluster="cv", split="val"),
               _pinder_row(91, cluster="ct", split="test"),
               _pinder_row(92, cluster="cz", inter_chain=0)]
    drop = tmp_path / "drop.parquet"
    pq.write_table(pa.Table.from_pylist([{"system_id": "sys00051"}]), drop)
    out = tmp_path / f"out_cap{cluster_cap}"
    stats = build(
        str(_arm(tmp_path, "afcdb", afcdb)),
        str(_arm(tmp_path, "pinder", pinder)),
        str(out),
        pinder_droplist=str(drop),
        shard_rows=10,
        cluster_cap=cluster_cap,
        threads=2,
    )
    return {"stats": stats, "out": out}


@pytest.fixture
def corpus(tmp_path: Path) -> dict:
    """The cap is switched off here: 31 clusters is far too few for a 0.1% cap
    to be inactive, and a binding cap hides the weighting law under test."""
    return _build(tmp_path, cluster_cap=1.0)


@pytest.fixture
def capped(tmp_path: Path) -> dict:
    """A cap that binds on exactly the two multi-member clusters."""
    return _build(tmp_path, cluster_cap=0.05)


def test_benchmark_splits_and_droplist_are_both_applied(corpus: dict) -> None:
    stats = corpus["stats"]
    assert tuple(stats["pinder_excluded_splits"]) == PINDER_EXCLUDED_SPLITS
    assert stats["pinder_dropped_benchmark_split"] == 2
    assert stats["pinder_dropped_eval_homolog"] == 1
    assert stats["pinder_documents_in"] == 18
    assert stats["pinder_documents_kept"] == 15
    assert stats["documents"] == 29 + 15 - 2


def test_shard_files_are_flat_and_carry_no_partition_column(corpus: dict) -> None:
    """PARTITION_BY needs `shard` in its input but must not leak it into rows."""
    shards = sorted((corpus["out"] / "train").glob("*.parquet"))
    assert [p.name for p in shards] == [
        f"shard-{i:05d}-of-{len(shards):05d}.parquet" for i in range(len(shards))
    ]
    con = duckdb.connect()
    glob = str(corpus["out"] / "train" / "*.parquet")
    columns = {c[0] for c in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{glob}')").fetchall()}
    assert "shard" not in columns and "raw_weight" not in columns
    assert con.execute(
        f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()[0] == 42


def test_cluster_concentration_is_monotone(corpus: dict) -> None:
    """Cumulative shares must not fall as the prefix grows.

    An `ORDER BY` inside a CTE is discarded when the CTE is re-read under a
    `LIMIT`, which returned arbitrary clusters and non-monotone shares.
    """
    stats = corpus["stats"]
    for kind in ("balanced_share", "document_share"):
        shares = [stats[f"{kind}_top_{k}_clusters" if k != 1 else
                        f"{kind}_top_1_cluster"] for k in (1, 10, 100)]
        assert 0 < shares[0] <= shares[1] <= shares[2] <= 1.0 + 1e-9, kind
        # The 1% tier of 31 clusters is the single top cluster.
        assert stats[f"{kind}_top_1pct_clusters"] == pytest.approx(shares[0])
    # Weighting only reshapes probability, so the redundant cluster must own a
    # larger share of documents than of sampling mass.
    assert (stats["document_share_top_1_cluster"]
            > stats["balanced_share_top_1_cluster"])


def test_redundant_clusters_are_downweighted_not_dropped(corpus: dict) -> None:
    con = duckdb.connect()
    glob = str(corpus["out"] / "train" / "*.parquet")
    big, small = con.execute(
        f"""
        SELECT max(CASE WHEN cluster_size = 8 THEN sampling_weight END),
               max(CASE WHEN cluster_size = 1 THEN sampling_weight END)
        FROM read_parquet('{glob}')
        """
    ).fetchone()
    assert big == pytest.approx(1.0 / 8 ** 0.5)
    assert small == pytest.approx(1.0)
    # All eight redundant documents survive; only their weight changes.
    assert con.execute(
        f"SELECT count(*) FROM read_parquet('{glob}') WHERE cluster_size = 8"
    ).fetchone()[0] == 8


def test_stats_json_is_written_and_parseable(corpus: dict) -> None:
    written = json.loads((corpus["out"] / "corpus_stats.json").read_text())
    assert written["documents"] == corpus["stats"]["documents"]


def test_cap_binds_only_on_over_represented_clusters(capped: dict, corpus: dict) -> None:
    con = duckdb.connect()

    def weights(out: Path) -> dict[int, float]:
        rows = con.execute(
            f"""
            SELECT cluster_size, max(sampling_weight)
            FROM read_parquet('{out / "train" / "*.parquet"}') GROUP BY 1
            """
        ).fetchall()
        return {int(size): float(weight) for size, weight in rows}

    uncapped, limited = weights(corpus["out"]), weights(capped["out"])
    # Singletons are under the cap, so their weight is untouched.
    assert limited[1] == pytest.approx(uncapped[1])
    # The 8- and 5-member clusters are over it, so theirs is cut.
    assert limited[8] < uncapped[8]
    assert limited[5] < uncapped[5]
    assert (capped["stats"]["balanced_share_top_1_cluster"]
            < corpus["stats"]["balanced_share_top_1_cluster"])


def test_contactless_models_are_dropped_from_both_arms(corpus: dict) -> None:
    """A complex whose chains never touch is not an interface example.

    The PINDER worker already rejects these; AFCDB's ipSAE/pDockQ2 gate lets a
    few through, so consolidation is where the two arms are made to agree.
    """
    stats = corpus["stats"]
    assert stats["min_interface_contacts"] == MIN_INTERFACE_CONTACTS
    assert stats["dropped_no_interface_contacts"] == 2
    con = duckdb.connect()
    assert con.execute(
        f"""SELECT count(*) FROM read_parquet(
                '{corpus["out"] / "train" / "*.parquet"}')
            WHERE contacts_emitted_inter_chain < {MIN_INTERFACE_CONTACTS}"""
    ).fetchone()[0] == 0


def test_the_two_manifests_are_different_sampling_specs(corpus: dict) -> None:
    """Natural is uniform over documents; only balanced carries cluster weight.

    Both once carried the balanced weight and differed only in row order, which
    left a reader nothing to choose between.
    """
    con = duckdb.connect()
    out = corpus["out"]
    natural = con.execute(
        f"SELECT min(sampling_weight), max(sampling_weight), count(*) "
        f"FROM read_parquet('{out / 'manifest_natural.parquet'}')").fetchone()
    balanced = con.execute(
        f"SELECT min(sampling_weight), max(sampling_weight), count(*) "
        f"FROM read_parquet('{out / 'manifest_balanced.parquet'}')").fetchone()
    assert natural[0] == natural[1] == 1.0
    assert balanced[0] < balanced[1] == pytest.approx(1.0)
    assert natural[2] == balanced[2] == corpus["stats"]["documents"]


def test_flatten_merges_a_partition_duckdb_split(tmp_path: Path) -> None:
    """One `shard=N/` directory may hold several part files.

    duckdb flushes a partition once it passes its write threshold, so at 48
    threads shard 165 of this corpus arrived as `part_0` + `part_1`. Assuming one
    file per partition dropped half that shard.
    """
    stage, train = tmp_path / "_staged", tmp_path / "train"
    (stage / "shard=0").mkdir(parents=True)
    (stage / "shard=1").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"i": 0}, {"i": 1}]),
                   stage / "shard=0" / "part_0.parquet")
    pq.write_table(pa.Table.from_pylist([{"i": 2}]),
                   stage / "shard=1" / "part_0.parquet")
    pq.write_table(pa.Table.from_pylist([{"i": 3}, {"i": 4}]),
                   stage / "shard=1" / "part_1.parquet")

    merged = flatten(duckdb.connect(), stage, train, 2)

    assert merged == 2
    assert not stage.exists()
    assert sorted(p.name for p in train.iterdir()) == [
        "shard-00000-of-00002.parquet", "shard-00001-of-00002.parquet"]
    con = duckdb.connect()
    assert sorted(r[0] for r in con.execute(
        f"SELECT i FROM read_parquet('{train / '*.parquet'}')").fetchall()) == [
        0, 1, 2, 3, 4]


def test_flatten_rejects_a_missing_partition(tmp_path: Path) -> None:
    stage, train = tmp_path / "_staged", tmp_path / "train"
    (stage / "shard=0").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"i": 0}]), stage / "shard=0" / "p.parquet")
    with pytest.raises(RuntimeError, match="shard 1: duckdb wrote no part file"):
        flatten(duckdb.connect(), stage, train, 2)
