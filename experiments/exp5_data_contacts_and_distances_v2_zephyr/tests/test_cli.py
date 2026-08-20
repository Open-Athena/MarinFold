# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CLI smoke: the full Zephyr pipeline composes off-cluster.

Uses ``file://`` URIs in a local parquet manifest so the URI-mode path
(``Dataset.from_files`` → ``load_parquet`` → ``map_shard`` w/ thread
pool fetch → write_parquet) exercises against a real fsspec backend
without needing GCS auth or network access.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import pyarrow as pa
import pyarrow.parquet as pq

import cli


def _make_local_manifest(tmp_path: Path, synthetic_cif: str, n_rows: int) -> Path:
    """Write `n_rows` copies of the synthetic cif as individual files +
    a parquet manifest pointing at them via ``file://`` URIs."""
    cifs_dir = tmp_path / "cifs"
    cifs_dir.mkdir()
    entry_ids, uris = [], []
    for i in range(n_rows):
        cif_path = cifs_dir / f"s{i}.cif"
        cif_path.write_text(synthetic_cif)
        entry_ids.append(f"AF-SMOKE{i:03d}-F1")
        uris.append(f"file://{cif_path}")
    manifest = tmp_path / "manifest.parquet"
    pq.write_table(
        pa.table({"entry_id": entry_ids, "gcs_uri": uris}),
        str(manifest),
    )
    return manifest


_REQUIRED_OUTPUT_COLUMNS = {
    "entry_id", "structure", "document",
    "sha1", "seq_len", "global_plddt", "contacts_emitted",
}


def test_cmd_generate_uri_mode_e2e(tmp_path, synthetic_cif):
    """URI mode: parquet manifest → fetch + parse + gen → single output parquet."""
    manifest = _make_local_manifest(tmp_path, synthetic_cif, n_rows=3)
    out = tmp_path / "docs.parquet"

    args = cli.build_parser().parse_args([
        "generate",
        "--input", str(manifest),
        "--out", str(out),
        # Default cif_uri_column='gcs_uri' picks our column.
        # ThreadPool of 2 keeps the test deterministic + fast.
        "--fetch-concurrency", "2",
    ])
    cli.cmd_generate(args)

    tbl = pq.read_table(str(out))
    assert tbl.num_rows == 3
    cols = set(tbl.column_names)
    assert _REQUIRED_OUTPUT_COLUMNS.issubset(cols)
    # gcs_uri is in the manifest (it's our cif source) and listed in
    # _OPTIONAL_PASSTHROUGH, so it should also appear in the output.
    assert "gcs_uri" in cols, f"expected gcs_uri passthrough; got {sorted(cols)}"
    structures = set(tbl.column("structure").to_pylist())
    assert structures == {"contacts-and-distances-v2"}
    # entry_id + gcs_uri are threaded through verbatim from the manifest.
    assert tbl.column("entry_id").to_pylist() == ["AF-SMOKE000-F1", "AF-SMOKE001-F1", "AF-SMOKE002-F1"]
    for u in tbl.column("gcs_uri").to_pylist():
        assert u.startswith("file://") and u.endswith(".cif")
    # Each doc starts with the v2 marker + the synthetic poly-ALA sequence,
    # ends with <end>, and its sha1 is consistent.
    import hashlib
    for d, sha in zip(tbl.column("document").to_pylist(), tbl.column("sha1").to_pylist()):
        assert d.startswith("<contacts-and-distances-v2> <begin_sequence> <ALA>")
        assert d.endswith("<end>")
        assert sha == hashlib.sha1(d.encode()).hexdigest()


def test_cmd_generate_inline_cif_text_fallback(tmp_path, synthetic_cif):
    """--cif-text-column path: bulk inline-cif read, no URI fetching."""
    manifest = tmp_path / "inline.parquet"
    pq.write_table(
        pa.table({
            "entry_id": ["AF-INL1-F1", "AF-INL2-F1"],
            "cif_content": [synthetic_cif, synthetic_cif],
        }),
        str(manifest),
    )
    out = tmp_path / "inline_docs.parquet"
    args = cli.build_parser().parse_args([
        "generate",
        "--input", str(manifest),
        "--cif-text-column", "cif_content",
        "--out", str(out),
    ])
    cli.cmd_generate(args)
    tbl = pq.read_table(str(out))
    assert tbl.num_rows == 2


def test_cmd_generate_passthrough_columns_propagated(tmp_path, synthetic_cif):
    """Manifest columns matching the passthrough wishlist are copied to output.

    Mirrors the afdb-1.6M shape (a 'split' + cluster ids alongside entry_id +
    gcs_uri). The schema peek in ``_resolve_input_columns`` should detect them
    and pass them through verbatim to every output row.
    """
    cifs_dir = tmp_path / "cifs"
    cifs_dir.mkdir()
    rows = []
    for i, split in enumerate(["train", "train", "val"]):
        cif_path = cifs_dir / f"r{i}.cif"
        cif_path.write_text(synthetic_cif)
        rows.append({
            "entry_id": f"AF-PT{i:02d}-F1",
            "gcs_uri": f"file://{cif_path}",
            "split": split,
            "seq_cluster_id": f"clust{i // 2}",
            "struct_cluster_id": f"sclust{i}",
        })
    manifest = tmp_path / "rich.parquet"
    pq.write_table(
        pa.table({k: [r[k] for r in rows] for k in rows[0]}),
        str(manifest),
    )
    out = tmp_path / "rich_out.parquet"
    args = cli.build_parser().parse_args([
        "generate", "--input", str(manifest), "--out", str(out),
        "--fetch-concurrency", "2",
    ])
    cli.cmd_generate(args)

    tbl = pq.read_table(str(out))
    assert tbl.num_rows == 3
    cols = set(tbl.column_names)
    assert _REQUIRED_OUTPUT_COLUMNS.issubset(cols)
    assert {"split", "seq_cluster_id", "struct_cluster_id"}.issubset(cols), (
        f"expected passthrough columns present in output, got {sorted(cols)}"
    )
    # Per-row values copied unchanged (and aligned to entry_id, since
    # ``executor.map`` preserves row order within a shard).
    assert tbl.column("split").to_pylist() == ["train", "train", "val"]
    assert tbl.column("seq_cluster_id").to_pylist() == ["clust0", "clust0", "clust1"]


def test_cmd_generate_num_docs_caps_total(tmp_path, synthetic_cif):
    """--num-docs N collapses to a single output file with up to N rows."""
    manifest = _make_local_manifest(tmp_path, synthetic_cif, n_rows=5)
    out = tmp_path / "capped.parquet"
    args = cli.build_parser().parse_args([
        "generate", "--input", str(manifest), "--out", str(out),
        "--num-docs", "2",
        "--fetch-concurrency", "2",
    ])
    cli.cmd_generate(args)
    assert pq.read_table(str(out)).num_rows == 2
