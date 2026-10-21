"""AFDB source-parser tests without network access."""

import gemmi
import pyarrow as pa
import pyarrow.parquet as pq

from afdb_fetch_rows import fetch_plan_files, parse_row


def test_parse_row_preserves_sequence_and_ca_coordinates() -> None:
    structure = gemmi.Structure()
    structure.name = "TEST"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for index, name in enumerate(["ALA", "GLY"], 1):
        residue = gemmi.Residue()
        residue.name = name
        residue.seqid = gemmi.SeqId(index, " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(float(index), 0, 0)
        atom.b_iso = 90
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    content = structure.make_mmcif_document().as_string().encode()
    output = parse_row(
        {
            "entry_id": "entry",
            "gcs_uri": "gs://example/entry.cif",
            "struct_cluster_id": "cluster",
            "seq_len": 2,
        },
        content,
    )
    assert output["sequence"] == "AG"
    assert output["ca_coords"] == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert output["per_residue_plddt"] == [90.0, 90.0]
    assert output["noncanonical_residues"] == 0


def test_fetch_plan_files_reads_complete_parquet(monkeypatch, tmp_path) -> None:
    path = tmp_path / "plan.parquet"
    pq.write_table(pa.Table.from_pylist([{"entry_id": "A"}, {"entry_id": "B"}]), path)

    class FakeFilesystem:
        def cat_file(self, source: str) -> bytes:
            assert source == "gs://bucket/plan.parquet"
            return path.read_bytes()

    monkeypatch.setattr("afdb_fetch_rows.filesystem", lambda: FakeFilesystem())
    monkeypatch.setattr(
        "afdb_fetch_rows.fetch_shard",
        lambda rows, fetch_concurrency: iter(rows),
    )

    output = list(
        fetch_plan_files(
            ["gs://bucket/plan.parquet"],
            columns=("entry_id",),
            fetch_concurrency=4,
            row_limit_per_shard=1,
        )
    )

    assert output == [{"entry_id": "A"}]
