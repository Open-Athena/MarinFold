"""Protect original cluster lineage and fail-loud Atlas decoding."""

import hashlib
import io

import brotli
import msgpack
import msgpack_numpy
import numpy as np
import pyarrow as pa
import pytest

from sample_esm import (
    choose_clusters,
    decode_protein,
    sample_members,
)
from structure_audit import noncanonical_sequence


def test_membership_keeps_original_representative_when_training_anchor_differs():
    clusters = [
        {"cluster_id": "original", "protein_hash": "selected", "cluster_size": 3}
    ]
    members, count = sample_members(
        io.BytesIO(
            b"original\toriginal\noriginal\tselected\nother\tother\noriginal\tomitted\n"
        ),
        clusters,
        32,
        292,
    )
    assert count == 4
    assert set(members["original"]) == {"original", "omitted"}
    with pytest.raises(ValueError, match="Incomplete original group"):
        sample_members(io.BytesIO(b"original\tselected\n"), clusters, 32, 292)


def test_choose_clusters_requires_actual_retained_anchor():
    rows = [
        {
            "protein_hash": h,
            "cluster_id": f"original-{h}",
            "seq_len": 100,
            "cluster_size": 5,
            "mean_plddt": 0.9,
        }
        for h in ["retained", "excluded"]
    ]
    selected = choose_clusters(pa.Table.from_pylist(rows), {"retained"}, 4, 292)
    assert [r["protein_hash"] for r in selected] == ["retained"]


def test_unknown_residue_cannot_be_a_full_chain_training_anchor():
    sequence = "MDERLEKALDFSNYMVTLNNQRRLIHEQFLENCVHYLNGGKFSVTRELINFCHMLVSTGQEDVVLIDDNNSPVKVDNIEDFLSEILDIYFTNSNEYLEKFSSLKKKRIKDLVNLXPRVYYFLHSTILT"
    assert len(sequence) == 128
    assert noncanonical_sequence(sequence)
    assert not noncanonical_sequence("ACDEFGHIKLMNPQRSTVWY")


def test_atom37_coordinates_and_sequence_validation():
    mask = np.zeros((2, 37), dtype=bool)
    mask[0, [0, 1, 2, 4]] = True
    mask[1, [0, 1, 2, 3, 4]] = True
    positions = np.arange(27).reshape(9, 3).astype(np.float16)
    decoded = {
        "sequence": "GA",
        "atom37_mask": mask,
        "atom37_positions": positions,
        "confidence": np.array([0.7, 0.9]),
        "chain_boundaries": [[0, 2]],
    }
    blob = brotli.compress(msgpack.packb(decoded, default=msgpack_numpy.encode))
    row = {
        "protein_hash": hashlib.md5(b"GA").hexdigest(),
        "sequence": "GA",
        "structure_blob": blob,
    }
    seq, coords, plddt = decode_protein(row)
    assert seq == "GA"
    np.testing.assert_array_equal(coords, positions[[1, 5]])
    np.testing.assert_allclose(plddt, [70, 90])
    with pytest.raises(ValueError, match="sequence mismatch"):
        decode_protein({**row, "sequence": "GG"})
