# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The pilot draw must be reproducible, stratified, and honest about shortfalls."""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from pilot import _allocate, build


def _selected(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "selected.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _row(i: int, *, quality: float, ctype: str, residues: int, tax: int) -> dict:
    return {
        "model_id": f"AF-{i:010d}",
        "complex_type": ctype,
        "quality_ratio": quality,
        "total_residues": residues,
        "tax_id_a": tax,
        "source_tar_uri": f"https://example/tar_{i % 7}.tar",
    }


def _corpus(n: int = 600) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            _row(
                i,
                quality=(1.2, 0.7, 0.4, 0.1)[i % 4],
                ctype="homodimer" if i % 3 else "heterodimer",
                residues=(300, 600, 1000, 1600)[i % 4],
                tax=9606 + (i % 5),
            )
        )
    return rows


def test_allocation_redistributes_thin_strata() -> None:
    """A stratum smaller than its share must not shrink the total draw."""
    assert sum(_allocate({"a": 1, "b": 100, "c": 100}, 60).values()) == 60
    quotas = _allocate({"a": 1, "b": 100, "c": 100}, 60)
    assert quotas["a"] == 1, "a thin stratum contributes everything it has"
    assert quotas["b"] + quotas["c"] == 59


def test_allocation_cannot_exceed_availability() -> None:
    assert sum(_allocate({"a": 3, "b": 4}, 100).values()) == 7


def test_pilot_is_deterministic_and_stratified(tmp_path: Path) -> None:
    selected = _selected(tmp_path, _corpus())
    first = build(selected, tmp_path / "one", size=120)
    second = build(selected, tmp_path / "two", size=120)

    assert first["drawn"] == 120
    ids_one = pq.read_table(first["pilot_manifest"]).column("model_id").to_pylist()
    ids_two = pq.read_table(second["pilot_manifest"]).column("model_id").to_pylist()
    assert ids_one == ids_two, "the draw must be reproducible from the manifest alone"
    assert len(set(ids_one)) == len(ids_one)

    # Every populated stratum contributes; that is the point of stratifying.
    drawn = {b["stratum"]: b["drawn"] for b in first["breakdown"]}
    assert all(n > 0 for n in drawn.values())
    assert first["strata"] == len(drawn)


def test_pilot_oversamples_the_low_confidence_tail(tmp_path: Path) -> None:
    """Tier A is not what the floor decision turns on, so it must not dominate."""
    rows = [_row(i, quality=1.5, ctype="homodimer", residues=600, tax=9606) for i in range(5000)]
    rows += [
        _row(10000 + i, quality=0.35, ctype="homodimer", residues=600, tax=9606)
        for i in range(200)
    ]
    stats = build(_selected(tmp_path, rows), tmp_path / "out", size=400)
    drawn = {b["quality_bin"]: b["drawn"] for b in stats["breakdown"]}
    assert drawn["B_0.3_0.5"] == 200, "the whole thin tail should be taken"
    assert drawn["A_ge_1.0"] == 200


def test_pilot_reports_a_shortfall_instead_of_silently_shrinking(tmp_path: Path) -> None:
    selected = _selected(tmp_path, _corpus(n=40))
    with pytest.raises(RuntimeError, match="pilot drew"):
        build(selected, tmp_path / "out", size=500)
    # The manifest is still written, so the frontier can be inspected.
    assert (tmp_path / "out" / "pilot_manifest.parquet").is_file()
    assert (tmp_path / "out" / "pilot.json").is_file()
