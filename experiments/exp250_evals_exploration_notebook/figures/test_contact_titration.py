# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The two pieces of the titration animation that fail silently, checked.

**The PDB writer.** PDB is a fixed-width format, and a record that is one character short is not
an error anywhere — it parses, into the wrong fields. Omitting the altLoc blank shifts `resSeq` by
one character, every atom lands in its own residue, and PyMOL draws 150 disconnected cartoon stubs
instead of a chain. That is what the first version of this did, and the only symptom was a picture
that looked like confetti.

**The superposition.** Kabsch through an SVD gives a reflection rather than a rotation whenever
the determinant comes out negative, and a mirror image fits *better* than the correct
superposition, so the failure is invisible in the residual. The trimming has its own quiet mode:
when no atom survives the cutoff the fit falls back to all atoms, and reporting the set it fitted
over would claim a 150-residue agreement for a prediction that has none.

    uv run --with pytest pytest test_contact_titration.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    """Import a script whose filename starts with a digit, which `import` cannot spell."""
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


titration = load_module("contact_titration", "8_plot_contact_titration.py")


def rotation_about_z(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def chain():
    """A crooked little backbone — nothing symmetric, so a mirror image is not also a rotation."""
    rng = np.random.default_rng(0)
    return np.cumsum(rng.normal(size=(40, 3)), axis=0)


def test_kabsch_recovers_a_rotation_and_translation(chain):
    rotation = rotation_about_z(0.7)
    shift = np.array([3.0, -2.0, 11.0])
    moved = chain @ rotation.T + shift
    recovered, translation = titration.kabsch(chain, moved)
    assert np.allclose(chain @ recovered.T + translation, moved, atol=1e-8)
    assert np.linalg.det(recovered) == pytest.approx(1.0)


def test_kabsch_refuses_to_mirror(chain):
    """A reflected copy must not be fitted by reflecting back — that is not a superposition.

    Without the determinant correction the SVD returns the improper transform, which lands every
    atom exactly and so looks like a perfect fit.
    """
    mirrored = chain * np.array([1.0, 1.0, -1.0])
    rotation, translation = titration.kabsch(chain, mirrored)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert not np.allclose(chain @ rotation.T + translation, mirrored, atol=1e-3)


def test_trimming_finds_a_core_that_least_squares_would_smear(chain):
    """Half the chain placed correctly, half thrown away: the correct half must land."""
    rotation, shift = rotation_about_z(1.1), np.array([1.0, 5.0, -3.0])
    target = chain @ rotation.T + shift
    mobile = chain.copy()
    mobile[20:] += np.linspace(6.0, 40.0, 20)[:, None]     # the second half goes wandering

    (found, translation), core = titration.robust_superpose(mobile, target)
    moved = mobile @ found.T + translation
    distance = np.linalg.norm(moved - target, axis=1)
    assert core.sum() >= 18, "the intact half should survive the trimming"
    assert distance[:20].max() < 1e-6, "the intact half should land exactly"
    assert core[:20].all() and not core[20:].any()

    # What plain least squares does with the same input, and why the trimming is here.
    plain, plain_translation = titration.kabsch(mobile, target)
    smeared = np.linalg.norm(mobile @ plain.T + plain_translation - target, axis=1)
    assert smeared[:20].max() > 1.0


def test_no_core_reports_no_core(chain):
    """A prediction with nothing in the right place must not be described as superposed."""
    rng = np.random.default_rng(1)
    target = rng.normal(scale=12.0, size=chain.shape)
    _, core = titration.robust_superpose(chain, target)
    assert core.sum() < 0.15 * len(chain)


def test_pdb_records_put_every_field_in_its_column():
    """The columns PyMOL reads, checked by slicing them back out of the record."""
    atom_index = pd.DataFrame({
        "chain_id": ["A"] * 4, "res_seq_id": [0, 0, 7, 7],
        "atom_name": ["N", "CA", "CB", "CA"], "element": ["N", "C", "C", "C"],
        "entity_type": ["protein"] * 4})
    coords = np.array([[1.0, 2.0, 3.0], [-4.5, 5.25, 6.125],
                       [10.0, 20.0, 30.0], [-1.0, -2.0, -3.0]])
    sequence = "GAWKLMNPQRST"
    lines = titration.pdb_lines(coords, atom_index, sequence).splitlines()

    assert lines[-1] == "END"
    for line, (name, res_seq_id, point) in zip(
            lines, [("N", 0, coords[0]), ("CA", 0, coords[1]),
                    ("CB", 7, coords[2]), ("CA", 7, coords[3])], strict=False):
        assert line[:6] == "ATOM  "
        assert line[12:16].strip() == name
        assert line[16] == " ", "column 17 is altLoc and must be blank, not the residue name"
        assert line[17:20] == titration.THREE_LETTER[sequence[res_seq_id]]
        assert line[21] == "A"
        assert int(line[22:26]) == res_seq_id + 1
        assert [float(line[30:38]), float(line[38:46]), float(line[46:54])] == \
            pytest.approx(list(point), abs=5e-4)
        assert line[76:78].strip() in {"N", "C"}


def test_ligand_atoms_are_not_written():
    """Six sodium ions are matched for scoring and are not part of the fold being drawn."""
    atom_index = pd.DataFrame({
        "chain_id": ["A", "A", "F"], "res_seq_id": [0, 0, 0],
        "atom_name": ["N", "CA", "NA"], "element": ["N", "C", "NA"],
        "entity_type": ["protein", "protein", "ligand"]})
    lines = titration.pdb_lines(np.zeros((3, 3)), atom_index, "GA").splitlines()
    assert len([line for line in lines if line.startswith("ATOM")]) == 2
