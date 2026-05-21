# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modal_app import _inject_precomputed_msa_paths, _seed_outputs_complete  # noqa: E402


def _job_data() -> list[dict]:
    return [{
        "name": "5sbj_A",
        "sequences": [{"proteinChain": {"sequence": "ACD", "count": 1}}],
        "covalent_bonds": [],
    }]


def test_seed_outputs_complete_requires_every_sample(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed_1"
    seed_dir.mkdir()
    (seed_dir / "5sbj_A_distogram.npz").write_bytes(b"npz")
    (seed_dir / "5sbj_A_sample_0.cif").write_text("sample 0")
    (seed_dir / "5sbj_A_summary_confidence_sample_0.json").write_text("{}")

    assert not _seed_outputs_complete(seed_dir, stem="5sbj_A", n_sample=2)

    (seed_dir / "5sbj_A_sample_1.cif").write_text("sample 1")
    (seed_dir / "5sbj_A_summary_confidence_sample_1.json").write_text("{}")

    assert _seed_outputs_complete(seed_dir, stem="5sbj_A", n_sample=2)


def test_inject_precomputed_msa_paths_warns_without_files(tmp_path: Path, capsys) -> None:
    job_data = _job_data()

    found_precomputed_msa = _inject_precomputed_msa_paths(
        job_data,
        stem="5sbj_A",
        msa_root=tmp_path,
    )

    assert not found_precomputed_msa
    assert "auto-search at inference time" in capsys.readouterr().out
    protein_chain = job_data[0]["sequences"][0]["proteinChain"]
    assert "pairedMsaPath" not in protein_chain
    assert "unpairedMsaPath" not in protein_chain


def test_inject_precomputed_msa_paths_uses_existing_files(tmp_path: Path) -> None:
    msa_base = tmp_path / "5sbj_A" / "msa" / "0"
    pairing = msa_base / "pairing.a3m"
    non_pairing = msa_base / "0" / "non_pairing.a3m"
    non_pairing.parent.mkdir(parents=True)
    pairing.write_text(">query\nACD\n")
    non_pairing.write_text(">query\nACD\n")

    job_data = _job_data()

    found_precomputed_msa = _inject_precomputed_msa_paths(
        job_data,
        stem="5sbj_A",
        msa_root=tmp_path,
    )

    assert found_precomputed_msa
    protein_chain = job_data[0]["sequences"][0]["proteinChain"]
    assert protein_chain["pairedMsaPath"] == str(pairing)
    assert protein_chain["unpairedMsaPath"] == str(non_pairing)
