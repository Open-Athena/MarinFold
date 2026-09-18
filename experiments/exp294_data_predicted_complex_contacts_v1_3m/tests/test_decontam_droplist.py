# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The drop list must catch a real homolog and spare an unrelated sequence."""

import gzip
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from decontam_droplist import accession_from_header, build, build_droplist

CACHED_MMSEQS = Path.home() / ".cache/marinfold/mmseqs/mmseqs/bin/mmseqs"
has_mmseqs = shutil.which("mmseqs") is not None or CACHED_MMSEQS.is_file()

# A real eval-set sequence and a near-identical variant of it, so the identity
# arm of the rule has something genuine to fire on.
EVAL_SEQ = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHS"
    "LAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGI"
)
DECOY_SEQ = "".join(reversed(EVAL_SEQ))


def test_accession_from_header_handles_mmseqs_target_names() -> None:
    """mmseqs reports the full first header token, AFDB prefix and all."""
    assert accession_from_header("AFDB:AF-A0A919MGV6-F1") == "A0A919MGV6"
    assert accession_from_header("AF-Q9XYZ1-F1 trailing") == "Q9XYZ1"
    assert accession_from_header("sp|P12345|NAME") == "P12345"


def test_build_droplist_keeps_the_strongest_hit(tmp_path: Path) -> None:
    """Two alignments to one accession must yield the lower-E reason."""
    m8 = tmp_path / "aln.m8"
    m8.write_text(
        "q1\tAFDB:AF-P1-F1\t0.95\t0.99\t1e-40\t400\n"
        "q2\tAFDB:AF-P1-F1\t0.32\t0.55\t1e-04\t90\n"
        # Below the reporting ceiling but failing both arms of the rule.
        "q3\tAFDB:AF-P2-F1\t0.10\t0.10\t5.0\t20\n"
        # Clears the rule but is above the reporting ceiling.
        "q4\tAFDB:AF-P3-F1\t0.99\t0.99\t50.0\t10\n"
    )
    dropped, stats = build_droplist(m8, report_ceiling=10.0)
    assert set(dropped) == {"P1"}
    assert "1e-40" in dropped["P1"], "the strongest hit should name the reason"
    assert stats["alignments_reported"] == 4
    assert stats["dropped_accessions"] == 1


@pytest.mark.skipif(not has_mmseqs, reason="mmseqs binary not available")
def test_end_to_end_drops_the_homolog_only(tmp_path: Path) -> None:
    reference = tmp_path / "eval2.fasta"
    reference.write_text(f">foldbench100__test_A\n{EVAL_SEQ}\n")

    sequences = tmp_path / "seqs"
    sequences.mkdir()
    with gzip.open(sequences / "afcdb-0000.fasta.gz", "wt") as handle:
        # HOMOLOG is the eval sequence with a handful of substitutions; DECOY is
        # its reverse, which has the same composition but no real alignment.
        homolog = EVAL_SEQ[:40] + "A" + EVAL_SEQ[41:100] + "G" + EVAL_SEQ[101:]
        handle.write(f">AFDB:AF-HOMOLOG-F1 x\n{homolog}\n")
        handle.write(f">AFDB:AF-DECOY-F1 x\n{DECOY_SEQ}\n")

    out = tmp_path / "droplist.parquet"
    provenance = build(
        str(sequences / "*.fasta.gz"),
        reference,
        tmp_path / "work",
        out,
        threads=2,
        split_memory_limit="4G",
    )
    dropped = dict(
        zip(
            pq.read_table(out).column("accession").to_pylist(),
            pq.read_table(out).column("eval_decontam_reason").to_pylist(),
        )
    )
    assert "HOMOLOG" in dropped, "a near-identical subunit must be dropped"
    assert "DECOY" not in dropped, "an unrelated sequence must survive"
    assert dropped["HOMOLOG"].startswith("eval_homolog:foldbench100__test_A")
    assert provenance["reference_version"] == "eval2-v1"
