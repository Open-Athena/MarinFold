# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The single seam onto exp213's overlap code and committed artifacts.

exp226 is an *extension* of exp213's audit, not a new one: it appends 222
queries to exp213's query set and searches them against the same 70.9 M-sequence
target database, so that the two tables join and the counts are comparable. That
only works if the identity conventions and the alignment reduction are literally
the same code, not a re-implementation that agrees today and drifts tomorrow.

So this module — and only this module — puts exp213's directory on ``sys.path``
and re-exports what exp226 needs. Everything else in exp226 imports from here.

Why not a kind library: `experiments/AGENTS.md` rule 7 says shared helpers move
to a kind dir once a second use case exists, which is now. But exp213 is still
unmerged (PR #216) and exp226 is stacked on its branch, so promoting
``overlap_lib`` would mean rewriting an open PR's files from a downstream one.
Left as a follow-up: when #216 lands, ``overlap_lib`` is the natural seed for an
``evals/`` kind library and both experiments should import it from there.
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP213_DIR = HERE.parent / "exp213_evals_train_sequence_overlap_audit"
if not EXP213_DIR.is_dir():  # pragma: no cover - branch-layout guard
    raise SystemExit(
        f"exp213 directory not found at {EXP213_DIR}. exp226 is stacked on "
        "exp213's branch (PR #216); check out "
        "claude/eval-sequence-overlap-analysis-33a77a or a commit that contains it."
    )
sys.path.insert(0, str(EXP213_DIR))

from overlap_lib import (  # noqa: E402
    ARM_AFDB,
    ARM_ESM,
    ARM_LABELS,
    ARMS,
    HOMOLOGY_EVALUE,
    MIN_QCOV,
    arm_of,
    ensure_mmseqs,
    identity_stratum,
    is_designed,
    run,
)
from search_overlap import (  # noqa: E402
    FIELDS,
    FORMAT,
    MANIFEST_STRATA,
    reduce_alignments,
)

__all__ = [
    "ARM_AFDB", "ARM_ESM", "ARM_LABELS", "ARMS", "HOMOLOGY_EVALUE", "MIN_QCOV",
    "arm_of", "ensure_mmseqs", "identity_stratum", "is_designed", "run",
    "FIELDS", "FORMAT", "MANIFEST_STRATA", "reduce_alignments",
    "EXP213_DIR", "EXP213_QUERIES", "EXP213_TABLE", "EXP213_QUERIES_SHA256",
    "read_exp213_queries", "check_exp213_queries",
]

#: exp213's 554-record query FASTA. The expanded set is this file *verbatim*
#: plus the net-new records, so the 554 rows of the expanded table are directly
#: comparable to exp213's — that is the validation anchor (284 / 264 survivors).
EXP213_QUERIES = EXP213_DIR / "data" / "eval_queries.fasta"

#: exp213's committed per-protein identity table (554 rows).
EXP213_TABLE = EXP213_DIR / "data" / "eval_train_identity.csv"

#: Pinned so a silent edit to exp213's query set cannot quietly invalidate the
#: parity check without the run failing first.
EXP213_QUERIES_SHA256 = (
    "17336c8bcfdcc024ddda5d7c1383181620658aa02c2ca78bd3d1da1509a3566a"
)


def check_exp213_queries(path: Path = EXP213_QUERIES) -> str:
    """Assert exp213's query FASTA is the one exp226's numbers assume."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXP213_QUERIES_SHA256:
        raise SystemExit(
            f"{path} has sha256 {digest}, expected {EXP213_QUERIES_SHA256}. "
            "exp213's query set changed; the 284/264 parity anchor no longer applies."
        )
    return digest


def read_exp213_queries(path: Path = EXP213_QUERIES) -> list[tuple[str, str]]:
    """exp213's queries as ``[(header, sequence), ...]`` in file order.

    Headers are ``{dataset}__{stem}``. Order is preserved so the expanded FASTA
    can reproduce exp213's file byte-for-byte in its first 554 records.
    """
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header, chunks = line[1:].strip(), []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records
