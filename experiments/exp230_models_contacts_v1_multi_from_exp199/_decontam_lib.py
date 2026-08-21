# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Tier-A decontamination primitives, vendored from exp225 (#225 / PR #229).

Vendored rather than imported because exp225 lives on an unmerged branch
(``claude/github-issue-225-fead58``) and libraries must not import from
experiments.  Copied verbatim so the thresholds this run filters at are
*provably* the ones #225 priced and #213/#65/#94 reported against — a
re-derivation would silently drift.

Only the sequence axis is carried over.  #225's structure axis (Tiers B and C)
is not used here: Tier C was declined at 37.31 % of AFDB, and Tier B's
structural increment measured 0.54 % on the only arm it could be measured on.

The one deliberate difference from #225 is the **query set**: #225 searched the
554-protein #89 benchmark, this searches #226's **776** (554 + the 222 net-new
FoldBench monomers), a strict superset.  See ``decontam.py``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# --- Tier A's thresholds (exp225 decontam_lib.py, verbatim) -----------------
#
# 0.30 is exp65's ``REDUNDANT_ID``, the twilight-zone novel-family boundary and
# the number the issue asks for ("no >= 30% sequence similarity to our eval
# set").  0.50 qcov is exp213's ``MIN_QCOV`` — a 95%-identical 12-residue local
# match is not homology.  1e-3 is exp213's ``HOMOLOGY_EVALUE``, a *catch-all*
# alongside the identity rule rather than a conjunct.
SEQ_MIN_IDENTITY = 0.30
SEQ_MIN_QCOV = 0.50
SEQ_MAX_EVALUE = 1e-3

#: exp65 / exp94 / exp213 / exp225 all search at -s 7.5.  Keep it so the
#: numbers stay comparable across the five experiments that quote them.
SEARCH_SENSITIVITY = "7.5"

#: The reporting ceiling.  exp65's ``seq_leakage.py`` — the source of the 30%
#: bar — searched at ``-e 10``, and #213 used the same, so ``E <= 10`` is what
#: makes a drop here mean what ``redundant_seq`` means elsewhere in the repo.
REPORT_EVALUE_CEILING = 10.0

#: Same field list as exp213/exp225, so alignments are comparable line-for-line.
CONVERTALIS_FORMAT = "query,target,fident,alnlen,qcov,tcov,evalue,bits"

MMSEQS_DOWNLOAD = (
    "https://github.com/soedinglab/MMseqs2/releases/download/"
    "15-6f452/mmseqs-linux-avx2.tar.gz"
)


def is_sequence_contaminant(identity: float, qcov: float, evalue: float) -> bool:
    """Tier A's rule for one alignment: identity-and-coverage **or** significance.

    The disjunction is the point.  The identity arm catches near-duplicates in
    the 30-40 % band that #91's 40 % funnel let through; the E-value arm catches
    remote homologs that align over too little of the query to clear coverage
    but are still evidence of a relative.
    """
    if evalue <= SEQ_MAX_EVALUE:
        return True
    return identity >= SEQ_MIN_IDENTITY and qcov >= SEQ_MIN_QCOV


@dataclass(frozen=True)
class TrainingRow:
    arm: str
    shard: int
    row: int
    entry_id: str


def parse_target(target: str) -> TrainingRow:
    """Invert exp213's FASTA header grammar.

    ``"esm_atlas|00123_45_0000052aa00ab212061f7c6987fd87ae"`` becomes
    ``TrainingRow(arm="esm_atlas", shard=123, row=45, entry_id="0000...")``.

    ``entry_id`` is everything after the second underscore, so an entry id
    containing underscores (``2m0s_A``) survives the round trip.
    """
    arm, _, rest = target.partition("|")
    shard_s, _, rest = rest.partition("_")
    row_s, _, entry_id = rest.partition("_")
    return TrainingRow(arm=arm, shard=int(shard_s), row=int(row_s), entry_id=entry_id)


#: FASTA tag for the PDB arm.  It is **not** ``pdb``: mmseqs' ``createdb``
#: recognises the NCBI database prefixes (``pdb|``, ``sp|``, ``tr|``, ``gb|``,
#: ...) and rewrites ``>pdb|00000_0_2ieq_B`` down to the bare accession
#: ``00000_0_2ieq_B``, which then fails :func:`parse_target` — the tag is simply
#: gone from every alignment.  #213's ``afdb|`` and ``esm_atlas|`` survive
#: because neither is a recognised prefix.  Costs an hour to rediscover, so the
#: PDB arm carries a tag mmseqs has no opinion about and it is mapped back to
#: the arm name at reduce time.
PDB_FASTA_TAG = "pdbmono"


def format_target(tag: str, shard: int, row: int, entry_id: str) -> str:
    """The inverse of :func:`parse_target` — used when we build our own DB."""
    return f"{tag}|{shard:05d}_{row}_{entry_id}"


def run(cmd: list[str], *, log=print) -> None:
    log("  $ " + " ".join(str(c) for c in cmd[:8]) + (" ..." if len(cmd) > 8 else ""))
    subprocess.run([str(c) for c in cmd], check=True)


def ensure_mmseqs(log=print) -> str:
    """Resolve an mmseqs binary, downloading once into the shared marinfold cache.

    Prefers ``$MMSEQS_BIN``, then the cache path exp213/exp225 already populated
    on this workstation, then a fresh download.
    """
    env = os.environ.get("MMSEQS_BIN")
    if env and Path(env).exists():
        return env
    on_path = shutil.which("mmseqs")
    if on_path:
        return on_path
    cache = Path.home() / ".cache" / "marinfold" / "mmseqs"
    binary = cache / "mmseqs" / "bin" / "mmseqs"
    if binary.exists():
        return str(binary)
    cache.mkdir(parents=True, exist_ok=True)
    tgz = cache / "mmseqs.tar.gz"
    log(f"[mmseqs] downloading {MMSEQS_DOWNLOAD}")
    t0 = time.time()
    urllib.request.urlretrieve(MMSEQS_DOWNLOAD, tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(cache)
    tgz.unlink()
    log(f"[mmseqs] installed in {time.time() - t0:.0f}s -> {binary}")
    return str(binary)
