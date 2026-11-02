# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for the exp213 train-set sequence-overlap audit (issue #213).

Four concerns, kept here so the four pipeline scripts can't disagree:

* **Training arms** — the two corpora `contacts-v1-exp199-1.5B` was trained on
  (``afdb`` = the AFDB/AlphaFold2 `contacts_v1` train split, ``esm_atlas`` =
  the ESM-Atlas/ESMFold2 corpus). Every training sequence carries its arm in
  its FASTA header so a hit can be attributed back to one.
* **The FASTA header grammar** — ``{arm}|{local_id}``. MMseqs2 keys a target on
  the header up to the first whitespace, so the arm survives the search and
  ``convertalis`` round trip with no side table.
* **The novelty ladder** — how a best-hit identity becomes a stratum label.
  Defined once, used by the table builder, the comparison and the plots.
* **The mmseqs binary** — same installer exp65/exp94 use.

Identity convention (matches exp65's ``seq_leakage.py``): a hit only counts
toward the identity axis if it covers at least :data:`MIN_QCOV` of the *query*.
A 95 %-identical 12-residue local match is not homology, and gating on query
coverage is the standard way to say so. The uncoverage-gated maximum is
recorded separately as a paranoid upper bound.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

# --- training arms ----------------------------------------------------------

ARM_AFDB = "afdb"
ARM_ESM = "esm_atlas"
ARMS = (ARM_AFDB, ARM_ESM)

#: Human labels for plots/tables, with the structure predictor whose output
#: became that arm's training labels.
ARM_LABELS = {
    ARM_AFDB: "AFDB (AlphaFold2 labels)",
    ARM_ESM: "ESM Atlas (ESMFold2 labels)",
}


def fasta_header(arm: str, local_id: str) -> str:
    """``{arm}|{local_id}`` — the target id mmseqs will report."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if "|" in local_id or any(c.isspace() for c in local_id):
        raise ValueError(f"local_id {local_id!r} must not contain '|' or whitespace")
    return f"{arm}|{local_id}"


def arm_of(target: str) -> str:
    """Inverse of :func:`fasta_header` — the arm a hit's target belongs to."""
    arm = target.split("|", 1)[0]
    if arm not in ARMS:
        raise ValueError(f"target {target!r} has no recognised arm prefix")
    return arm


# --- the novelty ladder -----------------------------------------------------

#: A hit must cover this fraction of the query to count toward identity.
MIN_QCOV = 0.50

#: A hit must be at least this significant to count as "detectable homology".
#: 1e-3 is the conventional MMseqs2/BLAST significance line; anything above it
#: is not evidence of a relative.
HOMOLOGY_EVALUE = 1e-3

#: Upper edges of the identity strata, applied to the coverage-gated best
#: identity. Order matters — :func:`identity_stratum` walks it in order.
IDENTITY_EDGES = (0.20, 0.30, 0.50, 0.70)

STRATUM_NO_HIT = "no_homolog"
STRATUM_REMOTE = "remote"
STRATUM_ORDER = (
    STRATUM_NO_HIT,
    STRATUM_REMOTE,
    "id_20_30",
    "id_30_50",
    "id_50_70",
    "id_70_100",
)

STRATUM_LABELS = {
    STRATUM_NO_HIT: "no homolog",
    STRATUM_REMOTE: "remote\n(<20% id)",
    "id_20_30": "20–30%",
    "id_30_50": "30–50%",
    "id_50_70": "50–70%",
    "id_70_100": "≥70%",
}


def identity_stratum(n_significant_hits: int, best_identity_covered: float | None) -> str:
    """Bucket one eval protein by how close its nearest training sequence is.

    ``n_significant_hits`` counts hits at or below :data:`HOMOLOGY_EVALUE`
    against *either* arm; ``best_identity_covered`` is the highest ``fident``
    among those hits that also cover at least :data:`MIN_QCOV` of the query
    (``None`` when no hit clears the coverage bar).

    The two ends of the ladder are the ones that carry the argument:
    ``no_homolog`` (nothing detectable at all — the headline homology-free
    subset) and ``remote`` (something is detectable, but either it aligns to
    less than half the query or it does so below 20 % identity).
    """
    if n_significant_hits <= 0:
        return STRATUM_NO_HIT
    if best_identity_covered is None or best_identity_covered < IDENTITY_EDGES[0]:
        return STRATUM_REMOTE
    for edge, name in zip(IDENTITY_EDGES[1:], STRATUM_ORDER[2:]):
        if best_identity_covered < edge:
            return name
    return STRATUM_ORDER[-1]


# --- designed vs natural ----------------------------------------------------

#: exp65's de novo PDB set is *designed* protein. Designed proteins have no
#: evolutionary lineage, so they populate the low-identity strata for a reason
#: that has nothing to do with our training set — and structure predictors find
#: their idealised backbones easy. Every stratified number is reported split on
#: this flag as well as pooled (issue #94's confound, made explicit).
DESIGNED_DATASETS = frozenset({"denovo_pdb"})


def is_designed(dataset: str) -> bool:
    return dataset in DESIGNED_DATASETS


# --- mmseqs binary ----------------------------------------------------------

MMSEQS_DOWNLOAD = "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
_CACHE = Path.home() / ".cache" / "marinfold" / "mmseqs"


def ensure_mmseqs() -> str:
    """Path to the mmseqs binary, installing the static build if needed.

    Mirrors exp65's / exp94's installer so all three experiments share one
    cached binary.
    """
    env_bin = os.environ.get("MMSEQS_BIN")
    if env_bin and Path(env_bin).exists():
        return str(Path(env_bin).resolve())
    on_path = shutil.which("mmseqs")
    if on_path:
        return on_path
    binary = _CACHE / "mmseqs" / "bin" / "mmseqs"
    if not binary.exists():
        _CACHE.mkdir(parents=True, exist_ok=True)
        tar = _CACHE / "mmseqs.tar.gz"
        print(f"[overlap] downloading mmseqs from {MMSEQS_DOWNLOAD}", flush=True)
        urllib.request.urlretrieve(MMSEQS_DOWNLOAD, tar)
        with tarfile.open(tar) as tf:
            try:
                tf.extractall(_CACHE, filter="data")
            except TypeError:  # pragma: no cover - Python < 3.12
                tf.extractall(_CACHE)
    return str(binary.resolve())


def run(cmd: list[str], *, quiet: bool = True) -> None:
    """Run a subprocess, echoing an abbreviated command line."""
    print("  $", " ".join(str(c) for c in cmd[:7]), "...", flush=True)
    subprocess.run(
        [str(c) for c in cmd],
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )
