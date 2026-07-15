# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the sequence-KNN contact baseline (issue #94).

Three concerns live here so the five pipeline scripts agree on the contracts:

* **Document parsing** — turn a contacts-v1 training ``document`` string into a
  one-letter sequence + 0-based contact pairs, undoing the resampled token order
  and the modulo-2000 N-terminus offset. Validated against ``contacts_emitted``
  on 2000/2000 rows of shard 0.
* **Alignment mapping** — walk an mmseqs ``qaln``/``taln`` pair into a
  ``target-residue -> query-residue`` map (both 0-based) so a training protein's
  contacts can be copied through the local alignment onto the eval protein.
* **Score tie-breaking + the mmseqs binary** — ``tiebreak_matrix`` is copied
  verbatim from exp82 (the integer-vote AUC fix); ``ensure_mmseqs`` mirrors
  exp65's installer.

Coordinate convention (verified against exp89's ``gt_universe.jsonl`` and
``compute_metrics.py``): every index is a 0-based position into the protein's
input sequence, dense over ``[0, L)``. The eval score matrices we emit are
``[L, L]`` in that exact space, so exp89's metric harness consumes them directly.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

# --- document parsing -------------------------------------------------------

# Standard-20 three-letter -> one-letter, inverted from contacts_v1/parse.py's
# `_ONE_LETTER_TO_THREE`. Anything else in a document (only ever <UNK>) -> "X".
ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
THREE_TO_ONE = {v: k for k, v in ONE_TO_THREE.items()}

# A sequence token is `<pN> <RES3>`; a contact statement is `<contact> <pN> <pN>`.
# RES_RE naturally skips `<n-term> <pN>` / `<c-term> <pN>` markers (those are
# `<word> <pN>`, not `<pN> <RES3>`). Both match exp82's eval_contact_prediction.
RES_RE = re.compile(r"<p(\d+)>\s+<([A-Z]{3})>")
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")
BEGIN = "<begin_statements>"
POS_MOD = 2000  # contacts-v1 position vocabulary size
MIN_SEP = 6     # exp89 metric's minimum sequence separation


def parse_document(document: str, seq_len: int, n_term_index: int):
    """``document`` text -> (one-letter sequence, list of 0-based ``(i, j)`` contacts).

    Tokens carry positions modulo 2000 relative to a per-document N-terminus, and
    the residues are emitted in resampled order, so we invert both:
    ``seqpos = (token - n_term_index) mod 2000`` and keep iff ``0 <= seqpos < seq_len``.

    Contacts are returned as ``sorted (i, j)`` tuples with ``i < j`` (no separation
    filter — that is re-applied in query coordinates after alignment mapping).
    Out-of-range tokens are dropped silently; on the train corpus there are none.
    """
    cut = document.find(BEGIN)
    seq_part = document if cut < 0 else document[:cut]
    state_part = "" if cut < 0 else document[cut:]

    seq = ["X"] * seq_len
    for tok, res3 in RES_RE.findall(seq_part):
        sp = (int(tok) - n_term_index) % POS_MOD
        if 0 <= sp < seq_len:
            seq[sp] = THREE_TO_ONE.get(res3, "X")

    contacts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for a, b in CONTACT_RE.findall(state_part):
        ia = (int(a) - n_term_index) % POS_MOD
        jb = (int(b) - n_term_index) % POS_MOD
        if 0 <= ia < seq_len and 0 <= jb < seq_len and ia != jb:
            key = (ia, jb) if ia < jb else (jb, ia)
            if key not in seen:
                seen.add(key)
                contacts.append(key)
    return "".join(seq), contacts


# --- alignment mapping ------------------------------------------------------

def target_to_query_map(qaln: str, taln: str, qstart: int, tstart: int) -> dict[int, int]:
    """0-based ``target residue -> query residue`` from one mmseqs local alignment.

    mmseqs ``qstart``/``tstart`` are 1-based inclusive, so we subtract 1 before
    walking the aligned columns. A column with residues on both sides advances
    both cursors and records the pair; a gap advances only the side that carries a
    residue. Only aligned (non-gap/non-gap) columns enter the map.
    """
    qpos = qstart - 1
    tpos = tstart - 1
    t2q: dict[int, int] = {}
    for qc, tc in zip(qaln, taln):
        q_res = qc != "-"
        t_res = tc != "-"
        if q_res and t_res:
            t2q[tpos] = qpos
            qpos += 1
            tpos += 1
        elif t_res:        # gap in query: consume only the target residue
            tpos += 1
        else:              # gap in target: consume only the query residue
            qpos += 1
    return t2q


# --- score tie-breaking (verbatim from exp82 build_comparison_table.py) ------

def tiebreak_matrix(count: np.ndarray, pairwise: np.ndarray) -> np.ndarray:
    """count primary; pairwise breaks ties — count + min-max(pairwise) scaled to [0, 0.5).

    The vote counts are integers (>= 1 per neighbor), so adding a pairwise term
    bounded to [0, 0.5) only reorders pairs that are *tied* on votes; it cannot
    move a pair across a count boundary. min-max is monotonic, so within a tie
    group this is identical to ranking by the raw pairwise score.
    """
    iu = np.triu_indices(count.shape[0], k=1)
    s = pairwise[iu]
    lo, hi = float(s.min()), float(s.max())
    return count + (pairwise - lo) / (hi - lo + 1e-9) * 0.5


# --- mmseqs binary (mirrors exp65 seq_leakage.ensure_mmseqs) ----------------

MMSEQS_DOWNLOAD = "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
_CACHE = Path.home() / ".cache" / "marinfold" / "mmseqs"


def ensure_mmseqs() -> str:
    """Return a path to the mmseqs binary, installing the static build if needed."""
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
        print(f"[knn] downloading mmseqs from {MMSEQS_DOWNLOAD}")
        urllib.request.urlretrieve(MMSEQS_DOWNLOAD, tar)
        with tarfile.open(tar) as tf:
            try:
                tf.extractall(_CACHE, filter="data")
            except TypeError:
                tf.extractall(_CACHE)
    return str(binary.resolve())


def run(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd[:7]), "...", flush=True)
    subprocess.run(cmd, check=True)
