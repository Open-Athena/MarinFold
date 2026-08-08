# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""TM-score via US-align, with the residue correspondence fixed by index.

Prediction and ground truth are the *same protein*, so the equivalent-residue
mapping is known: residue ``i`` of the prediction corresponds to residue ``i``
of the ground truth. That makes the sequence-*dependent* TM-score the right
quantity — the one the ``TMscore`` program computes and CASP assessors report
— rather than TM-align's sequence-*independent* structural alignment, which
would be free to slide the prediction along the chain and inflate the score.

US-align's ``-TMscore 1`` is exactly that mode ("a pair of residues with the
same residue index are equivalent"), on top of the TM-score-maximizing
superposition search. Since ``canonical_pdb`` numbers every residue by its
1-based input-sequence index in both files, no alignment step is needed here.

**Normalization.** We always report the score normalized by the *ground-truth*
length. Pass the prediction as structure 1 and the ground truth as structure 2
and read ``TM2``: residues the predictor never placed are simply unmatched, so
they contribute 0 to the numerator while still counting in the denominator.
TM-score is therefore the one headline metric that is inherently
coverage-penalized (unlike RMSD, which is computed over covered atoms only).

Build the binary once with ``bash setup_usalign.sh``.
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BINARY = Path(__file__).resolve().parent / "_bin" / "USalign"


@dataclass(frozen=True)
class TMResult:
    """One US-align ``-TMscore 1`` comparison.

    Attributes:
        tm_score: TM-score normalized by the ground-truth length (``TM2``).
            This is the headline number.
        tm_score_pred_normalized: TM-score normalized by the prediction's own
            length (``TM1``). Reported as a diagnostic: with partial coverage
            it says "how good is the part that was predicted", while
            :attr:`tm_score` says "how good is the prediction as a model of
            this protein".
        rmsd_superposed: RMSD over the matched residues under the
            *TM-score-maximizing* superposition — not the least-squares
            optimum. Kept for provenance; the harness's headline RMSDs come
            from a Kabsch fit in ``structure_metrics``.
        n_aligned: number of matched residue pairs.
        len_pred / len_gt: residue counts US-align saw in each file.
    """

    tm_score: float
    tm_score_pred_normalized: float
    rmsd_superposed: float
    n_aligned: int
    len_pred: int
    len_gt: int


def binary_version(binary: Path = DEFAULT_BINARY) -> str:
    """The US-align version string, for provenance in the results CSV."""
    out = subprocess.run(
        [str(binary)], capture_output=True, text=True, check=False
    ).stdout
    for line in out.splitlines():
        if "US-align (Version" in line:
            return line.strip(" *").strip()
    raise RuntimeError(f"{binary}: could not read a version banner")


def require_binary(binary: Path = DEFAULT_BINARY) -> Path:
    """Return ``binary`` if it is executable, else raise with the fix."""
    if binary.exists() and shutil.which(str(binary)):
        return binary
    raise FileNotFoundError(
        f"US-align not found at {binary}. Build it with:\n"
        f"    bash {Path(__file__).resolve().parent / 'setup_usalign.sh'}"
    )


def tm_score(pred_path: Path, gt_path: Path, *, binary: Path = DEFAULT_BINARY) -> TMResult:
    """Run US-align ``-TMscore 1`` on a prediction / ground-truth PDB pair.

    Args:
        pred_path: prediction, in the ``canonical_pdb`` contract.
        gt_path: ground truth, same contract and same residue numbering.
        binary: the US-align executable.

    Raises:
        RuntimeError: US-align exited non-zero or produced no result row —
            never swallowed, because a silently-zero TM-score would look like
            a terrible prediction rather than a broken harness.
    """
    proc = subprocess.run(
        [str(binary), str(pred_path), str(gt_path), "-TMscore", "1", "-outfmt", "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"US-align failed ({proc.returncode}) on {pred_path} vs {gt_path}:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    rows = [ln for ln in proc.stdout.splitlines() if ln and not ln.startswith("#")]
    if len(rows) != 1:
        raise RuntimeError(
            f"US-align returned {len(rows)} result rows (expected 1) for "
            f"{pred_path} vs {gt_path}:\n{proc.stdout}"
        )
    # #PDBchain1 PDBchain2 TM1 TM2 RMSD ID1 ID2 IDali L1 L2 Lali
    fields = rows[0].split("\t")
    return TMResult(
        tm_score=float(fields[3]),
        tm_score_pred_normalized=float(fields[2]),
        rmsd_superposed=float(fields[4]),
        n_aligned=int(fields[10]),
        len_pred=int(fields[8]),
        len_gt=int(fields[9]),
    )
