# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score every predicted structure against the ground truth, for every arm.

helico#14 published lDDT / TM-score / GDT_TS for its own arms, but only for the
FoldBench monomers and not for Protenix or ESMFold on these targets. Since the
dashboard already holds every structure renumbered onto one residue index, the
metrics can simply be computed here — which gives complete coverage across all
42 proteins and all arms instead of a patchwork.

Definitions, stated because they are not interchangeable between tools:

``lddt``
    biotite's superposition-free lDDT over Cα atoms, all-atom inclusion radius
    15 Å, thresholds 0.5/1/2/4 Å.
``gdt_ts``
    Mean fraction of Cα within 1, 2, 4 and 8 Å after superposition.
``tm_score``
    biotite's TM-score under the same superposition. Superposition is
    outlier-trimmed (``superimpose_without_outliers``) rather than the iterative
    search TM-align runs, so this tracks TM-align closely but is not identical
    to it.

The published helico numbers are the check: :func:`validate` reports how far
this implementation lands from them on the arms where both exist.

    uv run python dashboard/build_structure_metrics.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import lddt, superimpose_without_outliers, tm_score
from biotite.structure.io.pdb import PDBFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import upstream as U  # noqa: E402
from build_structure_files import HELICO_SCORES, HELICO_SCORES_363K, SCORE_ARMS  # noqa: E402

HERE = Path(__file__).resolve().parent
STRUCTURES = HERE / "structures"
GDT_CUTOFFS = (1.0, 2.0, 4.0, 8.0)


def alpha_carbons(path: Path):
    """Cα atoms of one arm, keyed by the evaluation residue number."""

    atoms = PDBFile.read(str(path)).get_structure(model=1)
    return atoms[atoms.atom_name == "CA"]


def common(reference, subject):
    """The Cα atoms both structures model, in evaluation order."""

    shared = np.intersect1d(reference.res_id, subject.res_id)
    return (
        reference[np.isin(reference.res_id, shared)],
        subject[np.isin(subject.res_id, shared)],
        shared,
    )


def gdt_ts(reference, subject) -> float:
    """Mean fraction of Cα under 1/2/4/8 Å, after superposition."""

    distances = np.linalg.norm(reference.coord - subject.coord, axis=-1)
    return float(np.mean([(distances <= cutoff).mean() for cutoff in GDT_CUTOFFS]))


def score(reference_path: Path, subject_path: Path) -> dict | None:
    """lDDT / GDT_TS / TM-score for one predicted structure."""

    reference, subject = alpha_carbons(reference_path), alpha_carbons(subject_path)
    reference, subject, shared = common(reference, subject)
    if len(shared) < 10:
        return None
    fitted, _, _ = superimpose_without_outliers(reference, subject)
    return {
        "lddt": round(float(lddt(reference, subject)), 3),
        "gdt_ts": round(gdt_ts(reference, fitted), 3),
        "tm_score": round(float(tm_score(reference, fitted, np.arange(len(shared)),
                                         np.arange(len(shared)))), 3),
        "n_scored_residues": int(len(shared)),
    }


def validate(computed: dict[tuple[str, str], dict]) -> dict:
    """Compare the computed lDDT to helico#14's published numbers."""

    published = pd.concat(
        [pd.read_csv(HELICO_SCORES), pd.read_csv(HELICO_SCORES_363K)], ignore_index=True
    )
    published = published[published.arm.isin(SCORE_ARMS) & (published.status == "ok")]
    rows = []
    for record in published.itertuples(index=False):
        key = (record.stem, SCORE_ARMS[record.arm])
        if key in computed:
            rows.append(
                {
                    "stem": record.stem,
                    "arm": SCORE_ARMS[record.arm],
                    "published": float(record.lddt),
                    "computed": computed[key]["lddt"],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"compared": 0}
    difference = (frame.computed - frame.published).abs()
    return {
        "compared": int(len(frame)),
        "mean_absolute_difference": round(float(difference.mean()), 4),
        "max_absolute_difference": round(float(difference.max()), 4),
        "correlation": round(float(frame.computed.corr(frame.published)), 4),
    }


def main() -> None:
    index = json.loads((HERE / "structure_index.json").read_text())
    computed: dict[tuple[str, str], dict] = {}
    for key, arms in index.items():
        stem = key.split("__", 1)[1]
        truth = next((a for a in arms if a["arm"] == "ground_truth"), None)
        if truth is None:
            continue
        reference_path = HERE / truth["file"]
        for arm in arms:
            if arm["arm"] == "ground_truth":
                continue
            metrics = score(reference_path, HERE / arm["file"])
            if metrics is None:
                continue
            computed[(stem, arm["arm"])] = metrics
            arm.update({f"computed_{k}": v for k, v in metrics.items()})

    report = validate(computed)
    (HERE / "structure_index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    (U.DATA / "structure_metrics_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"scored": len(computed), "validation": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
