# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate ``explore_predictions.ipynb`` — the 3D viewer notebook.

Writing the notebook from a script rather than by hand keeps the cell sources
reviewable in a normal diff and lets the artifact URIs be filled in from one
place. Execute the result (``jupyter nbconvert --execute --inplace``) before
committing, per ``experiments/AGENTS.md``: notebooks are committed **with**
their outputs so a reader can skim without running anything.

The notebook itself must run with **no authentication** (experiments/AGENTS.md
rule 2), so it reads everything from the public ``open-athena/MarinFold`` HF
bucket over plain HTTPS.

Usage::

    uv run python build_notebook.py --out explore_predictions.ipynb
"""

import argparse
import json
from pathlib import Path

# Anonymous read of a public HF bucket. The path after ``/resolve/`` is
# **fully URL-quoted** — slashes included — which is what huggingface_hub
# builds internally; the un-quoted form 404s. Doing it with plain ``requests``
# rather than the hub client keeps the notebook free of a huggingface_hub
# version pin and of any token (experiments/AGENTS.md rule 2).
BUCKET_RESOLVE = "https://huggingface.co/buckets/open-athena/MarinFold/resolve"
BUCKET_PREFIX = "data/exp174-structural-eval"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# exp174 — contacts-and-crops-v1 structure predictions vs ground truth

Interactive 3D viewer for [issue #174](https://github.com/Open-Athena/MarinFold/issues/174).

Pick a protein and an inference plan; the prediction and the experimental
structure are superimposed and drawn together, with that protein's metrics
underneath.

**Reading the colours.** The prediction is coloured by the resolution tier the
document actually reached, which is the thing this format lives or dies on:

* **blue** — refined by a Pass-2 crop (0.1 Å tenths available)
* **orange** — Pass-1 only, so the model localized the atom to a **10 Å box**
  and it is drawn at that box's centre
* **grey** — the ground truth

An orange atom sitting ~5 Å from grey is not a model error; it is the format's
coarse tier doing exactly what it says. The `box10` row of the ceiling table
(lDDT 0.32 even when *every* box is correct) is what that costs.

Runs anonymously — no login, no token. Open in Colab straight from GitHub.""",
    ),
    (
        "code",
        """# Colab has none of these; locally you probably already do.
%pip install -q py3Dmol biotite pandas requests""",
    ),
    (
        "code",
        f'''import io, tarfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

RESOLVE = "{BUCKET_RESOLVE}"
PREFIX = "{BUCKET_PREFIX}"
CACHE = Path("exp174_data"); CACHE.mkdir(exist_ok=True)


def fetch(name: str) -> Path:
    """Download one published artifact (cached), anonymously.

    The bucket path is quoted whole (``safe=""``), so its slashes become
    %2F — the form the resolve endpoint expects.
    """
    local = CACHE / name.replace("/", "_")
    if not local.exists():
        path = quote(f"{{PREFIX}}/{{name}}", safe="")
        response = requests.get(f"{{RESOLVE}}/{{path}}", timeout=600)
        response.raise_for_status()
        local.write_bytes(response.content)
    return local


def fetch_structures(name: str) -> Path:
    """Download and unpack a structure tarball, returning its root."""
    root = CACHE / name.replace("/", "_").replace(".tar.gz", "")
    if not root.exists():
        with tarfile.open(fetch(name)) as tar:
            tar.extractall(root)
    return root


# scores_all.csv carries every scored run, including the model-free
# quantization baselines from the ceiling table. Only the runs below have their
# structures published, so only those can be viewed in 3D.
VIEWABLE = [
    "oracle-doc",
    "e2-cc1mix5-step50000",
    "e1-cc1mix5-step50000",
    "f-cc1mix5-step50000",
    "c-cc1mix5-step50000",
    "a-cc1mix5-step50000",
    "a-3way-step20000",
]

scores = pd.read_csv(fetch("results/scores_all.csv"))
gt_root = fetch_structures("gt/gt_structures.tar.gz")
print(f"{{len(scores):,}} scored (record, run) rows across {{scores.run.nunique()}} runs")
print("viewable in 3D:", [r for r in VIEWABLE if r in set(scores.run)])''',
    ),
    (
        "markdown",
        "## Aggregate — where each plan lands against the format's ceiling",
    ),
    (
        "code",
        """summary = (
    scores[scores.run.isin(VIEWABLE)]
    .groupby("run")
    .agg(n=("record_id", "size"),
         atom_coverage=("atom_coverage", "mean"),
         lddt=("lddt_all", "mean"),
         lddt_ca=("lddt_ca", "mean"),
         tm_score=("tm_score", "mean"),
         rmsd_ca=("rmsd_ca", "median"))
    .sort_values("lddt", ascending=False)
)
summary.style.format({"atom_coverage": "{:.3f}", "lddt": "{:.3f}", "lddt_ca": "{:.3f}",
                      "tm_score": "{:.3f}", "rmsd_ca": "{:.2f}"})""",
    ),
    (
        "markdown",
        """## Pick a protein

`oracle-document` is the ceiling row: a **real document generated from the
ground truth** and decoded by the same code path. It is what a perfect model
would score, so it is the honest comparator for every model row.""",
    ),
    (
        "code",
        '''RUN = "f-cc1mix5-step50000"      # any name from VIEWABLE above
TOP_N = 25

if RUN not in VIEWABLE:
    raise SystemExit(f"{RUN!r} has no published structures; pick from {VIEWABLE}")

ranked = (
    scores[(scores.run == RUN) & (scores.status == "ok")]
    .sort_values("lddt_all", ascending=False)
    .loc[:, ["record_id", "L", "atom_coverage", "frac_refined_of_gt",
             "lddt_all", "lddt_ca", "tm_score", "rmsd_ca"]]
)
print(f"{RUN}: {len(ranked)} scored; best {TOP_N} by lDDT")
ranked.head(TOP_N).style.format(precision=3)''',
    ),
    (
        "code",
        '''RECORD_ID = ranked.iloc[0].record_id     # or e.g. "foldbench100/7t9r_A"
print("showing", RECORD_ID)''',
    ),
    (
        "markdown",
        "## 3D view — prediction (blue = refined, orange = box-only) vs ground truth (grey)",
    ),
    (
        "code",
        '''import numpy as np
import py3Dmol
from biotite.structure import superimpose
from biotite.structure.io.pdb import PDBFile

REFINED_MAX_SIGMA = 1.0   # B-factor is the predictor's positional sigma in A


def read(path):
    return PDBFile.read(str(path)).get_structure(model=1, extra_fields=["b_factor"])


def load_pair(run, record_id):
    dataset, stem = record_id.split("/")
    pred_root = fetch_structures(f"results/pred_{run}.tar.gz")
    pred = read(next(pred_root.rglob(f"{dataset}/{stem}.pdb")))
    gt = read(gt_root / "gt_structures" / dataset / f"{stem}.pdb")

    # Superimpose the prediction onto the ground truth over their common atoms.
    # Each document lives in its own random frame, so this is required, not
    # cosmetic -- and it is exactly what the RMSD metric does.
    gt_keys = {(r, a): i for i, (r, a) in enumerate(zip(gt.res_id.tolist(), gt.atom_name.tolist()))}
    pred_rows, gt_rows = [], []
    for i, (r, a) in enumerate(zip(pred.res_id.tolist(), pred.atom_name.tolist())):
        if (r, a) in gt_keys:
            pred_rows.append(i); gt_rows.append(gt_keys[(r, a)])
    if len(pred_rows) < 3:
        raise ValueError(f"only {len(pred_rows)} common atoms - nothing to superimpose")
    _, transform = superimpose(gt[gt_rows], pred[pred_rows])
    return transform.apply(pred), gt, len(pred_rows)


def to_pdb_string(array, mask=None):
    subset = array if mask is None else array[mask]
    handle = PDBFile()
    handle.set_structure(subset)
    buffer = io.StringIO()
    handle.write(buffer)
    return buffer.getvalue()


def show(record_id, run=RUN, width=900, height=600):
    pred, gt, n_common = load_pair(run, record_id)
    refined = pred.b_factor <= REFINED_MAX_SIGMA

    view = py3Dmol.view(width=width, height=height)
    view.addModel(to_pdb_string(gt), "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "lightgrey"},
                                 "sphere": {"color": "lightgrey", "radius": 0.22}})
    if refined.any():
        view.addModel(to_pdb_string(pred, refined), "pdb")
        view.setStyle({"model": 1}, {"sphere": {"color": "#1f77b4", "radius": 0.35}})
    if (~refined).any():
        view.addModel(to_pdb_string(pred, ~refined), "pdb")
        view.setStyle({"model": 2}, {"sphere": {"color": "#ff7f0e", "radius": 0.35}})
    view.zoomTo()

    row = scores[(scores.run == run) & (scores.record_id == record_id)].iloc[0]
    print(f"{record_id}  |  {run}  |  L={int(row.L)}  |  {n_common} common atoms")
    print(f"  coverage {row.atom_coverage:.3f}   refined {row.frac_refined_of_gt:.3f}"
          f"   ({int(refined.sum())} blue / {int((~refined).sum())} orange)")
    print(f"  lDDT {row.lddt_all:.3f}   lDDT-CA {row.lddt_ca:.3f}"
          f"   TM {row.tm_score:.3f}   CA-RMSD {row.rmsd_ca:.2f} A")
    return view.show()


show(RECORD_ID)''',
    ),
    (
        "markdown",
        """## Compare plans on the same protein

The interesting contrast is Plan A (one free-running document) against Plan F
(neighbour-conditioned iterative refinement) — same model, same protein, and
the difference is entirely in how much inference was spent.""",
    ),
    (
        "code",
        '''for run in [r for r in ["a-cc1mix5-step50000", "f-cc1mix5-step50000",
                        "e2-cc1mix5-step50000", "oracle-doc"]
            if r in VIEWABLE and r in set(scores.run)]:
    try:
        show(RECORD_ID, run=run)
    except (ValueError, StopIteration) as exc:
        print(f"{run}: {exc}")''',
    ),
    (
        "markdown",
        """## Accuracy vs length

Coverage falls with chain length by construction — the 8192-token budget is
fixed while the number of atoms is not — so every comparison has to hold length
fixed. This is the plot that shows why.""",
    ),
    (
        "code",
        '''import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for metric, axis in zip(["atom_coverage", "lddt_all", "tm_score"], axes):
    subset = scores[(scores.status == "ok") & scores.run.isin(VIEWABLE)]
    for run, group in subset.groupby("run"):
        binned = group.groupby(pd.cut(group.L, [0, 100, 200, 400, 10_000]),
                               observed=True)[metric].mean()
        axis.plot([str(i) for i in binned.index], binned.values, marker="o", label=run)
    axis.set_title(metric); axis.set_xlabel("sequence length"); axis.grid(alpha=0.3)
    axis.tick_params(axis="x", rotation=30)
axes[0].legend(fontsize=7)
fig.tight_layout()''',
    ),
]


def notebook(cells) -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, source in cells
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("explore_predictions.ipynb"))
    args = ap.parse_args(argv)
    args.out.write_text(json.dumps(notebook(CELLS), indent=1) + "\n")
    print(f"[notebook] wrote {args.out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
