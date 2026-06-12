# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the fetched candidate set: counts per source + length distributions.

Reads the three committed manifests (``data/*_manifest.csv``) and renders a
two-panel overview into ``plots/candidates_overview.png`` (+ a ``.meta.json``
sidecar for ``build_summary.py``). Rows without a downloaded structure
(``length`` < 0, e.g. the unresolved CASP FM domains) are counted but excluded
from the length panel.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
MANIFESTS = {
    "de novo (PDB)": HERE / "data" / "denovo_pdb_manifest.csv",
    "CASP FM": HERE / "data" / "casp_fm_manifest.csv",
    "CAMEO hard": HERE / "data" / "cameo_hard_manifest.csv",
}


def load() -> dict[str, pd.DataFrame]:
    out = {}
    for label, path in MANIFESTS.items():
        if path.exists():
            out[label] = pd.read_csv(path)
    if not out:
        raise SystemExit("no manifests found; run the fetch_*.py scripts first")
    return out


def plot(frames: dict[str, pd.DataFrame], out_path: Path) -> Path:
    labels = list(frames)
    fig, (ax_count, ax_len) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel 1: total vs with-structure counts per source.
    totals = [len(frames[l]) for l in labels]
    with_struct = [int((frames[l]["length"] >= 0).sum()) for l in labels]
    x = range(len(labels))
    ax_count.bar(x, totals, color="#bbbbbb", label="total")
    ax_count.bar(x, with_struct, color="#3366cc", label="with structure")
    ax_count.set_xticks(list(x))
    ax_count.set_xticklabels(labels, rotation=15, ha="right")
    ax_count.set_ylabel("candidates")
    ax_count.set_title("Candidates per source")
    for xi, t, w in zip(x, totals, with_struct):
        ax_count.text(xi, t, str(t), ha="center", va="bottom", fontsize=9)
    ax_count.legend(fontsize=8)

    # Panel 2: length distributions (structures only).
    colors = ["#3366cc", "#dc3912", "#109618"]
    for label, color in zip(labels, colors):
        lengths = frames[label].loc[frames[label]["length"] >= 0, "length"]
        if len(lengths):
            ax_len.hist(lengths, bins=25, alpha=0.55, label=f"{label} (n={len(lengths)})", color=color)
    ax_len.set_xlabel("chain length (residues)")
    ax_len.set_ylabel("count")
    ax_len.set_title("Length distribution")
    ax_len.legend(fontsize=8)

    fig.tight_layout()
    return save_plot_with_meta(
        fig,
        out_path,
        caption=(
            "Fetched exp65 candidate set. Left: total vs with-downloaded-"
            "structure counts per source (CASP FM's gap = oligomeric/late-release "
            "domains recorded without coords). Right: chain-length distributions."
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", type=Path, default=HERE / "plots" / "candidates_overview.png",
        help="Output PNG path (default: ./plots/candidates_overview.png).",
    )
    args = p.parse_args()
    path = plot(load(), args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
