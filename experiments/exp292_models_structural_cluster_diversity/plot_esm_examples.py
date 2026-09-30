"""Inspect ESM candidates with overlays and stored, unscaled PAE codes.

The Atlas matrices are uint8 codes. Until their physical-unit decoding is
verified, plots label them as stored codes and apply no angstrom threshold.
CSV inputs preserve the exact traces and block-averaged matrices used here.
"""

import argparse
import io
import zipfile
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tmtools import tm_align

from build_summary import save_plot_with_meta
from structure_audit import load_protein, read_csv, write_csv


def prepare(rows: list[dict], cache: Path, data_dir: Path) -> None:
    """Save compact plot inputs without changing or inferring PAE units."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        a = load_protein(cache, row["entry_id"])
        b = load_protein(cache, row["nearest_anchor"])
        aligned = tm_align(a.coords, b.coords, a.sequence, b.sequence)
        coords = a.coords @ aligned.u.T + aligned.t
        traces = []
        for role, h, xyz, confidence in [
            ("candidate", row["entry_id"], coords, a.plddt),
            ("anchor", row["nearest_anchor"], b.coords, b.plddt),
        ]:
            traces.extend(
                {
                    "role": role,
                    "residue": i + 1,
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]),
                    "plddt": float(confidence[i]),
                }
                for i, p in enumerate(xyz)
            )
            with zipfile.ZipFile(cache / f"{h}.pae.zip") as archive:
                pae = np.load(io.BytesIO(archive.read("arr.npy")), allow_pickle=False)
            if pae.shape != (len(xyz), len(xyz)) or pae.dtype != np.uint8:
                raise ValueError(
                    "Expected an Atlas uint8 PAE code matrix matching the chain"
                )
            edges = np.linspace(0, len(xyz), min(128, len(xyz)) + 1, dtype=int)
            reduced = np.array(
                [
                    [pae[x0:x1, y0:y1].mean() for y0, y1 in pairwise(edges)]
                    for x0, x1 in pairwise(edges)
                ]
            )
            np.savetxt(
                data_dir / f"{h}-pae-codes.csv", reduced, delimiter=",", fmt="%.4f"
            )
        write_csv(data_dir / f"{row['entry_id']}-traces.csv", traces)


def main() -> None:
    """Render saved overlays and PAE code patterns for provisional additions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        r
        for r in read_csv(args.sample_dir / "candidates.csv")
        if int(r["selected_order"]) > 0
    ]
    if args.cache:
        prepare(rows, args.cache, args.data_dir)
    fig = plt.figure(figsize=(14, 4.6 * len(rows)), layout="constrained")
    grid = fig.add_gridspec(len(rows), 3)
    for index, row in enumerate(rows):
        ax = fig.add_subplot(grid[index, 0], projection="3d")
        traces = read_csv(args.data_dir / f"{row['entry_id']}-traces.csv")
        for role, color in [("candidate", "#dc852c"), ("anchor", "#178c96")]:
            xyz = np.array(
                [
                    [float(r[k]) for k in ["x", "y", "z"]]
                    for r in traces
                    if r["role"] == role
                ]
            )
            ax.plot(*xyz.T, color=color, linewidth=0.8, label=role)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.legend(loc="lower center", fontsize=8)
        ax.set_title(
            f"{row['entry_id'][:12]}… ({row['seq_len']} aa)\nTM {float(row['nearest_anchor_tm']):.3f}; core {float(row['max_anchor_core_tm']):.3f}",
            fontsize=11,
        )
        for column, key, title in [
            (1, "entry_id", "Candidate"),
            (2, "nearest_anchor", "Training anchor"),
        ]:
            axis = fig.add_subplot(grid[index, column])
            values = np.loadtxt(
                args.data_dir / f"{row[key]}-pae-codes.csv", delimiter=","
            )
            im = axis.imshow(values, origin="lower", vmin=0, vmax=255, cmap="magma")
            axis.set_title(title + " · stored PAE codes", fontsize=11)
            axis.set_xlabel("Residue bins (up to 128)")
            axis.set_ylabel("Residue bins")
            fig.colorbar(im, ax=axis, shrink=0.7, label="Unscaled uint8 code")
    fig.suptitle(
        "ESMFold2 provisional additions: inspect domain arrangement and uncertainty",
        fontsize=15,
    )
    save_plot_with_meta(
        fig,
        args.output,
        caption="Orange candidate / teal retained training anchor, aligned by TM-align. PAE matrices are block means of stored uint8 codes, not values in angstroms; their physical-unit decoding remains unverified and no PAE threshold is applied. Inspect domain-block patterns alongside the original confidence-colored gallery. Neither prediction disagreement nor low TM establishes a true alternative state.",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
