"""Plot per-protein wall-time vs sequence length.

One PNG in ``plots/``:
- ``timing_vs_sequence_length.png`` — Protenix-only (single_seq vs msa).

Plot conventions:
- log-log axes (timing roughly polynomial in N for AF3-class models)
- one point per (mode, protein)
- legend captures model + mode
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("n_residues", "elapsed_seconds", "model_load_seconds", "total_seconds"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _scatter(ax, df: pd.DataFrame, *, label: str, color: str, marker: str = "o") -> None:
    if df.empty:
        return
    df = df.dropna(subset=["n_residues", "elapsed_seconds"]).sort_values("n_residues")
    ax.scatter(df["n_residues"], df["elapsed_seconds"],
               label=label, color=color, marker=marker, s=30, alpha=0.85)


def _styled(ax, *, title: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Wall time per protein (s, log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize=9)


def plot_protenix_only(scores_csv: Path, out_png: Path) -> None:
    df = _load(scores_csv)
    fig, ax = plt.subplots(figsize=(7, 5))
    ss = df[df["mode"] == "single_seq"]
    msa = df[df["mode"] == "msa"]
    _scatter(ax, ss, label="Protenix v2 — single_seq", color="C1", marker="s")
    _scatter(ax, msa, label="Protenix v2 — MSA", color="C0", marker="o")
    _styled(
        ax,
        title="Protenix v2 inference wall-time on FoldBench monomers\n"
              "(5 seeds × 8 diffusion samples, H100, Modal)",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, default=Path("data/timings.csv"))
    parser.add_argument("--out", type=Path, default=Path("plots"))
    args = parser.parse_args()

    plot_protenix_only(args.scores, args.out / "timing_vs_sequence_length.png")


if __name__ == "__main__":
    main()
