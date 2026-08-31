# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the issue #262 Phase 0 figures from the probe CSVs.

Reads ``data/phase0_attention_offsets.csv``, ``data/phase0_attention_lift.csv``
and ``data/phase0_position_interventions.csv``; writes the three plots the
README and ``summary.pdf`` refer to.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402


def plot_offsets(data_dir: Path, plots_dir: Path) -> None:
    """Where the previous-token heads are, and how extreme they get."""
    offsets = pd.read_csv(data_dir / "phase0_attention_offsets.csv")
    structure = offsets[offsets.group == "struct_pos"]
    grid = structure.pivot(index="layer", columns="head", values="mass_prev1")

    figure, (heat, bars) = plt.subplots(1, 2, figsize=(13, 4.6), width_ratios=[1.6, 1])
    image = heat.imshow(grid.values, aspect="auto", cmap="magma", vmin=0, vmax=1, origin="lower")
    heat.set_xlabel("head")
    heat.set_ylabel("layer")
    heat.set_title("attention mass on the previous token\n(structure-section position queries)")
    figure.colorbar(image, ax=heat, label="mass at offset 1")

    top = structure.nlargest(12, "mass_prev1").iloc[::-1]
    labels = [f"L{int(l)}H{int(h)}" for l, h in zip(top["layer"], top["head"])]
    y = np.arange(len(top))
    bars.barh(y, top.mass_prev1, color="#2b6cb0", label="offset 1")
    bars.barh(y, top.mass_prev2, left=top.mass_prev1, color="#63b3ed", label="offset 2")
    bars.set_yticks(y, labels, fontsize=8)
    bars.set_xlabel("attention mass")
    bars.set_xlim(0, 1)
    bars.set_title("the 12 strongest previous-token heads")
    bars.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    save_plot_with_meta(
        figure,
        plots_dir / "phase0_previous_token_heads.png",
        caption=(
            "Attention mass at offset 1 per (layer, head), over structure-section position queries "
            "of 24 ground-truth documents. Layer 1 holds two near-pure previous-token heads (0.999, "
            "0.996) and a third splitting 0.75/0.16 over offsets 1 and 2 — a width-3 smear built out "
            "of attention. Direct motivation for the smear half of #262."
        ),
        dpi=150,
    )
    plt.close(figure)


def plot_lift(data_dir: Path, plots_dir: Path) -> None:
    """Co-referent retrieval quality as a function of document distance."""
    lift = pd.read_csv(data_dir / "phase0_attention_lift.csv")
    structure = lift[(lift.group == "struct_pos") & (lift.bucket_index > 0)]
    per_head = structure.groupby(["layer", "head"], as_index=False)[
        ["mass_total", "mass_coref", "mass_expected"]
    ].sum()
    per_head["coref_share"] = per_head.mass_coref / per_head.mass_total
    top = per_head.nlargest(8, "coref_share")

    figure, (profile, pooled) = plt.subplots(1, 2, figsize=(13, 4.6))
    for _, row in top.iterrows():
        layer, head = int(row["layer"]), int(row["head"])
        series = structure[(structure["layer"] == layer) & (structure["head"] == head)].sort_values("bucket_index")
        profile.plot(
            series.bucket_index,
            series.mass_coref / series.mass_total,
            marker="o",
            markersize=3,
            label=f"L{layer}H{head}",
        )
    labels = structure.sort_values("bucket_index").bucket.unique()
    profile.set_xticks(sorted(structure.bucket_index.unique()), labels, rotation=45, fontsize=7)
    profile.set_xlabel("distance from query to key (tokens)")
    profile.set_ylabel("share of the head's mass on co-referents")
    profile.set_title("the 8 strongest co-referent-retrieval heads")
    profile.set_ylim(0, 1)
    profile.legend(fontsize=7, ncol=2)
    profile.grid(alpha=0.3)

    total = structure.groupby(["bucket_index", "bucket"], as_index=False)[
        ["mass_total", "mass_coref", "mass_expected"]
    ].sum()
    pooled.plot(total.bucket_index, total.mass_total / total.mass_total.sum(), marker="o", color="#805ad5")
    pooled.set_xticks(total.bucket_index, total.bucket, rotation=45, fontsize=7)
    pooled.set_xlabel("distance from query to key (tokens)")
    pooled.set_ylabel("share of all structure-section attention")
    pooled.set_title("where structure-section attention goes")
    pooled.grid(alpha=0.3)
    figure.tight_layout()
    save_plot_with_meta(
        figure,
        plots_dir / "phase0_coreferent_retrieval.png",
        caption=(
            "Left: for heads that retrieve earlier mentions of the query's own residue index, the "
            "share of their attention on those co-referents, by distance. Flat — L1H17 holds 0.90-0.93 "
            "out to 2048 tokens. Retrieval is already distance-uniform, so RoPE costs us no reach and "
            "the mechanistic case for dropping it fails. Right: where structure-section attention goes."
        ),
        dpi=150,
    )
    plt.close(figure)


def plot_interventions(data_dir: Path, plots_dir: Path) -> None:
    """What the position interventions cost, with the matched pairs adjacent."""
    frame = pd.read_csv(data_dir / "phase0_position_interventions.csv")
    baseline = frame[frame["mode"] == "baseline"].set_index("stem")
    frame["delta"] = frame.structure_nll - frame.stem.map(baseline.structure_nll)
    summary = frame.groupby("mode").agg(
        delta=("delta", "mean"),
        error=("delta", lambda values: values.std() / np.sqrt(len(values))),
    )

    order = [
        ("shift1024", "controls"), ("gap1", "controls"),
        ("fixedgap2", "fixed"), ("randgap2", "random"),
        ("fixedgap3", "fixed"), ("randgap3", "random"),
        ("fixedgap5", "fixed"), ("randgap5", "random"),
        ("jitter1", "degenerate"), ("jitter4", "degenerate"),
        ("flat", "out of distribution"), ("rope_off", "out of distribution"),
    ]
    colors = {
        "controls": "#a0aec0", "fixed": "#dd6b20", "random": "#2b6cb0",
        "degenerate": "#805ad5", "out of distribution": "#e53e3e",
    }
    figure, axes = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(order))
    values = [summary.loc[mode, "delta"] for mode, _ in order]
    errors = [summary.loc[mode, "error"] for mode, _ in order]
    axes.bar(x, values, yerr=errors, color=[colors[kind] for _, kind in order], capsize=3)
    axes.set_xticks(x, [mode for mode, _ in order], rotation=40, ha="right", fontsize=9)
    axes.set_ylabel("Δ structure-section NLL vs baseline (nats/token)")
    axes.set_yscale("symlog", linthresh=0.05)
    axes.axhline(0, color="black", linewidth=0.8)
    axes.set_title("cost of rewriting position_ids at inference (n=24 documents, paired)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
    axes.legend(handles, colors.keys(), fontsize=8, loc="upper left")
    for index, (mode, _) in enumerate(order):
        axes.annotate(
            f"{values[index]:+.3f}", (index, values[index]),
            textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7,
        )
    axes.set_ylim(-0.02, None)
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    save_plot_with_meta(
        figure,
        plots_dir / "phase0_position_interventions.png",
        caption=(
            "Paired change in structure-section NLL when position_ids are rewritten (n=24). fixedgapN "
            "and randgapN match on expected stretch and never repeat a position; only randgapN destroys "
            "exact cross-statement distances, and is never the worse of the pair — the model does not "
            "read them. jitterN instead repeats ids at fixed range and costs 10x more: position is an index."
        ),
        dpi=150,
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    arguments = parser.parse_args()
    arguments.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_offsets(arguments.data_dir, arguments.plots_dir)
    plot_lift(arguments.data_dir, arguments.plots_dir)
    plot_interventions(arguments.data_dir, arguments.plots_dir)
    print(f"wrote plots to {arguments.plots_dir}")


if __name__ == "__main__":
    main()
