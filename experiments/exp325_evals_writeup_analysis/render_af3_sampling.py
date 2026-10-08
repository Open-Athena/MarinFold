"""Fast AF3 sampling figures from precomputed CSVs; no model or metric calls."""

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from build_summary import save_plot_with_meta
from render_af3_context import load_tables
from theme import GRID, INK, PALETTE, PAPER

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent


def style(ax: plt.Axes) -> None:
    """Apply the writeup's palette and frame."""
    ax.set_facecolor(PAPER)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylim(0, 1)


def save(fig: plt.Figure, name: str, caption: str, *, include: bool = True) -> None:
    """Write independently usable raster and vector plots with provenance."""
    save_plot_with_meta(fig, HERE / "plots" / f"{name}.png", caption=caption,
                        script="render_af3_sampling.py", args=[], include_in_summary=include, dpi=180)
    fig.savefig(HERE / "plots" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(HERE / "plots" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def interactive(samples: pd.DataFrame, curves: pd.DataFrame, protocol: dict, context: pd.DataFrame) -> None:
    """Expose exact seeds and scores in an interactive per-protein view."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14)
    stems = sorted(samples.stem.unique())
    groups = []
    for index, stem in enumerate(stems):
        start = len(fig.data)
        frame = samples[samples.stem == stem]
        curve = curves[curves.stem == stem]
        final = curve.iloc[-1]
        pilot = frame.seed < protocol["seed_start"] + 100
        for mask, color, label in [(pilot, PALETTE[1], "First 100"), (~pilot, PALETTE[0], "Additional runs")]:
            subset = frame[mask]
            fig.add_trace(go.Scatter(x=subset.ptm.tolist(), y=subset.tm_score.tolist(), mode="markers", name=label,
                          marker=dict(color=color, size=6, opacity=0.45), customdata=subset[["seed"]].values.tolist(),
                          hovertemplate="Seed %{customdata[0]}<br>pTM %{x:.4f}<br>TM %{y:.4f}<extra></extra>"), row=1, col=1)
        for seed, color, symbol, label in [(final.best_seed, INK, "diamond-open", "Oracle best TM"),
                                          (final.ptm_selected_seed, PALETTE[3], "star", "pTM selected")]:
            row = frame[frame.seed == seed].iloc[0]
            fig.add_trace(go.Scatter(x=[row.ptm], y=[row.tm_score], mode="markers", name=label,
                          marker=dict(color=color, size=15, symbol=symbol), customdata=[[int(seed)]],
                          hovertemplate="Seed %{customdata[0]}<br>pTM %{x:.4f}<br>TM %{y:.4f}<extra>%{fullData.name}</extra>"), row=1, col=1)
        for metric, seed_column, color, label in [("best_tm", "best_seed", INK, "Oracle best TM"),
                                                ("ptm_selected_tm", "ptm_selected_seed", PALETTE[3], "pTM selected"),
                                                ("ranking_selected_tm", "ranking_selected_seed", PALETTE[1], "Official rank selected")]:
            fig.add_trace(go.Scatter(x=curve.budget.tolist(), y=curve[metric].tolist(), mode="lines", name=label,
                          showlegend=metric == "ranking_selected_tm", line=dict(color=color, shape="hv"),
                          customdata=curve[[seed_column]].values.tolist(),
                          hovertemplate="Budget %{x}<br>Seed %{customdata[0]}<br>TM %{y:.4f}<extra>%{fullData.name}</extra>"), row=1, col=2)
        for method, label, color in [("af3", "AF3 baseline (25)", "#817970"),
                                     ("esmfold2", "ESMFold2 baseline", PALETTE[2])]:
            baseline = context[(context.stem == stem) & (context.method == method)].iloc[0]
            fig.add_trace(go.Scatter(x=[1, int(curve.budget.max())], y=[baseline.tm_score] * 2,
                          name=label, mode="lines", line=dict(color=color, dash="dot"),
                          hovertemplate="TM %{y:.4f}<extra>%{fullData.name}</extra>"), row=1, col=2)
        groups.append((start, len(fig.data)))
        for trace in fig.data[start:]:
            trace.visible = index == 0
    buttons = [dict(label=stem, method="update", args=[{"visible": [start <= i < end for i in range(len(fig.data))]}])
               for stem, (start, end) in zip(stems, groups, strict=True)]
    fig.update_layout(paper_bgcolor=PAPER, plot_bgcolor=PAPER, font=dict(family="Lato, DejaVu Sans, Arial", size=12, color=INK),
                      height=490, margin=dict(l=60, r=20, t=80, b=100),
                      legend=dict(orientation="h", x=0, y=-0.25),
                      updatemenus=[dict(buttons=buttons, x=0, y=1.22, xanchor="left", yanchor="top")])
    fig.update_xaxes(title_text="AF3 pTM", range=[0, 1], row=1, col=1)
    ticks = [n for n in [1, 10, 100, 1000] if n <= curves.budget.max()]
    fig.update_xaxes(title_text="Full AF3 runs", type="log", tickvals=ticks,
                     ticktext=[f"{n:,}" for n in ticks], row=1, col=2)
    fig.update_yaxes(title_text="TM-score", range=[0, 1], gridcolor=GRID, zeroline=False)
    fig.add_hline(y=protocol["accuracy_threshold_tm"], line_dash="dash", line_color=INK, opacity=0.4)
    spec = json.loads(fig.to_json())
    spec["config"] = dict(responsive=True, displayModeBar=False, scrollZoom=False)
    mobile = json.loads(json.dumps(spec))
    mobile["layout"].update(height=780, margin=dict(l=48, r=15, t=80, b=110))
    mobile["layout"]["xaxis"]["domain"] = [0, 1]
    mobile["layout"]["xaxis2"]["domain"] = [0, 1]
    mobile["layout"]["yaxis"]["domain"] = [0.60, 1]
    mobile["layout"]["yaxis2"]["domain"] = [0, 0.40]
    mobile["layout"]["legend"].update(y=-0.15, font=dict(size=10))
    for suffix, payload in [("", spec), ("-mobile", mobile)]:
        (HERE / "site" / f"01b_af3_sampling{suffix}.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def main() -> None:
    """Draw one page per protein plus a compact blog comparison."""
    manifest = json.loads((HERE / "data/af3_sampling_analysis.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((HERE / "data" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared AF3 input changed: {name}; rerun prepare_af3_sampling.py")
    for font in (HERE / "data/inputs").glob("Lato-*.ttf"):
        font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": ["Lato", "DejaVu Sans"], "font.size": 11,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})
    protocol = json.loads((HERE / "data/af3_sampling_protocol.json").read_text())
    samples = pd.read_csv(HERE / "data/af3_sampling_samples.csv")
    curves = pd.read_csv(HERE / "data/af3_sampling_curves.csv")
    summary = pd.read_csv(HERE / "data/af3_sampling_summary.csv")
    original = pd.read_csv(HERE / "data/af3_sampling_original.csv")
    context, _ = load_tables()
    threshold = protocol["accuracy_threshold_tm"]
    interactive(samples, curves, protocol, context)
    for stem, frame in samples.groupby("stem", sort=True):
        curve = curves[curves.stem == stem]
        final = curve.iloc[-1]
        old = original[original.stem == stem]
        old_selected = old.loc[old.ranking_score.idxmax()]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=PAPER)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.26, top=0.77, wspace=0.24)
        fig.text(0.07, 0.94, f"{stem} · Can more AF3 runs find an accurate fold?", fontsize=19, weight="bold")
        fig.text(0.07, 0.87, f"MSA depth {int(final.msa_depth)} · {len(frame):,} independent full runs · one diffusion sample per seed", fontsize=12)
        for ax in axes:
            style(ax)
            ax.axhline(threshold, color=INK, lw=1, ls="--", alpha=0.45, zorder=1)
        for pilot, color, label in [(True, PALETTE[1], "First 100 runs"), (False, PALETTE[0], "Runs 101–1,000")]:
            mask = frame.seed < protocol["seed_start"] + 100
            subset = frame[mask if pilot else ~mask]
            if not subset.empty:
                axes[0].scatter(subset.ptm, subset.tm_score, s=15, color=color, alpha=0.45, linewidths=0,
                                zorder=2, label=label)
        selected = frame[frame.seed == final.ptm_selected_seed].iloc[0]
        best = frame[frame.seed == final.best_seed].iloc[0]
        axes[0].scatter([best.ptm], [best.tm_score], marker="D", s=100, facecolors="none", edgecolors=INK,
                        linewidths=1.5, zorder=5, label="Oracle best TM")
        axes[0].scatter([selected.ptm], [selected.tm_score], marker="*", s=180, color=PALETTE[3],
                        edgecolors=PAPER, zorder=6, label="Selected by pTM")
        axes[0].set(xlim=(0, 1), xlabel="AF3 pTM", ylabel="TM-score against reference")
        axes[0].legend(loc="upper left", bbox_to_anchor=(-0.02, -0.18), frameon=False, ncol=2, fontsize=9)
        axes[1].step(curve.budget, curve.best_tm, where="post", color=INK, lw=2, label="Oracle best TM")
        axes[1].step(curve.budget, curve.ptm_selected_tm, where="post", color=PALETTE[3], lw=1.8, label="Selected by pTM")
        axes[1].step(curve.budget, curve.ranking_selected_tm, where="post", color=PALETTE[1], lw=1.2, label="Official rank selected")
        axes[1].axhline(old_selected.tm_score, color="#817970", ls=":", lw=1.5, label="Original AF3 selection (25)")
        esmfold2 = context[(context.stem == stem) & (context.method == "esmfold2")].iloc[0]
        axes[1].axhline(esmfold2.tm_score, color=PALETTE[2], ls="--", lw=1.5, label=f"ESMFold2 baseline ({esmfold2.tm_score:.3f})")
        axes[1].set(xscale="log", xlim=(1, len(frame)), xlabel="Number of full AF3 runs", ylabel="TM-score")
        ticks = [n for n in [1, 10, 100, 1000] if n <= len(frame)]
        axes[1].set_xticks(ticks, labels=[f"{n:,}" for n in ticks])
        axes[1].legend(loc="upper left", bbox_to_anchor=(-0.02, -0.18), frameon=False, fontsize=8, ncol=2)
        fig.text(0.07, 0.045, f"TM ≥ {threshold:.1f}: {int(final.hits_tm_80)}/{len(frame):,} runs   |   "
                 f"Best TM {final.best_tm:.3f}   |   pTM-selected TM {final.ptm_selected_tm:.3f}", fontsize=11)
        caption = (f"{stem}: each point is one independent AF3 trunk + diffusion run, seeds starting at 10000, ten recycles, "
                   "fixed archived MSA and no templates. Full-precision pTM selects without using truth. "
                   "Oracle best TM uses truth only for this diagnostic. Dashed line: prespecified TM >=0.8. "
                   "Curves use the ascending-seed prefix; original dotted baseline selected among five seeds x five samples "
                   "by official ranking_score. Data: af3_sampling_samples.csv and af3_sampling_curves.csv; "
                   "original baseline: af3_sampling_original.csv. ESMFold2 baseline: af3_context_rows.csv. "
                   "Official ranking_score selection is shown separately from pTM. All are in data/; "
                   "preprocessing: prepare_af3_sampling.py and prepare_af3_context.py.")
        save(fig, f"01b_af3_sampling_{stem}", caption)
    fig, ax = plt.subplots(figsize=(10.4, 5.4), facecolor=PAPER)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.77, bottom=0.27)
    ax.set_facecolor(PAPER)
    stems = sorted(samples.stem.unique())
    colors = [PALETTE[1], INK, PALETTE[3]]
    max_budget = int(curves.budget.max())
    descriptions = ["Best of 100", f"Best of {max_budget:,}", f"pTM-selected at {max_budget:,}"]
    for i, stem in enumerate(stems):
        group = summary[summary.stem == stem]
        final = group.iloc[-1]
        pilot = group[group.budget == 100].iloc[0]
        values = [pilot.best_tm, final.best_tm, final.ptm_selected_tm]
        ax.plot([min(values), max(values)], [i, i], color=GRID, lw=2, zorder=1)
        for j, (value, color, marker) in enumerate(zip(values, colors, ["o", "D", "*"])):
            ax.scatter(value, i + (j - 1) * 0.12, s=95 if marker != "*" else 180, color=color,
                       marker=marker, label=descriptions[j] if i == 0 else None, zorder=3)
    ax.axvline(threshold, color=INK, ls="--", lw=1, alpha=0.45)
    ax.set(xlim=(0.25, 1), xlabel="TM-score against reference", yticks=range(len(stems)), yticklabels=stems)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=10)
    fig.text(0.08, 0.94, "Does more AF3 sampling recover accurate folds?", fontsize=20, weight="bold")
    fig.text(0.08, 0.87, "Five proteins with MSA depth <10 · fixed MSAs · independent full runs", fontsize=12)
    save(fig, "01b_af3_sampling", f"Best of 100 and {max_budget:,} independent AF3 runs versus highest-pTM selection at {max_budget:,}. "
         "Only five biological examples. Oracle best uses the reference; pTM selection does not. "
         "Dashed line: prespecified TM >=0.8. Source: data/af3_sampling_summary.csv.", include=False)


if __name__ == "__main__":
    main()
