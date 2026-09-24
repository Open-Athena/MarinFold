"""Render all blog figures from prepared tables, without network or analysis.

Static PNG/SVG, site-compatible Plotly JSON, and an offline interactive review
page share one palette and the same precomputed means and intervals. Plotly's
local JS bundle is generated from the installed package and is not committed.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches, font_manager, ticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from build_summary import save_plot_with_meta
from theme import CAPTIONS, COHORTS, FONT, GRID, INK, METHODS, METRICS, ORDER, PALETTE, PAPER, TIERS, TITLES

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
SITE = HERE / "site"
matplotlib.use("Agg")


def verify_data() -> dict:
    """Fail before drawing if prepared figure data have been modified."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, record in manifest["outputs"].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Prepared data changed: {name}; run prepare.py")
    return manifest


def style_axes(ax: plt.Axes) -> None:
    """Apply a low-ink frame to a static panel."""
    ax.set_facecolor(PAPER)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK, length=0, pad=9)


def save(fig: plt.Figure, figure: str, caption: str, suffix: str = "", *, include_in_summary: bool = True) -> None:
    """Save vector/raster artwork and its source-table metadata."""
    name = figure + suffix
    save_plot_with_meta(fig, PLOTS / f"{name}.png", caption=caption,
                        script="render.py", args=[], include_in_summary=include_in_summary, dpi=180)
    fig.savefig(PLOTS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def static_depth(summary: pd.DataFrame, figure: str, metric: str, cohort: str = "natural") -> None:
    """Draw the default matched-population view from summary rows."""
    frame = summary[(summary.figure == figure) & (summary.metric == metric) &
                    (summary.cohort == cohort) & summary.tier.isin(TIERS)]
    crowded = len(ORDER[figure]) > 6
    fig, ax = plt.subplots(figsize=(10.4, 6.8 if crowded else 6.0), facecolor=PAPER)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.36 if crowded else 0.33, top=0.79)
    fig.text(0.09, 0.945, TITLES[figure], fontsize=20, color=INK, weight="bold")
    population = "Natural FoldBench monomers" if figure in {"01_predictors", "02_oracle"} else "Natural FoldBench · 248B-token model"
    fig.text(0.09, 0.887, f"{population}  ·  mean and 95% interval", fontsize=11, color=INK)
    style_axes(ax)
    methods = ORDER[figure]
    counts = frame.groupby("tier").n.first().to_dict()
    for i, method in enumerate(methods):
        group = frame[frame.method == method].set_index("tier").reindex(TIERS)
        x = np.arange(4) + (i - (len(methods) - 1) / 2) * 0.065
        label, color, marker = METHODS[method]
        # Slight horizontal offsets keep CIs visible; no line suggests a fitted trend.
        ax.errorbar(x, group["mean"], yerr=[group["mean"] - group.ci_low, group.ci_high - group["mean"]],
                    fmt=marker, color=color, markersize=6.5, capsize=3, elinewidth=1.3,
                    label=label, zorder=3)
    ax.set_xticks(np.arange(4), [f"{t}\nn = {counts.get(t, 0)}" for t in TIERS])
    ax.set_xlim(-0.42, 3.42)
    for i, tier in enumerate(TIERS):
        if not counts.get(tier, 0):
            ax.text(i, 0.48, "Not yet\nevaluated", ha="center", va="center", fontsize=10, color="#817970")
    ax.set_ylabel(METRICS[metric], labelpad=12)
    ax.set_xlabel("MSA depth (sequences)", labelpad=13)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.075, 0.016),
               ncol=2, frameon=False, fontsize=10, columnspacing=2.5)
    suffix = {"lddt": "_lddt", "r_precision_long": "_long"}.get(metric, "")
    save(fig, figure, CAPTIONS[figure], suffix)


def plotly_layout(metric: str) -> dict:
    """Site-compatible transparent plot layout; the host supplies the frame."""
    return dict(template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family=FONT, size=13, color=INK), height=510,
                margin=dict(l=65, r=25, t=65, b=165),
                hoverlabel=dict(bgcolor=INK, font=dict(color=PAPER)),
                xaxis=dict(title="MSA depth (sequences)", range=[-0.45, 3.45], tickmode="array",
                           tickvals=list(range(4)), ticktext=TIERS, showgrid=False, zeroline=False),
                yaxis=dict(title=METRICS[metric], range=[0, 1.02], dtick=0.2,
                           gridcolor=GRID, zeroline=False),
                legend=dict(orientation="h", x=0, y=-0.25, font=dict(size=11)),
                hovermode="closest")


def export_depth(summary: pd.DataFrame, figure: str) -> go.Figure:
    """Export selectable precomputed populations and metrics, including mobile layout."""
    frame = summary[(summary.figure == figure) & summary.tier.isin(TIERS)]
    metrics = ["r_precision", "r_precision_long"] if figure == "04_contacts" else ["gdt_ts", "lddt"]
    fig = go.Figure(layout=plotly_layout(metrics[0]))
    states = []
    for cohort in COHORTS:
        for metric in metrics:
            subset = frame[(frame.cohort == cohort) & (frame.metric == metric)]
            if subset.empty:
                continue
            start = len(fig.data)
            counts = subset.groupby("tier").n.first().to_dict()
            for i, method in enumerate(ORDER[figure]):
                part = subset[subset.method == method].set_index("tier").reindex(TIERS)
                label, color, marker = METHODS[method]
                visible = cohort == "natural" and metric == metrics[0]
                custom = [[tier, None if pd.isna(n) else int(n), lo, hi, figure, cohort, metric, method]
                          for tier, n, lo, hi in zip(TIERS, part.n, part.ci_low, part.ci_high)]
                fig.add_trace(go.Scatter(x=(np.arange(4) + (i - (len(ORDER[figure]) - 1) / 2) * 0.065).tolist(),
                                        y=part["mean"].tolist(), mode="markers", name=label,
                                        marker=dict(size=9, color=color, symbol={"o": "circle", "s": "square", "D": "diamond", "v": "triangle-down", "^": "triangle-up"}[marker]),
                                        error_y=dict(type="data", array=(part.ci_high - part["mean"]).tolist(),
                                                     arrayminus=(part["mean"] - part.ci_low).tolist(), width=3),
                                        visible=visible, legendgroup=method, customdata=custom,
                                        hovertemplate="%{fullData.name}<br>Depth %{customdata[0]} · n=%{customdata[1]}<br>Mean %{y:.3f}<br>95% CI [%{customdata[2]:.3f}, %{customdata[3]:.3f}]<br><br>summary.csv key:<br>%{customdata[4]} / %{customdata[5]}<br>%{customdata[6]} / %{customdata[7]}<extra></extra>"))
            states.append((cohort, metric, start, len(fig.data), counts))
    buttons = []
    for cohort, metric, start, end, counts in states:
        buttons.append(dict(label=f"{COHORTS[cohort]} · {METRICS[metric]}", method="update",
                            args=[{"visible": [start <= i < end for i in range(len(fig.data))]},
                                  {"yaxis.title.text": METRICS[metric],
                                   "xaxis.ticktext": [f"{t}<br>n={counts.get(t, 0)}" for t in TIERS]}]))
    initial_counts = states[0][4]
    fig.update_xaxes(ticktext=[f"{t}<br>n={initial_counts.get(t, 0)}" for t in TIERS])
    fig.update_layout(updatemenus=[dict(buttons=buttons, direction="down", x=0, xanchor="left", y=1.16,
                                       bgcolor=PAPER, bordercolor=GRID, font=dict(size=12))])
    return fig


def method_figure(training: pd.DataFrame) -> go.Figure:
    """Draw a source-linked method schematic, with no quantitative flow widths."""
    fig, ax = plt.subplots(figsize=(10.4, 5.6), facecolor=PAPER)
    ax.set(xlim=(0, 10), ylim=(0, 5.2))
    ax.axis("off")
    stages = [
        (0.1, 3.0, 2.8, "Predicted structures", "Decontaminated AFDB + ESM Atlas\nNative + ProteinMPNN sequences", PALETTE[0]),
        (3.5, 3.0, 2.8, "Contact documents", "Sequence + ordered contact tokens\n248.584B raw tokens · one epoch", PALETTE[2]),
        (6.9, 3.0, 2.8, "Train MarinFold", "1.47B autoregressive transformer\nPredict the next token", PALETTE[3]),
        (0.1, 0.4, 2.05, "Single sequence", "No MSA at inference", PALETTE[0]),
        (2.65, 0.4, 2.05, "100 samples", "Resample the prompt\nGenerate contact sets", PALETTE[3]),
        (5.2, 0.4, 2.05, "Vote on contacts", "Rank residue pairs\nPass top-L to Helico", PALETTE[3]),
        (7.75, 0.4, 2.05, "3D structure", "Helico conditions on\nthe predicted contacts", PALETTE[2]),
    ]
    for x, y, width, title, body, color in stages:
        ax.add_patch(patches.FancyBboxPatch((x, y), width, 1.15, boxstyle="round,pad=0.02,rounding_size=0.06",
                                           facecolor=PAPER, edgecolor=color, linewidth=1.5))
        ax.text(x + width / 2, y + 0.83, title, ha="center", fontsize=11, weight="bold", color=color)
        ax.text(x + width / 2, y + 0.35, body, ha="center", va="center", fontsize=9, linespacing=1.5)
    for x1, x2, y in [(2.95, 3.4, 3.58), (6.35, 6.8, 3.58), (2.2, 2.55, 0.98), (4.75, 5.1, 0.98), (7.3, 7.65, 0.98)]:
        ax.annotate("", (x2, y), (x1, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(0.1, 4.65, "TRAIN", fontsize=10, weight="bold", color=INK)
    ax.text(0.1, 1.95, "GENERATE → AGGREGATE → FOLD", fontsize=10, weight="bold", color=INK)
    # The amounts come directly from the cached source inventory.
    inventory = " · ".join(f"{row.corpus}: {row.documents / 1e6:.1f}M docs" for row in training.itertuples())
    ax.text(0.1, 2.58, inventory, fontsize=10, color=INK)
    fig.suptitle(TITLES["03_method"], x=0.135, ha="left", fontsize=20, weight="bold")
    save(fig, "03_method", CAPTIONS["03_method"])
    # Plotly keeps the same diagram portable to the site's native figure shortcode.
    result = go.Figure(layout=dict(template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(family=FONT, color=INK), height=470, margin=dict(l=10, r=10, t=10, b=10),
                                   xaxis=dict(range=[-0.1, 10], visible=False, fixedrange=True),
                                   yaxis=dict(range=[0, 5.1], visible=False, fixedrange=True)))
    for x, y, width, title, body, color in stages:
        result.add_shape(type="rect", x0=x, x1=x + width, y0=y, y1=y + 1.15, line=dict(color=color, width=1.5))
        result.add_annotation(x=x + width / 2, y=y + 0.57,
                              text=f"<b>{title}</b><br>{body.replace(chr(10), '<br>')}",
                              font=dict(size=11, color=color), showarrow=False)
    for x1, x2, y in [(2.95, 3.4, 3.58), (6.35, 6.8, 3.58), (2.2, 2.55, 0.98), (4.75, 5.1, 0.98), (7.3, 7.65, 0.98)]:
        result.add_annotation(x=x2, y=y, ax=x1, ay=y, axref="x", ayref="y", arrowhead=2, text="")
    for y, text in [(4.65, "<b>TRAIN</b>"), (2.58, inventory), (1.95, "<b>GENERATE → AGGREGATE → FOLD</b>")]:
        result.add_annotation(x=0.1, y=y, xanchor="left", text=text, showarrow=False)
    return result


def sampling_figure(summary: pd.DataFrame, rows: pd.DataFrame) -> go.Figure:
    """Pair a mean comparison with the underlying per-protein scatter."""
    subset = summary[(summary.figure == "06_sampling") & (summary.cohort == "natural") &
                     (summary.tier == "All depths") & (summary.metric == "r_precision")].set_index("method")
    paired = rows[(rows.figure == "06_sampling") & (rows.metric == "r_precision")].pivot(
        index="stem", columns="method", values="value")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), facecolor=PAPER)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.77, wspace=0.4)
    fig.text(0.09, 0.945, TITLES["06_sampling"], fontsize=19, weight="bold")
    fig.text(0.09, 0.884, f"248B-token model · step 266,344 · {len(paired)} natural proteins", fontsize=11)
    for ax in axes:
        style_axes(ax)
    for i, method in enumerate(ORDER["06_sampling"]):
        record = subset.loc[method]
        label, color, marker = METHODS[method]
        axes[0].errorbar(i, record["mean"], yerr=[[record["mean"] - record.ci_low], [record.ci_high - record["mean"]]],
                        fmt=marker, color=color, markersize=9, capsize=5)
        axes[0].text(i, record.ci_high + 0.05, f"{record['mean']:.3f}", ha="center", fontsize=12, color=color)
    axes[0].set_xticks([0, 1, 2], ["Mean\nsingle", "Consensus\nof 100", "Oracle\nbest of 100*"])
    axes[0].set_xlim(-0.5, 2.5)
    axes[0].set_ylabel("R-precision")
    axes[1].plot([0, 1], [0, 1], color=INK, linewidth=1, linestyle="--", alpha=0.6)
    axes[1].scatter(paired.consensus, paired.best100, color=PALETTE[3], s=20, alpha=0.65, linewidth=0)
    axes[1].set(xlim=(0, 1), xlabel="Consensus of 100", ylabel="Oracle best of 100*")
    axes[1].text(0.02, 0.96, "Above line: oracle wins", fontsize=10, va="top")
    save(fig, "06_sampling", CAPTIONS["06_sampling"])
    interactive = make_subplots(rows=1, cols=2, horizontal_spacing=0.16)
    layout = plotly_layout("r_precision")
    layout.pop("xaxis")
    layout.update(showlegend=True, margin=dict(l=60, r=20, t=50, b=120), legend=dict(orientation="h", y=-0.28, title="MSA depth"))
    interactive.update_layout(**layout)
    for i, method in enumerate(ORDER["06_sampling"]):
        record = subset.loc[method]
        label, color, marker = METHODS[method]
        interactive.add_trace(go.Scatter(x=[i], y=[float(record["mean"])], mode="markers", name=label, showlegend=False,
                                        marker=dict(color=color, size=12),
                                        error_y=dict(type="data", array=[record.ci_high - record["mean"]],
                                                     arrayminus=[record["mean"] - record.ci_low]),
                                        hovertemplate=f"{label}<br>R-precision %{{y:.3f}}<br>n={len(paired)}<extra></extra>"), row=1, col=1)
    details = rows[(rows.figure == "06_sampling") & (rows.metric == "r_precision")].drop_duplicates("stem").set_index("stem")
    interactive.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=INK, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=2)
    for i, tier in enumerate(TIERS):
        stems = paired.index.intersection(details.index[details.tier == tier])
        group = paired.loc[stems]
        interactive.add_trace(go.Scatter(x=group.consensus, y=group.best100, mode="markers", name=tier,
                                        marker=dict(color=PALETTE[i], size=7, opacity=0.7),
                                        customdata=[[stem, tier, details.loc[stem, "eval_set"]] for stem in stems],
                                        hovertemplate="%{customdata[0]} · %{customdata[2]}<br>MSA depth %{customdata[1]}<br>Consensus %{x:.3f}<br>Oracle best %{y:.3f}<extra></extra>"), row=1, col=2)
    interactive.update_xaxes(row=1, col=1, range=[-0.5, 2.5], tickvals=[0, 1, 2],
                             ticktext=["Mean<br>single", "Consensus<br>of 100", "Oracle best<br>of 100*"])
    interactive.update_xaxes(row=1, col=2, range=[0, 1], title="Consensus of 100", showgrid=False)
    interactive.update_yaxes(range=[0, 1.02], dtick=0.2, gridcolor=GRID, zeroline=False)
    interactive.update_yaxes(title="Oracle best of 100*", row=1, col=2)
    return interactive


def confidence_figure() -> go.Figure:
    """Show each map's maximum pTM and the cached oracle rank."""
    maps = pd.read_csv(DATA / "structured_confidence_per_map.csv")
    ranks = pd.read_csv(DATA / "structured_confidence_ranks.csv").set_index("stem")
    stems = sorted(maps.stem.unique())
    fig, ax = plt.subplots(figsize=(10.4, 6.2), facecolor=PAPER)
    fig.subplots_adjust(left=0.16, right=0.82, bottom=0.2, top=0.78)
    fig.text(0.08, 0.95, TITLES["02b_confidence"], fontsize=19, weight="bold")
    fig.text(0.08, 0.885, "Five natural proteins · MSA depth <10 · 100 ESMFold2 maps per protein", fontsize=11)
    style_axes(ax)
    ax.grid(False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    interactive = go.Figure()
    annotations = []
    for index, stem in enumerate(stems):
        group = maps[maps.stem == stem].sort_values(["arm", "map_seed"])
        decoys = group[group.arm == "esmfold2"]
        oracle = group[group.arm == "oracle"].iloc[0]
        rank = ranks.loc[stem]
        y = index + 0.25 * np.sin(decoys.map_seed.to_numpy() * 2.3999632297)
        rank_label = (f"{int(rank.rank_best)}" if rank.rank_best == rank.rank_worst
                      else f"{int(rank.rank_best)}–{int(rank.rank_worst)}") + " / 101"
        annotations.append(dict(x=1.02, y=index, xref="paper", yref="y", text=rank_label,
                                xanchor="left", showarrow=False, font=dict(color=PALETTE[3], size=12)))
        ax.scatter(decoys.ptm, y, s=19, alpha=0.48, color=PALETTE[2], linewidths=0,
                   label="ESMFold2-derived maps" if index == 0 else None)
        ax.scatter([oracle.ptm], [index], s=85, color=PALETTE[3], marker="D",
                   edgecolor=PAPER, linewidth=1.2, zorder=4, label="Oracle map" if index == 0 else None)
        ax.text(1.03, index, rank_label, transform=ax.get_yaxis_transform(), va="center",
                color=PALETTE[3], weight="bold", fontsize=11)
        custom = decoys[["stem", "map_seed", "ptm_rank", "tm_score", "contact_jaccard", "source_row", "sample_idx"]].to_numpy()
        interactive.add_trace(go.Scatter(x=decoys.ptm.tolist(), y=y.tolist(), mode="markers",
            name="ESMFold2-derived maps", legendgroup="decoys", showlegend=index == 0,
            marker=dict(color=PALETTE[2], size=6, opacity=0.55), customdata=custom.tolist(),
            hovertemplate="%{customdata[0]} · ESMFold2 seed %{customdata[1]}<br>Helico pTM %{x:.4f}<br>Rank %{customdata[2]} / 101<br>Helico TM-score %{customdata[3]:.3f}<br>Contact Jaccard vs oracle %{customdata[4]:.3f}<br>Source row %{customdata[5]} · sample %{customdata[6]}<extra></extra>"))
        interactive.add_trace(go.Scatter(x=[oracle.ptm], y=[index], mode="markers",
            name="Oracle map", legendgroup="oracle", showlegend=index == 0,
            marker=dict(color=PALETTE[3], size=13, symbol="diamond", line=dict(color=PAPER, width=1)),
            text=[f"{stem} · oracle<br>Rank {rank_label}<br>Helico TM-score {oracle.tm_score:.3f}<br>Source row {oracle.source_row}"],
            hovertemplate="%{text}<br>Helico pTM %{x:.4f}<extra></extra>"))
    annotations.append(dict(x=1.02, y=1.075, xref="paper", yref="paper", text="Oracle rank",
                            showarrow=False, xanchor="left", font=dict(size=11)))
    labels = [f"{stem}  ·  MSA {int(maps.loc[maps.stem == stem, 'msa_depth'].iloc[0])}" for stem in stems]
    ax.set_yticks(range(5), labels)
    ax.set_ylim(4.65, -0.65)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Helico pTM (highest of three samples per map)")
    ax.text(1.03, 1.08, "Oracle rank", transform=ax.transAxes, fontsize=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.12, 0.015), frameon=False, ncol=2)
    save(fig, "02b_confidence", CAPTIONS["02b_confidence"])
    interactive.update_layout(template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK), height=530, margin=dict(l=145,r=105,t=65,b=100),
        xaxis=dict(title="Helico pTM (highest of three samples per map)", range=[0,1.02], gridcolor=GRID, zeroline=False),
        yaxis=dict(tickvals=list(range(5)), ticktext=labels, range=[4.65,-0.65], showgrid=False, zeroline=False),
        legend=dict(orientation="h", y=-0.22), annotations=annotations)
    return interactive


TM_VIEWS = [("source_structure_tm_score", "Original ESMFold2", "Original structure TM-score"),
            ("tm_score", "After Helico", "Helico structure TM-score")]


def ptm_limits(values: pd.Series) -> tuple[float, float]:
    """Pad the observed pTM range without changing point coordinates."""
    padding = max(0.006, float(values.max() - values.min()) * 0.08)
    return max(0, float(values.min()) - padding), min(1.01, float(values.max()) + padding)


def accuracy_confidence_figure() -> go.Figure:
    """Separate original and reconstructed TM-score against selected Helico pTM."""
    table = pd.read_csv(DATA / "structured_accuracy_confidence.csv")
    stems = sorted(table.stem.unique())
    interactive = go.Figure()
    buttons = []
    for view_index, (metric, label, axis_label) in enumerate(TM_VIEWS):
        view_traces = []
        fig, ax = plt.subplots(figsize=(8.8, 6.5), facecolor=PAPER)
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.25, top=0.77)
        fig.text(0.1, 0.95, TITLES["02c_accuracy_confidence"], fontsize=19, weight="bold")
        fig.text(0.1, 0.89, "MSA depth <10 · 100 ESMFold2 maps per protein · diamonds = oracle", fontsize=11)
        style_axes(ax)
        for index, stem in enumerate(stems):
            group = table[table.stem == stem]
            color = PALETTE[index]
            decoys, oracle = group[group.arm == "esmfold2"], group[group.arm == "oracle"]
            ax.scatter(decoys.ptm, decoys[metric], s=19, alpha=0.55, color=color, linewidths=0, label=stem)
            ax.scatter(oracle.ptm, oracle[metric], s=100, marker="D", color=color,
                       edgecolor=INK, linewidth=1.3, zorder=5, label="Oracle contacts" if index == 0 else None)
            for is_oracle, points in ((False, decoys), (True, oracle)):
                accuracy_row = "accuracy_source_row" if metric.startswith("source_structure_") else "source_row"
                custom = points[["stem", "map_seed", "source_structure_tm_score", "tm_score", "source_row", accuracy_row, "sample_idx"]].fillna("reference").to_numpy().tolist()
                view_traces.append(go.Scatter(x=points.ptm.tolist(), y=points[metric].tolist(), mode="markers",
                    name="Oracle contacts" if is_oracle else stem, legendgroup="oracle" if is_oracle else stem,
                    showlegend=True, visible=index == 0,
                    marker=dict(color=color, size=12 if is_oracle else 6, symbol="diamond" if is_oracle else "circle",
                                opacity=1 if is_oracle else 0.55, line=dict(color=INK, width=1.2 if is_oracle else 0)),
                    customdata=custom,
                    hovertemplate=("%{customdata[0]} · " + ("oracle contacts" if is_oracle else "ESMFold2 seed %{customdata[1]}")
                                   + "<br>Helico pTM %{x:.4f}<br>TM-score %{y:.4f}<br>Original TM-score %{customdata[2]:.3f}"
                                     "<br>Helico TM-score %{customdata[3]:.3f}<br>Helico source row %{customdata[4]} · sample %{customdata[6]}"
                                     "<br>Accuracy source row %{customdata[5]}<extra></extra>")))
        ax.set(xlim=ptm_limits(table.ptm), ylim=(0, 1.035), xlabel="Helico pTM", ylabel=axis_label)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.53, 0.01), ncol=3, frameon=False)
        save(fig, "02c_accuracy_confidence", CAPTIONS["02c_accuracy_confidence"],
             suffix="" if view_index == 0 else "_tm_score", include_in_summary=False)
        if view_index == 0:
            interactive.add_traces(view_traces)
        buttons.append(dict(label=label, method="update", args=[
            {"y": [list(trace.y) for trace in view_traces],
             "customdata": [list(trace.customdata) for trace in view_traces]},
            {"yaxis.title.text": axis_label}]))
    protein_buttons = []
    for protein_index, stem in enumerate(stems):
        protein_buttons.append(dict(label=stem, method="update", args=[
            {"visible": [i // 2 == protein_index for i in range(10)]},
            {"xaxis.range": list(ptm_limits(table.loc[table.stem == stem, "ptm"]))}]))
    protein_buttons.append(dict(label="All five proteins", method="update", args=[
        {"visible": [True] * 10}, {"xaxis.range": list(ptm_limits(table.ptm))}]))
    interactive.update_layout(template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK), height=570, margin=dict(l=65,r=25,t=85,b=130),
        xaxis=dict(title="Helico pTM", range=list(ptm_limits(table.loc[table.stem == stems[0], "ptm"])), gridcolor=GRID, zeroline=False),
        yaxis=dict(title=TM_VIEWS[0][2], range=[0,1.035], gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h", y=-0.24),
        updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.2, yanchor="top", font=dict(size=11)),
                     dict(buttons=protein_buttons, x=0.62, xanchor="left", y=1.2, yanchor="top", font=dict(size=11))])
    return interactive


def per_protein_accuracy_figures() -> None:
    """Render two TM-versus-pTM panels per protein from cached analysis tables."""
    table = pd.read_csv(DATA / "structured_accuracy_confidence.csv")
    summaries = pd.read_csv(DATA / "structured_accuracy_summary.csv").set_index(["stem", "metric"])
    ranks = pd.read_csv(DATA / "structured_confidence_ranks.csv").set_index("stem")
    for index, (stem, group) in enumerate(table.groupby("stem", sort=True)):
        decoys, oracle = group[group.arm == "esmfold2"], group[group.arm == "oracle"]
        rank = ranks.loc[stem]
        rank_text = (str(int(rank.rank_best)) if rank.rank_best == rank.rank_worst
                     else f"{int(rank.rank_best)}–{int(rank.rank_worst)}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.1), sharex=True, sharey=True, facecolor=PAPER)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.2, top=0.73, wspace=0.17)
        fig.text(0.075, 0.95, f"{stem} · TM-score versus pTM", fontsize=20, weight="bold")
        fig.text(0.075, 0.875,
                 f"MSA depth {int(rank.msa_depth)} · 100 ESMFold2 maps · oracle pTM rank {rank_text} / 101",
                 fontsize=11)
        for ax, (metric, title, _) in zip(axes, TM_VIEWS, strict=True):
            style_axes(ax)
            ax.scatter(decoys.ptm, decoys[metric], s=26, alpha=0.62, color=PALETTE[index],
                       linewidths=0, label="ESMFold2-derived map")
            ax.scatter(oracle.ptm, oracle[metric], s=100, marker="D", color=PALETTE[index],
                       edgecolor=INK, linewidth=1.4, zorder=5, label="Oracle map")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.set(xlim=ptm_limits(group.ptm), ylim=(0, 1.035), xlabel="Helico pTM", title=title)
            rho = summaries.loc[(stem, metric), "spearman_vs_helico_ptm"]
            ax.text(0.025, 0.05, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes,
                    va="bottom", fontsize=10, bbox=dict(facecolor=PAPER, edgecolor="none", alpha=0.85, pad=2))
        axes[0].set_ylabel("TM-score versus experimental structure")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.54, 0.015), ncol=2, frameon=False)
        save(fig, f"02c_accuracy_confidence_protein_{stem}",
             "Highest pTM selects one of three Helico samples for every map; pTM also ranks the 101 maps. "
             "Left: original ESMFold2 TM-score (oracle reference = 1). Right: selected Helico reconstruction TM-score. "
             "Diamonds identify oracle contacts; correlations use only the 100 ESMFold2 maps. "
             "TM-score uses matched protein CA atoms; pTM retains all Helico input tokens. No ipTM or clash penalty.")


def export_plotly(fig: go.Figure, name: str) -> None:
    """Write figures in the exact data/layout/config envelope expected by the blog."""
    spec = json.loads(fig.to_json())
    spec["config"] = {"responsive": True, "displayModeBar": False, "scrollZoom": False}
    (SITE / f"{name}.json").write_text(json.dumps(spec, separators=(",", ":")) + "\n")
    mobile = json.loads(json.dumps(spec))
    mobile["layout"]["height"] = 600
    mobile["layout"]["font"]["size"] = 11
    mobile["layout"]["margin"] = dict(l=47, r=12, t=75, b=230)
    if name in ORDER and len(ORDER[name]) > 6:
        mobile["layout"]["height"] = 685
        mobile["layout"]["margin"]["b"] = 315
    if "legend" in mobile["layout"]:
        mobile["layout"]["legend"].update(y=-0.28, font=dict(size=10))
    if "updatemenus" in mobile["layout"]:
        mobile["layout"]["updatemenus"][0]["font"]["size"] = 10
    if name == "06_sampling":
        mobile["layout"].update(height=760, margin=dict(l=55, r=15, t=20, b=90))
        mobile["layout"]["xaxis"]["domain"] = [0, 1]
        mobile["layout"]["xaxis2"]["domain"] = [0, 1]
        mobile["layout"]["yaxis"]["domain"] = [0.58, 1]
        mobile["layout"]["yaxis2"]["domain"] = [0, 0.4]
        if "legend" in mobile["layout"]:
            mobile["layout"]["legend"]["y"] = -0.12
    if name == "02b_confidence":
        mobile["layout"].update(height=570, margin=dict(l=65, r=72, t=90, b=110))
        mobile["layout"]["yaxis"]["ticktext"] = [label.replace("_A", "").replace("  ·  ", "<br>") for label in spec["layout"]["yaxis"]["ticktext"]]
        mobile["layout"]["legend"].update(y=-0.25, x=0, font=dict(size=10))
        for annotation in mobile["layout"]["annotations"]:
            annotation["font"]["size"] = 10
        for menu in mobile["layout"].get("updatemenus", []):
            for button in menu["buttons"]:
                for annotation in button["args"][1]["annotations"]:
                    annotation["font"]["size"] = 10
    if name == "02c_accuracy_confidence":
        mobile["layout"].update(height=685, margin=dict(l=55,r=14,t=125,b=185))
        mobile["layout"]["legend"].update(y=-0.27, font=dict(size=10))
        for index, menu in enumerate(mobile["layout"]["updatemenus"]):
            menu.update(x=0, y=1.3 - index * 0.15, font=dict(size=10))
    if name == "03_method":
        steps = [a for a in spec["layout"]["annotations"] if "<b>" in a.get("text", "") and "<br>" in a["text"]]
        mobile["layout"].update(height=850, margin=dict(l=10,r=10,t=15,b=15),
                               shapes=[], annotations=[], xaxis=dict(range=[0,1],visible=False),
                               yaxis=dict(range=[0,8],visible=False))
        for i, item in enumerate(steps):
            y = 7.4 - i * 1.05
            color = item["font"]["color"]
            mobile["layout"]["shapes"].append(dict(type="rect",x0=0.04,x1=0.96,y0=y-0.42,y1=y+0.42,line=dict(color=color,width=1.5)))
            mobile["layout"]["annotations"].append(dict(x=0.5,y=y,text=item["text"],font=dict(size=11,color=color),showarrow=False))
            if i not in (2,6):
                mobile["layout"]["annotations"].append(dict(x=0.5,y=y-0.61,ax=0.5,ay=y-0.43,axref="x",ayref="y",text="",arrowhead=2))
    (SITE / f"{name}-mobile.json").write_text(json.dumps(mobile, separators=(",", ":")) + "\n")


def preview(names: list[str]) -> None:
    """Create a browser-ready review page; no server, CDN or external account needed."""
    (SITE / "plotly.min.js").write_text(get_plotlyjs())
    sections = []
    specs = {}
    mobile_specs = {}
    for name in names:
        specs[name] = json.loads((SITE / f"{name}.json").read_text())
        mobile_specs[name] = json.loads((SITE / f"{name}-mobile.json").read_text())
        lineage = (f'Data: <a href="../data/summary.csv">summary.csv</a> filtered by figure = {name}; '
                   'underlying proteins and original row IDs: <a href="../data/figure_rows.csv">figure_rows.csv</a>. ')
        if name in ("02b_confidence", "02c_accuracy_confidence"):
            lineage = ('Each dot: <a href="../data/structured_confidence_per_map.csv">selected map scores and source rows</a>. '
                       'Accuracy/confidence join: <a href="../data/structured_accuracy_confidence.csv">paired measurements</a>. '
                       'Oracle labels: <a href="../data/structured_confidence_ranks.csv">ranks and ties</a>. '
                       'Contact extraction: <a href="../data/structured_decoy_maps.csv">seeds, map/structure hashes and contact counts</a>. ')
        elif name == "03_method":
            lineage = 'Training inventory: <a href="../data/training_sources.csv">training_sources.csv</a>. '
        sections.append(f'<section id="section-{name}"><p class="number">FIGURE {name[:2]}</p>'
                        f'<h2>{html.escape(TITLES[name])}</h2><div class="frame"><div id="{name}" class="chart"></div></div>'
                        f'<p class="caption">{html.escape(CAPTIONS[name])}</p>'
                        f'<details><summary>Figure data and provenance</summary><p>{lineage}'
                        '<a href="../data/manifest.json">Input hashes and metric definitions</a>.</p>'
                        f'<p><a href="../plots/{name}.svg">SVG</a> · <a href="../plots/{name}.png">PNG</a> · '
                        f'<a href="{name}.json">Plotly JSON</a></p></details></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MarinFold — writeup figures</title><style>
@font-face{font-family:Lato;src:url(../data/inputs/Lato-Regular.ttf)}@font-face{font-family:Lato;font-weight:700;src:url(../data/inputs/Lato-Bold.ttf)}*{box-sizing:border-box}body{margin:0;background:#F1E8DF;color:#1F1E1B;font-family:Lato,"DejaVu Sans",Arial,sans-serif;font-size:16px;line-height:1.55}
main{max-width:1080px;margin:auto;padding:54px 28px}h1,h2{font-family:Georgia,serif;font-weight:400;line-height:1.2}h1{font-size:48px;max-width:800px}h2{font-size:30px;margin:6px 0 22px}a{color:#385C8F}section{margin:68px 0}.number{font-size:11px;letter-spacing:2px;color:#817970}.frame{background:#BDB1A5;padding:12px;border-radius:3px}.chart{background:#C4B9AE;min-height:470px}.caption{font-size:14px;max-width:930px}details{font-size:12px;border-top:1px solid #D2C8BC;padding-top:10px}nav{display:flex;gap:18px;flex-wrap:wrap;font-size:13px}.note{border-left:3px solid #8F6B38;padding:4px 16px}.kicker{letter-spacing:2px;font-size:12px}@media(max-width:600px){main{padding:24px 14px}h1{font-size:36px}h2{font-size:25px}.frame{padding:5px}section{margin:44px 0}.chart{min-height:540px}}
</style><main><p class="kicker">OPEN ATHENA / WORKING FIGURES / EXP325</p><h1>Sampling contacts from a single sequence</h1>
<p>A figure-first draft. Natural proteins lead; designed proteins are a separate selectable view. Plot menus switch among cached views and metrics.</p>
<nav><a href="../DRAFT.md">Terse post outline</a><a href="../README.md">Analysis notes</a><a href="../plots/summary.pdf">Slide deck</a><a href="../data/paired_deltas.csv">Paired comparisons</a></nav>
<p class="note">All MarinFold panels use the 248B-token model, exp277 step 266,344. Natural eval-val and eval-test are included; designed proteins remain separate. All plots read precomputed tables. Oracle comparisons use ground truth and are explicitly labeled.</p>
<p class="caption">AlphaFold3-derived results carry the <a href="../data/af3_notice.txt">required notice</a> and <a href="../data/af3_output_terms.md">output terms</a>.</p>
''' + "\n".join(sections) + '''<section><h2>Next: useful diversity and search</h2><p>Improve candidate contact sets, then select them. Inference-time search and post-training remain directions to test.</p></section></main>
<script src="plotly.min.js"></script><script>
const specs=innerWidth<600?MOBILE_DATA:SPEC_DATA;
for(const [name,spec] of Object.entries(specs)){
 Plotly.newPlot(name,spec.data,spec.layout,spec.config);
}
</script></html>'''
    (SITE / "index.html").write_text(page.replace("SPEC_DATA", json.dumps(specs).replace("</", "<\\/")).replace("MOBILE_DATA", json.dumps(mobile_specs).replace("</", "<\\/")))


def main() -> None:
    """Render default panels, metric alternatives, native site assets and preview."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    verify_data()
    for font in (DATA / "inputs").glob("Lato-*.ttf"):
        font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": "Lato", "font.size": 11, "text.color": INK,
                         "axes.labelcolor": INK, "svg.fonttype": "none", "savefig.facecolor": PAPER})
    PLOTS.mkdir(exist_ok=True)
    SITE.mkdir(exist_ok=True)
    summary = pd.read_csv(DATA / "summary.csv")
    rows = pd.read_csv(DATA / "figure_rows.csv")
    for figure in ["01_predictors", "02_oracle", "04_contacts", "05_folding"]:
        metrics = ["r_precision", "r_precision_long"] if figure == "04_contacts" else ["gdt_ts", "lddt"]
        for metric in metrics:
            static_depth(summary, figure, metric)
        export_plotly(export_depth(summary, figure), figure)
    export_plotly(method_figure(pd.read_csv(DATA / "training_sources.csv")), "03_method")
    export_plotly(sampling_figure(summary, rows), "06_sampling")
    names = list(TITLES)
    if (DATA / "structured_confidence_ranks.csv").exists():
        export_plotly(confidence_figure(), "02b_confidence")
        export_plotly(accuracy_confidence_figure(), "02c_accuracy_confidence")
        per_protein_accuracy_figures()
    else:
        names.remove("02b_confidence")
        names.remove("02c_accuracy_confidence")
    preview(names)
    print(f"Rendered {len(names)} figures, SVG/PNG, Plotly JSON, and {SITE / 'index.html'}")


if __name__ == "__main__":
    main()
