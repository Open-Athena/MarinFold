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
from matplotlib import patches, font_manager
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


def save(fig: plt.Figure, figure: str, caption: str, suffix: str = "") -> None:
    """Save vector/raster artwork and its source-table metadata."""
    name = figure + suffix
    save_plot_with_meta(fig, PLOTS / f"{name}.png", caption=caption,
                        script="render.py", args=[], dpi=180)
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
    """Show confidence discrimination and the quality of the matched controls."""
    table = pd.read_csv(DATA / "confidence_summary.csv")
    paired = pd.read_csv(DATA / "confidence_per_protein.csv")
    scores = paired[paired.confidence == "ranking_score"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), facecolor=PAPER)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.77, wspace=0.4)
    fig.text(0.09, 0.945, TITLES["02b_confidence"], fontsize=19, weight="bold")
    fig.text(0.09, 0.885, "20 natural proteins · equal map information and diffusion budgets", fontsize=11)
    for ax in axes:
        style_axes(ax)
    interactive = make_subplots(rows=1, cols=2, horizontal_spacing=0.17)
    arms = [("uniform", "Uniform random", PALETTE[1]), ("separation_matched", "Separation matched", PALETTE[3])]
    for index, (arm, label, color) in enumerate(arms):
        group = table[(table.arm == arm) & (table.confidence == "ranking_score")].set_index("tier").loc[TIERS]
        x = np.arange(4) + (index - 0.5) * 0.12
        axes[0].errorbar(x, group["mean"], yerr=[group["mean"] - group.ci_low, group.ci_high - group["mean"]],
                        fmt="o", color=color, capsize=3, label=label)
        interactive.add_trace(go.Scatter(x=x, y=group["mean"], mode="markers", name=label,
                                         marker=dict(color=color), error_y=dict(type="data", array=group.ci_high-group["mean"], arrayminus=group["mean"]-group.ci_low),
                                         customdata=TIERS, hovertemplate="%{fullData.name}<br>MSA depth %{customdata}<br>Oracle win rate %{y:.2f}<extra></extra>"), row=1, col=1)
        group = scores[scores.arm == arm]
        axes[1].scatter(group.random_mean_gdt_ts, group.oracle_gdt_ts, color=color, alpha=0.75, s=24)
        interactive.add_trace(go.Scatter(x=group.random_mean_gdt_ts, y=group.oracle_gdt_ts, mode="markers", showlegend=False,
                                         marker=dict(color=color), customdata=group.stem,
                                         hovertemplate="%{customdata}<br>Random GDT-TS %{x:.3f}<br>Oracle GDT-TS %{y:.3f}<extra></extra>"), row=1, col=2)
    axes[0].axhline(0.5, color=INK, ls="--", lw=1)
    axes[0].set_xticks(range(4), TIERS)
    axes[0].set(xlabel="MSA depth (5 proteins per tier)", ylabel="Oracle wins by confidence")
    axes[1].plot([0, 1], [0, 1], color=INK, ls="--", lw=1)
    axes[1].set(xlim=(0, 1), xlabel="Random-map mean GDT-TS", ylabel="Oracle-map GDT-TS")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.07, 0.02), frameon=False, ncol=2)
    save(fig, "02b_confidence", CAPTIONS["02b_confidence"])
    interactive.update_layout(template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(family=FONT, color=INK), height=510, margin=dict(l=60,r=20,t=35,b=120),
                              legend=dict(orientation="h", y=-0.25))
    interactive.update_yaxes(range=[0,1.02], gridcolor=GRID, zeroline=False)
    interactive.update_xaxes(tickvals=list(range(4)), ticktext=TIERS, title="MSA depth", row=1,col=1)
    interactive.update_yaxes(title="Oracle confidence win rate", row=1,col=1)
    interactive.update_xaxes(range=[0,1], title="Random mean GDT-TS", row=1,col=2)
    interactive.update_yaxes(title="Oracle GDT-TS", row=1,col=2)
    return interactive


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
    if name in ("06_sampling", "02b_confidence"):
        mobile["layout"].update(height=760, margin=dict(l=55, r=15, t=20, b=90))
        mobile["layout"]["xaxis"]["domain"] = [0, 1]
        mobile["layout"]["xaxis2"]["domain"] = [0, 1]
        mobile["layout"]["yaxis"]["domain"] = [0.58, 1]
        mobile["layout"]["yaxis2"]["domain"] = [0, 0.4]
        if "legend" in mobile["layout"]:
            mobile["layout"]["legend"]["y"] = -0.12
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
        sections.append(f'<section id="section-{name}"><p class="number">FIGURE {name[:2]}</p>'
                        f'<h2>{html.escape(TITLES[name])}</h2><div class="frame"><div id="{name}" class="chart"></div></div>'
                        f'<p class="caption">{html.escape(CAPTIONS[name])}</p>'
                        f'<details><summary>Figure data and provenance</summary><p>Data: <a href="../data/summary.csv">summary.csv</a> '
                        f'filtered by figure = {name}; underlying proteins and original row IDs: '
                        '<a href="../data/figure_rows.csv">figure_rows.csv</a>. '
                        '<a href="../data/manifest.json">Input hashes and metric definitions</a>.</p>'
                        f'<p><a href="../plots/{name}.svg">SVG</a> · <a href="../plots/{name}.png">PNG</a> · '
                        f'<a href="{name}.json">Plotly JSON</a></p></details></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MarinFold — writeup figures</title><style>
@font-face{font-family:Lato;src:url(../data/inputs/Lato-Regular.ttf)}@font-face{font-family:Lato;font-weight:700;src:url(../data/inputs/Lato-Bold.ttf)}*{box-sizing:border-box}body{margin:0;background:#F1E8DF;color:#1F1E1B;font-family:Lato,"DejaVu Sans",Arial,sans-serif;font-size:16px;line-height:1.55}
main{max-width:1080px;margin:auto;padding:54px 28px}h1,h2{font-family:Georgia,serif;font-weight:400;line-height:1.2}h1{font-size:48px;max-width:800px}h2{font-size:30px;margin:6px 0 22px}a{color:#385C8F}section{margin:68px 0}.number{font-size:11px;letter-spacing:2px;color:#817970}.frame{background:#BDB1A5;padding:12px;border-radius:3px}.chart{background:#C4B9AE;min-height:470px}.caption{font-size:14px;max-width:930px}details{font-size:12px;border-top:1px solid #D2C8BC;padding-top:10px}nav{display:flex;gap:18px;flex-wrap:wrap;font-size:13px}.note{border-left:3px solid #8F6B38;padding:4px 16px}.kicker{letter-spacing:2px;font-size:12px}@media(max-width:600px){main{padding:24px 14px}h1{font-size:36px}h2{font-size:25px}.frame{padding:5px}section{margin:44px 0}.chart{min-height:540px}}
</style><main><p class="kicker">OPEN ATHENA / WORKING FIGURES / EXP325</p><h1>Sampling contacts from a single sequence</h1>
<p>A figure-first draft. Natural proteins lead; designed proteins are a separate selectable view. Each plot menu switches between cached populations and metrics.</p>
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
    if (DATA / "confidence_summary.csv").exists():
        export_plotly(confidence_figure(), "02b_confidence")
    else:
        names.remove("02b_confidence")
    preview(names)
    print(f"Rendered {len(names)} figures, SVG/PNG, Plotly JSON, and {SITE / 'index.html'}")


if __name__ == "__main__":
    main()
