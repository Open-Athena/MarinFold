"""Fast low-MSA oracle-budget figures from prepared per-map/per-protein tables."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from build_summary import save_plot_with_meta
from poster_style import save_poster_vectors
from theme import CAPTIONS, GRID, INK, METHODS, PAPER, TITLES

HERE = Path(__file__).resolve().parent
BUDGET_ORDER = ["no_contacts", "oracle_5", "oracle_10", "oracle_L5", "oracle_L2", "oracle_positive_all", "oracle"]
CONTEXT_ORDER = ["oracle", "oracle_positive_all", "oracle_L2", "oracle_L5", "oracle_10", "oracle_5", "no_contacts",
                 "af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "protenix_ss", "marinfold_helico"]
STYLES = {**METHODS,
    "oracle": ("Helico + full oracle map*", INK, "D"),
    "oracle_positive_all": ("Helico + all true contacts*", "#47388F", "P"),
    "oracle_5": ("Helico + 5 true contacts*", "#B37CA0", "v"),
    "oracle_10": ("Helico + 10 true contacts*", "#8F386D", "o"),
    "oracle_L5": ("Helico + L/5 true contacts*", "#388F8D", "s"),
    "oracle_L2": ("Helico + L/2 true contacts*", "#8F6B38", "^"),
}
TICKS = ["0", "5", "10", "L/5", "L/2", "All\npositive", "Full\nmap"]
METRICS = {"gdt_ts": "GDT-TS", "lddt": "lDDT"}


def save(fig: plt.Figure, name: str, metric: str, poster_dir: Path | None) -> None:
    """Write native artwork and analysis provenance, or white poster variants."""
    filename = name + ("_lddt" if metric == "lddt" else "")
    if poster_dir is not None:
        save_poster_vectors(fig, poster_dir, filename)
    else:
        save_plot_with_meta(fig, HERE / "plots" / f"{filename}.png", caption=CAPTIONS[name],
                            script="render.py", args=[], dpi=180)
        for extension in ("pdf", "svg"):
            fig.savefig(HERE / "plots" / f"{filename}.{extension}", bbox_inches="tight")
    plt.close(fig)


def axes_style(ax: plt.Axes, metric: str) -> None:
    """Use identical accuracy scales in every protein panel."""
    ax.set_facecolor(PAPER)
    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(METRICS[metric])
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=7)


def protein_panels(summary: pd.DataFrame, rows: pd.DataFrame, selected: pd.DataFrame,
                   metric: str, poster_dir: Path | None) -> None:
    """Show every protein, both random subsets, and the protein-weighted mean."""
    data = rows[(rows.selector == "ranking_score") & (rows.metric == metric) & rows.method.isin(BUDGET_ORDER)]
    samples = selected[selected.selector == "ranking_score"]
    stems = sorted(data.stem.unique())
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 12.4), facecolor=PAPER)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.10, hspace=0.47, wspace=0.18)
    fig.text(0.075, 0.965, TITLES["02e_oracle_budget"], fontsize=22, weight="bold")
    fig.text(0.075, 0.925, "Five natural proteins · MSA depth <10 · Helico sees no MSA", fontsize=12)
    for ax, stem in zip(axes.flat, stems):
        group = data[data.stem == stem].set_index("method").reindex(BUDGET_ORDER)
        axes_style(ax, metric)
        ax.plot(range(7), group.value, color=GRID, lw=1, zorder=1)
        for index, method in enumerate(BUDGET_ORDER):
            _, color, marker = STYLES[method]
            draws = samples[(samples.stem == stem) & (samples.method == method)]
            if len(draws) == 2:
                ax.scatter(index + np.array([-0.09, 0.09]), draws[metric], facecolors="none",
                           edgecolors=color, s=28, linewidths=1, zorder=3)
            ax.scatter(index, group.loc[method, "value"], marker=marker, color=color, s=60, zorder=4)
        ax.set_xticks(range(7), TICKS, fontsize=10)
        ax.set_xlim(-0.4, 6.4)
        ax.set_title(f"{stem} · depth {int(group.msa_depth.iloc[0])} · L = {int(group.L_exp245.iloc[0])}",
                     loc="left", fontsize=12, weight="bold", pad=12)
    ax = axes.flat[-1]
    axes_style(ax, metric)
    means = summary[(summary.selector == "ranking_score") & (summary.cohort == "natural") &
                    (summary.metric == metric) & (summary.tier == "<10")].set_index("method").reindex(BUDGET_ORDER)
    for index, method in enumerate(BUDGET_ORDER):
        row = means.loc[method]
        _, color, marker = STYLES[method]
        ax.errorbar(index, row["mean"], yerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]],
                    fmt=marker, color=color, markersize=7, capsize=3, elinewidth=1.2)
        ax.text(index, row.ci_high + 0.035, f"{row['mean']:.2f}", ha="center", fontsize=9, color=color)
    ax.set(xlim=(-0.4, 6.4), xticks=range(7), xticklabels=TICKS)
    ax.set_title("Mean of five proteins · 95% protein interval", loc="left", fontsize=12, weight="bold", pad=12)
    fig.text(0.075, 0.045, "True contacts supplied →   Sparse arms: two random subsets; hollow dots are the individual subset results.", fontsize=10)
    fig.text(0.075, 0.020, "Unselected pairs stay unknown. Full map also supplies non-contacts. All contact arms use ground truth.", fontsize=10)
    save(fig, "02e_oracle_budget", metric, poster_dir)


def context_panel(summary: pd.DataFrame, rows: pd.DataFrame, metric: str, poster_dir: Path | None) -> None:
    """Compare all arms on the exact same five proteins; show individual values."""
    means = summary[(summary.selector == "ranking_score") & (summary.cohort == "natural") &
                    (summary.metric == metric) & (summary.tier == "<10")].set_index("method").reindex(CONTEXT_ORDER)
    data = rows[(rows.selector == "ranking_score") & (rows.metric == metric)]
    fig, ax = plt.subplots(figsize=(12.0, 9.0), facecolor=PAPER)
    fig.subplots_adjust(left=0.34, right=0.965, top=0.82, bottom=0.13)
    ax.set_facecolor(PAPER)
    fig.text(0.055, 0.952, TITLES["02f_oracle_budget_context"], fontsize=21, weight="bold")
    fig.text(0.055, 0.902, "Same five natural proteins with MSA depth <10 · mean and 95% protein interval", fontsize=12)
    for index, method in enumerate(CONTEXT_ORDER):
        row = means.loc[method]
        _, color, marker = STYLES[method]
        values = data[data.method == method].sort_values("stem").value
        ax.scatter(values, np.full(len(values), index) + np.linspace(-0.11, 0.11, len(values)),
                   color=color, s=18, alpha=0.30, zorder=2)
        ax.errorbar(row["mean"], index, xerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]],
                    fmt=marker, color=color, markersize=7, capsize=3, elinewidth=1.5, zorder=4)
    ax.axhline(6.5, color=GRID, lw=1)
    ax.set(xlim=(0, 1.02), ylim=(len(CONTEXT_ORDER) - 0.5, -0.5), xlabel=METRICS[metric],
           yticks=range(len(CONTEXT_ORDER)), yticklabels=[STYLES[m][0] for m in CONTEXT_ORDER])
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=9)
    fig.text(0.055, 0.062, "*Ground-truth diagnostic. Sparse arms average two random subsets per protein. Faint dots show all five proteins.", fontsize=10)
    fig.text(0.055, 0.025, "Predictor entries retain their archived inputs and sampling budgets. All Helico arms use single-sequence input.", fontsize=10)
    save(fig, "02f_oracle_budget_context", metric, poster_dir)


def interactive(summary: pd.DataFrame, rows: pd.DataFrame, selected: pd.DataFrame, name: str) -> None:
    """Expose per-protein values and pTM sensitivity with source rows in hover."""
    context = name == "02f_oracle_budget_context"
    methods = CONTEXT_ORDER if context else BUDGET_ORDER
    chart = go.Figure()
    buttons = []
    for selector in ("ranking_score", "ptm"):
        for metric in METRICS:
            data = rows[(rows.selector == selector) & (rows.metric == metric)]
            for stem in ["Mean of five", *sorted(data.stem.unique())]:
                traces = []
                if stem == "Mean of five":
                    values = summary[(summary.selector == selector) & (summary.cohort == "natural") &
                        (summary.metric == metric) & (summary.tier == "<10")].set_index("method").reindex(methods)
                    y, lo, hi = values["mean"], values.ci_low, values.ci_high
                    custom = [[m, 5, "oracle_budget_per_protein.csv", "all five protein rows", None] for m in methods]
                else:
                    values = data[data.stem == stem].set_index("method").reindex(methods)
                    y, lo, hi = values.value, values.min_draw, values.max_draw
                    custom = [[m, int(values.loc[m, "n_draws"]), values.loc[m, "source"],
                               values.loc[m, "source_rows"], float(values.loc[m, "n_contacts"]) if pd.notna(values.loc[m, "n_contacts"]) else None]
                              for m in methods]
                colors = [STYLES[m][1] for m in methods]
                if context:
                    trace = go.Scatter(x=y, y=list(range(len(methods))), mode="markers",
                        marker=dict(color=colors, size=10), customdata=custom,
                        error_x=dict(type="data", array=hi - y, arrayminus=y - lo, color=INK, width=3),
                        hovertemplate="%{customdata[0]}<br>Accuracy %{x:.3f}<br>Source: %{customdata[2]}<br>Rows: %{customdata[3]}<extra></extra>")
                else:
                    trace = go.Scatter(x=list(range(len(methods))), y=y, mode="markers+lines",
                        marker=dict(color=colors, size=10), line=dict(color=GRID, width=1), customdata=custom,
                        error_y=dict(type="data", array=hi - y, arrayminus=y - lo, color=INK, width=3),
                        hovertemplate="%{customdata[0]}<br>Accuracy %{y:.3f}<br>Contacts %{customdata[4]}<br>Source: %{customdata[2]}<br>Rows: %{customdata[3]}<extra></extra>")
                traces.append(trace)
                if not buttons:
                    chart.add_traces(traces)
                axis = "xaxis" if context else "yaxis"
                error = "error_x" if context else "error_y"
                values_axis = "x" if context else "y"
                interval = "95% protein-bootstrap interval" if stem == "Mean of five" else "range across the two random subsets"
                buttons.append(dict(label=f"{METRICS[metric]} · {stem} · {selector}", method="update", args=[
                    {values_axis: [list(y)], error: [getattr(trace, error).to_plotly_json()], "customdata": [custom]},
                    {f"{axis}.title.text": METRICS[metric], "title.text": f"{stem} · {interval}"}]))
    chart.update_layout(template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Lato, DejaVu Sans, Arial", color=INK, size=12), showlegend=False,
        title=dict(text="Mean of five · 95% protein-bootstrap interval", font=dict(size=13)),
        height=650 if context else 480, margin=dict(l=270 if context else 65, r=20, t=110, b=70),
        updatemenus=[dict(buttons=buttons, x=0, y=1.20, xanchor="left", bgcolor=PAPER)])
    if context:
        chart.update_xaxes(title="GDT-TS", range=[0, 1.02], dtick=0.2, gridcolor=GRID)
        chart.update_yaxes(tickvals=list(range(len(methods))), ticktext=[STYLES[m][0] for m in methods],
                          range=[len(methods) - 0.5, -0.5])
    else:
        chart.update_xaxes(title="True contacts supplied", tickvals=list(range(7)), ticktext=[t.replace("\n", "<br>") for t in TICKS])
        chart.update_yaxes(title="GDT-TS", range=[0, 1.06], dtick=0.2, gridcolor=GRID)
    spec = json.loads(chart.to_json())
    spec["config"] = dict(responsive=True, displayModeBar=False)
    mobile = json.loads(json.dumps(spec))
    mobile["layout"].update(margin=dict(l=165 if context else 42, r=8, t=115, b=70), font=dict(size=9, color=INK))
    mobile["layout"]["title"]["font"] = dict(size=10)
    mobile["layout"]["updatemenus"][0]["font"] = dict(size=9)
    for suffix, payload in [("", spec), ("-mobile", mobile)]:
        (HERE / "site" / f"{name}{suffix}.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")


def main(*, poster_dir: Path | None = None) -> None:
    """Validate cached inputs, then render the low-depth sweep and its context."""
    manifest = json.loads((HERE / "data/oracle_budget_analysis.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((HERE / "data" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Stale oracle-budget analysis: {name}")
    summary = pd.read_csv(HERE / "data/oracle_budget_summary.csv")
    rows = pd.read_csv(HERE / "data/oracle_budget_per_protein.csv")
    selected = pd.read_csv(HERE / "data/oracle_budget_selected.csv")
    for metric in METRICS:
        protein_panels(summary, rows, selected, metric, poster_dir)
        context_panel(summary, rows, metric, poster_dir)
    for name in ("02e_oracle_budget", "02f_oracle_budget_context"):
        interactive(summary, rows, selected, name)
