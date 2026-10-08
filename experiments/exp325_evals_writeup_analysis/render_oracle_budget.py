"""Render oracle-contact budget comparisons from the cached analysis tables."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from build_summary import save_plot_with_meta
from poster_style import save_poster_vectors
from theme import GRID, INK, METHODS, PAPER, TIERS

HERE = Path(__file__).resolve().parent
BUDGET_ORDER = ["no_contacts", "oracle_5", "oracle_10", "oracle_L5", "oracle_L2", "oracle_positive_all", "oracle"]
CONTEXT_ORDER = ["oracle", "oracle_positive_all", "oracle_L2", "oracle_L5", "oracle_10", "oracle_5",
                 "af3", "af2", "boltz2", "protenix_msa", "no_contacts"]
STYLES = {**METHODS,
    "oracle": ("Helico + full oracle map*", INK, "D"),
    "oracle_positive_all": ("Helico + all true contacts*", "#47388F", "P"),
    "oracle_5": ("Helico + 5 true contacts*", "#B37CA0", "v"),
    "oracle_10": ("Helico + 10 true contacts*", "#8F386D", "o"),
    "oracle_L5": ("Helico + L/5 true contacts*", "#388F8D", "s"),
    "oracle_L2": ("Helico + L/2 true contacts*", "#8F6B38", "^"),
}
TITLES = {"02e_oracle_budget": "How many true contacts does Helico need?",
          "02f_oracle_budget_context": "Sparse oracle contacts in predictor context"}
CAPTION = ("Same 305 natural proteins as Figure 02. Sparse arms uniformly sample true contacts without replacement: "
    "5, 10, floor(L/5), or floor(L/2), with L the frozen input sequence length. Unselected pairs are unknown, "
    "not negative. Two independently sampled subsets per protein/budget; nested budgets within each draw. "
    "Each map gets three Helico diffusion samples, six recycles, seed42, no MSA; highest ranking_score selects "
    "one sample. Average the two selected structures' accuracies within protein, never select the best subset. "
    "Fresh no-contact, all-positive-contact and full positive/negative oracle controls use the same setup. "
    "Relative budgets cap at the available positives; exact counts are in oracle_budget_maps.csv. "
    "Points are equal-weight protein means with 95% protein-bootstrap intervals. *Uses ground truth. "
    "Data: oracle_budget_summary.csv and oracle_budget_per_protein.csv; preprocessing: prepare_oracle_budget_analysis.py. "
    "Interactive menus expose pTM selection (Helico only), split views and GDT-TS/lDDT.")


def draw_panel(summary: pd.DataFrame, name: str, metric: str, poster_dir: Path | None) -> None:
    """Draw a depth-tier panel with every method on the same protein population."""
    methods = BUDGET_ORDER if name == "02e_oracle_budget" else CONTEXT_ORDER
    data = summary[(summary.selector == "ranking_score") & (summary.cohort == "natural") &
                   (summary.metric == metric) & summary.tier.isin(TIERS)]
    fig, ax = plt.subplots(figsize=(12.2, 7.5), facecolor=PAPER)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.37, top=0.80)
    ax.set_facecolor(PAPER)
    fig.text(0.09, 0.95, TITLES[name], fontsize=21, weight="bold", color=INK)
    fig.text(0.09, 0.895, "Random true contacts only · unselected pairs unknown · no MSA", fontsize=12, color=INK)
    fig.text(0.09, 0.845, "Two subsets per protein/budget · mean and 95% protein interval", fontsize=11, color=INK)
    for index, method in enumerate(methods):
        group = data[data.method == method].set_index("tier").reindex(TIERS)
        label, color, marker = STYLES[method]
        x = np.arange(4) + (index - (len(methods) - 1) / 2) * 0.061
        ax.errorbar(x, group["mean"], yerr=[group["mean"] - group.ci_low, group.ci_high - group["mean"]],
                    fmt=marker, color=color, markersize=6, capsize=2.5, elinewidth=1.2, label=label, zorder=3)
    counts = data.groupby("tier").n.first()
    ax.set_xticks(range(4), [f"{tier}\nn = {counts[tier]}" for tier in TIERS])
    ax.set(xlim=(-0.46, 3.46), ylim=(0, 1.02), ylabel="GDT-TS" if metric == "gdt_ts" else "lDDT",
           xlabel="MSA depth (sequences)")
    ax.xaxis.labelpad = 12
    ax.yaxis.labelpad = 12
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=8)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.078, 0.062),
               ncol=3 if len(methods) > 8 else 2, frameon=False, fontsize=9.5, columnspacing=1.8)
    fig.text(0.09, 0.023, "*Ground-truth diagnostic. Full oracle map also supplies non-contacts. L is input sequence length.", fontsize=9.5)
    filename = name + ("_lddt" if metric == "lddt" else "")
    if poster_dir is not None:
        save_poster_vectors(fig, poster_dir, filename)
    else:
        save_plot_with_meta(fig, HERE / "plots" / f"{filename}.png", caption=CAPTION,
                            script="render.py", args=[], dpi=180)
        for extension in ("pdf", "svg"):
            fig.savefig(HERE / "plots" / f"{filename}.{extension}", bbox_inches="tight")
    plt.close(fig)


def interactive(summary: pd.DataFrame, name: str) -> None:
    """Expose selector, metric and split choices without recomputing statistics."""
    methods = BUDGET_ORDER if name == "02e_oracle_budget" else CONTEXT_ORDER
    chart = go.Figure()
    buttons = []
    symbols = {"D": "diamond", "o": "circle", "s": "square", "^": "triangle-up", "v": "triangle-down", "P": "cross"}
    for selector in ("ranking_score", "ptm"):
        for metric in ("gdt_ts", "lddt"):
            for cohort in ("natural", "eval-val", "eval-test"):
                payloads = []
                for index, method in enumerate(methods):
                    group = summary[(summary.selector == selector) & (summary.metric == metric) &
                                    (summary.cohort == cohort) & (summary.method == method)].set_index("tier").reindex(TIERS)
                    label, color, marker = STYLES[method]
                    custom = [[tier, int(n) if pd.notna(n) else 0, method, selector, cohort]
                              for tier, n in zip(TIERS, group.n)]
                    values = [float(v) if pd.notna(v) else None for v in group["mean"]]
                    above = [float(v) if pd.notna(v) else None for v in group.ci_high - group["mean"]]
                    below = [float(v) if pd.notna(v) else None for v in group["mean"] - group.ci_low]
                    trace = go.Scatter(x=np.arange(4) + (index - (len(methods) - 1) / 2) * 0.061,
                        y=values, mode="markers", name=label,
                        marker=dict(color=color, symbol=symbols[marker], size=8),
                        error_y=dict(type="data", array=above, arrayminus=below, color=color, thickness=1.2, width=3),
                        customdata=custom, hovertemplate="%{fullData.name}<br>MSA %{customdata[0]} · n=%{customdata[1]}<br>Accuracy %{y:.3f}<extra></extra>")
                    payloads.append(trace)
                if not buttons:
                    chart.add_traces(payloads)
                label = f"{'GDT-TS' if metric == 'gdt_ts' else 'lDDT'} · {cohort} · {selector}"
                buttons.append(dict(label=label, method="update", args=[
                    {"y": [list(trace.y) for trace in payloads], "error_y": [trace.error_y.to_plotly_json() for trace in payloads],
                     "customdata": [trace.customdata for trace in payloads]},
                    {"yaxis.title.text": "GDT-TS" if metric == "gdt_ts" else "lDDT"}]))
    chart.update_layout(template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Lato, DejaVu Sans, Arial", color=INK, size=12), height=590,
        margin=dict(l=65, r=20, t=65, b=175),
        xaxis=dict(title="MSA depth (sequences)", tickvals=list(range(4)), ticktext=TIERS, range=[-0.46, 3.46]),
        yaxis=dict(title="GDT-TS", range=[0, 1.02], dtick=0.2, gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h", y=-0.30, font=dict(size=10)),
        updatemenus=[dict(buttons=buttons, x=0, y=1.15, xanchor="left", bgcolor=PAPER)])
    spec = json.loads(chart.to_json())
    spec["config"] = dict(responsive=True, displayModeBar=False)
    mobile = json.loads(json.dumps(spec))
    mobile["layout"].update(height=650, margin=dict(l=47, r=8, t=90, b=245), font=dict(size=10, color=INK))
    mobile["layout"]["legend"].update(y=-0.34, font=dict(size=8))
    for suffix, payload in [("", spec), ("-mobile", mobile)]:
        (HERE / "site" / f"{name}{suffix}.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")


def main(*, poster_dir: Path | None = None) -> None:
    """Validate cached outputs and render both the focused and context panels."""
    manifest = json.loads((HERE / "data/oracle_budget_analysis.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((HERE / "data" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Stale oracle-budget analysis: {name}")
    summary = pd.read_csv(HERE / "data/oracle_budget_summary.csv")
    for name in TITLES:
        for metric in ("gdt_ts", "lddt"):
            draw_panel(summary, name, metric, poster_dir)
        interactive(summary, name)
