"""Fast paired seed-completion figures from cached small analysis tables."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from build_summary import save_plot_with_meta
from poster_style import save_poster_vectors
from theme import CAPTIONS, GRID, INK, PAPER, TITLES

HERE = Path(__file__).resolve().parent
BUDGETS = ["0", "5", "10", "L5"]
TICKS = ["0", "5", "10", "L/5"]
STYLES = {"direct": ("Helico: supplied contacts only", "#385C8F", "s"),
          "complete": ("MarinFold fills to L/2 → Helico", "#8F386D", "o")}
METRICS = {"gdt_ts": "GDT-TS", "tm_score": "TM-score", "lddt": "lDDT"}


def save(fig: plt.Figure, name: str, suffix: str, poster_dir: Path | None) -> None:
    """Write native vector artwork or plain-white poster variants."""
    filename = name + suffix
    if poster_dir is not None:
        save_poster_vectors(fig, poster_dir, filename)
    else:
        save_plot_with_meta(fig, HERE / "plots" / f"{filename}.png", caption=CAPTIONS[name],
                            script="render.py", args=[], dpi=180)
        for extension in ("pdf", "svg"):
            fig.savefig(HERE / "plots" / f"{filename}.{extension}", bbox_inches="tight")
    plt.close(fig)


def style(ax: plt.Axes, label: str) -> None:
    """Keep common bounds and budget positions for comparable protein panels."""
    ax.set_facecolor(PAPER)
    ax.set(xlim=(-.25, 3.25), ylim=(0, 1.04), xticks=range(4), xticklabels=TICKS,
           yticks=np.arange(0, 1.01, .2), ylabel=label)
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, pad=7)


def structural(rows: pd.DataFrame, summary: pd.DataFrame, selected: pd.DataFrame,
               metric: str, poster_dir: Path | None) -> None:
    """Show the paired response separately for every protein and for their mean."""
    data = rows[(rows.selector == "ranking_score") & (rows.metric == metric)]
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 11.4), facecolor=PAPER)
    fig.subplots_adjust(left=.075, right=.98, bottom=.09, top=.84, hspace=.47, wspace=.18)
    fig.text(.075, .965, TITLES["02g_seed_completion"], fontsize=22, weight="bold")
    fig.text(.075, .922, "Same known contacts in both arms · five proteins with MSA depth <10 · no MSA input", fontsize=12)
    for ax, stem in zip(axes.flat, sorted(data.stem.unique())):
        group = data[data.stem == stem]
        style(ax, METRICS[metric])
        for method, (label, color, marker) in STYLES.items():
            values = group[group.method == method].set_index("budget").loc[BUDGETS]
            ax.plot(range(4), values.value, color=color, marker=marker, lw=1.8, ms=6, label=label, zorder=3)
            draws = selected[(selected.stem == stem) & (selected.method == method) & (selected.selector == "ranking_score")]
            for x, budget in enumerate(BUDGETS[1:], 1):
                points = draws[draws.budget == budget].sort_values("map_seed")
                ax.scatter(x + np.array([-.055, .055]), points[metric], s=24, facecolors="none", edgecolors=color, zorder=4)
        target = group.iloc[0]
        ax.set_title(f"{stem} · depth {int(target.msa_depth)} · L = {int(target.L_exp245)}",
                     loc="left", fontsize=12, weight="bold", pad=10)
    ax = axes.flat[-1]
    style(ax, METRICS[metric])
    for method, (label, color, marker) in STYLES.items():
        values = summary[(summary.selector == "ranking_score") & (summary.metric == metric) &
                         (summary.method == method)].set_index("budget").loc[BUDGETS]
        x = np.arange(4) + (-.055 if method == "direct" else .055)
        ax.errorbar(x, values["mean"], yerr=[values["mean"] - values.ci_low, values.ci_high - values["mean"]],
                    color=color, marker=marker, lw=1.8, ms=6, capsize=3, label=label)
    ax.set_title("Mean of five proteins · 95% protein interval", loc="left", fontsize=12, weight="bold", pad=10)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.066, .90),
               ncol=2, frameon=False, fontsize=12)
    fig.text(.075, .040, "Number of true contacts supplied →   Filled points average two subsets; hollow points show each subset.", fontsize=10)
    fig.text(.075, .017, "MarinFold: 248B-token checkpoint, 100-rollout consensus. Helico: confidence-select one of three structures per map.", fontsize=10)
    save(fig, "02g_seed_completion", "" if metric == "gdt_ts" else f"_{metric}", poster_dir)


def contact_precision(rows: pd.DataFrame, targets: pd.DataFrame, poster_dir: Path | None) -> None:
    """Separate newly predicted contact correctness from guaranteed seed contacts."""
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 10.8), facecolor=PAPER)
    fig.subplots_adjust(left=.075, right=.98, bottom=.08, top=.84, hspace=.47, wspace=.18)
    fig.text(.075, .965, TITLES["02h_seed_contact_precision"], fontsize=22, weight="bold")
    fig.text(.075, .922, "The final L/2 set includes both the known contacts and MarinFold's additions", fontsize=12)
    styles = [("added_precision", "Newly predicted contacts only", "#8F386D", "o"),
              ("total_precision", "All L/2 contacts, including seeds", "#388F8D", "s")]
    for ax, stem in zip(axes.flat, [*sorted(rows.stem.unique()), "Mean"]):
        group = rows.groupby("budget").mean(numeric_only=True) if stem == "Mean" else rows[rows.stem == stem].set_index("budget")
        group = group.loc[BUDGETS]
        style(ax, "Contact precision")
        for column, label, color, marker in styles:
            ax.plot(range(4), group[column], color=color, marker=marker, lw=1.8, ms=6, label=label)
        title = "Mean of five proteins" if stem == "Mean" else f"{stem} · depth {int(targets.loc[stem, 'msa_depth'])}"
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=10)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.066, .90),
               ncol=2, frameon=False, fontsize=12)
    fig.text(.075, .028, "Number of true contacts supplied →   Same two subsets per protein. Ground truth scores additions only after selection.", fontsize=10)
    save(fig, "02h_seed_contact_precision", "", poster_dir)


def interactive(rows: pd.DataFrame, summary: pd.DataFrame, contacts: pd.DataFrame, name: str) -> None:
    """Expose protein and metric choices with cached source rows in every hover."""
    chart, buttons = go.Figure(), []
    precision = name == "02h_seed_contact_precision"
    for selector in (["ranking_score"] if precision else ["ranking_score", "ptm"]):
        for metric in (["precision"] if precision else METRICS):
            for stem in ["Mean of five", *sorted(rows.stem.unique())]:
                traces = []
                methods = ["added_precision", "total_precision"] if precision else list(STYLES)
                for method in methods:
                    if precision:
                        values = contacts.groupby("budget").mean(numeric_only=True) if stem == "Mean of five" else contacts[contacts.stem == stem].set_index("budget")
                        y = values.loc[BUDGETS, method].tolist()
                        label = "New contacts only" if method == "added_precision" else "All L/2 contacts"
                        color = "#8F386D" if method == "added_precision" else "#388F8D"
                        custom = [["seed_completion_contact_precision.csv", stem, budget] for budget in BUDGETS]
                        error = None
                    else:
                        label, color, _ = STYLES[method]
                        if stem == "Mean of five":
                            values = summary[(summary.selector == selector) & (summary.metric == metric) & (summary.method == method)].set_index("budget").loc[BUDGETS]
                            y = values["mean"].tolist()
                            lo, hi = values.ci_low, values.ci_high
                            custom = [["seed_completion_per_protein.csv", "all five protein rows", budget] for budget in BUDGETS]
                        else:
                            values = rows[(rows.selector == selector) & (rows.metric == metric) & (rows.method == method) & (rows.stem == stem)].set_index("budget").loc[BUDGETS]
                            y = values.value.tolist()
                            lo, hi = values.min_draw, values.max_draw
                            custom = [[v.source, v.source_rows, v.n_seed] for v in values.itertuples()]
                        error = dict(type="data", array=(hi.to_numpy() - y).tolist(), arrayminus=(np.asarray(y) - lo.to_numpy()).tolist(), width=3)
                    traces.append(go.Scatter(x=list(range(4)), y=y, mode="lines+markers", name=label,
                        line=dict(color=color), marker=dict(size=9), error_y=error, customdata=custom,
                        hovertemplate="%{y:.3f}<br>Source: %{customdata[0]}<br>Rows: %{customdata[1]}<br>Seeds: %{customdata[2]}<extra>%{fullData.name}</extra>"))
                label = "Contact precision" if precision else METRICS[metric]
                if not buttons:
                    chart.add_traces(traces)
                buttons.append(dict(label=f"{label} · {stem}" + ("" if precision else f" · {selector}"), method="update",
                    args=[{"y": [list(t.y) for t in traces], "error_y": [t.error_y.to_plotly_json() for t in traces],
                           "customdata": [t.customdata for t in traces]},
                          {"yaxis.title.text": label, "title.text": stem}]))
    chart.update_layout(template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Lato, DejaVu Sans, Arial", color=INK, size=12),
        title=dict(text="Mean of five", font=dict(size=13)), height=510,
        margin=dict(l=60, r=20, t=105, b=125), legend=dict(orientation="h", y=-.30),
        updatemenus=[dict(buttons=buttons, x=0, y=1.23, xanchor="left", bgcolor=PAPER)])
    chart.update_xaxes(title="True contacts supplied", tickvals=list(range(4)), ticktext=TICKS)
    chart.update_yaxes(title="Contact precision" if precision else "GDT-TS", range=[0, 1.04], gridcolor=GRID, dtick=.2)
    spec = json.loads(chart.to_json())
    spec["config"] = dict(responsive=True, displayModeBar=False)
    mobile = json.loads(json.dumps(spec))
    mobile["layout"].update(margin=dict(l=43, r=8, t=100, b=130), font=dict(size=10, color=INK))
    mobile["layout"]["updatemenus"][0]["font"] = dict(size=9)
    for suffix, payload in [("", spec), ("-mobile", mobile)]:
        (HERE / "site" / f"{name}{suffix}.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")


def main(*, poster_dir: Path | None = None) -> None:
    """Verify cached table hashes; render without running models or analysis."""
    manifest = json.loads((HERE / "data/seed_completion_analysis.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((HERE / "data" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Stale seed-completion table: {name}")
    rows, summary, selected, contacts = [pd.read_csv(HERE / f"data/seed_completion_{name}.csv", dtype={"budget": str})
                                       for name in ("per_protein", "summary", "selected", "contact_precision")]
    for metric in METRICS:
        structural(rows, summary, selected, metric, poster_dir)
    contact_precision(contacts, rows.drop_duplicates("stem").set_index("stem"), poster_dir)
    for name in ("02g_seed_completion", "02h_seed_contact_precision"):
        interactive(rows, summary, contacts, name)
