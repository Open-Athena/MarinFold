"""Render AF3 context from cached tables; no prediction, scoring or resampling."""

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from build_summary import save_plot_with_meta
from poster_style import save_poster_vectors
from theme import CAPTIONS, INK, METHODS, PAPER, TIERS, TITLES

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
BASELINES = ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "protenix_ss", "marinfold_helico"]
EXTRA = [f"af3_{n}_{selector}" for n in (100, 1000) for selector in ("ranking_score", "ptm")]
DIAGNOSTICS = ["af3_1000_tm_score", "oracle"]
LABELS = {key: item[0] for key, item in METHODS.items()}
LABELS.update({"af3": "AF3 · 25 · official rank", "af3_1000_tm_score": "AF3 · best of 1,000*",
               "oracle": "Helico + oracle map*", "marinfold_helico": "Helico + MarinFold (248B)"})
for _n in (100, 1000):
    LABELS[f"af3_{_n}_ranking_score"] = f"AF3 · {_n:,} · official rank"
    LABELS[f"af3_{_n}_ptm"] = f"AF3 · {_n:,} · pTM"
COLORS = [[0, "#F5EFE8"], [0.5, "#B2C5C9"], [1, "#385C8F"]]


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject stale prepared inputs before drawing any figure."""
    manifest = json.loads((HERE / "data/af3_context_analysis.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((HERE / "data" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared context changed: {name}; rerun prepare_af3_context.py")
    return pd.read_csv(HERE / "data/af3_context_rows.csv"), pd.read_csv(HERE / "data/af3_context_summary.csv")


def panel(frame: pd.DataFrame, name: str, methods: list[str], columns: list[str],
          xlabels: list[str], column: str, value: str, subtitle: str, *,
          poster_dir: Path | None = None) -> None:
    """Draw a shared-scale matrix and interactive cell-level provenance."""
    matrix = frame.pivot(index="method", columns=column, values=value).reindex(index=methods, columns=columns)
    cmap = LinearSegmentedColormap.from_list("athena_tm", [color for _, color in COLORS])
    cmap.set_bad("#E6DED4")
    fig, ax = plt.subplots(figsize=(12, 8.4), facecolor=PAPER)
    fig.subplots_adjust(left=0.32, right=0.89, top=0.80, bottom=0.16)
    fig.text(0.035, 0.945, TITLES[name], fontsize=21, weight="bold")
    fig.text(0.035, 0.895, subtitle, fontsize=12)
    im = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), xlabels, fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods], fontsize=11)
    ax.tick_params(length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    texts, hover = [], []
    for i, method in enumerate(methods):
        text_row, hover_row = [], []
        for j, category in enumerate(columns):
            score = matrix.loc[method, category]
            label = f"{score:.3f}" if np.isfinite(score) else "not run"
            ax.text(j, i, label, ha="center", va="center", fontsize=12 if np.isfinite(score) else 10,
                    color="white" if score > 0.7 else INK)
            text_row.append(label)
            row = frame[(frame.method == method) & (frame[column] == category)]
            if row.empty:
                hover_row.append("Extended sampling was only run at depth <10")
            elif value == "mean":
                r = row.iloc[0]
                hover_row.append(f"n={r.n} proteins; 95% protein-bootstrap CI [{r.lo:.3f}, {r.hi:.3f}]"
                                 f"<br>TM ≥0.8: {r.hits_tm_80}/{r.n}<br>Source: af3_context_summary.csv; {method}, {category}")
            else:
                r = row.iloc[0]
                hover_row.append(f"MSA depth {int(r.msa_depth)}<br>context_row={r.context_row}"
                                 f"<br>{r.source}, row {r.source_row}, {r.source_column}")
        texts.append(text_row)
        hover.append(hover_row)
    boundaries = [len(BASELINES) - 0.5, len(methods) - len(DIAGNOSTICS) - 0.5]
    for boundary in boundaries:
        ax.axhline(boundary, color=PAPER, lw=5)
    cax = fig.add_axes([0.915, 0.20, 0.014, 0.55])
    fig.colorbar(im, cax=cax, label="TM-score", ticks=[0, 0.5, 1])
    cax.spines[:].set_visible(False)
    fig.text(0.035, 0.075, "*Uses ground truth: diagnostic only. Predictor budgets differ. All AF3 runs use shared MSAs and no templates.", fontsize=10)
    fig.text(0.035, 0.035, "MSA tiers contain different proteins. Extended sampling covers only the five proteins at depth <10.", fontsize=10)
    if poster_dir is not None:
        save_poster_vectors(fig, poster_dir, name)
    else:
        save_plot_with_meta(fig, HERE / "plots" / f"{name}.png", caption=CAPTIONS[name],
                            script="render_af3_context.py", args=[], dpi=180)
        for extension in ("pdf", "svg"):
            fig.savefig(HERE / "plots" / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)

    chart = go.Figure(go.Heatmap(z=matrix.where(matrix.notna(), None).to_numpy(dtype=object).tolist(),
        x=xlabels, y=[LABELS[m] for m in methods], zmin=0, zmax=1, colorscale=COLORS,
        text=texts, texttemplate="%{text}", customdata=hover, xgap=2, ygap=2,
        colorbar=dict(title="TM", thickness=12),
        hovertemplate="%{y}<br>%{x}<br>TM %{text}<br>%{customdata}<extra></extra>"))
    chart.update_layout(paper_bgcolor=PAPER, plot_bgcolor="#E6DED4",
        font=dict(family="Lato, DejaVu Sans, Arial", color=INK, size=12), height=680,
        margin=dict(l=250, r=55, t=55, b=25), xaxis=dict(side="top", fixedrange=True),
        yaxis=dict(autorange="reversed", fixedrange=True))
    spec = json.loads(chart.to_json())
    spec["config"] = dict(responsive=True, displayModeBar=False, scrollZoom=False)
    mobile = json.loads(json.dumps(spec))
    mobile["layout"].update(height=720, margin=dict(l=132, r=8, t=60, b=15), font=dict(size=8, color=INK))
    mobile["data"][0].update(showscale=False, textfont=dict(size=9))
    for suffix, payload in [("", spec), ("-mobile", mobile)]:
        (HERE / "site" / f"{name}{suffix}.json").write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")


def main(*, poster_dir: Path | None = None) -> None:
    """Build two context panels from source-traced TM-score tables."""
    rows, summary = load_tables()
    for font in (HERE / "data/inputs").glob("Lato-*.ttf"):
        font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": ["Lato", "DejaVu Sans"], "font.size": 11,
                         "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.labelcolor": INK, "pdf.fonttype": 42, "svg.fonttype": "none"})
    low = rows[rows.tier == "<10"]
    stems = sorted(low.stem.unique())
    depths = low.drop_duplicates("stem").set_index("stem").msa_depth
    panel(low, "01c_af3_context", BASELINES + EXTRA + DIAGNOSTICS, stems,
          [f"{stem}\ndepth {int(depths[stem])}" for stem in stems], "stem", "tm_score",
          "Five low-depth proteins · archived predictor entries versus 100 / 1,000 fresh AF3 runs", poster_dir=poster_dir)
    counts = summary[summary.method == "af3"].set_index("tier").n
    panel(summary, "01d_af3_depth_context", BASELINES + EXTRA[2:] + DIAGNOSTICS, TIERS,
          [f"{tier}\nn={counts[tier]}" for tier in TIERS], "tier", "mean",
          "Mean TM-score · 305 matched natural proteins · same 0–1 scale in every cell", poster_dir=poster_dir)


if __name__ == "__main__":
    main()
