# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the Stage-A metadata census.

Reads the normalized Parquet written by ``metadata.py normalize`` and emits the
census figures into ``plots/`` with ``build_summary.py`` sidecars::

    uv run python plots.py --input '/data/exp294_predicted_complexes/metadata/normalized_*.parquet'

Every figure is metadata-only; no coordinates are read.
"""

import argparse
import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
TARGET_DOCS = 3_000_000
HETERODIMER_FLOOR = 500_000
HOMO = "#3b6ea5"
HET = "#c4622d"
HARD = (
    "source_total_residues BETWEEN 4 AND 1998 "
    "AND coalesce(num_interactions, 0) >= 1 "
    "AND coalesce(clashes_backbone, 1000000) <= 10 "
    "AND quality_ratio IS NOT NULL"
)


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_yield_frontier(con: duckdb.DuckDBPyConnection, src: str) -> None:
    """Total and heterodimer yield against the relaxed quality floor."""
    floors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    rows = con.execute(
        f"""
        SELECT q AS floor,
            count(*) FILTER (WHERE quality_ratio >= q) AS total,
            count(*) FILTER (WHERE quality_ratio >= q
                AND complex_type = 'heterodimer') AS het
        FROM {src}, (SELECT unnest({floors}) AS q)
        WHERE {HARD}
        GROUP BY q ORDER BY q DESC
        """
    ).fetchall()
    floor = [r[0] for r in rows]
    total = [r[1] / 1e6 for r in rows]
    het = [r[2] / 1e6 for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.plot(floor, total, "o-", color=HOMO, lw=2, ms=5)
    ax.axhline(TARGET_DOCS / 1e6, color="#b3262a", ls="--", lw=1.4)
    ax.annotate(
        "3M target", xy=(0.985, TARGET_DOCS / 1e6),
        xycoords=("axes fraction", "data"), xytext=(0, 4),
        textcoords="offset points", color="#b3262a", va="bottom", ha="right",
        fontsize=9,
    )
    ax.invert_xaxis()
    ax.set_xlabel("relaxed quality floor  (1.0 = nominal Tier-A gate)")
    ax.set_ylabel("candidates (millions)")
    ax.set_title("Total yield reaches 3M only below floor 0.4", fontsize=11)
    _style(ax)

    ax2.plot(floor, het, "o-", color=HET, lw=2, ms=5)
    ax2.axhline(HETERODIMER_FLOOR / 1e6, color="#b3262a", ls="--", lw=1.4)
    ax2.annotate(
        "500k heterodimer floor", xy=(0.985, HETERODIMER_FLOOR / 1e6),
        xycoords=("axes fraction", "data"), xytext=(0, 4),
        textcoords="offset points", color="#b3262a", va="bottom", ha="right",
        fontsize=9,
    )
    ax2.set_ylim(0, 0.62)
    ax2.invert_xaxis()
    ax2.set_xlabel("relaxed quality floor")
    ax2.set_ylabel("heterodimer candidates (millions)")
    ax2.set_title("Heterodimers never approach 500k", fontsize=11)
    _style(ax2)
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "yield_frontier.png", dpi=150,
        caption=(
            "Hard-eligible AFCDB candidates vs the relaxed quality floor. The "
            "3M target needs a floor at or below ~0.4; 500k heterodimers is "
            "unreachable at any floor."
        ),
    )
    plt.close(fig)


def plot_score_distribution(con: duckdb.DuckDBPyConnection, src: str) -> None:
    """Where the confidence mass actually sits, per complex type."""
    rows = con.execute(
        f"""
        SELECT complex_type,
            least(20, CAST(floor(ipsae_score * 20) AS INT)) AS bin,
            count(*) AS n
        FROM {src} WHERE {HARD} GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).fetchall()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for ctype, color in (("homodimer", HOMO), ("heterodimer", HET)):
        pts = {b: n for c, b, n in rows if c == ctype}
        xs = sorted(pts)
        ax.plot(
            [x / 20 for x in xs], [pts[x] for x in xs],
            "o-", color=color, lw=2, ms=4, label=ctype,
        )
    ax.axvline(0.6, color="#666", ls=":", lw=1.2)
    ax.annotate(
        "Tier-A ipSAE gate", xy=(0.6, 0.97), xycoords=("data", "axes fraction"),
        xytext=(-4, 0), textcoords="offset points", fontsize=9, color="#666",
        ha="right", va="top",
    )
    ax.set_yscale("log")
    ax.set_xlabel("ipSAE selection score")
    ax.set_ylabel("hard-eligible models (log)")
    ax.set_title("Confidence mass is concentrated at ipSAE = 0", fontsize=11)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "ipsae_distribution.png", dpi=150,
        caption=(
            "ipSAE over hard-eligible models (min over directions for "
            "homodimers, max for heterodimers; the two differ by 0.0014 on "
            "average). 76% of heterodimers score exactly 0, which is why "
            "relaxing the floor buys so little."
        ),
    )
    plt.close(fig)


def plot_filter_attrition(con: duckdb.DuckDBPyConnection, src: str) -> None:
    """What each hard filter costs, split by confidence band."""
    rows = con.execute(
        f"""
        SELECT complex_type,
            CASE WHEN quality_ratio >= 1.0 THEN 'Tier A (>=1.0)'
                 WHEN quality_ratio >= 0.5 THEN '0.5 - 1.0'
                 WHEN quality_ratio >= 0.25 THEN '0.25 - 0.5'
                 ELSE '< 0.25' END AS band,
            round(count(*) FILTER (WHERE clashes_backbone <= 10) * 1.0
                  / count(*), 4) AS pass_clash
        FROM {src}
        WHERE source_total_residues BETWEEN 4 AND 1998
          AND coalesce(num_interactions, 0) >= 1 AND quality_ratio IS NOT NULL
        GROUP BY 1, 2
        """
    ).fetchall()
    bands = ["Tier A (>=1.0)", "0.5 - 1.0", "0.25 - 0.5", "< 0.25"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.38
    for i, (ctype, color) in enumerate((("homodimer", HOMO), ("heterodimer", HET))):
        vals = [next((r[2] for r in rows if r[0] == ctype and r[1] == b), 0) for b in bands]
        xs = [x + (i - 0.5) * width for x in range(len(bands))]
        ax.bar(xs, vals, width, color=color, label=ctype)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.0%}", ha="center", fontsize=8.5, color=color)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction passing the backbone-clash filter")
    ax.set_xlabel("confidence band")
    ax.set_title("The clash filter removes the low-confidence tail", fontsize=11)
    ax.legend(frameon=False, loc="lower left")
    _style(ax)
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "clash_filter_by_band.png", dpi=150,
        caption=(
            "Backbone-clash pass rate by confidence band. The filter is "
            "quality-correlated rather than mis-calibrated: it keeps ~95-97% "
            "of Tier A and rejects most of the weakest band."
        ),
    )
    plt.close(fig)


def plot_selection_funnel(selection_json: Path) -> None:
    """Where the 29,025,020 source models went."""
    stats = json.loads(selection_json.read_text())
    ledger = sorted(stats["ledger"], key=lambda r: -r["rows"])
    labels, values, colors = [], [], []
    for row in ledger:
        label = row["reason"].replace("_", " ")
        labels.append(label)
        values.append(row["rows"] / 1e6)
        colors.append(
            "#2e7d4f" if row["status"] == "selected"
            else ("#c4622d" if row["status"] == "rejected" else "#9aa0a6")
        )
    fig, ax = plt.subplots(figsize=(9, 5.0))
    ypos = range(len(labels))
    ax.barh(list(ypos), values, color=colors)
    for y, v in zip(ypos, values):
        ax.text(v + 0.15, y, f"{v:.2f}M", va="center", fontsize=9)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("source models (millions)")
    ax.set_title(
        f"Every one of {sum(r['rows'] for r in ledger):,} source models has a "
        "terminal reason", fontsize=11,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c)
        for c in ("#2e7d4f", "#c4622d", "#9aa0a6")
    ]
    ax.legend(handles, ["selected", "rejected", "not selected"], frameon=False, loc="lower right")
    _style(ax)
    ax.grid(axis="y", alpha=0)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "selection_funnel.png", dpi=150,
        caption=(
            "Terminal status of every AFCDB source model. The two dominant "
            "rejections are the relaxed quality floor and backbone clashes; "
            "3.0M are selected."
        ),
    )
    plt.close(fig)


def plot_pilot_composition(pilot_json: Path) -> None:
    """The stratified draw against the corpus it is drawn from."""
    stats = json.loads(pilot_json.read_text())
    bins = ["A_ge_1.0", "B_0.5_1.0", "B_0.3_0.5"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    width = 0.38
    peak = 0
    for i, (ctype, color) in enumerate((("homodimer", HOMO), ("heterodimer", HET))):
        drawn = [
            sum(b["drawn"] for b in stats["breakdown"]
                if b["complex_type"] == ctype and b["quality_bin"] == q)
            for q in bins
        ]
        peak = max(peak, *drawn)
        xs = [x + (i - 0.5) * width for x in range(len(bins))]
        ax.bar(xs, drawn, width, color=color, label=ctype)
    ax.set_ylim(0, peak * 1.28)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(["Tier A\n(>=1.0)", "Tier B\n(0.5-1.0)", "Tier B\n(0.3-0.5)"])
    ax.set_ylabel("pilot models drawn")
    ax.set_title("Equalised, not proportional", fontsize=11)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    _style(ax)

    names = ["corpus\n(the real run)", "throughput\nprobe", "stratified\npilot"]
    density = [
        stats["corpus_models_per_tar"],
        stats["probe_models_per_tar"],
        stats["pilot_models_per_tar"],
    ]
    ax2.bar(names, density, color=["#9aa0a6", "#2e7d4f", "#c4622d"], width=0.55)
    for x, v in enumerate(density):
        ax2.text(x, v + 4, f"{v:g}", ha="center", fontsize=10)
    ax2.set_ylabel("selected models per source tar")
    ax2.set_ylim(0, max(density) * 1.2)
    ax2.set_title("Why throughput needs its own sample", fontsize=11)
    _style(ax2)
    fig.tight_layout()
    save_plot_with_meta(
        fig, PLOTS / "pilot_composition.png", dpi=150,
        caption=(
            "Left: the pilot equalises across strata to over-sample the "
            "low-confidence tail. Right: extraction cost is per tar, so the "
            "stratified draw's 3.9 models/tar cannot time the run's 180."
        ),
    )
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Normalized Parquet file or glob.")
    parser.add_argument("--selection-json", type=Path, default=None)
    parser.add_argument("--pilot-json", type=Path, default=None)
    args = parser.parse_args(argv)
    src = f"read_parquet({_sql_literal(args.input)}, union_by_name=true)"
    con = duckdb.connect()
    plot_yield_frontier(con, src)
    plot_score_distribution(con, src)
    plot_filter_attrition(con, src)
    if args.selection_json:
        plot_selection_funnel(args.selection_json)
    if args.pilot_json:
        plot_pilot_composition(args.pilot_json)
    print(f"wrote figures to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
