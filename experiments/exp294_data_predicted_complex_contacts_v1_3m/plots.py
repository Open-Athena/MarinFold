# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the Stage-A metadata census.

Reads the normalized Parquet written by ``metadata.py normalize`` and emits the
census figures into ``plots/`` with ``build_summary.py`` sidecars::

    uv run python plots.py --input '/data/exp294_predicted_complexes/metadata/normalized_*.parquet'

Every figure is metadata-only; no coordinates are read.
"""

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Normalized Parquet file or glob.")
    args = parser.parse_args(argv)
    src = f"read_parquet({_sql_literal(args.input)}, union_by_name=true)"
    con = duckdb.connect()
    plot_yield_frontier(con, src)
    plot_score_distribution(con, src)
    plot_filter_attrition(con, src)
    print(f"wrote census figures to {PLOTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
