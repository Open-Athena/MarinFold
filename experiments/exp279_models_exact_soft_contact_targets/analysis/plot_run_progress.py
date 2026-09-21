# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot exp279's schedule progress against wall clock, with every restart marked."""

import csv
import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from experiments.exp279_models_exact_soft_contact_targets.build_summary import (  # noqa: E402
    save_plot_with_meta,
)

EXPERIMENT = Path(__file__).resolve().parents[1]
BASE_END = 217800
RECOVERY_END = 333960
FINAL_END = 363000

# Each terminal failure: when the successor was relaunched, the update it
# resumed at, which attempt that was, how the predecessor died, and the label
# offset (the last one hugs the right edge).
RESTARTS = [
    ("2026-09-12T04:51Z", 55266, "a02", "exit 139", (6, -30)),
    ("2026-09-13T00:37Z", 109721, "a03", "exit 137 (OOM)", (6, -30)),
    ("2026-09-15T12:25Z", 125490, "a04", "exit 139", (6, -30)),
    ("2026-09-16T13:36Z", 183290, "a05", "exit 137 (OOM)", (6, -30)),
    ("2026-09-21T14:04Z", 235058, "a08", "restore blocker", (-64, -32)),
]


def main() -> None:
    source = EXPERIMENT / "data/run_progress.csv"
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    times = [datetime.datetime.fromisoformat(r["utc"]) for r in rows]
    steps = [int(r["global_step"]) for r in rows]

    fig, axis = plt.subplots(figsize=(11, 6), layout="constrained")
    axis.plot(times, steps, color="#087E8B", linewidth=2, zorder=3)

    for label, level, style in (
        ("base phase ends", BASE_END, "--"),
        ("recovery phase ends", RECOVERY_END, ":"),
        ("run complete", FINAL_END, "-."),
    ):
        axis.axhline(level, color="#9CA3AF", linewidth=1, linestyle=style, zorder=1)
        axis.text(times[0], level, f" {label} ({level:,})", fontsize=8.5,
                  color="#6B7280", va="bottom")

    for stamp, update, job, cause, offset in RESTARTS:
        when = datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        axis.plot([when], [update], marker="v", color="#D97706", markersize=9, zorder=4)
        axis.annotate(f"{cause}\n→ {job}", xy=(when, update), xytext=offset,
                      textcoords="offset points", fontsize=7.5, color="#92400E")

    axis.set_ylabel("Global step (update)")
    axis.set_xlabel("Wall clock (UTC)")
    axis.set_ylim(0, FINAL_END * 1.04)
    axis.grid(alpha=0.25)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.set_title(
        f"exp279 production progress — {steps[-1]:,} of {FINAL_END:,} updates "
        f"({steps[-1] / FINAL_END:.0%}), 5 recoveries"
    )
    save_plot_with_meta(
        fig,
        EXPERIMENT / "plots/run_progress.png",
        caption=(
            "Schedule position against wall clock. Triangles mark each terminal "
            "failure at the update it resumed from; flat stretches are downtime, "
            "not slow training (throughput held near 3.5 s/update)."
        ),
        script=str(Path(__file__).resolve().relative_to(EXPERIMENT.parents[1])),
        args=[],
        dpi=180,
    )
    plt.close(fig)
    print(f"Wrote plots/run_progress.png (through step {steps[-1]:,})")


if __name__ == "__main__":
    main()
