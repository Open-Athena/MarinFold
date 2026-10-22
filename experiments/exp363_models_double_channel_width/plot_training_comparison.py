"""Plot matched validation losses from the committed W&B export.

Run with ``uv run --no-project --with matplotlib python plot_training_comparison.py``.
The CSV contains every non-null full-validation observation in the snapshot;
matching uses exact optimizer steps, without interpolation or smoothing.
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta


HERE = Path(__file__).resolve().parent
BASELINE = "contacts-v1-exp277-m2-p06-full-epoch-1.5B"
WIDE = "contacts-v1-exp363-m2-p06-full-epoch-5.86B"


def load_curves() -> dict[str, dict[int, float]]:
    """Read finite validation observations, rejecting duplicate steps."""
    curves: dict[str, dict[int, float]] = {}
    with (HERE / "data/training_comparison_curves.csv").open() as handle:
        for row in csv.DictReader(handle):
            curve = curves.setdefault(row["run_id"], {})
            step, loss = int(row["global_step"]), float(row["validation_loss"])
            if step in curve or not math.isfinite(loss):
                raise ValueError(f"Invalid validation observation: {row}")
            curve[step] = loss
    return curves


def main() -> None:
    """Save the matched CSV and a full-curve plus late-gap figure."""
    curves = load_curves()
    provenance = json.loads((HERE / "data/training_comparison_provenance.json").read_text())
    exported = datetime.fromisoformat(provenance["observed_at"]).astimezone(ZoneInfo("America/New_York"))
    export_date = exported.strftime("%B %d, %Y")
    steps = sorted(curves[WIDE])
    missing = set(steps) - curves[BASELINE].keys()
    if missing:
        raise ValueError(f"Baseline is missing exact comparison steps: {missing}")
    baseline = [curves[BASELINE][step] for step in steps]
    wide = [curves[WIDE][step] for step in steps]
    deltas = [w - b for w, b in zip(wide, baseline, strict=True)]
    with (HERE / "data/matched_width_comparison.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "global_step", "exp277_validation_loss", "exp363_validation_loss",
            "wide_minus_default_nats", "perplexity_increase_percent",
        ])
        for step, b, w, delta in zip(steps, baseline, wide, deltas, strict=True):
            writer.writerow([step, b, w, delta, 100 * math.expm1(delta)])

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    x = [step / 1000 for step in steps]
    left.plot(x, baseline, "o-", color="#2864a5", markersize=3, label="Default: 1.471B (exp277)")
    left.plot(x, wide, "o-", color="#d36b2c", markersize=3, label="Double width: 5.862B (exp363)")
    left.set(title="Matched full-validation loss", ylabel="Full validation loss (nats/token)")
    left.legend(frameon=False, fontsize=10)

    # Show the small late reversal at a readable scale; the left panel preserves
    # all observations, including the wider model's larger early advantage.
    late = [(s, d) for s, d in zip(steps, deltas, strict=True) if s >= 25000]
    right.axhline(0, color="#333333", linewidth=1)
    right.plot([s / 1000 for s, _ in late], [d for _, d in late], "o-", color="#8a4d98")
    right.set(title="Late gap: wider minus default", ylabel="Validation loss difference (nats/token)")
    right.set_ylim(min(-0.016, min(d for _, d in late) - 0.004), max(0.017, max(d for _, d in late) + 0.008))
    right.text(0.02, 0.92, "Above zero: wider model worse", transform=right.transAxes, fontsize=10)
    right.text(0.02, 0.05, "Below zero: wider model better", transform=right.transAxes, fontsize=10)
    right.annotate(
        f"{steps[-1]:,}: {deltas[-1]:+.5f}",
        xy=(x[-1], deltas[-1]), xytext=(-115, -35), textcoords="offset points", fontsize=10,
        arrowprops={"arrowstyle": "-", "color": "#777777"},
    )
    for axis in (left, right):
        axis.set_xlabel("Optimizer step (thousands)")
        axis.grid(alpha=0.18)
    fig.suptitle(
        f"Updated {export_date}\nSame native + MPNN corpus · batch 128 · sequence 8192 · matched LR schedule",
        fontsize=11,
    )
    save_plot_with_meta(
        fig, HERE / "plots/training_comparison.png", dpi=180,
        caption=(
            f"Full-validation observations through step {steps[-1]:,}, exported {export_date}. "
            "Exact-step comparison; no interpolation or smoothing. Same logical training recipe, "
            "different hardware/gradient accumulation; single seed, incomplete epoch. "
            "The right panel zooms into observations after step 25,000."
        ),
    )
    plt.close(fig)
    print(f"Matched {len(steps)} evaluations; latest delta {deltas[-1]:+.8f} nats/token")


if __name__ == "__main__":
    main()
