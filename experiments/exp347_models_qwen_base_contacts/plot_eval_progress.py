"""Plot all committed 4B pilot and continuation eval-val measurements."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta


def measurements(here: Path) -> list[dict]:
    """Read complete checkpoint measurements and retain their exposure/provenance."""
    rows = []
    for document_format in ("contacts_v1", "prompted"):
        files = [
            here / "data/eval_val_pilot" / document_format / "summary.json",
            here / "data/eval_val_interim_20261021" / document_format / "summary.json",
            *sorted(
                (here / "data/eval_val_periodic" / document_format).glob(
                    "step-*/summary.json"
                )
            ),
        ]
        seen = set()
        for filename in files:
            result = json.loads(filename.read_text())
            if (
                not result["complete"]
                or result["n_units"] != 97
                or result["n_rollouts"] != 100
                or result["unfinished_rollouts"] != 0
                or result["document_format"] != document_format
            ):
                raise ValueError(f"Incomplete or mismatched measurement: {filename}")
            tokens = 0 if "eval_val_pilot" in filename.parts else result["tokens"]
            if tokens in seen:
                raise ValueError(
                    f"Duplicate checkpoint exposure: {document_format}, {tokens}"
                )
            seen.add(tokens)
            for contact_range in ("all", "long"):
                rows.append(
                    {
                        "format": document_format,
                        "phase_tokens": tokens,
                        "phase_tokens_billions": tokens / 1e9,
                        "step": result.get("step", ""),
                        "range": contact_range,
                        "n_proteins": result["valid_r_precision"][contact_range],
                        "r_precision": result["r_precision"][contact_range],
                        "checkpoint": result["checkpoint"],
                        "summary": str(filename.relative_to(here)),
                    }
                )
    return sorted(
        rows, key=lambda row: (row["format"], row["range"], row["phase_tokens"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status-note", help="Dated collection status, separate from measured scores."
    )
    args = parser.parse_args()
    here = Path(__file__).parent
    rows = measurements(here)
    reference = json.loads((here / "data/eval_val_default_reference.json").read_text())
    if reference["eval_set"] != "eval-val" or reference["n_proteins"] != 97:
        raise ValueError(
            "Default-model reference must use the same 97-protein eval-val set"
        )
    with (here / "data/eval_val_progress.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    highest = max(
        max(row["r_precision"] for row in rows),
        max(reference["r_precision"].values()),
    )
    upper = math.ceil((highest + 0.025) * 20) / 20
    for document_format, label, color in [
        ("contacts_v1", "contacts-v1", "#2166ac"),
        ("prompted", "Prompted", "#b35806"),
    ]:
        for ax, contact_range in zip(axes, ["all", "long"], strict=True):
            points = [
                row
                for row in rows
                if row["format"] == document_format and row["range"] == contact_range
            ]
            ax.plot(
                [row["phase_tokens_billions"] for row in points],
                [row["r_precision"] for row in points],
                marker="o",
                markersize=4,
                color=color,
                label=label,
            )
            last = points[-1]
            ax.annotate(
                f"{last['phase_tokens_billions']:.0f}B: {last['r_precision']:.3f}",
                (last["phase_tokens_billions"], last["r_precision"]),
                xytext=(5, 6),
                textcoords="offset points",
                color=color,
                fontsize=10,
            )
            ax.set_title(f"{contact_range.capitalize()}-range contacts")
            ax.set_xlabel("Additional full-corpus native tokens (billions)")
            ax.set_ylabel("Mean per-protein R-precision")
            ax.set_ylim(bottom=0, top=upper)
            ax.margins(x=0.13)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.2)
    for ax, contact_range in zip(axes, ["all", "long"], strict=True):
        value = reference["r_precision"][contact_range]
        ax.axhline(
            value,
            color="#555555",
            linestyle=":",
            linewidth=1.8,
            label=reference["label"],
        )
        ax.annotate(
            f"{value:.3f}",
            (0.98, value),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            color="#555555",
            fontsize=10,
        )
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Qwen3.5-4B: 97 eval-val proteins × 100 completed rollouts")
    if args.status_note:
        fig.text(0.5, 0.92, args.status_note, ha="center", fontsize=10, color="#555555")
    fig.text(
        0.5,
        0.015,
        "0 = corresponding 1B-token AFDB pilot; equal native-token counts do not imply equal protein exposure.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.9 if args.status_note else 1))
    for extension in ("png", "svg"):
        save_plot_with_meta(
            fig,
            here / f"plots/eval_val_progress.{extension}",
            caption=(
                "Measured checkpoints only; zero additional tokens is the corresponding "
                "1B-token AFDB pilot. Native-token counts imply different protein exposure "
                "between formats. Differences below approximately 0.005 are evaluator ties. "
                "Dotted lines show the current default exp277 1.5B step-266344 on the same "
                "97 eval-val proteins, with pinned provenance in data/eval_val_default_reference.json. "
                "Neither continuation is near its 100B target. "
                + (args.status_note or "")
            ),
            dpi=160,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
