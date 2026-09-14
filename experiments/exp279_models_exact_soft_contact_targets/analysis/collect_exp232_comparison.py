"""Collect matched ordinary validation CE from exp279 and exp232 m2/p06."""

import csv
import math
from pathlib import Path
from typing import Any

import wandb


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = "open-athena/MarinFold"
SOFT_RUN = "exp279-soft-s0-cw-h100x32-b02"
ONE_HOT_RUN = "prot-exp232-cw-cv1-decontam-s02-m2-p06-aug"
TOKENS_PER_STEP = 128 * 8192


def validation_history(run: Any) -> dict[int, float]:
    """Return the finite ordinary validation losses keyed by global step."""
    history: dict[int, float] = {}
    for row in run.scan_history(keys=["global_step", "eval/loss"], page_size=1000):
        step = row.get("global_step")
        loss = row.get("eval/loss")
        if step is None or loss is None:
            continue
        step, loss = int(step), float(loss)
        if not math.isfinite(loss):
            raise ValueError(f"Nonfinite eval/loss at step {step} in {run.id}")
        if step in history:
            raise ValueError(f"Duplicate eval/loss at step {step} in {run.id}")
        history[step] = loss
    if not history:
        raise ValueError(f"No eval/loss history found in {run.id}")
    return history


def main() -> None:
    """Join both W&B histories on exact global step and write the source table."""
    api = wandb.Api(timeout=60)
    soft = validation_history(api.run(f"{PROJECT}/{SOFT_RUN}"))
    one_hot = validation_history(api.run(f"{PROJECT}/{ONE_HOT_RUN}"))
    matched_steps = sorted(soft.keys() & one_hot.keys())
    if len(matched_steps) < 2:
        raise ValueError("The runs do not have enough exactly matched evaluations")

    rows = []
    for step in matched_steps:
        soft_loss, one_hot_loss = soft[step], one_hot[step]
        rows.append(
            {
                "global_step": step,
                "nominal_tokens": step * TOKENS_PER_STEP,
                "soft_eval_loss": soft_loss,
                "one_hot_eval_loss": one_hot_loss,
                "soft_minus_one_hot": soft_loss - one_hot_loss,
                "soft_perplexity": math.exp(soft_loss),
                "one_hot_perplexity": math.exp(one_hot_loss),
            }
        )

    output = EXPERIMENT / "data/exp232_m2_p06_matched_validation.csv"
    with output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} matched evaluations through step {matched_steps[-1]} to {output}")


if __name__ == "__main__":
    main()
