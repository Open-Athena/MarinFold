"""Collect the existing continuation W&B run without creating another run."""

import ast
import csv
import json
import tempfile
from pathlib import Path

import wandb


def write_csv(path: Path, rows: list[dict]) -> None:
    """Preserve all observed fields while leaving absent diagnostics blank."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "data"
    run = wandb.Api(timeout=30).run("open-athena/MarinFold/exp281-format-s02")
    history = list(run.scan_history(page_size=1000))
    (output / "format_s02_history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    # A recovered optimizer step may be logged more than once. Keep its latest
    # values; a baseline-only resume event must not erase the training fields.
    latest = {}
    for row in history:
        if row.get("train/step") is None:
            continue
        step = int(row["train/step"])
        latest.setdefault(step, {}).update({key: value for key, value in row.items()
                                           if not key.startswith("_") and value is not None})
    write_csv(output / "format_s02_training.csv", [latest[step] for step in sorted(latest)])
    snapshot = {"url": run.url, "state": run.state, "summary": dict(run.summary), "config": dict(run.config)}
    (output / "format_s02_summary.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    if run.state not in {"finished", "failed", "crashed", "killed"}:
        print(f"Captured history through step {max(latest)}; console is collected after the attempt ends.")
        return
    with tempfile.TemporaryDirectory(prefix="exp281-continuation-console-") as directory:
        run.file("output.log").download(root=directory, replace=True)
        publications = []
        for line in (Path(directory) / "output.log").read_text().splitlines():
            start = line.find("{'checkpoint/step':")
            if start >= 0:
                publications.append(ast.literal_eval(line[start:]))
    if publications:
        path = output / "format_s02_checkpoint_publications.csv"
        saved = {}
        if path.exists():
            with path.open() as handle:
                saved = {int(row["checkpoint/step"]): row for row in csv.DictReader(handle)}
        saved.update({int(row["checkpoint/step"]): row for row in publications})
        publications = [saved[step] for step in sorted(saved)]
        write_csv(path, publications)
    print(json.dumps({"state": run.state, "logged_global_steps": len(latest),
                      "last_step": max(latest), "checkpoint_publications": publications}, indent=2))


if __name__ == "__main__":
    main()
