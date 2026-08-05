# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch normalized contacts-v1 sweep results from W&B.

The three source experiments use two validation-loss keys and two generations
of tag naming. This script normalizes them into one auditable CSV, keeps only
finished sweep runs, and selects the latest sweep subversion independently for
each issue.

Run from this directory::

    uv run python fetch_wandb.py
"""

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "data" / "wandb_runs.csv"

LOSS_KEYS = (
    "eval/tokenized/contacts-v1-val/loss",
    "eval/contacts-v1-val/loss",
)
TRAIN_LOSS_KEYS = ("train/loss",)
EXCLUDED_MARKERS = ("smoke", "probe", "profile", "batchcal")


@dataclass(frozen=True)
class Source:
    """One issue-tagged W&B sweep source."""

    issue: int
    project: str
    tag: str


SOURCES = (
    Source(issue=75, project="eric-czech/marin", tag="exp75"),
    Source(issue=117, project="eric-czech/marin", tag="exp117"),
    Source(issue=146, project="eric-czech/marin", tag="exp146"),
)

FIELDNAMES = (
    "issue",
    "project",
    "run_id",
    "run_name",
    "run_url",
    "state",
    "created_at",
    "finished_at",
    "sweep_subversion",
    "sweep_version",
    "data_subversion",
    "model_size",
    "num_params",
    "num_params_source",
    "num_tokens",
    "num_tokens_source",
    "epochs",
    "epochs_source",
    "weight_decay",
    "weight_decay_source",
    "learning_rate",
    "learning_rate_source",
    "batch_size",
    "batch_size_source",
    "train_loss",
    "train_loss_key",
    "val_loss",
    "val_loss_key",
    "summary_step",
    "tags_json",
)


def parse_tags(tags: Sequence[str]) -> dict[str, str]:
    """Parse ``key=value`` tags and reject conflicting duplicate keys."""
    parsed: dict[str, str] = {}
    for tag in tags:
        if "=" not in tag:
            continue
        key, value = tag.split("=", 1)
        previous = parsed.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"conflicting values for tag {key!r}: {previous!r}, {value!r}")
        parsed[key] = value
    return parsed


def parse_count(value: Any) -> int:
    """Parse an integer count, including decimal K/M/B/T suffixes."""
    if isinstance(value, bool):
        raise ValueError(f"boolean is not a count: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite count: {value!r}")
        return round(value)

    text = str(value).strip().replace("_", "").replace(",", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kKmMbBtT]?)", text)
    if not match:
        raise ValueError(f"invalid count: {value!r}")
    multipliers = {"": 1, "k": 10**3, "m": 10**6, "b": 10**9, "t": 10**12}
    return round(float(match.group(1)) * multipliers[match.group(2).lower()])


def parse_int(value: Any) -> int:
    """Parse a value as an integer without silently truncating it."""
    parsed = float(value)
    if not parsed.is_integer():
        raise ValueError(f"expected an integer, got {value!r}")
    return int(parsed)


def parse_float(value: Any) -> float:
    """Parse a finite floating-point value."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"expected a finite number, got {value!r}")
    return parsed


def nested_value(config: Mapping[str, Any], path: str) -> Any | None:
    """Read a dotted path from a nested W&B config mapping."""
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def resolve_tag_value(
    *,
    tag_values: Mapping[str, str],
    tag_keys: Sequence[str],
    parser: Callable[[Any], Any],
    field: str,
    run_name: str,
) -> tuple[Any, str]:
    """Resolve a normalized field from its first available tag alias."""
    for key in tag_keys:
        if key in tag_values:
            return parser(tag_values[key]), f"tag:{key}"
    raise ValueError(f"{run_name}: cannot resolve required field {field!r}")


def resolve_config_value(
    *,
    config: Mapping[str, Any],
    config_path: str,
    tag_values: Mapping[str, str],
    tag_keys: Sequence[str],
    parser: Callable[[Any], Any],
    field: str,
    run_name: str,
) -> tuple[Any, str]:
    """Resolve a config field and verify every available tag alias agrees."""
    raw_value = nested_value(config, config_path)
    if raw_value is None:
        raise ValueError(f"{run_name}: missing required config field {config_path!r}")
    value = parser(raw_value)
    for key in tag_keys:
        if key not in tag_values:
            continue
        tag_value = parser(tag_values[key])
        if tag_value != value:
            raise ValueError(
                f"{run_name}: {field} mismatch: config:{config_path}={value!r}, "
                f"tag:{key}={tag_value!r}"
            )
    return value, f"config:{config_path}"


def sweep_subversion(tag_values: Mapping[str, str], run_name: str) -> int:
    """Normalize the old ``sweep=v1`` and new ``sweep_subversion=2`` tags."""
    if "sweep_subversion" in tag_values:
        return parse_int(tag_values["sweep_subversion"])
    if "sweep" in tag_values:
        match = re.fullmatch(r"v?(\d+)", tag_values["sweep"], flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    match = re.search(r"(?:^|-)s(\d+)(?:-|$)", run_name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def summary_dict(run: Any) -> Mapping[str, Any]:
    """Return the public W&B summary as a plain mapping."""
    summary = run.summary
    data = getattr(summary, "_json_dict", summary)
    if not isinstance(data, Mapping):
        raise TypeError(f"{run.name}: W&B summary is not a mapping")
    return data


def select_loss(summary: Mapping[str, Any], run_name: str) -> tuple[float, str] | None:
    """Select the contacts-v1 validation loss, accepting both known keys."""
    found = [(key, parse_float(summary[key])) for key in LOSS_KEYS if key in summary]
    if not found:
        return None
    values = {value for _, value in found}
    if len(values) != 1:
        raise ValueError(f"{run_name}: conflicting validation losses: {found!r}")
    key, value = found[0]
    return value, key


def select_train_loss(summary: Mapping[str, Any], run_name: str) -> tuple[float, str] | None:
    """Select the final training loss when W&B recorded one."""
    found = [(key, parse_float(summary[key])) for key in TRAIN_LOSS_KEYS if key in summary]
    if not found:
        return None
    values = {value for _, value in found}
    if len(values) != 1:
        raise ValueError(f"{run_name}: conflicting training losses: {found!r}")
    key, value = found[0]
    return value, key


def is_excluded_run(run_name: str, tags: Sequence[str]) -> bool:
    """Return whether a finished job is a smoke/calibration run, not a sweep cell."""
    searchable = " ".join((run_name, *tags)).lower()
    return any(marker in searchable for marker in EXCLUDED_MARKERS)


def normalize_run(source: Source, run: Any) -> dict[str, Any] | None:
    """Normalize one W&B run, or return ``None`` when it has no target loss."""
    if run.state != "finished":
        raise ValueError(f"{run.name}: expected state='finished', got {run.state!r}")

    tags = [str(tag) for tag in run.tags]
    if is_excluded_run(run.name, tags):
        return None

    summary = summary_dict(run)
    selected_loss = select_loss(summary, run.name)
    if selected_loss is None:
        return None
    val_loss, val_loss_key = selected_loss
    selected_train_loss = select_train_loss(summary, run.name)
    train_loss, train_loss_key = selected_train_loss if selected_train_loss is not None else ("", "")

    tag_values = parse_tags(tags)
    config = run.config if isinstance(run.config, Mapping) else {}
    num_params, num_params_source = resolve_tag_value(
        tag_values=tag_values,
        tag_keys=("params_exact", "params", "num_params"),
        parser=parse_count,
        field="num_params",
        run_name=run.name,
    )
    num_tokens, num_tokens_source = resolve_tag_value(
        tag_values=tag_values,
        tag_keys=("tokens_exact", "tokens", "num_tokens"),
        parser=parse_count,
        field="num_tokens",
        run_name=run.name,
    )
    epochs, epochs_source = resolve_tag_value(
        tag_values=tag_values,
        tag_keys=("epochs", "num_epochs"),
        parser=parse_int,
        field="epochs",
        run_name=run.name,
    )
    weight_decay, weight_decay_source = resolve_config_value(
        config=config,
        config_path="optimizer.weight_decay",
        tag_values=tag_values,
        tag_keys=("wd", "weight_decay"),
        parser=parse_float,
        field="weight_decay",
        run_name=run.name,
    )
    learning_rate, learning_rate_source = resolve_config_value(
        config=config,
        config_path="optimizer.learning_rate",
        tag_values=tag_values,
        tag_keys=("lr", "learning_rate"),
        parser=parse_float,
        field="learning_rate",
        run_name=run.name,
    )
    batch_size, batch_size_source = resolve_config_value(
        config=config,
        config_path="trainer.train_batch_size",
        tag_values=tag_values,
        tag_keys=("global_batch", "batch_size", "bs"),
        parser=parse_int,
        field="batch_size",
        run_name=run.name,
    )

    run_id = str(run.id)
    return {
        "issue": source.issue,
        "project": source.project,
        "run_id": run_id,
        "run_name": run.name,
        "run_url": f"https://wandb.ai/{source.project}/runs/{run_id}",
        "state": run.state,
        "created_at": str(getattr(run, "created_at", "") or ""),
        "finished_at": str(getattr(run, "heartbeatAt", "") or ""),
        "sweep_subversion": sweep_subversion(tag_values, run.name),
        "sweep_version": tag_values.get("sweep_version", tag_values.get("sweep", "")),
        "data_subversion": tag_values.get("data_subversion", ""),
        "model_size": tag_values.get("model_size", next((t for t in tags if t in {"1_5b", "3b"}), "")),
        "num_params": num_params,
        "num_params_source": num_params_source,
        "num_tokens": num_tokens,
        "num_tokens_source": num_tokens_source,
        "epochs": epochs,
        "epochs_source": epochs_source,
        "weight_decay": weight_decay,
        "weight_decay_source": weight_decay_source,
        "learning_rate": learning_rate,
        "learning_rate_source": learning_rate_source,
        "batch_size": batch_size,
        "batch_size_source": batch_size_source,
        "train_loss": train_loss,
        "train_loss_key": train_loss_key,
        "val_loss": val_loss,
        "val_loss_key": val_loss_key,
        "summary_step": summary.get("_step", ""),
        "tags_json": json.dumps(tags, separators=(",", ":")),
    }


def latest_subversion_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Keep only the greatest normalized sweep subversion for each issue."""
    latest: dict[int, int] = {}
    for row in rows:
        issue = int(row["issue"])
        version = int(row["sweep_subversion"])
        latest[issue] = max(latest.get(issue, version), version)
    selected = [row for row in rows if row["sweep_subversion"] == latest[int(row["issue"])]]
    return selected, latest


def fetch_rows() -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Fetch all configured sources and return latest-subversion rows."""
    import wandb

    api = wandb.Api(timeout=180)
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        runs = api.runs(
            source.project,
            filters={"tags": source.tag, "state": "finished"},
            per_page=200,
        )
        for run in runs:
            row = normalize_run(source, run)
            if row is not None:
                rows.append(row)

    selected, latest = latest_subversion_rows(rows)
    missing = {source.issue for source in SOURCES} - set(latest)
    if missing:
        raise RuntimeError(f"no finished runs with a target loss for issues {sorted(missing)}")
    selected.sort(
        key=lambda row: (
            row["issue"],
            row["num_params"],
            row["epochs"],
            row["batch_size"],
            row["learning_rate"],
            row["weight_decay"],
            row["run_name"],
        )
    )
    return selected, latest


def write_csv(rows: Sequence[dict[str, Any]], output: Path) -> None:
    """Write normalized rows to ``output`` with a stable schema."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows, latest = fetch_rows()
    write_csv(rows, args.output)
    counts = {issue: sum(row["issue"] == issue for row in rows) for issue in sorted(latest)}
    print(f"Wrote {len(rows)} finished runs to {args.output}")
    for issue in sorted(latest):
        print(f"  issue #{issue}: subversion {latest[issue]}, {counts[issue]} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
