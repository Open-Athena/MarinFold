# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch exp166 augmentation runs and their matched exp117 comparators.

Run from the exp154 analysis directory::

    uv run --with wandb python fetch_exp166_wandb.py

The exp166 launcher treats region as an execution detail and ``trial_id`` as
the logical-run identity. When more than one regional race finishes, this
script marks the earliest finished attempt as canonical and retains every
other finished attempt in the final-comparison table.
"""

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASELINE_RUNS_CSV = HERE / "data" / "wandb_runs.csv"
RUNS_OUTPUT = HERE / "data" / "exp166_wandb_runs.csv"
FINAL_OUTPUT = HERE / "data" / "exp166_final_comparisons.csv"
HISTORY_OUTPUT = HERE / "data" / "exp166_validation_trajectories.csv"

PROJECT = "eric-czech/marin"
RUN_PREFIX = "prot-exp166-cv1-"
LOSS_KEY = "eval/tokenized/contacts-v1-val/loss"
HIGHLIGHT_RUN_ID = "prot-exp166-cv1-aaaug-1_5b-e8-lr3p162e-3-wd0p1-bs128-exp117-init-us-east1"
MODEL_SIZE = "1_5b"
LOCAL_EPOCHS = 8
FETCH_WORKERS = 6

RUN_FIELDS = (
    "issue",
    "project",
    "run_id",
    "run_name",
    "run_url",
    "state",
    "created_at",
    "finished_at",
    "trial_id",
    "initialization",
    "region",
    "model_size",
    "epochs",
    "num_params",
    "num_tokens",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "train_loss",
    "val_loss",
    "val_loss_key",
    "source_exp117_loss",
    "source_checkpoint",
    "version",
    "is_canonical",
    "is_highlighted_best",
    "canonical_rule",
    "tags_json",
)

FINAL_FIELDS = (
    "config_rank",
    "config_id",
    "config_label",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "condition",
    "condition_label",
    "training_epochs_this_run",
    "effective_total_epochs",
    "val_loss",
    "delta_vs_exp117_e8",
    "run_id",
    "run_name",
    "run_url",
    "region",
    "trial_id",
    "is_canonical",
    "is_highlighted_best",
    "is_exact_config_match",
)

HISTORY_FIELDS = (
    "config_rank",
    "config_id",
    "config_label",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "condition",
    "condition_label",
    "run_id",
    "run_name",
    "run_url",
    "region",
    "global_step",
    "local_epoch",
    "effective_epoch",
    "val_loss",
    "val_loss_key",
    "is_derived_anchor",
    "is_highlighted_best",
)


@dataclass(frozen=True)
class ConfigKey:
    """Configured hyperparameters used to join exp166 and exp117 runs."""

    learning_rate: float
    weight_decay: float
    batch_size: int

    @property
    def config_id(self) -> str:
        return f"lr={self.learning_rate:g}|wd={self.weight_decay:g}|bs={self.batch_size}"

    @property
    def label(self) -> str:
        return f"LR {self.learning_rate:.3g} · WD {self.weight_decay:g} · BS {self.batch_size}"


@dataclass(frozen=True)
class HistorySpec:
    """One W&B run whose validation trajectory is needed for a plot."""

    config: ConfigKey
    config_rank: int
    condition: str
    condition_label: str
    project: str
    run_id: str
    run_name: str
    run_url: str
    region: str
    local_epochs: int
    epoch_offset: int
    total_steps: int
    is_highlighted_best: bool


def parse_tags(tags: Sequence[str]) -> dict[str, str]:
    """Parse key-value tags and reject conflicting duplicate keys."""
    values: dict[str, str] = {}
    for tag in tags:
        if "=" not in tag:
            continue
        key, value = tag.split("=", 1)
        previous = values.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"conflicting values for tag {key!r}: {previous!r}, {value!r}")
        values[key] = value
    return values


def nested_value(config: Mapping[str, Any], path: str) -> Any:
    """Read a required dotted path from a nested W&B config."""
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"missing W&B config field {path!r}")
        current = current[part]
    return current


def as_float(value: Any, label: str) -> float:
    """Parse a finite float."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {label}: {value!r}")
    return parsed


def assert_close(actual: float, expected: float, label: str, run_name: str) -> None:
    """Require config and tag metadata to agree."""
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{run_name}: {label} mismatch: {actual!r} != {expected!r}")


def timestamp_key(value: str) -> datetime:
    """Parse an ISO timestamp, sorting missing values after real timestamps."""
    if not value:
        return datetime.max.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def summary_dict(run: Any) -> Mapping[str, Any]:
    """Return a W&B run summary as a plain mapping."""
    summary = getattr(run.summary, "_json_dict", run.summary)
    if not isinstance(summary, Mapping):
        raise TypeError(f"{run.name}: W&B summary is not a mapping")
    return summary


def config_key(row: Mapping[str, Any]) -> ConfigKey:
    """Build the exact hyperparameter join key from a normalized row."""
    return ConfigKey(
        learning_rate=float(row["learning_rate"]),
        weight_decay=float(row["weight_decay"]),
        batch_size=int(row["batch_size"]),
    )


def normalize_exp166_run(run: Any) -> dict[str, Any]:
    """Normalize one finished exp166 run using config and tag metadata."""
    if run.state != "finished":
        raise ValueError(f"{run.name}: expected a finished run, got {run.state!r}")
    if not str(run.name).startswith(RUN_PREFIX):
        raise ValueError(f"unexpected exp166 run name {run.name!r}")

    tags = [str(tag) for tag in run.tags]
    tag_values = parse_tags(tags)
    required_tags = {
        "exp166",
        "aa-augmentation",
        "contacts-v1",
        "trial_id",
        "initialization",
        "region",
        "model_size",
        "epochs",
        "params",
        "tokens",
        "lr",
        "wd",
        "global_batch",
        "steps",
        "version",
    }
    missing = required_tags - (set(tags) | set(tag_values))
    if missing:
        raise ValueError(f"{run.name}: missing required tags {sorted(missing)}")

    config = run.config if isinstance(run.config, Mapping) else {}
    learning_rate = as_float(nested_value(config, "optimizer.learning_rate"), "learning rate")
    weight_decay = as_float(nested_value(config, "optimizer.weight_decay"), "weight decay")
    batch_size = int(nested_value(config, "trainer.train_batch_size"))
    assert_close(learning_rate, float(tag_values["lr"]), "LR config/tag", run.name)
    assert_close(weight_decay, float(tag_values["wd"]), "WD config/tag", run.name)
    if batch_size != int(tag_values["global_batch"]):
        raise ValueError(f"{run.name}: batch-size config/tag mismatch")

    if tag_values["model_size"] != MODEL_SIZE or int(tag_values["epochs"]) != LOCAL_EPOCHS:
        raise ValueError(f"{run.name}: expected {MODEL_SIZE}, {LOCAL_EPOCHS} epochs")
    initialization = tag_values["initialization"]
    if initialization not in {"scratch", "exp117-init"}:
        raise ValueError(f"{run.name}: unexpected initialization {initialization!r}")

    summary = summary_dict(run)
    if LOSS_KEY not in summary:
        raise ValueError(f"{run.name}: missing required validation key {LOSS_KEY!r}")
    val_loss = as_float(summary[LOSS_KEY], "validation loss")
    train_loss = summary.get("train/loss", "")
    if train_loss != "":
        train_loss = as_float(train_loss, "training loss")

    run_id = str(run.id)
    return {
        "issue": 166,
        "project": PROJECT,
        "run_id": run_id,
        "run_name": str(run.name),
        "run_url": f"https://wandb.ai/{PROJECT}/runs/{run_id}",
        "state": str(run.state),
        "created_at": str(getattr(run, "created_at", "") or ""),
        "finished_at": str(getattr(run, "heartbeatAt", "") or ""),
        "trial_id": tag_values["trial_id"],
        "initialization": initialization,
        "region": tag_values["region"],
        "model_size": tag_values["model_size"],
        "epochs": int(tag_values["epochs"]),
        "num_params": int(tag_values["params"]),
        "num_tokens": int(tag_values["tokens"]),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_loss_key": LOSS_KEY,
        "source_exp117_loss": float(tag_values["exp117_loss"]),
        "source_checkpoint": tag_values.get("source_checkpoint", ""),
        "version": tag_values["version"],
        "is_canonical": False,
        "is_highlighted_best": run_id == HIGHLIGHT_RUN_ID,
        "canonical_rule": "earliest finished regional attempt per trial_id",
        "tags_json": json.dumps(tags, separators=(",", ":")),
    }


def mark_canonical_runs(rows: list[dict[str, Any]]) -> None:
    """Mark the earliest finished regional attempt for each logical trial."""
    by_trial: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_trial.setdefault(str(row["trial_id"]), []).append(row)
    for attempts in by_trial.values():
        winner = min(
            attempts,
            key=lambda row: (
                timestamp_key(str(row["finished_at"])),
                timestamp_key(str(row["created_at"])),
                str(row["run_id"]),
            ),
        )
        winner["is_canonical"] = True


def fetch_exp166_rows() -> list[dict[str, Any]]:
    """Fetch all finished production exp166 attempts and select logical winners."""
    import wandb

    api = wandb.Api(timeout=180)
    runs = api.runs(
        PROJECT,
        filters={"display_name": {"$regex": f"^{RUN_PREFIX}"}, "state": "finished"},
        per_page=100,
    )
    rows = [normalize_exp166_run(run) for run in runs]
    if not rows:
        raise RuntimeError("no finished exp166 runs found")
    mark_canonical_runs(rows)

    canonical = [row for row in rows if row["is_canonical"]]
    trials = {str(row["trial_id"]) for row in canonical}
    configs = {config_key(row) for row in canonical}
    if len(trials) != 12 or len(configs) != 6:
        raise ValueError(f"expected 12 logical trials across 6 configs, found {len(trials)} and {len(configs)}")
    for config in configs:
        modes = {str(row["initialization"]) for row in canonical if config_key(row) == config}
        if modes != {"scratch", "exp117-init"}:
            raise ValueError(f"{config.config_id}: missing initialization mode; found {sorted(modes)}")

    highlighted = [row for row in rows if row["is_highlighted_best"]]
    continued = [row for row in rows if row["initialization"] == "exp117-init"]
    if len(highlighted) != 1 or float(highlighted[0]["val_loss"]) != min(float(row["val_loss"]) for row in continued):
        raise ValueError(f"highlight run {HIGHLIGHT_RUN_ID!r} is missing or is not the best finished warm-start run")

    rows.sort(
        key=lambda row: (
            float(row["source_exp117_loss"]),
            str(row["initialization"]),
            not bool(row["is_canonical"]),
            timestamp_key(str(row["finished_at"])),
        )
    )
    return rows


def load_baseline_rows(path: Path) -> list[dict[str, str]]:
    """Load the existing normalized exp117 table."""
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def exact_baseline(
    baseline_rows: Sequence[Mapping[str, str]],
    config: ConfigKey,
    epochs: int,
) -> Mapping[str, str] | None:
    """Return the exact exp117 config match at an epoch count, if it exists."""
    matches = [
        row
        for row in baseline_rows
        if int(row["issue"]) == 117
        and row["model_size"] == MODEL_SIZE
        and int(row["epochs"]) == epochs
        and math.isclose(float(row["learning_rate"]), config.learning_rate, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(float(row["weight_decay"]), config.weight_decay, rel_tol=1e-9, abs_tol=1e-12)
        and int(row["batch_size"]) == config.batch_size
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple exp117 e{epochs} matches for {config.config_id}")
    if not matches:
        return None
    match = matches[0]
    if match["val_loss_key"] != LOSS_KEY:
        raise ValueError(f"{match['run_name']}: unexpected validation key {match['val_loss_key']!r}")
    return match


def total_steps_from_baseline(row: Mapping[str, str]) -> int:
    """Read total steps from the normalized baseline row's tags."""
    tags = [str(tag) for tag in json.loads(row["tags_json"])]
    values = parse_tags(tags)
    if "steps" not in values:
        raise ValueError(f"{row['run_name']}: missing steps tag")
    return int(values["steps"])


def build_final_rows(
    exp166_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[ConfigKey, int]]:
    """Build long-form final-loss plot data with exact matched baselines."""
    canonical = [row for row in exp166_rows if row["is_canonical"]]
    configs = sorted(
        {config_key(row) for row in canonical},
        key=lambda config: float(exact_baseline(baseline_rows, config, 8)["val_loss"]),  # type: ignore[index]
    )
    ranks = {config: rank for rank, config in enumerate(configs, start=1)}
    rows: list[dict[str, Any]] = []

    for config in configs:
        e8 = exact_baseline(baseline_rows, config, 8)
        if e8 is None:
            raise ValueError(f"missing exp117 e8 source for {config.config_id}")
        e8_loss = float(e8["val_loss"])
        source_losses = {
            float(row["source_exp117_loss"])
            for row in canonical
            if config_key(row) == config
        }
        if len(source_losses) != 1:
            raise ValueError(f"{config.config_id}: conflicting source loss tags {source_losses}")
        source_loss = source_losses.pop()
        if not math.isclose(source_loss, e8_loss, rel_tol=0.0, abs_tol=5e-8):
            raise ValueError(
                f"{config.config_id}: exp117 source-loss tag/table mismatch: "
                f"{source_loss!r} != {e8_loss!r}"
            )

        shared = {
            "config_rank": ranks[config],
            "config_id": config.config_id,
            "config_label": config.label,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
        }
        rows.append(
            {
                **shared,
                "condition": "exp117_e8",
                "condition_label": "exp117 · 8 epochs · no augmentation",
                "training_epochs_this_run": 8,
                "effective_total_epochs": 8,
                "val_loss": e8_loss,
                "delta_vs_exp117_e8": 0.0,
                "run_id": e8["run_id"],
                "run_name": e8["run_name"],
                "run_url": e8["run_url"],
                "region": "",
                "trial_id": "",
                "is_canonical": True,
                "is_highlighted_best": False,
                "is_exact_config_match": True,
            }
        )

        for run in (row for row in exp166_rows if config_key(row) == config):
            condition = "exp166_scratch" if run["initialization"] == "scratch" else "exp166_continued"
            label = (
                "exp166 · 8 epochs from scratch · augmentation"
                if condition == "exp166_scratch"
                else "exp166 · 8 more epochs from exp117 · augmentation"
            )
            rows.append(
                {
                    **shared,
                    "condition": condition,
                    "condition_label": label,
                    "training_epochs_this_run": 8,
                    "effective_total_epochs": 8 if condition == "exp166_scratch" else 16,
                    "val_loss": run["val_loss"],
                    "delta_vs_exp117_e8": float(run["val_loss"]) - e8_loss,
                    "run_id": run["run_id"],
                    "run_name": run["run_name"],
                    "run_url": run["run_url"],
                    "region": run["region"],
                    "trial_id": run["trial_id"],
                    "is_canonical": run["is_canonical"],
                    "is_highlighted_best": run["is_highlighted_best"],
                    "is_exact_config_match": True,
                }
            )

        e16 = exact_baseline(baseline_rows, config, 16)
        if e16 is not None:
            rows.append(
                {
                    **shared,
                    "condition": "exp117_e16",
                    "condition_label": "exp117 · 16 epochs · no augmentation",
                    "training_epochs_this_run": 16,
                    "effective_total_epochs": 16,
                    "val_loss": float(e16["val_loss"]),
                    "delta_vs_exp117_e8": float(e16["val_loss"]) - e8_loss,
                    "run_id": e16["run_id"],
                    "run_name": e16["run_name"],
                    "run_url": e16["run_url"],
                    "region": "",
                    "trial_id": "",
                    "is_canonical": True,
                    "is_highlighted_best": False,
                    "is_exact_config_match": True,
                }
            )

    rows.sort(key=lambda row: (int(row["config_rank"]), str(row["condition"]), not bool(row["is_canonical"])))
    return rows, ranks


def build_history_specs(
    exp166_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[Mapping[str, str]],
    ranks: Mapping[ConfigKey, int],
) -> list[HistorySpec]:
    """Build the unique set of W&B trajectories needed by the plot."""
    specs: list[HistorySpec] = []
    selected_exp166 = [row for row in exp166_rows if row["is_canonical"] or row["is_highlighted_best"]]
    for config, rank in ranks.items():
        e8 = exact_baseline(baseline_rows, config, 8)
        if e8 is None:
            raise ValueError(f"missing exp117 e8 source for {config.config_id}")
        specs.append(
            HistorySpec(
                config=config,
                config_rank=rank,
                condition="exp117_e8",
                condition_label="exp117 · 8 epochs · no augmentation",
                project=str(e8["project"]),
                run_id=str(e8["run_id"]),
                run_name=str(e8["run_name"]),
                run_url=str(e8["run_url"]),
                region="",
                local_epochs=8,
                epoch_offset=0,
                total_steps=total_steps_from_baseline(e8),
                is_highlighted_best=False,
            )
        )
        e16 = exact_baseline(baseline_rows, config, 16)
        if e16 is not None:
            specs.append(
                HistorySpec(
                    config=config,
                    config_rank=rank,
                    condition="exp117_e16",
                    condition_label="exp117 · 16 epochs · no augmentation",
                    project=str(e16["project"]),
                    run_id=str(e16["run_id"]),
                    run_name=str(e16["run_name"]),
                    run_url=str(e16["run_url"]),
                    region="",
                    local_epochs=16,
                    epoch_offset=0,
                    total_steps=total_steps_from_baseline(e16),
                    is_highlighted_best=False,
                )
            )

        for run in (row for row in selected_exp166 if config_key(row) == config):
            condition = "exp166_scratch" if run["initialization"] == "scratch" else "exp166_continued"
            specs.append(
                HistorySpec(
                    config=config,
                    config_rank=rank,
                    condition=condition,
                    condition_label=(
                        "exp166 · 8 epochs from scratch · augmentation"
                        if condition == "exp166_scratch"
                        else "exp166 · 8 more epochs from exp117 · augmentation"
                    ),
                    project=str(run["project"]),
                    run_id=str(run["run_id"]),
                    run_name=str(run["run_name"]),
                    run_url=str(run["run_url"]),
                    region=str(run["region"]),
                    local_epochs=8,
                    epoch_offset=0 if condition == "exp166_scratch" else 8,
                    total_steps=int(parse_tags(json.loads(str(run["tags_json"]))) ["steps"]),
                    is_highlighted_best=bool(run["is_highlighted_best"]),
                )
            )
    return specs


def fetch_history(spec: HistorySpec) -> list[dict[str, Any]]:
    """Fetch one run's contacts-v1 validation trajectory."""
    import wandb

    api = wandb.Api(timeout=180)
    run = api.run(f"{spec.project}/{spec.run_id}")
    sampled = run.history(samples=200, keys=["global_step", LOSS_KEY], pandas=False)
    by_step: dict[int, float] = {}
    for row in sampled:
        raw_step = row.get("global_step")
        raw_loss = row.get(LOSS_KEY)
        if raw_step is None or raw_loss is None:
            continue
        step = int(raw_step)
        loss = float(raw_loss)
        if step >= 0 and math.isfinite(loss):
            by_step[step] = loss
    if len(by_step) < spec.local_epochs:
        raise ValueError(f"{spec.run_name}: only {len(by_step)} validation-history points")

    rows: list[dict[str, Any]] = []
    for step in sorted(by_step):
        local_epoch = spec.local_epochs * step / spec.total_steps
        rows.append(
            {
                "config_rank": spec.config_rank,
                "config_id": spec.config.config_id,
                "config_label": spec.config.label,
                "learning_rate": spec.config.learning_rate,
                "weight_decay": spec.config.weight_decay,
                "batch_size": spec.config.batch_size,
                "condition": spec.condition,
                "condition_label": spec.condition_label,
                "run_id": spec.run_id,
                "run_name": spec.run_name,
                "run_url": spec.run_url,
                "region": spec.region,
                "global_step": step,
                "local_epoch": local_epoch,
                "effective_epoch": spec.epoch_offset + local_epoch,
                "val_loss": by_step[step],
                "val_loss_key": LOSS_KEY,
                "is_derived_anchor": False,
                "is_highlighted_best": spec.is_highlighted_best,
            }
        )
    return rows


def fetch_histories(
    specs: Sequence[HistorySpec],
    final_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fetch validation histories and anchor continued runs at their source loss."""
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {executor.submit(fetch_history, spec): spec for spec in specs}
        for future in as_completed(futures):
            rows.extend(future.result())

    for spec in specs:
        if spec.condition != "exp166_continued":
            continue
        source = next(
            row
            for row in final_rows
            if row["config_id"] == spec.config.config_id and row["condition"] == "exp117_e8"
        )
        rows.append(
            {
                "config_rank": spec.config_rank,
                "config_id": spec.config.config_id,
                "config_label": spec.config.label,
                "learning_rate": spec.config.learning_rate,
                "weight_decay": spec.config.weight_decay,
                "batch_size": spec.config.batch_size,
                "condition": spec.condition,
                "condition_label": spec.condition_label,
                "run_id": spec.run_id,
                "run_name": spec.run_name,
                "run_url": spec.run_url,
                "region": spec.region,
                "global_step": "",
                "local_epoch": 0.0,
                "effective_epoch": 8.0,
                "val_loss": source["val_loss"],
                "val_loss_key": LOSS_KEY,
                "is_derived_anchor": True,
                "is_highlighted_best": spec.is_highlighted_best,
            }
        )

    rows.sort(
        key=lambda row: (
            int(row["config_rank"]),
            str(row["condition"]),
            str(row["run_id"]),
            float(row["effective_epoch"]),
        )
    )
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path, fields: Sequence[str]) -> None:
    """Write a stable CSV with an explicit schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-runs", type=Path, default=BASELINE_RUNS_CSV)
    parser.add_argument("--runs-output", type=Path, default=RUNS_OUTPUT)
    parser.add_argument("--final-output", type=Path, default=FINAL_OUTPUT)
    parser.add_argument("--history-output", type=Path, default=HISTORY_OUTPUT)
    args = parser.parse_args()

    exp166_rows = fetch_exp166_rows()
    baseline_rows = load_baseline_rows(args.baseline_runs)
    final_rows, ranks = build_final_rows(exp166_rows, baseline_rows)
    history_specs = build_history_specs(exp166_rows, baseline_rows, ranks)
    history_rows = fetch_histories(history_specs, final_rows)

    write_csv(exp166_rows, args.runs_output, RUN_FIELDS)
    write_csv(final_rows, args.final_output, FINAL_FIELDS)
    write_csv(history_rows, args.history_output, HISTORY_FIELDS)

    canonical = [row for row in exp166_rows if row["is_canonical"]]
    duplicates = len(exp166_rows) - len(canonical)
    exact_e16 = sum(row["condition"] == "exp117_e16" for row in final_rows)
    print(f"Wrote {len(exp166_rows)} finished exp166 attempts ({len(canonical)} logical trials)")
    print(f"Canonical rule: earliest finished regional attempt per trial_id; {duplicates} alternates retained")
    print(f"Exact exp117 16-epoch config matches: {exact_e16} of 6")
    print(f"Validation key: {LOSS_KEY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
