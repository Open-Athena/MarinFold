"""Fetch completed public evaluations and summarize natural/design R-precision."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data/contact_eval"
RESULT_FILES = ("manifest.json", "contact_precision_all.csv", "timings.csv")
LOW_MSA = HERE.parent / "exp260_evals_msa_depth_stratified/data/low_msa_depth_set.csv"


def result_directory(checkpoint: dict, eval_set: str) -> Path:
    """Keep independently evaluated sets separate from the combined run."""
    directory = DATA / "results" / checkpoint["trial"]
    return directory if eval_set == "all" else directory / "sets" / eval_set


def collect(plan: dict, eval_set: str = "all") -> list[str]:
    """Download only complete small result sets; never download model weights."""
    fs = fsspec.filesystem("hf", token=False)
    completed = []
    for checkpoint in plan["checkpoints"]:
        remote = plan["artifact_root"] + "/" + checkpoint["label"]
        if eval_set != "all":
            remote += "/sets/" + eval_set
        if not all(
            fs.exists(remote.removeprefix("hf://") + "/" + name)
            for name in RESULT_FILES
        ):
            print("PENDING", checkpoint["trial"])
            continue
        destination = result_directory(checkpoint, eval_set)
        destination.mkdir(parents=True, exist_ok=True)
        for name in RESULT_FILES:
            subprocess.run(
                ["hf", "buckets", "cp", remote + "/" + name, str(destination / name)],
                check=True,
            )
        completed.append(checkpoint["trial"])
    return completed


def validate_result(
    checkpoint: dict,
    precision: pd.DataFrame,
    timings: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    """Require the requested universe and full, uncapped per-input measurements."""
    units = set(zip(targets.dataset, targets.stem, strict=True))
    if len(precision) != len(targets) * 20 or set(precision.model) != {
        checkpoint["label"]
    }:
        raise ValueError("Wrong checkpoint identity or metric-row count")
    keys = ["dataset", "stem", "range", "cut"]
    if precision.duplicated(keys).any():
        raise ValueError("Duplicate per-protein metric")
    expected_rows = {
        (*unit, contact_range, cut)
        for unit in units
        for contact_range in ("all", "short", "medium", "long")
        for cut in ("L", "L/2", "L/5", "R", "AUC")
    }
    if set(precision[keys].itertuples(index=False, name=None)) != expected_rows:
        raise ValueError("Metric ranges/cuts do not cover every requested protein")
    if set(zip(precision.dataset, precision.stem, strict=True)) != units:
        raise ValueError("Metric coverage differs from the requested universe")
    if (
        len(timings) != len(targets)
        or set(zip(timings.dataset, timings.stem, strict=True)) != units
    ):
        raise ValueError("Timing coverage differs from the requested universe")
    if not (
        (timings.n_rollouts == 100)
        & (timings.stopped_rollouts == 100)
        & (timings.unfinished_rollouts == 0)
    ).all():
        raise ValueError("Incomplete or capped rollouts cannot enter the report")
    if set(timings.checkpoint) != {checkpoint["label"]}:
        raise ValueError("Timings identify a different checkpoint")
    expected = targets[["dataset", "stem", "eval_set", "is_viral"]].rename(
        columns={"eval_set": "expected_set", "is_viral": "expected_is_viral"}
    )
    joined = precision.merge(expected, on=["dataset", "stem"], validate="many_to_one")
    if not (joined.eval_set == joined.expected_set).all():
        raise ValueError("Evaluation-set labels changed")
    if not (joined.is_viral == joined.expected_is_viral).all():
        raise ValueError("Viral labels changed")


def summarize(plan: dict, eval_set: str = "all") -> None:
    """Report macro means and protein-bootstrap intervals without pooling sets."""
    targets = pd.read_csv(DATA / "targets.csv")
    if eval_set != "all":
        targets = targets[targets.eval_set == eval_set].copy()
    expected_units = {"all": 116, "eval-val": 97, "eval-denovo": 19}[eval_set]
    if len(targets) != expected_units:
        raise ValueError("Unexpected target count for the selected evaluation set")
    rng = np.random.default_rng(20261021)
    low_msa = pd.read_csv(LOW_MSA)
    low_msa_units = set(zip(low_msa.dataset, low_msa.stem, strict=True))
    rows, subgroup_rows, identities = [], [], []
    for checkpoint in plan["checkpoints"]:
        directory = result_directory(checkpoint, eval_set)
        if not all((directory / name).exists() for name in RESULT_FILES):
            continue
        precision = pd.read_csv(directory / "contact_precision_all.csv")
        timings = pd.read_csv(directory / "timings.csv")
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["checkpoint"]["label"] != checkpoint["label"]:
            raise ValueError("Result manifest identifies another checkpoint")
        if (
            manifest["units"],
            manifest["rollouts"],
            manifest["unfinished_rollouts"],
        ) != (expected_units, expected_units * 100, 0):
            raise ValueError("Result manifest has incomplete or capped rollouts")
        if manifest["sampling"] != plan["sampling"]:
            raise ValueError("Sampling recipe differs from the approved plan")
        validate_result(checkpoint, precision, timings, targets)
        if "capped_sample_policy" in plan["sampling"]:
            if not timings.extended_rollouts.between(0, 100).all():
                raise ValueError("Invalid number of extended samples")
            if int(timings.extended_rollouts.sum()) != manifest["extended_rollouts"]:
                raise ValueError("Extension counts disagree with the result manifest")
        for (result_set, contact_range), group in precision[
            (precision.cut == "R") & precision.range.isin(["all", "long"])
        ].groupby(["eval_set", "range"], sort=True):
            values = group.precision.dropna().to_numpy()
            if len(values) != len(group) or not np.isfinite(values).all():
                raise ValueError("Undefined headline R-precision")
            means = values[
                rng.integers(0, len(values), size=(10000, len(values)))
            ].mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975])
            rows.append(
                {
                    "trial": checkpoint["trial"],
                    "step": checkpoint["step"],
                    "eval_set": result_set,
                    "range": contact_range,
                    "n": len(values),
                    "r_precision": values.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "extended_rollouts": int(
                        timings.merge(
                            group[["dataset", "stem"]],
                            on=["dataset", "stem"],
                            validate="one_to_one",
                        ).extended_rollouts.sum()
                    )
                    if "capped_sample_policy" in plan["sampling"]
                    else 0,
                }
            )
        annotated = precision.copy()
        annotated["low_msa"] = [
            unit in low_msa_units
            for unit in zip(annotated.dataset, annotated.stem, strict=True)
        ]
        for result_set in sorted(targets.eval_set.unique()):
            for subgroup, mask in (
                ("viral", annotated.is_viral == 1),
                ("nonviral", annotated.is_viral == 0),
                ("low_msa", annotated.low_msa),
            ):
                selected = annotated[
                    mask & (annotated.eval_set == result_set) & (annotated.cut == "R")
                ]
                for contact_range in ("all", "long"):
                    values = selected.loc[selected.range == contact_range, "precision"]
                    subgroup_rows.append(
                        {
                            "trial": checkpoint["trial"],
                            "step": checkpoint["step"],
                            "eval_set": result_set,
                            "subgroup": subgroup,
                            "range": contact_range,
                            "n": len(values),
                            "r_precision": values.mean(),
                            "status": "scored"
                            if len(values)
                            else "outside requested sets",
                        }
                    )
        identities.append(
            {
                "trial": checkpoint["trial"],
                "files": {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in RESULT_FILES
                },
            }
        )
    if not rows:
        print("No complete checkpoint results yet.")
        return
    frame = pd.DataFrame(rows)
    suffix = "" if eval_set == "all" else "_" + eval_set
    frame.to_csv(DATA / f"r_precision{suffix}.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(
        DATA / f"r_precision_subgroups{suffix}.csv", index=False
    )
    (DATA / f"report_manifest{suffix}.json").write_text(
        json.dumps(
            {
                "completed_checkpoints": len(identities),
                "requested_checkpoints": len(plan["checkpoints"]),
                "results": identities,
                "bootstrap": {"replicates": 10000, "seed": 20261021, "unit": "protein"},
                "low_msa_membership_sha256": hashlib.sha256(
                    LOW_MSA.read_bytes()
                ).hexdigest(),
                "scope": "eval-val and eval-denovo only; no eval-test or legacy read",
                "eval_set": eval_set,
                "matched_sequence_knn_null": None,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        frame.sort_values(
            ["eval_set", "trial", "range"], ascending=[False, True, True]
        ).to_string(index=False)
    )


def publish(plan: dict, eval_set: str = "all") -> None:
    """Publish the small aggregate report beside its existing source results."""
    suffix = "" if eval_set == "all" else "_" + eval_set
    names = (
        f"r_precision{suffix}.csv",
        f"r_precision_subgroups{suffix}.csv",
        f"report_manifest{suffix}.json",
    )
    manifest = json.loads((DATA / names[-1]).read_text())
    if {r["trial"] for r in manifest["results"]} != {
        c["trial"] for c in plan["checkpoints"]
    }:
        raise ValueError("Cannot publish an incomplete aggregate report")
    for name in names:
        subprocess.run(
            [
                "hf", "buckets", "cp", str(DATA / name),
                plan["artifact_root"] + "/reports/" + name,
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--eval-set", choices=("all", "eval-val", "eval-denovo"), default="all"
    )
    parser.add_argument("--trials", nargs="+")
    args = parser.parse_args()
    if args.publish and args.trials:
        parser.error("--publish requires the complete checkpoint plan")
    plan = json.loads((DATA / "plan.json").read_text())
    if args.trials:
        plan = {
            **plan,
            "checkpoints": [
                c for c in plan["checkpoints"] if c["trial"] in args.trials
            ],
        }
        if len(plan["checkpoints"]) != len(set(args.trials)):
            raise ValueError("Unknown requested trial")
    if args.fetch:
        collect(plan, args.eval_set)
    summarize(plan, args.eval_set)
    if args.publish:
        publish(plan, args.eval_set)


if __name__ == "__main__":
    main()
