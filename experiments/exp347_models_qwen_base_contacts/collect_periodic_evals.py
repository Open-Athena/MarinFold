"""Collect committed eval-val measurements and verify their per-protein evidence.

This is a one-shot read/collect command, not a scheduler. It copies only small
results, timings, and provenance from the checkpoint-local CoreWeave bucket.
Incomplete evaluations are reported as progress and never added to the curve.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import fsspec

from common import FORMATS, OUTPUT_ROOTS, output_root

HERE = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict]:
    """Read a small committed metric or timing table."""
    with path.open() as handle:
        return list(csv.DictReader(handle))


def validate_artifacts(folder: Path, result: dict, membership: dict) -> None:
    """Recompute headline means and require all 97 targets and their timings."""
    if (
        not result["complete"]
        or result["n_units"] != 97
        or result["n_rollouts"] != 100
        or result["unfinished_rollouts"] != 0
    ):
        raise ValueError("Incomplete evaluation result")
    metrics = read_csv(folder / "metrics.csv")
    timings = read_csv(folder / "timings.csv")
    if len(metrics) != 97 * 20 or len(timings) != 97:
        raise ValueError(f"Missing metric or timing rows: {folder}")
    for rows in (metrics, timings):
        if {r["stem"] for r in rows} != set(membership):
            raise ValueError(f"Wrong eval-val membership: {folder}")
        if any(r["dataset"] != "foldbench_monomer" for r in rows):
            raise ValueError("Unexpected evaluation universe")
    expected = {
        (stem, distance, cut)
        for stem in membership
        for distance in ("all", "short", "medium", "long")
        for cut in ("L", "L/2", "L/5", "R", "AUC")
    }
    if {(r["stem"], r["range"], r["cut"]) for r in metrics} != expected:
        raise ValueError("Missing, duplicate, or unexpected metric rows")
    dispatch = json.loads((folder / "dispatch.json").read_text())
    proofs = [
        json.loads(path.read_text()) for path in folder.glob("provenance-shard-*.json")
    ]
    if len(proofs) != dispatch["shards"]:
        raise ValueError("Missing worker provenance")
    identities = set()
    for proof in proofs:
        if (
            proof["checkpoint"]["uri"] != result["checkpoint"]
            or proof["document_format"] != result["document_format"]
        ):
            raise ValueError("Worker checkpoint/format differs from result")
        identities.add(json.dumps(proof["checkpoint"]["files"], sort_keys=True))
    if len(identities) != 1:
        raise ValueError("Workers loaded different checkpoint objects")
    for timing in timings:
        if int(timing["n_rollouts"]) != 100 or int(timing["unfinished_rollouts"]) != 0:
            raise ValueError("Incomplete inference timings")
        if (
            not math.isfinite(float(timing["elapsed_seconds"]))
            or float(timing["elapsed_seconds"]) <= 0
        ):
            raise ValueError("Invalid per-protein inference time")
    for distance in ("all", "short", "medium", "long"):
        values = [
            float(r["precision"])
            for r in metrics
            if r["range"] == distance and r["cut"] == "R"
        ]
        if len(values) != 97 or not all(
            math.isfinite(v) and 0 <= v <= 1 for v in values
        ):
            raise ValueError(f"Missing/invalid per-protein {distance} R-precision")
        if not math.isclose(
            sum(values) / len(values),
            result["r_precision"][distance],
            abs_tol=1e-12,
            rel_tol=0,
        ):
            raise ValueError(f"Aggregate differs from per-protein scores: {folder}")


def write_strata(root: Path, membership: dict) -> None:
    """Refresh viral/nonviral cuts from complete measurements only."""
    rows = []
    for summary in sorted(root.glob("*/step-*/summary.json")):
        result = json.loads(summary.read_text())
        metrics = read_csv(summary.parent / "metrics.csv")
        for distance in ("all", "long"):
            for split in ("all", "viral", "nonviral"):
                values = [
                    float(row["precision"])
                    for row in metrics
                    if row["range"] == distance
                    and row["cut"] == "R"
                    and (
                        split == "all"
                        or (membership[row["stem"]]["is_viral"] == "1")
                        == (split == "viral")
                    )
                ]
                rows.append(
                    {
                        "format": result["document_format"],
                        "step": result["step"],
                        "tokens": result["tokens"],
                        "split": split,
                        "range": distance,
                        "n": len(values),
                        "r_precision": sum(values) / len(values),
                    }
                )
    with (root / "strata.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Fetch new complete results and print a bounded backlog snapshot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", choices=OUTPUT_ROOTS, default="cw-rno2a")
    parser.add_argument("--profile", default="cw")
    args = parser.parse_args()
    fs = fsspec.filesystem(
        "s3",
        profile=args.profile,
        client_kwargs={"endpoint_url": "https://cwobject.com"},
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )
    root = output_root(args.cluster)
    local = HERE / "data/eval_val_periodic"
    membership = {
        r["stem"]: r
        for r in read_csv(
            HERE.parent / "exp245_evals_foldbench_held_out_monomers/data/eval_sets.csv"
        )
        if r["eval_set"] == "eval-val"
    }
    if len(membership) != 97:
        raise ValueError("Eval-val membership changed")
    for document_format in FORMATS:
        run = f"exp347-qwen35-4b-{document_format}-corpus235m-100bt"
        requests = sorted(
            (
                json.loads(value)
                for value in fs.cat(
                    fs.glob(f"{root}/eval_requests/{run}/step-*.json")
                ).values()
            ),
            key=lambda r: r["tokens"],
        )
        results = {
            result["step"]: result
            for value in fs.cat(
                fs.glob(f"{root}/eval_results/{run}/step-*.json")
            ).values()
            for result in [json.loads(value)]
        }
        completed, imported, pending = [], [], []
        for request in requests:
            result = results.get(request["step"])
            if result is None:
                output = f"{root}/eval-val/{run}/step-{request['step']}"
                pending.append(
                    {
                        "step": request["step"],
                        "tokens": request["tokens"],
                        "completed_proteins": len(fs.glob(output + "/*/complete.json")),
                    }
                )
                continue
            if (
                result["checkpoint"] != request["checkpoint"]
                or result["tokens"] != request["tokens"]
                or result["document_format"] != document_format
            ):
                raise ValueError("Evaluation does not match its request")
            if not result["output"].startswith(root + "/eval-val/"):
                raise ValueError("Evaluation output is outside the regional experiment")
            folder = local / document_format / f"step-{request['step']}"
            summary = folder / "summary.json"
            if not summary.exists():
                folder.mkdir(parents=True, exist_ok=True)
                output = result["output"]
                paths = [
                    output + "/" + name
                    for name in ("metrics.csv", "timings.csv", "dispatch.json")
                ]
                paths += fs.glob(output + "/provenance-shard-*.json")
                for path, value in fs.cat(paths).items():
                    (folder / path.rsplit("/", 1)[-1]).write_bytes(value)
                validate_artifacts(folder, result, membership)
                summary.write_text(json.dumps(result, indent=2) + "\n")
                imported.append(request["step"])
            else:
                saved = json.loads(summary.read_text())
                if saved != result:
                    raise ValueError(f"Committed result changed: {summary}")
                validate_artifacts(folder, result, membership)
            completed.append(result)
        print(
            json.dumps(
                {
                    "format": document_format,
                    "completed": len(completed),
                    "imported_steps": imported,
                    "latest": {
                        k: completed[-1][k] for k in ("step", "tokens", "r_precision")
                    },
                    "pending": pending,
                }
            ),
            flush=True,
        )
    write_strata(local, membership)


if __name__ == "__main__":
    main()
