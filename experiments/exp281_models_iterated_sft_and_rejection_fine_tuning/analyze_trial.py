"""Collect the bounded W&B report and audit the preregistered step-256 format gate.

Download open-athena/MarinFold/exp281-trial-s03-report:v0 with the W&B API,
then pass its directory. Raw trajectories stay in the artifact; small per-input
diagnostics, metrics, and captured timings are preserved for public review.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq
from marinfold.document_structures.contacts_v1_multi import BEGIN, END, FINAL, parse_history


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write heterogeneous timing records without dropping worker metadata."""
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def collect(report: Path, output: Path, run_slug: str, checkpoint_step: int, source_commit: str,
            bootstrap_targets: int) -> dict[str, Any]:
    """Validate report coverage and calculate format gates over all completions."""
    index = json.loads((report / "index.json").read_text())
    candidates: dict[str, list[dict]] = {"natural": [], "forced": []}
    timings = []
    bootstrap = Counter()
    for name, uri in sorted(index.items()):
        path = report / name
        if "/eval/" in uri:
            mode = uri.split("/eval/")[1].split("/")[0]
            if uri.endswith(".parquet"):
                candidates[mode].extend(pq.read_table(path).to_pylist())
            if "/metrics." in uri:
                (output / f"{run_slug}_{mode}_metrics{path.suffix}").write_text(path.read_text())
        if uri.endswith("-timings.csv"):
            with path.open() as handle:
                timings.extend({**row, "source_uri": uri} for row in csv.DictReader(handle))
        if "/eval/" not in uri and "/candidates/" in uri and uri.endswith(".json"):
            meta = json.loads(path.read_text())
            for key in ("targets", "candidates", "bootstrap_rejected_samples", "bootstrap_empty_samples", "invalid"):
                bootstrap[key] += meta[key]
    if bootstrap["targets"] != bootstrap_targets or bootstrap["candidates"] != bootstrap_targets:
        raise ValueError(f"incomplete bootstrap report: {bootstrap}")
    if len(timings) != bootstrap_targets + 25 + 25:
        raise ValueError(f"incomplete timing report: {len(timings)} rows")
    write_csv(output / f"{run_slug}_timings.csv", timings)
    diagnostics = []
    result: dict[str, Any] = {"checkpoint_step": checkpoint_step, "source_commit": source_commit,
                              "bootstrap": dict(bootstrap), "modes": {}}
    for mode, records in candidates.items():
        counts = Counter(row["target_id"] for row in records)
        keys = {(row["target_id"], row["candidate_id"]) for row in records}
        if len(counts) != 25 or set(counts.values()) != {8} or len(keys) != 200:
            raise ValueError(f"incomplete/duplicated {mode} evaluation: {counts}")
        mode_rows = []
        for row in sorted(records, key=lambda r: (r["target_id"], r["candidate_id"])):
            if row["bootstrap"] or row["forced"] != (mode == "forced"):
                raise ValueError("evaluation contains bootstrap or wrong-mode candidates")
            parsed = parse_history(row["generated"]) if row["valid"] else None
            diagnostic = {
                "mode": mode, "target_id": row["target_id"], "candidate_id": row["candidate_id"],
                "n_residues": row["n_residues"], "budget": row["budget"],
                "valid": row["valid"], "error": row["error"],
                "final_marker_present": FINAL in row["generated"],
                "terminated": bool(row["generated"]) and row["generated"][-1] == END,
                "generated_tokens": len(row["generated"]),
                "raw_hypothesis_markers": row["generated"].count(BEGIN),
                "hypotheses": len(parsed.hypotheses) if parsed else 0,
                "nonempty_hypotheses": sum(bool(h) for h in parsed.hypotheses) if parsed else 0,
                "final_contacts": len(parsed.final or ()) if parsed else 0,
                "final_f1": row["score"]["f1"] if row["valid"] else 0.0,
            }
            mode_rows.append(diagnostic)
        diagnostics.extend(mode_rows)
        valid = sum(row["valid"] for row in mode_rows)
        multi = sum(row["nonempty_hypotheses"] >= 2 for row in mode_rows)
        result["modes"][mode] = {
            "targets": 25, "completions": len(records), "valid": valid,
            "valid_fraction": valid / len(records), "multiple_nonempty_hypotheses": multi,
            "multi_fraction": multi / len(records),
            "marker_present": sum(row["final_marker_present"] for row in mode_rows),
            "terminated": sum(row["terminated"] for row in mode_rows),
            "macro_final_f1": mean(row["final_f1"] for row in mode_rows),
            "mean_hypotheses": mean(row["hypotheses"] for row in mode_rows),
            "raw_multiple_hypothesis_markers": sum(row["raw_hypothesis_markers"] >= 2 for row in mode_rows),
            "errors": dict(Counter(row["error"].split(" at history offset")[0]
                                   for row in mode_rows if not row["valid"])),
            "budget_target_counts": dict(sorted(Counter(budget for _, budget in
                                           {(row["target_id"], row["budget"]) for row in records}).items())),
            "passes_validity_gate": valid >= 198,
        }
        with (output / f"{run_slug}_{mode}_metrics.csv").open() as handle:
            metrics = list(csv.DictReader(handle))
        metric_valid = sum(round(float(row["valid_fraction"]) * int(row["candidates"])) for row in metrics)
        metric_multi = sum(round(float(row["multi_fraction"]) * int(row["candidates"])) for row in metrics)
        metric_f1 = mean(float(row["final_f1"]) for row in metrics)
        if (metric_valid, metric_multi) != (valid, multi) or abs(metric_f1 - result["modes"][mode]["macro_final_f1"]) > 1e-12:
            raise ValueError("raw trajectories disagree with independently scored metrics")
    natural_targets = {row["target_id"] for row in candidates["natural"]}
    if natural_targets != {row["target_id"] for row in candidates["forced"]}:
        raise ValueError("natural and forced evaluation targets differ")
    result["passes_format_gate"] = (
        all(mode["passes_validity_gate"] for mode in result["modes"].values())
        and result["modes"]["natural"]["multiple_nonempty_hypotheses"] >= 180
    )
    write_csv(output / f"{run_slug}_diagnostics.csv", diagnostics)
    (output / f"{run_slug}_format_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--run-slug", default="trial_s03")
    parser.add_argument("--checkpoint-step", type=int, default=256)
    parser.add_argument("--source-commit", default="bab3f50a")
    parser.add_argument("--bootstrap-targets", type=int, default=2048)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(collect(args.report, args.output, args.run_slug, args.checkpoint_step, args.source_commit,
                             args.bootstrap_targets), indent=2))


if __name__ == "__main__":
    main()
