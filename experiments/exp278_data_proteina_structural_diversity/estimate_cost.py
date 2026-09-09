"""Estimate one-million-document compute from measured stage times and retention.

The projection targets a uniform distribution over integer lengths 60–500. It
interpolates cost per retained document (rather than dividing global mean cost
by global mean yield), since expensive low-yield lengths require oversampling.
Screening-set duplicate rejection is not a forecast of million-scale novelty.
"""

import csv
import json
from pathlib import Path

import numpy as np

from analyze_screen import write_csv

HERE = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    sampling = read_csv(HERE / "data/sampling-timings.csv")
    rows = []
    cases = []
    for length in [60, 100, 200, 300, 400, 500]:
        report = HERE / f"data/l{length}-screen"
        retention = json.loads((report / "retention.json").read_text())
        folding = read_csv(report / "fold-timings.csv")
        warm = [
            float(row["elapsed_seconds"])
            for row in sampling
            if int(row["length"]) == length
            and row["warm_batch"] == "True"
            and row["known_cache_fallback"] == "False"
        ]
        design = [
            float(row["elapsed_seconds"])
            for row in folding
            if row["mode"] == "sequence_design"
        ]
        refold = [
            float(row["elapsed_seconds"]) for row in folding if row["mode"] == "refold"
        ]
        if not warm or not design or not refold or retention["after_cluster_cap"] == 0:
            raise ValueError(
                f"Cannot price length {length}: missing timings or zero yield"
            )
        observed_yield = retention["retained_fraction"]
        seconds = float(np.mean(warm) + np.mean(design) + np.mean(refold))
        retained_rows = read_csv(report / "retention.csv")
        for condition in ["unconditional", "1.x.x.x", "2.x.x.x", "3.x.x.x"]:
            members = [row for row in retained_rows if row["condition"] == condition]
            stems = {row["stem"] for row in members}
            retained = sum(row["selected"] == "True" for row in members)
            arm_warm = [
                float(row["elapsed_seconds"])
                for row in sampling
                if int(row["length"]) == length
                and row["cath"] == condition
                and row["warm_batch"] == "True"
                and row["known_cache_fallback"] == "False"
            ]
            arm_design = [
                float(row["elapsed_seconds"])
                for row in folding
                if row["stem"] in stems and row["mode"] == "sequence_design"
            ]
            arm_refold = [
                float(row["elapsed_seconds"])
                for row in folding
                if row["stem"] in stems and row["mode"] == "refold"
            ]
            arm_seconds = float(
                np.mean(arm_warm or warm) + np.mean(arm_design) + np.mean(arm_refold)
            )
            cases.append(
                {
                    "length": length,
                    "condition": condition,
                    "candidates": len(members),
                    "retained": retained,
                    "retained_fraction": retained / len(members),
                    "sampling_time_imputed_from_other_arms_at_same_length": not bool(
                        arm_warm
                    ),
                    "gpu_seconds_per_raw_candidate": arm_seconds,
                    "gpu_seconds_per_retained_document_20pct_overhead": arm_seconds
                    * len(members)
                    / retained
                    * 1.2
                    if retained
                    else None,
                }
            )
        rows.append(
            {
                "length": length,
                "candidates": retention["candidates"],
                "retained": retention["after_cluster_cap"],
                "retained_fraction": observed_yield,
                "sampling_warm_seconds": np.mean(warm),
                "mpnn_seconds": np.mean(design),
                "esmfold_seconds": np.mean(refold),
                "gpu_seconds_per_raw_candidate": seconds,
                "gpu_seconds_per_retained_document": seconds / observed_yield,
                "gpu_seconds_per_retained_document_20pct_overhead": seconds
                / observed_yield
                * 1.2,
            }
        )
    write_csv(HERE / "data/cost-by-length.csv", rows)
    write_csv(HERE / "data/cost-by-case.csv", cases)
    lengths = np.arange(60, 501)
    raw_seconds = np.interp(
        lengths,
        [row["length"] for row in rows],
        [row["gpu_seconds_per_raw_candidate"] for row in rows],
    )
    accepted_seconds = np.interp(
        lengths,
        [row["length"] for row in rows],
        [row["gpu_seconds_per_retained_document_20pct_overhead"] for row in rows],
    )
    hours = float(accepted_seconds.mean() * 1e6 / 3600)
    zero_yield_cases = [
        {"length": row["length"], "condition": row["condition"]}
        for row in cases
        if row["retained"] == 0
    ]
    balanced_hours = None
    if not zero_yield_cases:
        per_length = [
            np.mean(
                [
                    row["gpu_seconds_per_retained_document_20pct_overhead"]
                    for row in cases
                    if row["length"] == length
                ]
            )
            for length in [60, 100, 200, 300, 400, 500]
        ]
        balanced_hours = float(
            np.interp(lengths, [60, 100, 200, 300, 400, 500], per_length).mean()
            * 1e6
            / 3600
        )
    result = {
        "target_documents": 1000000,
        "target_length_distribution": "uniform integers 60..500; interpolation between six measured lengths",
        "primary_projection_class_mix": "observed surviving mixture from equal raw sampling across four requested arms; not balanced accepted classes",
        "checkpoint_assignment": "short checkpoint measured through 200; long checkpoint from 300; crossover remains unvalidated",
        "h100_hours_for_one_million_raw_pipeline_candidates_without_overhead": float(
            raw_seconds.mean() * 1e6 / 3600
        ),
        "h100_hours_for_one_million_retained_at_screen_yield_with_20pct_overhead": hours,
        "days_on_32_h100": hours / (32 * 24),
        "h100_hours_if_accepted_contributions_balanced_across_four_requested_arms": balanced_hours,
        "zero_yield_length_condition_bins": zero_yield_cases,
        "balanced_projection_note": "equal retained contribution from unconditional/alpha/beta/mixed requested arms; requested classes are not independently assigned CATH labels",
        "illustrative_accounting_dollars_at_2_per_h100_hour": hours * 2,
        "illustrative_accounting_dollars_at_4_per_h100_hour": hours * 4,
        "pricing_note": "illustrative rates, not vendor quotes; available fleet is prepaid",
        "important_limit": "retention and duplicate growth measured only on the screening set; no million-scale or diversity-target success is implied",
        "tf32": "separate 500-aa arm is excluded from this reference-precision projection",
        "extra_attempts": "one sequence design/refold per backbone; two-attempt pilot priced separately",
    }
    (HERE / "data/cost-projection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
