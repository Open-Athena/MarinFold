"""Reduce new checkpoints with the original scorers and retain all four models."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_stats import diversity, thin_matched
from analyze import analyze_one, candidate_sets, load
from extend_analysis import reduce_record

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = DATA / "two_epoch"


def reduce_one(task: tuple[str, dict, list[int]]) -> dict:
    """Cache complete protein-level reductions keyed by raw and analysis sources."""
    model, truth, grid = task
    stem = truth["stem"]
    sources = sorted((HERE / "scratch/raw_two_epoch" / model).glob(stem + "-*.json.gz"))
    assert len(sources) == 6
    sha = hashlib.sha256()
    for path in sources + [
        Path(__file__),
        HERE / "analyze.py",
        HERE / "extend_analysis.py",
        HERE / "analysis_stats.py",
        HERE / "contacts.py",
        HERE / "inputs/ground_truth.json",
    ]:
        sha.update(path.read_bytes())
    cache = HERE / "scratch/two_epoch_reduced" / f"{model}-{stem}.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["sha256"] == sha.hexdigest():
            return saved["tables"]
    tables = {
        k: []
        for k in [
            "diversity",
            "metrics",
            "ensemble_curves",
            "single_rollout_summary",
            "count_controls",
            "ensemble_curves_1000",
            "rollout_f1_1000",
        ]
    }
    for path in sources:
        full = load(path)
        record = {**full, "samples": full["samples"][:300]}
        sets, true, ids, labels = candidate_sets(record, truth)
        div, curves, diagnostics, metrics = analyze_one(
            record, truth, sets, true, ids, labels
        )
        tables["diversity"].extend(div)
        tables["ensemble_curves"].extend(curves)
        tables["metrics"].extend(metrics)
        groups = ["eval_set", "model", "stem", "temperature", "mode"]
        tables["single_rollout_summary"].extend(
            pd.DataFrame(diagnostics)
            .groupby(groups, as_index=False)
            .mean(numeric_only=True)
            .to_dict("records")
        )
        reference = load(HERE / "scratch/raw/default" / path.name)
        reference_sets, _, _, _ = candidate_sets(reference, truth)
        for region, minimum in [("all", 6), ("long", 24)]:
            a, b = thin_matched(
                *[
                    [
                        {x for x in s if x % truth["L"] - x // truth["L"] >= minimum}
                        for s in samples
                    ]
                    for samples in [reference_sets, sets]
                ],
                373,
            )
            stats_a, stats_b = (
                diversity(a, truth["L"] ** 2),
                diversity(b, truth["L"] ** 2),
            )
            tables["count_controls"].append(
                dict(
                    stem=stem,
                    model=model,
                    reference="default",
                    range=region,
                    mode=record["mode"],
                    temperature=record["temperature"],
                    eval_set=record["target"]["eval_set"],
                    jaccard_difference=stats_b["jaccard"] - stats_a["jaccard"],
                )
            )
        if full["temperature"] == 1 and full["mode"] == "fixed":
            curves_1000, f1 = reduce_record(full, truth, grid)
            tables["ensemble_curves_1000"].extend(curves_1000)
            tables["rollout_f1_1000"].extend(f1)
    cache.write_text(json.dumps({"sha256": sha.hexdigest(), "tables": tables}))
    return tables


def panel(models: list[str], truths: list[dict], stems: list[str]) -> None:
    """Preserve the frozen examples and expose the same 300-sample contact maps."""
    rows = []
    for truth in truths:
        stem, length = truth["stem"], truth["L"]
        if stem not in stems:
            continue
        matrices = {}
        for model in models:
            raw = (
                HERE
                / "scratch"
                / ("raw" if model in ["default", "mpnn"] else "raw_two_epoch")
            )
            record = load(raw / model / f"{stem}-t1.0-fixed.json.gz")
            record = {**record, "samples": record["samples"][:300]}
            sets, true, _, _ = candidate_sets(record, truth)
            counts = np.zeros(length**2)
            for sample in sets:
                if sample:
                    counts[list(sample)] += 1
            matrices[model] = counts / len(sets)
        visible = set.union(true, *(set(np.flatnonzero(v)) for v in matrices.values()))
        rows.extend(
            dict(
                stem=stem,
                i=int(pair // length),
                j=int(pair % length),
                truth=int(pair in true),
                **{model: values[pair] for model, values in matrices.items()},
            )
            for pair in sorted(visible)
        )
    pd.DataFrame(rows).to_csv(
        OUT / "panel_contacts.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


def main() -> None:
    """Join new reductions to immutable original source tables without resampling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-ready", action="store_true")
    args = parser.parse_args()
    plan = json.loads((DATA / "two_epoch_plan.json").read_text())
    if not args.cache_ready:
        validation = json.loads((DATA / "validation_two_epoch.json").read_text())
        assert validation["samples"] == 440800
    truths = json.loads((HERE / "inputs/ground_truth.json").read_text())
    grid = json.loads((DATA / "extension_plan.json").read_text())["discovery_k"]
    tasks = [(model, truth, grid) for truth in truths for model in plan["models"]]
    if args.cache_ready:
        tasks = [
            (model, truth, grid)
            for model, truth, grid in tasks
            if len(
                list(
                    (HERE / "scratch/raw_two_epoch" / model).glob(
                        truth["stem"] + "-*.json.gz"
                    )
                )
            )
            == 6
        ]
    (HERE / "scratch/two_epoch_reduced").mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    tables = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(reduce_one, tasks), 1):
            for name, rows in result.items():
                tables.setdefault(name, []).extend(rows)
            if index % 10 == 0:
                print("REDUCED", index, "/", len(tasks), flush=True)
    if args.cache_ready:
        print(
            "Cached complete conditions; final tables require full validation",
            flush=True,
        )
        return
    for name, rows in tables.items():
        new = pd.DataFrame(rows)
        suffix = ".csv.gz" if name == "rollout_f1_1000" else ".csv"
        old = pd.read_csv(DATA / (name + suffix))
        if name == "rollout_f1_1000":
            new = new[old.columns]
        if name == "count_controls":
            keys = ["stem", "eval_set", "temperature", "mode", "range"]
            old = old.pivot(index=keys, columns="model", values="jaccard")
            old = (
                old.assign(
                    jaccard_difference=old.mpnn - old.default,
                    model="mpnn",
                    reference="default",
                )
                .reset_index()
                .drop(columns=["default", "mpnn"])
            )
        combined = pd.concat([old, new], ignore_index=True)
        combined.to_csv(
            OUT / (name + suffix),
            index=False,
            compression={"method": "gzip", "mtime": 0}
            if suffix.endswith("gz")
            else None,
        )
    models = ["default", "mpnn", *plan["models"]]
    metrics = pd.read_csv(OUT / "metrics.csv")
    assert len(metrics) == 2784 * 20
    assert metrics.groupby(["model", "stem", "temperature", "mode"]).size().eq(20).all()
    curves = pd.read_csv(OUT / "ensemble_curves_1000.csv")
    assert len(curves) == 464 * 2 * len(grid)
    short = pd.read_csv(OUT / "ensemble_curves.csv")
    keys = ["stem", "model", "temperature", "mode", "range", "k"]
    a = (
        short[short.temperature.eq(1) & short["mode"].eq("fixed")]
        .set_index(keys)
        .sort_index()
    )
    b = curves[curves.k.le(300)].set_index(keys).sort_index()
    assert a.index.equals(b.index)
    for endpoint in [
        "r_precision",
        "true_coverage",
        "true_discovered",
        "false_discovered",
        "discovery_precision",
        "n_true",
        "n_candidate",
    ]:
        np.testing.assert_allclose(a[endpoint], b[endpoint], rtol=0, atol=1e-12)
    panel(models, truths, plan["representative_panel"])
    result = dict(
        models=models,
        metric_rows=len(metrics),
        first_300_curve_rows_verified=len(a),
        f1_scores=len(tables["rollout_f1_1000"]) + 464000,
        old_tables_reused=True,
    )
    (OUT / "analysis_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
