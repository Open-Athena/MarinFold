# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile the exp324 10k feature/performance table.

This script joins the local 10k manifest + biology annotations with per-protein
contact metrics from ESMFold2 and MarinFold rollout evaluations. ESMFold2 n=100
may be split across the original shard prefix plus a rescue prefix; rows are
deduplicated by stem/range/cut, preferring rescue rows when present.
"""

import argparse
import json
from pathlib import Path

import fsspec
import pandas as pd

ROOT = "s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample"
DEFAULT_ESMFOLD_ORIGINAL = f"{ROOT}/pdb_deduped_10k_n100_scored/score_shards"
DEFAULT_ESMFOLD_RESCUE = f"{ROOT}/pdb_deduped_10k_n100_scored_rescue_lesshalf_20260924/score_shards"
DEFAULT_MARINFOLD_PREFIX = f"{ROOT}/marinfold_10k_rollout_scores"

MARINFOLD_LABELS = [
    "delta-v2-step78499-n100",
    "exp117-e16-final-step35679-n100",
    "exp277-full-epoch-step266344-n100",
    "exp177-cv1-step71359-n100",
    "exp157-rope-delta-step71359-n100",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_manifest.csv"))
    parser.add_argument("--biology", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_biology_features.parquet"))
    parser.add_argument("--uniref-depth-proxy", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_uniref_depth_proxy.csv"))
    parser.add_argument("--esmfold-original", default=DEFAULT_ESMFOLD_ORIGINAL)
    parser.add_argument("--esmfold-rescue", default=DEFAULT_ESMFOLD_RESCUE)
    parser.add_argument("--marinfold-prefix", default=DEFAULT_MARINFOLD_PREFIX)
    parser.add_argument("--out", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_feature_performance_table.parquet"))
    parser.add_argument("--csv-out", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_feature_performance_table.csv"))
    parser.add_argument("--allow-incomplete-esmfold", action="store_true")
    return parser.parse_args()


def glob_existing(pattern: str) -> list[str]:
    if pattern.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        stripped = pattern.removeprefix("s3://")
        return sorted(f"s3://{path}" for path in fs.glob(stripped))
    return sorted(fsspec.filesystem("file").glob(pattern))


def read_esmfold_metrics(original_prefix: str, rescue_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    metric_paths = []
    meta_paths = []
    for source, prefix in [("original", original_prefix), ("rescue", rescue_prefix)]:
        for path in glob_existing(f"{prefix}/shard-*-of-*.contact_precision.csv"):
            # Ignore smoke shard from early testing.
            if "shard-000-of-001" in path:
                continue
            metric_paths.append((source, path))
        for path in glob_existing(f"{prefix}/shard-*-of-*.meta.csv"):
            if "shard-000-of-001" in path:
                continue
            meta_paths.append((source, path))

    metric_frames = []
    for source, path in metric_paths:
        frame = pd.read_csv(path)
        frame["source"] = source
        metric_frames.append(frame)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    if len(metrics):
        metrics["model"] = "esmfold2-n100"
        metrics["source_priority"] = metrics["source"].map({"original": 0, "rescue": 1}).fillna(0)
        metrics = metrics.sort_values("source_priority").drop_duplicates(["stem", "range", "cut"], keep="last")

    meta_frames = []
    for source, path in meta_paths:
        frame = pd.read_csv(path)
        frame["source"] = source
        meta_frames.append(frame)
    meta = pd.concat(meta_frames, ignore_index=True) if meta_frames else pd.DataFrame()
    if len(meta):
        meta["source_priority"] = meta["source"].map({"original": 0, "rescue": 1}).fillna(0)
        meta = meta.sort_values("source_priority").drop_duplicates(["stem"], keep="last")

    counts = {
        "metric_files": len(metric_paths),
        "meta_files": len(meta_paths),
        "metric_stems": int(metrics["stem"].nunique()) if len(metrics) else 0,
        "meta_stems": int(meta["stem"].nunique()) if len(meta) else 0,
    }
    return metrics, meta, counts


def read_marinfold_metrics(prefix: str, labels: list[str]) -> pd.DataFrame:
    frames = []
    for label in labels:
        path = f"{prefix}/{label}_eval/contact_precision.csv"
        try:
            frame = pd.read_csv(path)
        except FileNotFoundError:
            continue
        frame["model"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pivot_metrics(metrics: pd.DataFrame, model_prefix: str | None = None) -> pd.DataFrame:
    if not len(metrics):
        return pd.DataFrame(columns=["stem"])
    rows = metrics.copy()
    rows["metric_prefix"] = rows["model"].str.replace("-", "_", regex=False).str.replace(".", "_", regex=False)
    if model_prefix is not None:
        rows["metric_prefix"] = model_prefix
    rows["base"] = rows["metric_prefix"] + "_" + rows["range"] + "_" + rows["cut"].str.lower().str.replace("/", "", regex=False)
    prec = rows.pivot_table(index="stem", columns="base", values="precision", aggfunc="first")
    auc_rows = rows.drop_duplicates(["stem", "metric_prefix", "range"])
    auc_rows["base"] = auc_rows["metric_prefix"] + "_" + auc_rows["range"] + "_auc"
    auc = auc_rows.pivot_table(index="stem", columns="base", values="auc", aggfunc="first")
    out = pd.concat([prec, auc], axis=1).reset_index()
    out.columns.name = None
    return out


def main() -> int:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    biology = pd.read_parquet(args.biology)
    base = manifest.merge(biology.drop(columns=[c for c in ["pdb_id", "chain_id", "entry_id"] if c in biology.columns]), on="stem", how="left")
    if args.uniref_depth_proxy.exists():
        uniref = pd.read_csv(args.uniref_depth_proxy)
        base = base.merge(uniref, on="stem", how="left", suffixes=("", "_uniref"))

    esm_metrics, esm_meta, esm_counts = read_esmfold_metrics(args.esmfold_original, args.esmfold_rescue)
    if esm_counts["metric_stems"] < len(base) and not args.allow_incomplete_esmfold:
        raise SystemExit(f"ESMFold2 incomplete: {esm_counts['metric_stems']}/{len(base)} metric stems; pass --allow-incomplete-esmfold to write partial table")

    marinfold_metrics = read_marinfold_metrics(args.marinfold_prefix, MARINFOLD_LABELS)
    wide = base.merge(pivot_metrics(esm_metrics, "esmfold2_n100"), on="stem", how="left")
    wide = wide.merge(pivot_metrics(marinfold_metrics), on="stem", how="left")

    if len(esm_meta):
        meta_cols = [
            "stem", "n_samples", "mean_sample_confidence", "mean_pred_align_identity",
            "mean_pred_contacts_raw", "elapsed_seconds", "status", "source",
        ]
        keep = [c for c in meta_cols if c in esm_meta.columns]
        wide = wide.merge(esm_meta[keep].rename(columns={c: f"esmfold2_n100_{c}" for c in keep if c != "stem"}), on="stem", how="left")

    # Handy paired deltas for the headline R-precision-all comparisons.
    for left, right in [
        ("delta_v2_step78499_n100_all_r", "esmfold2_n100_all_r"),
        ("exp277_full_epoch_step266344_n100_all_r", "esmfold2_n100_all_r"),
        ("exp117_e16_final_step35679_n100_all_r", "esmfold2_n100_all_r"),
        ("delta_v2_step78499_n100_all_r", "exp117_e16_final_step35679_n100_all_r"),
        ("exp277_full_epoch_step266344_n100_all_r", "delta_v2_step78499_n100_all_r"),
    ]:
        if left in wide.columns and right in wide.columns:
            wide[f"{left}_minus_{right}"] = wide[left] - wide[right]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(args.out, index=False)
    wide.to_csv(args.csv_out, index=False)
    summary = {
        "n_rows": int(len(wide)),
        "esmfold2": esm_counts,
        "marinfold_metric_stems": int(marinfold_metrics["stem"].nunique()) if len(marinfold_metrics) else 0,
        "uniref_depth_proxy": str(args.uniref_depth_proxy) if args.uniref_depth_proxy.exists() else None,
        "stems_with_uniref50": int(wide["uniref50_size"].notna().sum()) if "uniref50_size" in wide.columns else 0,
        "out": str(args.out),
        "csv_out": str(args.csv_out),
    }
    (args.out.parent / "sample_10000_feature_performance_table.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
