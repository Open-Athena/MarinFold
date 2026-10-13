"""Regroup frozen scores by measured Neff80, caching all intervals before rendering."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import Sources, bootstrap, cohort_slices, depth_tier, sha256, summarize
from theme import NEFF_FIGURES, ORDER

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def regroup(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Preserve scores and original tier while joining the exact alignment Neff."""
    result = frame.rename(columns={"tier": "raw_depth_tier"}).merge(
        targets[["stem", "msa_neff", "neff_tier", "neff_source_row"]], on="stem", validate="many_to_one")
    if len(result) != len(frame) or result.msa_neff.isna().any():
        raise ValueError("Neff join lost score rows")
    return result.rename(columns={"neff_tier": "tier"})


def main() -> None:
    """Write unchanged per-protein values with new grouping and source provenance."""
    sources = Sources()
    targets = pd.read_csv(DATA / "targets.csv", float_precision="round_trip")
    measurement = sources.csv(DATA / "msa_neff.csv", round_trip=True).rename(columns={"source_row": "neff_source_row"})
    targets = targets.merge(measurement.drop(columns="source"), on="stem", validate="one_to_one", suffixes=("", "_measured"))
    if len(targets) != 333 or targets.msa_neff.isna().any():
        raise ValueError("Require Neff for all 333 targets")
    if not (targets.msa_depth == targets.msa_depth_measured).all() or not (targets.msa_query_length == targets.msa_query_length_measured).all():
        raise ValueError("Neff was measured on different alignment counts or query lengths")
    if not np.isfinite(targets.msa_neff).all() or not targets.msa_neff.between(1 - 1e-10, targets.msa_depth).all():
        raise ValueError("Invalid Neff")
    targets["neff_tier"] = depth_tier(targets.msa_neff)
    targets = targets.rename(columns={"tier": "raw_depth_tier"})
    sources.records[str((DATA / "targets.csv").relative_to(HERE.parents[1]))] = {"sha256": sha256(DATA / "targets.csv")}
    targets.to_csv(DATA / "neff_targets.csv", index=False)
    parts = []
    for filename in ("figure_rows.csv", "pl5_figure_rows.csv", "pl2_figure_rows.csv"):
        frame = pd.read_csv(DATA / filename, float_precision="round_trip")
        sources.records[str((DATA / filename).relative_to(HERE.parents[1]))] = {"sha256": sha256(DATA / filename)}
        frame["analysis_source"] = f"data/{filename}"
        frame["analysis_source_row"] = np.arange(len(frame))
        frame = frame[frame.figure.isin(NEFF_FIGURES)]
        keep = [method in ORDER[figure] for figure, method in zip(frame.figure, frame.method, strict=True)]
        frame = regroup(frame.loc[keep], targets)
        frame["figure"] += "_neff"
        parts.append(frame)
    rows = pd.concat(parts, ignore_index=True)
    if rows.duplicated(["figure", "metric", "method", "stem"]).any():
        raise ValueError("Duplicate Neff figure scores")
    rows.to_csv(DATA / "neff_figure_rows.csv", index=False)
    summary = summarize(rows)
    summary.to_csv(DATA / "neff_summary.csv", index=False)
    migration = []
    for figure, frame in rows.groupby("figure"):
        for cohort, subset in cohort_slices(frame).items():
            for (raw_tier, neff_tier), group in subset.drop_duplicates("stem").groupby(["raw_depth_tier", "tier"]):
                migration.append(dict(figure=figure, cohort=cohort, raw_depth_tier=raw_tier,
                                      neff_tier=neff_tier, n=len(group), stems="|".join(sorted(group.stem))))
    pd.DataFrame(migration).to_csv(DATA / "neff_tier_migration.csv", index=False)

    # Extended AF3 sampling still covers its original five proteins, even though
    # the full Neff<10 tier is larger. Preserve per-cell n for explicit display.
    context = pd.read_csv(DATA / "af3_context_rows.csv", float_precision="round_trip")
    sources.records[str((DATA / "af3_context_rows.csv").relative_to(HERE.parents[1]))] = {"sha256": sha256(DATA / "af3_context_rows.csv")}
    context = regroup(context, targets)
    context.to_csv(DATA / "neff_af3_context_rows.csv", index=False)
    records = []
    for (method, tier), group in context.groupby(["method", "tier"], sort=True):
        group = group.sort_values("stem")
        mean, lo, hi = bootstrap(group.tm_score.to_numpy())
        records.append(dict(method=method, tier=tier, n=len(group), mean=mean, lo=lo, hi=hi,
                            hits_tm_80=int((group.tm_score >= 0.8).sum()),
                            uses_ground_truth=bool(group.uses_ground_truth.iloc[0]),
                            context_rows=json.dumps(group.context_row.tolist()), stems=json.dumps(group.stem.tolist())))
    pd.DataFrame(records).to_csv(DATA / "neff_af3_context_summary.csv", index=False)
    files = ["msa_neff.csv", "neff_targets.csv", "neff_figure_rows.csv", "neff_summary.csv",
             "neff_tier_migration.csv", "neff_af3_context_rows.csv", "neff_af3_context_summary.csv"]
    manifest = dict(sources=sources.records, files={name: sha256(DATA / name) for name in files},
                    preprocessors={name: sha256(HERE / name) for name in ("prepare_neff.py", "generation/measure_neff.py")},
                    msa_inputs_sha256=json.loads((DATA / "alphafold_inputs.json").read_text())["inputs_sha256"],
                    definition="Neff80=sum_i 1/count_j(identity(i,j)>=0.8); identity over columns non-gap in both; insertions removed; all original records retained",
                    tiers="[1,10), [10,100), [100,1000), [1000,infinity); same prespecified cutoffs as raw depth",
                    uncertainty="5000 protein-bootstrap replicates, seed325; matched populations within each figure",
                    prediction_changes="None: original metrics, checkpoint, ranks, contacts and selected structures reused",
                    af3_sampling="Only the original five raw-depth<10 proteins; Neff tier cell is explicitly labeled n=5",
                    old_exp260_table="Not used: its alignments differ from the frozen exp325 predictor inputs")
    (DATA / "neff_analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary[(summary.cohort == "natural") & (summary.tier == "<10") &
                  (summary.figure.isin(["04_contacts_pl2_neff", "05_folding_neff"])) &
                  summary.metric.isin(["p_at_l2", "gdt_ts"])][["figure", "metric", "method", "n", "mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
