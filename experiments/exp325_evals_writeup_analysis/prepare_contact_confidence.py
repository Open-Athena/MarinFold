"""Cache confidence-ranked P@L/5 summaries and paired changes for rendering."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import bootstrap, cohort_slices, sha256, summarize, TIERS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = {"plddt_product": "04c_pl5_plddt_neff", "plddt_product_degree": "04d_pl5_plddt_degree_neff"}


def main() -> None:
    """Preserve paired proteins and retain original score lineage in every row."""
    old = pd.read_csv(DATA / "neff_figure_rows.csv", float_precision="round_trip")
    old = old[old.figure == "04_contacts_pl5_neff"].copy()
    old["neff_analysis_row"] = old.index
    metrics = pd.read_csv(DATA / "contact_confidence_metrics.csv", float_precision="round_trip")
    if len(metrics) != 333*7*3*2 or metrics.duplicated(["stem","method","range","ranking"]).any():
        raise ValueError("Incomplete confidence ranking comparison")
    metrics["contact_confidence_row"] = np.arange(len(metrics))
    metrics["metric"] = metrics["range"].map({"all":"p_at_l5", "long":"p_at_l5_long"})
    measured = metrics.merge(old[["stem","method","metric","value","eval_set","designed","is_viral","tier","raw_depth_tier","msa_neff"]],
                             on=["stem","method","metric"], validate="many_to_one")
    if len(measured) != len(metrics) or not np.allclose(measured.original_precision, measured.value, atol=1e-12, rtol=0):
        raise ValueError("Original P@L/5 changed")
    parts = []
    for ranking, figure in FIGURES.items():
        control = old[old.method.isin(["marinfold","knn"])].copy()
        scores = metrics[metrics.ranking == ranking][["stem","method","metric","precision","contact_confidence_row"]]
        frame = old[~old.method.isin(["marinfold","knn"])].merge(scores, on=["stem","method","metric"], validate="one_to_one")
        frame["original_precision"] = frame.value
        frame["value"] = frame.precision
        part = pd.concat([frame, control], ignore_index=True)
        part["figure"], part["ranking"] = figure, ranking
        parts.append(part)
    rows = pd.concat(parts, ignore_index=True)
    rows.to_csv(DATA / "contact_confidence_figure_rows.csv", index=False)
    summary = summarize(rows)
    summary.to_csv(DATA / "contact_confidence_summary.csv", index=False)
    deltas = []
    for axis, tier_column in [("neff","tier"), ("msa_depth","raw_depth_tier")]:
        for cohort, subset in cohort_slices(measured).items():
            for tier in [*TIERS, "All depths"]:
                group = subset if tier == "All depths" else subset[subset[tier_column] == tier]
                for (method, region, ranking), part in group.groupby(["method","range","ranking"]):
                    part = part.sort_values("stem")
                    mean, lo, hi = bootstrap((part.precision-part.original_precision).to_numpy())
                    deltas.append(dict(axis=axis, cohort=cohort, tier=tier, method=method, range=region, ranking=ranking,
                                       n=len(part), mean=mean, ci_low=lo, ci_high=hi,
                                       original_mean=part.original_precision.mean(), new_mean=part.precision.mean(),
                                       n_improved=int((part.precision>part.original_precision+1e-12).sum()),
                                       n_worse=int((part.precision<part.original_precision-1e-12).sum()),
                                       shortfall=int((part.n_selected<part.k).sum()),
                                       stems="|".join(part.stem)))
    pd.DataFrame(deltas).to_csv(DATA / "contact_confidence_deltas.csv", index=False)
    files = ["contact_confidence_metrics.csv", "contact_confidence_sources.csv", "contact_confidence_validation.csv",
             "contact_confidence_figure_rows.csv", "contact_confidence_summary.csv", "contact_confidence_deltas.csv"]
    sources = ["data/neff_figure_rows.csv", "data/pl5_figure_rows.csv", "data/figure_rows.csv", "data/msa_neff.csv",
               "scratch/contact_confidence/sources.csv", "scratch/contact_confidence/archived_contacts.parquet",
               "scratch/contacts/inputs/gt_universe_scored.jsonl"]
    manifest = dict(files={name:sha256(DATA/name) for name in files},
                    sources={name:sha256(HERE/name) for name in sources},
                    code={name:sha256(HERE/name) for name in ["prepare_contact_confidence.py",
                          "generation/contact_confidence_sources.py", "generation/score_contact_confidence.py"]},
                    definition="Rank all ConFind degree>0 predicted contacts by q_i*q_j or q_i*q_j*degree; q=residue mean heavy-atom pLDDT normalized to [0,1]",
                    eligibility="Frozen resolved pair universe; separation >=6 (all) or >=24 (long); filter before selecting k=max(1,floor(input_L/5)); unfilled ranks receive zero credit",
                    truth="Frozen ConFind degree>=0.001; no ground truth in contact ranking; original structure selection unchanged",
                    ties="Stable input-sequence (i,j) ordering, same as original degree ranking",
                    checks="All 2331 structures reproduce archived P@L/5 and R-precision in both ranges; original degree contacts and scores retained",
                    uncertainty="Equal protein weights; 5000 paired bootstrap replicates; seed325",
                    scope="333 proteins, 314 natural and 19 designs separately; 7 structure arms; MarinFold and Sequence KNN scores unchanged")
    (DATA / "contact_confidence_analysis.json").write_text(json.dumps(manifest, indent=2)+"\n")
    output = pd.DataFrame(deltas)
    print(output[(output.axis=="neff") & (output.cohort=="natural") & output.tier.isin(["All depths","<10"])][
        ["tier","method","range","ranking","original_mean","new_mean","mean","ci_low","ci_high","shortfall"]].to_string(index=False))


if __name__ == "__main__":
    main()
