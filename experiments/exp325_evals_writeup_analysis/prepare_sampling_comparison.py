"""Join cached L/2 sampling and three-way-ranked predictor scores for display.

No predictions, scores or bootstrap intervals are recomputed. Require the same
314 natural proteins for every method and preserve source-row pointers for
both individual scores and the plotted means/intervals.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import sha256

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STANDARD_METHODS = ["af3", "af2", "boltz2", "protenix_msa", "esmfold2", "esmfold", "protenix_ss"]
SAMPLING_METHODS = ["single", "consensus", "best100"]


def main() -> None:
    """Cache matched-cohort comparison rows with unchanged means and intervals."""
    inputs = {}
    rows, summaries = [], []
    for manifest_name, rows_name, summary_name, figure, methods in [
        ("pl2_sampling_analysis.json", "pl2_sampling_figure_rows.csv", "pl2_sampling_summary.csv",
         "06_sampling_pl2", SAMPLING_METHODS),
        ("poster_product_analysis.json", "poster_product_figure_rows.csv", "poster_product_summary.csv",
         "04_contacts_pl2_neff", STANDARD_METHODS),
    ]:
        manifest = json.loads((DATA / manifest_name).read_text())
        inputs[manifest_name] = sha256(DATA / manifest_name)
        for name in [rows_name, summary_name]:
            inputs[name] = sha256(DATA / name)
            if inputs[name] != manifest["files"][name]:
                raise ValueError(f"Changed cached input: {name}")
        source = pd.read_csv(DATA / rows_name, float_precision="round_trip")
        selected = source[(source.figure == figure) & (source.designed == 0) & source.method.isin(methods)].copy()
        selected["source_figure_rows"] = rows_name
        selected["source_figure_row"] = selected.index
        selected["figure"] = "06_sampling_pl2"
        rows.append(selected[["figure", "stem", "method", "metric", "value", "L", "n_top", "eval_set",
                              "msa_neff", "tier", "source_figure_rows", "source_figure_row"]])
        source_summary = pd.read_csv(DATA / summary_name, float_precision="round_trip")
        selected_summary = source_summary[(source_summary.figure == figure) & source_summary.method.isin(methods)
                                          & (source_summary.cohort == "natural")
                                          & (source_summary.tier == "All depths")].copy()
        selected_summary["source_summary"] = summary_name
        selected_summary["source_summary_row"] = selected_summary.index
        selected_summary["figure"] = "06_sampling_pl2"
        summaries.append(selected_summary)
    combined = pd.concat(rows, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    if len(combined) != 314 * 10 * 2 or combined.duplicated(["stem", "method", "metric"]).any():
        raise ValueError("Incomplete or duplicate comparison rows")
    if len(summary) != 20 or summary.duplicated(["method", "metric"]).any():
        raise ValueError("Incomplete or duplicate comparison means")
    stems = set(combined.loc[combined.method == "consensus", "stem"])
    for row in summary.itertuples():
        group = combined[(combined.method == row.method) & (combined.metric == row.metric)]
        if set(group.stem) != stems or set(row.stems.split("|")) != stems or row.n != 314:
            raise ValueError(f"Unmatched comparison cohort: {row.method}/{row.metric}")
        if not np.isclose(row.mean, group.value.mean(), atol=1e-12, rtol=0):
            raise ValueError(f"Cached mean differs: {row.method}/{row.metric}")
    if not (combined.n_top == combined.L // 2).all():
        raise ValueError("Comparison does not use the same L/2 denominator")
    combined.to_csv(DATA / "pl2_sampling_comparison_rows.csv", index=False)
    summary.to_csv(DATA / "pl2_sampling_comparison_summary.csv", index=False)
    names = ["pl2_sampling_comparison_rows.csv", "pl2_sampling_comparison_summary.csv"]
    manifest = dict(inputs=inputs, files={name: sha256(DATA / name) for name in names},
                    preprocessor_sha256=sha256(Path(__file__)),
                    methods=SAMPLING_METHODS + STANDARD_METHODS,
                    cohort="Same 314 natural proteins, 97 validation + 217 test, in every arm",
                    standard_ranking="pLDDT_i * pLDDT_j * ConFind degree; same selected archived structures",
                    sampling="Original 100 iid samples; sampling consensus policy unchanged; oracle uses ground truth",
                    uncertainty="Existing equal-protein means and 95% bootstrap intervals copied exactly",
                    validation="Every mean, interval and individual score retains its cached source row; no new scoring")
    (DATA / "pl2_sampling_comparison_analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary[["method", "metric", "n", "mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
