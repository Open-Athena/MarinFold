"""Cache P@L/2 predictor comparisons from the same archived evaluations as P@L/5.

No predictor runs or contact rankings change. Select each source's L/2 scores,
verify its R-precision against the original panel, and bootstrap proteins once.
Renderers consume the resulting small tables without repeating this analysis.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import Sources, sha256, summarize
from prepare_pl5 import contact_sources

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main() -> None:
    """Write source-traced L/2 precision rows and cached tier summaries."""
    sources = Sources()
    targets = sources.csv(DATA / "targets.csv", round_trip=True).drop(columns=["source", "source_row"])
    rows = contact_sources(sources, targets, divisor=2).assign(figure="04_contacts_pl2")
    rows["requested_n_top"] = np.maximum(1, rows.L // 2)
    if not (rows.n_top == np.minimum(rows.requested_n_top, rows.n_candidate)).all():
        raise ValueError("L/2 source cutoff disagrees with sequence length and candidate count")
    natural = rows[rows.designed == 0]
    if not (natural.n_top == natural.requested_n_top).all():
        raise ValueError("Every natural-protein L/2 score must use the full sequence-length cutoff")
    if rows.value.isna().any() or not rows.value.between(0, 1).all():
        raise ValueError("Invalid P@L/2 scores")
    if rows.duplicated(["stem", "method", "metric"]).any():
        raise ValueError("Duplicate P@L/2 scores")
    rows.to_csv(DATA / "pl2_figure_rows.csv", index=False)
    summary = summarize(rows)
    summary.to_csv(DATA / "pl2_summary.csv", index=False)
    names = ["pl2_figure_rows.csv", "pl2_summary.csv"]
    manifest = dict(
        preprocessors={name: sha256(HERE / name) for name in ("prepare_pl2.py", "prepare_pl5.py", "prepare.py")},
        sources=sources.records,
        files={name: sha256(DATA / name) for name in names},
        checkpoint="contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344",
        cutoff="k=min(max(1,floor(L/2)),n_candidate); L=frozen input sequence length, including unresolved input residues; all natural rows use full L/2",
        capped_rows=rows.loc[rows.n_top < rows.requested_n_top,
                            ["stem", "method", "metric", "L", "requested_n_top", "n_top", "n_candidate"]].to_dict("records"),
        contact_definition="pyconfind degree >=0.001; resolved pairs; sequence separation >=6 (all) or >=24 (long)",
        ranking="Same archived scores as P@L/5: structural contact degree, MarinFold consensus votes, and sequence-KNN transferred scores; stable candidate-order ties",
        population="314 natural proteins (97 validation, 217 test); tiers 5/21/62/226; 19 designs separate",
        uncertainty="Equal-weight protein means; 5000 protein-bootstrap replicates, seed325",
        regression="All 5994 original predictor R-precision cells reproduced within 1e-12 before selecting L/2 scores",
    )
    (DATA / "pl2_analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary[(summary.cohort == "natural") & (summary.tier == "All depths")]
          [["metric", "method", "n", "mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
