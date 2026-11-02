"""Summarize archived same-set pyconfind controls; no new predictor runs."""

import pandas as pd
from analyze import CUTS, bootstrap
from prepare_inputs import DATA, EXPERIMENTS, digest


def main() -> None:
    """Save a metric-matched reference table and paired checkpoint differences."""
    current = pd.read_csv(DATA / "per_protein.csv.gz")
    current = current[
        current.definition.eq("pyconfind") & current.cut.isin(CUTS)
    ].copy()
    stems = set(current.stem)
    current["predictor"] = "exp277 step266344 (current default)"
    sources = {
        "exp232 step363000 (previous default)": (
            EXPERIMENTS
            / "exp232_sweep_cv1_decontam/evals/2026-08-24_rollout_v2/data/coreweave_results/marinfold_precision.csv.gz",
            "marinfold-exp232-decontam-train-m2-p06-step363000",
        ),
        "seq-KNN decontaminated native corpus": (
            EXPERIMENTS
            / "exp245_evals_foldbench_held_out_monomers/data/knn_precision_new.csv.gz",
            "seq-knn-k10-decontam",
        ),
    }
    frames = [current[["predictor", "stem", "range", "cut", "precision"]]]
    source_rows = []
    for name, (path, model) in sources.items():
        frame = pd.read_csv(path)
        frame = frame[
            frame.dataset.eq("foldbench_monomer")
            & frame.stem.isin(stems)
            & frame.model.eq(model)
            & frame.cut.isin(CUTS)
        ].copy()
        assert len(frame) == 97 * 12 and frame.stem.nunique() == 97
        frame["predictor"] = name
        frames.append(frame[["predictor", "stem", "range", "cut", "precision"]])
        source_rows.append(
            {
                "predictor": name,
                "source": str(path.relative_to(EXPERIMENTS.parent)),
                "sha256": digest(path.read_bytes()),
            }
        )
    joined = pd.concat(frames, ignore_index=True)
    joined.to_csv(
        DATA / "native_reference_per_protein.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    rows = []
    for (name, span, cut), group in joined.groupby(["predictor", "range", "cut"]):
        low, high = bootstrap(group.precision.to_numpy(), 20000)
        rows.append(
            {
                "predictor": name,
                "definition": "pyconfind",
                "range": span,
                "cut": cut,
                "n": len(group),
                "mean": group.precision.mean(),
                "ci_low": low,
                "ci_high": high,
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "native_reference_summary.csv", index=False)
    deltas = []
    for name in sources:
        reference = joined[joined.predictor.eq(name)]
        paired = current.merge(
            reference,
            on=["stem", "range", "cut"],
            suffixes=("", "_reference"),
            validate="one_to_one",
        )
        for (span, cut), group in paired.groupby(["range", "cut"]):
            values = (group.precision - group.precision_reference).to_numpy()
            low, high = bootstrap(values, 20000)
            deltas.append(
                {
                    "reference": name,
                    "range": span,
                    "cut": cut,
                    "n": len(values),
                    "delta": values.mean(),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    pd.DataFrame(deltas).to_csv(DATA / "native_reference_deltas.csv", index=False)
    pd.DataFrame(source_rows).to_csv(DATA / "native_reference_sources.csv", index=False)
    print(pd.DataFrame(rows).query("range == 'long'").to_string(index=False))


if __name__ == "__main__":
    main()
