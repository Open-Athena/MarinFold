"""Record the coverage of existing similarity evidence against the training sources."""

import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main() -> None:
    """Restrict the available evidence to eval-val and expose redesign search gaps."""
    targets = pd.read_csv(HERE / "data/targets.csv")
    paths = {
        "native_residual_identity": REPO
        / "experiments/exp245_evals_foldbench_held_out_monomers/data/residual_identity.csv",
        "training_corpora": REPO
        / "experiments/exp277_models_single_mpnn_pilot/config.py",
        "training_counts": REPO
        / "experiments/exp277_models_single_mpnn_pilot/data/epoch_corpus_counts.csv",
        "knn_search": REPO
        / "experiments/exp245_evals_foldbench_held_out_monomers/run_knn_baseline.py",
        "msa_metadata": REPO
        / "experiments/exp260_evals_msa_depth_stratified/data/msa_depth.csv",
    }
    residual = pd.read_csv(paths["native_residual_identity"])
    residual = residual[residual.eval_set.eq("eval-val")]
    assert len(residual) == 97 and set(residual.stem) == set(
        targets.loc[targets.eval_set.eq("eval-val"), "stem"]
    )
    counts = pd.read_csv(paths["training_counts"])
    rows = []
    for row in counts.to_dict("records"):
        redesigned = row["corpus"].startswith("mpnn")
        rows.append(
            {
                **row,
                "in_default": True,
                "in_mpnn_only": redesigned,
                "eval_val_sequence_search": "unmeasured"
                if redesigned
                else "exp245 historical residual alignment audit",
                "sequence_knn_coverage": "exp245 AFDB-only native null"
                if row["corpus"] == "native-afdb"
                else "unmeasured",
            }
        )
    pd.DataFrame(rows).to_csv(
        HERE / "data/training_similarity_coverage.csv", index=False
    )
    summaries = []
    for coverage in ["50", "40", "30", "20", "00"]:
        col = "best_surviving_identity_cov" + coverage
        values = residual[col].dropna()
        summaries.append(
            {
                "minimum_coverage": float(coverage) / 100,
                "n_proteins": 97,
                "n_with_reported_surviving_match": len(values),
                "median_best_identity": values.median(),
                "maximum_best_identity": values.max(),
                "n_identity_ge_0.3": int(values.ge(0.3).sum()),
                "n_identity_ge_0.9": int(values.ge(0.9).sum()),
            }
        )
    pd.DataFrame(summaries).to_csv(
        HERE / "data/native_similarity_coverage.csv", index=False
    )
    report = {
        "sources": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths.values()
        },
        "natural_eval_val": 97,
        "native_search": "Existing exp245 alignments against native AFDB and ESM-Atlas after the exp225 removal rule (identity >=0.3 over >=0.5 of shorter sequence). Historical evidence; final token-cache rows were not independently searched again.",
        "redesign_search": "No complete query-versus-redesign-sequence search or redesign-corpus KNN null was found in the available repository artifacts. Coverage is unmeasured for all 162,574,724 redesign documents, not zero similarity.",
        "native_knn": "Existing decontaminated sequence-KNN null covers AFDB native documents only, excluding ESM-Atlas and all redesigns. It is contextual evidence, not a matched null for either complete recipe.",
        "ancestry": "Both models trained on the same redesign sources inherited from decontaminated native backbones. Removing native sequence documents does not remove structural ancestry or prove absence of local sequence matches.",
        "neff_join": "One-to-one stem join to the FoldBench-volume exp260 depth table. Evaluation sequence equality and MSA query length are checked against exp245. Original A3M query sequences were not re-fetched for a new identity audit.",
        "caveat": "Neither Neff nor native similarity substitutes for similarity to the actual redesign sequences. Training exposure/schedules and one selected checkpoint per recipe prevent a causal isolation of native-document removal.",
    }
    (HERE / "data/similarity_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
