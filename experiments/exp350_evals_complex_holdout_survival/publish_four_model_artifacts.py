"""Publish validated comparison data, figures, and predicted structures."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION = "hf://buckets/open-athena/MarinFold/data/evals/exp350_foldbench_pair_holdout/contact_eval_v1/four_model_v1"


def main() -> None:
    """Add reviewable results to the validated staging directory and upload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE / "data/four_model_v1/manifest.json").read_text())
    if not manifest["all_models_complete"]:
        raise ValueError("Comparison is incomplete")
    shutil.copytree(
        HERE / "data/four_model_v1", args.stage / "data", dirs_exist_ok=True
    )
    (args.stage / "plots").mkdir(exist_ok=True)
    for path in (HERE / "plots").glob("four_model_*"):
        if path.is_file():
            shutil.copy2(path, args.stage / "plots" / path.name)
    (
        args.stage / "README.md"
    ).write_text("""# Four-model contact comparison on FoldBench complexes

All 23 frozen complexes are predicted as whole dimers; figures use the 17 test complexes.
See plots/four_model_comparison.pdf and data/per_target.csv. Native resolved-residue masks
are shared by every model and both contact classes. The per-chain plots use the two
partners from these same complex predictions; no monomer-only targets are included.

MarinFold multichain: exp343 step 280154, 100 rollouts. MarinFold default: exp277
step 266344, chain A + ten glycines + chain B, 100 rollouts, linker excluded from scoring.
ESMFold2: native dimer, single sequence, best of five by model confidence.
AlphaFold3: native dimer, ColabFold paired/unpaired MSAs, no templates, one seed with
five diffusion samples, selected by the model's ranking_score. AF3 receives MSA information.

AlphaFold3 output is distributed with the output terms under each target directory.
Please cite Abramson et al., Accurate structure prediction of biomolecular interactions
with AlphaFold 3, Nature 630, 493–500 (2024), https://doi.org/10.1038/s41586-024-07487-w.
ESMFold2: https://huggingface.co/biohub/ESMFold2 and https://github.com/Biohub/esm.
FoldBench: https://github.com/BEAM-Labs/FoldBench.

Reproduction code is in Open-Athena/MarinFold, experiment 350, PR 351.
""")
    subprocess.run(
        [
            "uvx",
            "--from",
            "huggingface-hub>=2.1,<3",
            "hf",
            "buckets",
            "sync",
            str(args.stage),
            DESTINATION,
        ],
        check=True,
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()
