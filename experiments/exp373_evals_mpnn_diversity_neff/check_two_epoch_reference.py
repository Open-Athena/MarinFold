"""Compare independent exp364 prompt draws with the matched exp373 evaluation."""

import hashlib
import io
import json
from pathlib import Path

from huggingface_hub import HfFileSystem
import numpy as np
import pandas as pd

from summarize import estimate

HERE = Path(__file__).resolve().parent
OUT = HERE / "data/two_epoch"
ROOT = (
    "buckets/open-athena/MarinFold/data/exp364-models-mpnn-only-lr-sweep/"
    "contact-eval/2026-11-02/final-01"
)


def main() -> None:
    """Verify identical scoring universes and retain per-protein reference scores."""
    ours = pd.read_csv(OUT / "metrics.csv")
    ours = ours[ours.temperature.eq(1) & ours["mode"].eq("randomized")]
    fs = HfFileSystem(token=False)
    comparisons, sources = [], []
    keys = ["stem", "eval_set", "range", "cut"]
    for trial in ["lr5e-4", "lr1e-3"]:
        model = "mpnn2-" + trial
        path = (
            f"{ROOT}/contacts-v1-exp364-mpnn-only-2ep-{trial}-1.5B-step-373768/"
            "contact_precision_all.csv"
        )
        content = fs.read_bytes(path)
        sources.append(
            dict(path="hf://" + path, sha256=hashlib.sha256(content).hexdigest())
        )
        ref = pd.read_csv(io.BytesIO(content))
        assert len(ref) == 2320 and ref["mode"].eq("single_seq").all()
        paired = ours[ours.model.eq(model)].merge(
            ref[keys + ["precision", "n_candidate", "n_true", "n_top"]],
            on=keys,
            validate="one_to_one",
            suffixes=("", "_exp364"),
        )
        assert len(paired) == 2320
        for column in ["n_candidate", "n_true", "n_top"]:
            np.testing.assert_array_equal(paired[column], paired[column + "_exp364"])
        paired["difference"] = paired.precision - paired.precision_exp364
        comparisons.append(paired)
    paired = pd.concat(comparisons, ignore_index=True)
    paired.to_csv(OUT / "reference_per_protein.csv", index=False)
    rows = []
    for (model, cohort, region), part in paired[
        paired.cut.eq("R") & paired["range"].isin(["all", "long"])
    ].groupby(["model", "eval_set", "range"]):
        rows.append(
            dict(
                model=model,
                eval_set=cohort,
                range=region,
                ours=part.precision.mean(),
                exp364=part.precision_exp364.mean(),
                **estimate(part.sort_values("stem").difference.to_numpy()),
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "reference_difference_summary.csv", index=False)
    (OUT / "reference_validation.json").write_text(
        json.dumps(
            dict(
                sources=sources,
                metric_rows_verified=len(paired),
                scoring_universes_identical=True,
                note=(
                    "Exp364 uses seed 0; exp373 uses its frozen paired SHA256 seeds. "
                    "Intervals resample proteins conditional on these two sets of "
                    "draws, not repeated independent evaluations."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
