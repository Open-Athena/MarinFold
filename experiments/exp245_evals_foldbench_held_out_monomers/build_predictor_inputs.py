# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 5 -- the inputs the baseline predictors need, and what can be reused.

Five baselines are reported beside the checkpoints: Protenix-v2 single-sequence
and +MSA, ESMFold, ESMFold2, and the sequence-KNN null. 126 of the 333 scored
monomers already have all five, published by #74 / #78 / #94 / #213 / #226; the
rest have never been predicted by anything.

**Reuse is gated on the input sequence, not the stem.** A published score is
only this experiment's score if the predictor was given the same sequence and
the metric was computed against the same ground truth. Both hold exactly for
the units whose frozen ground-truth record ``build_ground_truth.py`` reproduced
byte-for-byte; the eleven ``denovo_pdb`` units built from exp65's own construct
sequences are *not* reusable and are re-predicted here, even though a row with
their stem exists upstream.

Writes, for the units that need running:

``data/predictor_manifest_new.csv``
    ``dataset, stem, gt_cif, gt_chain, input_seq, n_residues`` -- the
    eval-manifest shape exp74's and exp78's Modal drivers and their pyconfind
    scorers consume, with ``gt_cif`` relative to the mmCIF cache.
``data/predictor_targets_new.parquet``
    ``dataset, stem, L, input_seq`` -- what exp94's KNN driver reads.
``data/baseline_reuse.csv``
    Every scored unit, with the source its baseline scores come from.

    uv run python build_predictor_inputs.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import upstream as U

DATA = U.DATA
MANIFEST = DATA / "predictor_manifest_new.csv"
TARGETS = DATA / "predictor_targets_new.parquet"
REUSE = DATA / "baseline_reuse.csv"
SUMMARY = DATA / "baseline_reuse.summary.json"


def reusable_stems() -> dict[str, str]:
    """``{stem: frozen dataset}`` for units whose upstream scores still apply.

    Taken from the ground-truth control: a unit qualifies when the frozen record
    was built from the same input sequence *and* reproduced identically here.
    """
    report = json.loads((DATA / "gt_report.json").read_text())
    control = report["control"]
    if not control["passed"]:
        raise SystemExit("ground-truth control did not pass; fix that first")
    variants = {record["stem"] for record in control["sequence_variants"]}
    manifest = pd.read_csv(DATA / "gt_manifest.csv")
    return {
        row.stem: "frozen"
        for row in manifest.itertuples()
        if row.in_frozen_universe == 1 and row.stem not in variants
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-root", type=Path, default=U.WORK / "cif",
                        help="where build_ground_truth.py cached the assembly mmCIFs")
    args = parser.parse_args()

    sets = pd.read_csv(DATA / "eval_sets.csv")
    scored = sets[sets.scorable == 1]
    gt_manifest = pd.read_csv(DATA / "gt_manifest.csv").set_index("stem")
    reuse = reusable_stems()

    rows = []
    for row in scored.itertuples():
        record = gt_manifest.loc[row.stem]
        rows.append({
            "eval_set": row.eval_set, "dataset": record.dataset, "stem": row.stem,
            "source": "published" if row.stem in reuse else "new",
            "gt_cif": record.gt_cif, "gt_chain": record.gt_chain,
            "input_seq": record.input_seq, "n_residues": int(record.n_residues),
        })
    frame = pd.DataFrame(rows)
    frame[["eval_set", "stem", "source"]].to_csv(REUSE, index=False)

    new = frame[frame.source == "new"]
    missing = [r.gt_cif for r in new.itertuples()
               if not (args.gt_root / r.gt_cif).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} ground-truth structures missing under {args.gt_root}: "
            f"{missing[:3]}; rerun build_ground_truth.py"
        )
    new[["dataset", "stem", "gt_cif", "gt_chain", "input_seq", "n_residues"]].to_csv(
        MANIFEST, index=False)
    new.assign(L=new.n_residues)[["dataset", "stem", "L", "input_seq"]].to_parquet(
        TARGETS, index=False)

    summary = {
        "scored_units": int(len(frame)),
        "reused": int((frame.source == "published").sum()),
        "to_predict": int(len(new)),
        "by_set": {
            name: group.source.value_counts().to_dict()
            for name, group in frame.groupby("eval_set")
        },
        "residues_to_predict": int(new.n_residues.sum()),
        "longest_to_predict": int(new.n_residues.max()),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[inputs] {len(new)} proteins -> {MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
