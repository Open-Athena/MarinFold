# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble everything the low-MSA-depth dashboard shows into one JSON.

The page is a single self-contained file, so this pulls the pieces together
once, offline, rather than having a browser fetch nine sources:

* identity and depth, from the frozen set;
* the evaluation sequence and the ground-truth contacts, from the published
  ground-truth universes (#89 for the legacy half, #245 for FoldBench);
* MarinFold's rollout vote matrix, thinned to its top pairs in-cluster by
  ``rollout/export_low_depth_maps.py``;
* the baselines' predicted contacts where they exist as per-pair records —
  #245's `contacts_raw.parquet` covers the FoldBench members it scored;
* per-protein R-precision for every predictor;
* Cα coordinates in evaluation numbering, from ``build_structures.py``;
* the alignment itself, which for these proteins is at most nine sequences.

Contact sets are capped at ``top-L`` per predictor, which is the cut
R-precision is defined at and as many as the page ever draws at once.

    uv run python dashboard/build_dashboard_data.py
"""

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import upstream as U  # noqa: E402

HERE = Path(__file__).resolve().parent

#: #89's contact definition: a pair counts as a contact at >= 0.001 degree and
#: at least 6 residues of sequence separation. Same thresholds as
#: ``compute_metrics.py``, so the map and the score agree.
MIN_DEGREE = 0.001
MIN_SEPARATION = 6

#: Per-pair baseline predictions. #74 and #78 published theirs over the legacy
#: 554 — which is where 24 of the 29 live — and #245's local run covers the
#: FoldBench members it scored. The two families label their arms in different
#: columns (`mode` for Protenix, `model` for ESMFold), so each source names its
#: own.
BASELINE_SOURCES = (
    {
        "source": f"{U.BUCKET}/data/protenix-contacts-eval-exp74/contacts_raw_all.parquet",
        "column": "mode",
        "labels": {
            "msa": "Protenix-v2 + MSA",
            "single_seq": "Protenix-v2 single-seq",
        },
    },
    {
        "source": f"{U.BUCKET}/data/esmfold-contacts-eval-exp78/contacts_raw_all.parquet",
        "column": "model",
        "labels": {"esmfold2": "ESMFold2", "esmfold": "ESMFold"},
    },
    {
        "source": "/data/exp245/baseline_scores/protenix/contacts_raw.parquet",
        "column": "model",
        "labels": {
            "protenix-v2_msa": "Protenix-v2 + MSA",
            "protenix-v2_single_seq": "Protenix-v2 single-seq",
        },
        "optional": True,
    },
    {
        "source": "/data/exp245/baseline_scores/esm/contacts_raw.parquet",
        "column": "model",
        "labels": {"esmfold2": "ESMFold2", "esmfold": "ESMFold"},
        "optional": True,
    },
)


def ground_truth(units: set[tuple[str, str]]) -> dict[str, dict]:
    """Load the ground-truth records for the units we need."""

    records: dict[str, dict] = {}
    for url in (U.LEGACY_GROUND_TRUTH_URL, U.FOLDBENCH_GROUND_TRUTH_URL):
        with urllib.request.urlopen(url) as response:
            for line in response.read().decode().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["dataset"], record["stem"])
                if key in units:
                    records[f"{key[0]}__{key[1]}"] = record
    missing = {f"{d}__{s}" for d, s in units} - set(records)
    if missing:
        raise ValueError(f"no ground-truth record for {sorted(missing)}")
    return records


def true_contacts(record: dict) -> list[list[float]]:
    """The pairs #89 counts as true contacts, as ``[i, j, degree]``."""

    length = record["L"]
    return [
        [int(i), int(j), round(float(degree), 4)]
        for i, j, degree in record["contacts"]
        if degree >= MIN_DEGREE and (j - i) >= MIN_SEPARATION and i < j < length
    ]


#: Evaluation length per unit, filled in ``main`` — the top-L cut needs it.
_LENGTHS: dict[tuple[str, str], int] = {}


def baseline_contacts(units: set[tuple[str, str]]) -> dict[str, dict[str, list]]:
    """Per-pair baseline predictions, capped at top-L, where they exist.

    Keyed by stem rather than by ``(dataset, stem)``: #74 and #78 label the
    FoldBench-100 proteins ``foldbench100`` while this experiment calls the same
    proteins ``foldbench_monomer``. The stems are unambiguous within this set.
    """

    by_stem = {stem: dataset for dataset, stem in units}
    out: dict[str, dict[str, list]] = {}
    for spec in BASELINE_SOURCES:
        source = spec["source"]
        if not source.startswith("http") and not Path(source).exists():
            if spec.get("optional"):
                print(f"[baselines] {source} is absent — skipping", flush=True)
                continue
            raise FileNotFoundError(source)
        frame = pd.read_parquet(source)
        column, labels = spec["column"], spec["labels"]
        frame = frame[
            (frame.role == "pred")
            & frame[column].isin(labels)
            & frame.stem.isin(by_stem)
            & (frame.sep >= MIN_SEPARATION)
        ]
        for (stem, arm), group in frame.groupby(["stem", column]):
            key = f"{by_stem[stem]}__{stem}"
            label = labels[arm]
            if label in out.get(key, {}):
                continue
            top = group.nlargest(_LENGTHS[(by_stem[stem], stem)], "degree")
            out.setdefault(key, {})[label] = [
                [int(i), int(j), round(float(degree), 4)]
                for i, j, degree in zip(top.i, top.j, top.degree, strict=True)
            ]
    return out


def main() -> None:
    low = pd.read_csv(U.DATA / "low_msa_depth_set.csv")
    sequences = pd.read_csv(U.DATA / "low_depth_sequences.csv").set_index("stem")
    units = set(zip(low.dataset, low.stem, strict=True))
    _LENGTHS.update({(d, s): int(l) for d, s, l in zip(low.dataset, low.stem, low.L, strict=True)})

    truth = ground_truth(units)
    marinfold = json.loads(
        urllib.request.urlopen(
            f"{U.RESULTS_URL}/analysis/marinfold_low_depth_contacts.json"
        )
        .read()
        .decode()
    )["proteins"]
    alignments = json.loads((U.DATA / "low_msa_depth_a3m.json").read_text())
    structures = json.loads((U.DATA / "low_depth_structures.json").read_text())
    baselines = baseline_contacts(units)

    scores = pd.read_csv(U.DATA / "per_protein_depth.csv")
    scores = scores[(scores["range"] == "all") & (scores["cut"] == "R")]

    proteins = []
    for record in low.itertuples(index=False):
        key = f"{record.dataset}__{record.stem}"
        gt = truth[key]
        per_protein = scores[
            (scores.dataset == record.dataset) & (scores.stem == record.stem)
        ]
        marinfold_pairs = marinfold[key]["top_pairs"][: int(record.L)]
        proteins.append(
            {
                "key": key,
                "stem": record.stem,
                "dataset": record.dataset,
                "source": {
                    "cameo_hard": "CAMEO hard",
                    "casp_fm": "CASP free-modeling",
                    "foldbench_monomer": "FoldBench",
                }[record.dataset],
                "in_foldbench": bool(record.dataset == "foldbench_monomer"),
                "eval_set": record.eval_set if isinstance(record.eval_set, str) else None,
                "L": int(record.L),
                "sequence": sequences.loc[record.stem, "input_seq"],
                "msa_depth": int(record.msa_depth),
                "msa_neff": round(float(record.msa_neff), 2),
                "a3m": alignments[record.stem],
                "structure": structures[key],
                "n_true_contacts": len(true_contacts(gt)),
                "contacts": {
                    "Ground truth": true_contacts(gt),
                    "MarinFold #232 m2-p06": [
                        [int(i), int(j), round(float(votes) / 100, 3)]
                        for i, j, votes in marinfold_pairs
                    ],
                    **baselines.get(key, {}),
                },
                "r_precision": {
                    row.predictor: round(float(row.precision), 4)
                    for row in per_protein.itertuples(index=False)
                },
            }
        )

    payload = {
        "generated_from": {
            "scores": U.RESULTS_URL,
            "checkpoint": "prot-exp232-trc-cv1-decontam-train-s01-m2-p06-"
            "srcpeak-augcont-lr005-us-east1 step 363000",
            "experiment": "exp260",
        },
        "contact_definition": {
            "min_degree": MIN_DEGREE,
            "min_separation": MIN_SEPARATION,
            "predicted_pairs_shown": "top-L per predictor",
        },
        "proteins": proteins,
    }
    destination = HERE / "data.json"
    destination.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))

    predictors = sorted({name for p in proteins for name in p["contacts"]})
    print(
        json.dumps(
            {
                "proteins": len(proteins),
                "contact_layers": predictors,
                "with_baselines": sum(1 for p in proteins if len(p["contacts"]) > 2),
                "with_structures": sum(
                    1 for p in proteins if p["structure"]["available"]
                ),
                "bytes": destination.stat().st_size,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
