# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3 -- build the eval targets and publish the three sets to the HF bucket.

The CoreWeave evaluation cannot reach this workstation, so every input it reads
has to be public and immutable: it mirrors each file by URL and refuses to start
unless the bytes match a pinned size and digest. This writes those files, pushes
them to the public bucket, and prints the pins to paste into
``rollout/checkpoint_specs.py``.

Published under ``data/contacts-v1-foldbench-monomers-exp245/``:

``eval_targets_foldbench_monomers.parquet``
    One row of ``(dataset, stem, L, input_seq)`` per **scorable** unit -- the
    schema exp82's rollout worker reads. ``dataset`` is ``foldbench_monomer``
    for every row: the three eval sets are cuts of one universe, not three
    ground-truth files.
``gt_universe_foldbench_monomers.jsonl``
    The matching pyconfind ground-truth records, in #89's schema.
``eval_sets.csv``
    All 334 monomers -- the set assignment, the annotation behind it, the viral
    flag the reporting cuts split on, and ``scorable`` / ``exclusion_reason``
    for the ones held out of the scored universe.

**One protein is excluded.** ``8uxt_A`` (1,596 residues) is the only monomer
whose contacts-v1 document does not fit an 8,192-token context:
``build_document`` truncates it to 1,664 of its 3,809 contacts, so no rollout
can produce it in full and any score for it would measure the format's context
limit. It stays in ``eval_sets.csv``, flagged, and out of the scored universe.
See ``check_context_budget.py``.
``eval_sets.fasta``
    The same sequences for the baseline predictors.

    uv run python publish_eval_inputs.py --dry-run   # build + validate locally
    uv run python publish_eval_inputs.py             # ... and push to the bucket
    uv run python publish_eval_inputs.py --print-pins
"""
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import upstream as U

DATA = U.DATA
DATASET = "foldbench_monomer"
BUCKET_PREFIX = "data/contacts-v1-foldbench-monomers-exp245"
BUCKET_URI = f"hf://buckets/open-athena/MarinFold/{BUCKET_PREFIX}"

TARGETS = DATA / "eval_targets_foldbench_monomers.parquet"
PUBLISHED_SETS = DATA / "eval_sets.csv"
FASTA = DATA / "eval_sets.fasta"
PINS = DATA / "published_inputs.json"

#: ``hf`` outside the venv: marinfold pins huggingface_hub<1, which has no
#: ``buckets`` subcommand. Same resolution #226 documented.
HF_BIN_CANDIDATES = ("/home/bizon/anaconda3/bin/hf", "hf")


def mark_scorable(sets: pd.DataFrame, budget: pd.DataFrame) -> pd.DataFrame:
    """Join the context-budget verdict onto the set manifest."""
    verdict = budget.set_index("stem")
    missing = [s for s in sets.stem if s not in verdict.index]
    if missing:
        raise AssertionError(
            f"no context-budget verdict for {missing[:5]}; run check_context_budget.py")
    sets = sets.copy()
    sets["scorable"] = [int(verdict.loc[s, "scorable"]) for s in sets.stem]
    sets["exclusion_reason"] = [
        "" if verdict.loc[s, "scorable"]
        else ("contacts-v1 document truncated at the 8192-token context: "
              f"{int(verdict.loc[s, 'n_contacts_emitted'])} of "
              f"{int(verdict.loc[s, 'n_gt_contacts'])} contacts representable")
        for s in sets.stem
    ]
    return sets


def build_targets(sets: pd.DataFrame, universe: Path) -> pd.DataFrame:
    """``(dataset, stem, L, input_seq)`` for every unit with ground truth."""
    lengths = {}
    for line in universe.read_text().splitlines():
        record = json.loads(line)
        lengths[record["stem"]] = record["L"]
    rows = []
    for row in sets.itertuples():
        if not row.scorable:
            continue
        if row.stem not in lengths:
            raise AssertionError(f"{row.stem} has no ground-truth record")
        if lengths[row.stem] != len(row.sequence):
            raise AssertionError(
                f"{row.stem}: ground truth L={lengths[row.stem]} but the eval "
                f"sequence is {len(row.sequence)} residues"
            )
        rows.append({
            "dataset": DATASET, "stem": row.stem,
            "L": len(row.sequence), "input_seq": row.sequence,
        })
    targets = pd.DataFrame(rows)
    if targets.stem.nunique() != len(targets):
        raise AssertionError("duplicate stems in the targets table")
    return targets


def hf_binary() -> str:
    for candidate in HF_BIN_CANDIDATES:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    raise SystemExit(
        "no usable `hf` CLI found. The venv's huggingface_hub is pinned <1 and "
        "has no `buckets` subcommand; use a system install (hub >= 1.5)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="build and validate, but do not upload")
    parser.add_argument("--print-pins", action="store_true",
                        help="print the constants for rollout/checkpoint_specs.py")
    args = parser.parse_args()

    sets = mark_scorable(
        pd.read_csv(DATA / "eval_sets.csv"),
        pd.read_csv(DATA / "context_budget.csv"),
    )
    sets.to_csv(PUBLISHED_SETS, index=False)
    universe = DATA / "gt_universe_foldbench_monomers.jsonl"
    scored_universe = DATA / "gt_universe_scored.jsonl"
    targets = build_targets(sets, universe)
    scored_stems = set(targets.stem)
    with scored_universe.open("w") as handle:
        for line in universe.read_text().splitlines():
            if json.loads(line)["stem"] in scored_stems:
                handle.write(line + "\n")
    pq.write_table(pa.Table.from_pandas(targets, preserve_index=False), TARGETS)
    with FASTA.open("w") as handle:
        for row in targets.itertuples():
            handle.write(f">{row.stem}\n{row.input_seq}\n")

    published = [
        TARGETS, scored_universe, PUBLISHED_SETS, FASTA,
        DATA / "decontamination_check.json", DATA / "gt_report.json",
        DATA / "context_budget.csv",
    ]
    # Results, once they exist: the per-protein table is what anyone rescoring or
    # re-slicing these sets actually needs, and it is small enough to publish.
    published += [
        path for path in (
            DATA / "per_protein.csv.gz", DATA / "headline.csv",
            DATA / "paired_deltas.csv", DATA / "val_vs_test.csv",
            DATA / "residual_identity.csv", DATA / "path_validation.json",
            DATA / "analysis_summary.json",
        ) if path.exists()
    ]
    pins = {
        path.name: {"size": path.stat().st_size, "sha256": U.sha256(path)}
        for path in published
    }
    pins["_units"] = int(len(targets))
    pins["_sets"] = sets.eval_set.value_counts().to_dict()
    pins["_scored_sets"] = (
        sets[sets.scorable == 1].eval_set.value_counts().to_dict())
    pins["_excluded"] = sets.loc[sets.scorable == 0,
                                 ["stem", "eval_set", "exclusion_reason"]
                                 ].to_dict(orient="records")
    pins["_bucket"] = BUCKET_URI
    PINS.write_text(json.dumps(pins, indent=2) + "\n")

    if args.print_pins:
        for name, key in (
            ("eval_targets_foldbench_monomers.parquet", "TARGETS"),
            ("gt_universe_scored.jsonl", "GROUND_TRUTH"),
            ("eval_sets.csv", "SETS_MANIFEST"),
        ):
            print(f"{key}_SIZE = {pins[name]['size']:_}")
            print(f'{key}_SHA256 = "{pins[name]["sha256"]}"')

    if args.dry_run:
        print(f"[publish] dry run; {len(targets)} targets -> {TARGETS}", flush=True)
        return 0

    binary = hf_binary()
    for path in published:
        subprocess.run(
            [binary, "buckets", "cp", str(path), f"{BUCKET_URI}/{path.name}"],
            check=True,
        )
        print(f"[publish] {path.name} -> {BUCKET_URI}/{path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
