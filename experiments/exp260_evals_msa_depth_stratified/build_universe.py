# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1 — define the proteins whose MSA depth this experiment measures.

Our eval universe is two files that overlap: the legacy 554 (#89) and the 334
FoldBench monomers (#245). Cut to natural proteins, they partition cleanly:

``foldbench_natural`` (314)
    Every natural FoldBench monomer — ``eval-val`` (97) plus ``eval-test``
    (217). The 97 are the same proteins the legacy set calls ``foldbench100``,
    so they are counted once, here, where the eval-set labels live.
``nonfoldbench_natural`` (58)
    ``cameo_hard`` (32) and ``casp_fm`` (26): CAMEO hard targets and CASP
    free-modeling domains, collected in #65 precisely because they are the
    regime where MSA-based methods struggle.
``foldbench_designed`` (19)
    ``eval-denovo``. Not part of the natural stratification — designed proteins
    have no evolutionary lineage, so their MSA depth is a property of the design
    process, not of a protein family. Measured anyway as a control.

The legacy ``denovo_pdb`` 396 are left out: they are designs too, they would
outnumber every natural protein four to one in any pooled bin, and #74 already
published their Neff.

    uv run python build_universe.py
"""

import argparse
import hashlib
import io
import json
import urllib.request

import pandas as pd
import upstream as U

EXPECTED_SUBSET_SIZES = {
    "foldbench_natural": 314,
    "nonfoldbench_natural": 58,
    "foldbench_designed": 19,
}


def fetch(url: str, expected_sha256: str | None = None) -> bytes:
    """Download one published input and check its digest when we have one."""

    with urllib.request.urlopen(url) as response:
        payload = response.read()
    if expected_sha256 is not None:
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"{url} changed: {digest} != {expected_sha256}")
    return payload


def build() -> pd.DataFrame:
    """Return one row per protein whose MSA depth we measure."""

    legacy = pd.read_parquet(
        io.BytesIO(fetch(U.LEGACY_TARGETS_URL, U.LEGACY_TARGETS_SHA256))
    )
    sets = pd.read_csv(
        io.BytesIO(fetch(U.FOLDBENCH_SETS_URL, U.FOLDBENCH_SETS_SHA256))
    )

    foldbench = sets[sets.scorable == 1].copy()
    foldbench["subset"] = [
        "foldbench_designed" if designed else "foldbench_natural"
        for designed in foldbench.designed
    ]
    foldbench = foldbench.rename(columns={"seq_len": "L"})
    foldbench["dataset"] = "foldbench_monomer"
    foldbench["msa_volume"] = "foldbench"
    foldbench = foldbench[
        ["stem", "dataset", "subset", "eval_set", "L", "is_viral", "kingdom",
         "msa_volume"]
    ]

    other = legacy[legacy.dataset.isin(U.NONFOLDBENCH_NATURAL_DATASETS)].copy()
    other["subset"] = "nonfoldbench_natural"
    other["eval_set"] = ""
    other["is_viral"] = pd.NA
    other["kingdom"] = ""
    other["msa_volume"] = "exp74"
    other = other[
        ["stem", "dataset", "subset", "eval_set", "L", "is_viral", "kingdom",
         "msa_volume"]
    ]

    universe = pd.concat([foldbench, other], ignore_index=True)
    universe["is_viral"] = universe.is_viral.astype("Int64")
    universe = universe.sort_values(["subset", "dataset", "stem"], ignore_index=True)

    sizes = universe.subset.value_counts().to_dict()
    if sizes != EXPECTED_SUBSET_SIZES:
        raise ValueError(f"universe changed: {sizes} != {EXPECTED_SUBSET_SIZES}")
    if universe.stem.duplicated().any():
        raise ValueError("a stem appears in more than one subset")
    natural = universe[universe.subset != "foldbench_designed"]
    if len(natural) != 372:
        raise ValueError(f"expected 372 natural proteins, got {len(natural)}")
    return universe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(U.DATA / "universe.csv"))
    args = parser.parse_args()
    universe = build()
    U.DATA.mkdir(parents=True, exist_ok=True)
    universe.to_csv(args.out, index=False)
    summary = {
        "n": len(universe),
        "subsets": universe.subset.value_counts().to_dict(),
        "eval_sets": universe.eval_set.value_counts().to_dict(),
        "datasets": universe.dataset.value_counts().to_dict(),
        "msa_volumes": universe.msa_volume.value_counts().to_dict(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
