# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect the per-protein inputs the dashboard needs beyond the score tables.

Two small files, both derived from the same digest-pinned published inputs the
evaluation itself read:

``data/low_depth_sequences.csv``
    The evaluation sequence for each of the 29. Everything downstream — the
    structure alignment, the contact-map axes, the sequence ruler in the page —
    is indexed against this exact string.
``data/foldbench_chains.csv``
    ``pdb_id`` / ``chain`` for the FoldBench members, which #65's manifests do
    not cover (they only describe the CAMEO and CASP halves).

    uv run python dashboard/build_inputs.py
"""

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import upstream as U  # noqa: E402
from build_universe import fetch  # noqa: E402


def main() -> None:
    low = pd.read_csv(U.DATA / "low_msa_depth_set.csv")

    legacy = pd.read_parquet(
        io.BytesIO(fetch(U.LEGACY_TARGETS_URL, U.LEGACY_TARGETS_SHA256))
    )[["dataset", "stem", "input_seq"]]
    foldbench_targets = pd.read_parquet(
        io.BytesIO(fetch(U.FOLDBENCH_TARGETS_URL, U.FOLDBENCH_TARGETS_SHA256))
    )[["dataset", "stem", "input_seq"]]
    sequences = pd.concat([legacy, foldbench_targets], ignore_index=True)
    sequences = low[["dataset", "stem", "L"]].merge(
        sequences, on=["dataset", "stem"], how="left"
    )
    if sequences.input_seq.isna().any():
        missing = sequences.loc[sequences.input_seq.isna(), "stem"].tolist()
        raise ValueError(f"no evaluation sequence for {missing}")
    if (sequences.input_seq.str.len() != sequences.L).any():
        raise ValueError("evaluation sequence lengths disagree with the set's L")
    sequences.to_csv(U.DATA / "low_depth_sequences.csv", index=False)

    sets = pd.read_csv(
        io.BytesIO(fetch(U.FOLDBENCH_SETS_URL, U.FOLDBENCH_SETS_SHA256))
    )
    chains = sets[sets.stem.isin(low.stem)][["stem", "pdb_id", "chain_id"]]
    chains = chains.rename(columns={"chain_id": "chain"})
    chains.to_csv(U.DATA / "foldbench_chains.csv", index=False)

    print(
        f"{len(sequences)} sequences, {len(chains)} FoldBench chains -> {U.DATA}"
    )


if __name__ == "__main__":
    main()
