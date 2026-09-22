#!/usr/bin/env python
"""Freeze exp304 cohorts and export a sequence-only target file for GPU workers."""

import argparse
import csv
import hashlib
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"
DATA = HERE / "data"
MMSEQS = Path("/home/bizon/exp292_scratch/mmseqs/mmseqs/bin/mmseqs")


def sequence_groups(records: pd.DataFrame) -> dict[str, str]:
    """Group query sequences connected by >=30% identity and >=50% coverage."""
    parents = {str(pid): str(pid) for pid in records.pair_id}

    def root(pid: str) -> str:
        while parents[pid] != pid:
            pid = parents[pid]
        return pid

    with tempfile.TemporaryDirectory(prefix="exp304-mmseqs-") as tmp:
        path = Path(tmp)
        fasta = path / "queries.fasta"
        hits = path / "hits.tsv"
        with fasta.open("w") as handle:
            for row in records.itertuples():
                handle.write(f">{row.pair_id}\n{row.sequence}\n")
        subprocess.run(
            [str(MMSEQS), "easy-search", str(fasta), str(fasta), str(hits),
             str(path / "work"), "-s", "7.5", "--max-seqs", "1000",
             "--format-output", "query,target,fident,qcov,tcov", "--threads", "8"],
            check=True, stdout=subprocess.DEVNULL,
        )
        for line in hits.read_text().splitlines():
            query, target, identity, qcov, tcov = line.split("\t")
            if query == target or float(identity) < 0.3 or max(float(qcov), float(tcov)) < 0.5:
                continue
            a, b = root(query), root(target)
            parents[max(a, b)] = min(a, b)
    return {pid: root(pid) for pid in parents}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    pref = pd.read_csv(args.source / "fold_preference.csv")
    targets = pd.read_parquet(args.source / "eval_targets.parquet")
    targets = targets[targets.role == "foldswitch"].merge(
        pref[["pair_id", "n_a_fs", "n_b_fs", "budget_capped"]], on="pair_id", validate="one_to_one"
    )
    targets["primary"] = (
        (targets.seq_class == "identical") & (~targets.budget_capped)
        & (targets.n_a_fs >= 10) & (targets.n_b_fs >= 10)
    )
    assert len(targets) == 68 and int(targets.primary.sum()) == 44
    groups = sequence_groups(targets)
    targets["sequence_group"] = targets.pair_id.map(groups)
    # A deterministic sequence-group split. Only development truth may be used
    # to choose global search settings; no pair in a group straddles the split.
    primary_by_group = targets.groupby("sequence_group")["primary"].sum().to_dict()
    ordered_groups = sorted(groups.values(), key=lambda group: hashlib.sha256(group.encode()).digest())
    group_split: dict[str, str] = {}
    n_primary_dev = 0
    for group in dict.fromkeys(ordered_groups):
        take = n_primary_dev < round(int(targets.primary.sum()) / 3)
        group_split[group] = "dev" if take else "test"
        if take:
            n_primary_dev += int(primary_by_group[group])
    targets["split"] = targets.sequence_group.map(group_split)
    manifest = targets[["pair_id", "tier", "seq_class", "L", "primary", "budget_capped",
                        "n_a_fs", "n_b_fs", "sequence_group", "split"]].copy()
    manifest.to_csv(DATA / "cohort.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    # The worker receives no reference structure, contacts, switching-region
    # annotation, cohort label, or result from exp301. The file is its access
    # boundary, not just a convention in the search code.
    blind = targets[(~targets.budget_capped)][["pair_id", "sequence", "L"]].sort_values("pair_id")
    assert len(blind) == 67 and set(blind.columns) == {"pair_id", "sequence", "L"}
    blind.to_parquet(DATA / "search_targets.parquet", index=False)
    print(manifest.groupby(["primary", "split"]).size().to_string())
    print(f"wrote {len(blind)} sequence-only targets and {len(manifest)} cohort rows")


if __name__ == "__main__":
    main()
