# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 6b -- the sequence-KNN null for the proteins that have never been run.

The null asks how much of a protein's contact map you get by copying the
contacts of its nearest training-set sequences. #94 built it and #213 published
its per-protein scores for the 554, but this runs it over **all 333 scored
units** rather than reusing those rows: the second null below has no published
counterpart, and a null that covered only part of the eval set would not be
comparable across the three sets.

So this reuses #94's machinery wholesale -- its MMseqs2 index over the 4.13 M
AFDB training documents, its per-document contact store, and its scoring and
metric scripts -- and only supplies the new query set and ground truth. Same
approach #226 used for its 23, including symlinking #94's built artifacts into
a fresh working directory rather than writing into #94's own.

**Two nulls, because the two model families trained on different corpora.**
#94's index is over the *unfiltered* AFDB corpus -- what the #199 cooldown saw.
Running the same KNN over it says what copying a training homolog could have
bought that model. Filtering the alignments down to the rows #225 *kept* gives
the same null for the corpus the #232 checkpoints actually trained on. The gap
between the two nulls is the memorisable signal decontamination removed, priced
in the metric everything else here is reported in.

The filter is exact rather than approximate: #94's alignment targets are named
``{shard}_{row}`` in the AFDB corpus's own coordinates, and #225's drop list
carries the same ``(shard, row)`` for every document it removed, so a dropped
neighbour is identified by coordinate and not by a re-derived identity. 9,712 of
the 22,484 distinct neighbours these 209 queries hit are dropped rows -- 43 %,
against the 4.04 % the drop list is of the corpus, which is what a real
homology signal looks like.

    uv run python run_knn_baseline.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

import upstream as U

DATA = U.DATA
EXP94_DIR = U.EXPERIMENTS / "exp94_evals_sequence_knn_baseline"
#: #94's built index and contact store. It lives in a separate worktree because
#: #94 has never been merged; both are inputs, not artifacts.
EXP94_SCRATCH = Path(
    "/home/bizon/git/MarinFold-exp94/experiments/"
    "exp94_evals_sequence_knn_baseline/_scratch"
)
WORK = Path("/data/exp245_knn")
#: The k #213 published, so the column is comparable to its rows.
K = 10
MODE = "noself"

#: #225's applied drop list, the same file `confirm_decontamination.py` verifies
#: the corpora against.
DROPLIST = U.DROPLIST_FINAL


def run(command: list[str], cwd: Path) -> None:
    print(f"[knn] $ {' '.join(command[:6])} ...", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--skip-search", action="store_true",
                        help="reuse an existing aln.m8 in the work dir")
    parser.add_argument("--decontaminated", action="store_true",
                        help="also score the null over the decontaminated corpus")
    parser.add_argument("--manifest", type=Path,
                        default=DATA / "predictor_manifest_all.csv",
                        help="which proteins to run the null on (default: every "
                             "scored unit, so both nulls cover the whole eval set)")
    args = parser.parse_args()

    for required in (EXP94_SCRATCH / "trainDB", EXP94_SCRATCH / "contacts_store"):
        if not required.exists():
            raise SystemExit(
                f"#94's {required.name} not found at {required}; the KNN null "
                "needs its built index."
            )
    args.work.mkdir(parents=True, exist_ok=True)
    for name in ("trainDB", "contacts_store", "train_seqs.fasta"):
        link = args.work / name
        if not link.exists():
            link.symlink_to(EXP94_SCRATCH / name)
    for suffix in ("", ".dbtype", ".index", ".lookup", ".source", "_h", "_h.dbtype",
                   "_h.index"):
        source = EXP94_SCRATCH / f"trainDB{suffix}"
        link = args.work / f"trainDB{suffix}"
        if source.exists() and not link.exists():
            link.symlink_to(source)

    manifest = pd.read_csv(args.manifest)
    # #94's run_mmseqs builds the query FASTA itself from the manifest, keyed
    # `{dataset}__{stem}`, which is what its scorer joins on -- so the manifest
    # is the input and the FASTA is its output, not ours.
    queries = args.work / "queries.fasta"

    universe = DATA / "gt_universe_foldbench_monomers.jsonl"
    subset = args.work / "gt_universe.jsonl"
    wanted = set(manifest.stem)
    with subset.open("w") as handle:
        for line in universe.read_text().splitlines():
            if json.loads(line)["stem"] in wanted:
                handle.write(line + "\n")

    if not args.skip_search:
        run([sys.executable, str(EXP94_DIR / "run_mmseqs.py"),
             "--scratch", str(args.work), "--query-fasta", str(queries),
             "--manifests", str(args.manifest),
             "-s", "7.5", "--threads", str(args.threads)], cwd=EXP94_DIR)
    run([sys.executable, str(EXP94_DIR / "build_knn_scores.py"),
         "--scratch", str(args.work), "--gt", str(subset), "--ks", str(K)],
        cwd=EXP94_DIR)

    scores_root = args.work / "scores" / f"k{K}_{MODE}"
    if not scores_root.is_dir():
        raise SystemExit(f"expected KNN scores under {scores_root}")
    print(f"[knn] scores -> {scores_root}", flush=True)

    if args.decontaminated:
        decontam = args.work / "decontam"
        decontam.mkdir(parents=True, exist_ok=True)
        for name in ("contacts_store",):
            link = decontam / name
            if not link.exists():
                link.symlink_to(EXP94_SCRATCH / name)
        droplist = pd.read_parquet(DROPLIST, columns=["arm", "shard", "row"])
        dropped = set(
            zip(droplist.loc[droplist.arm == "afdb", "shard"],
                droplist.loc[droplist.arm == "afdb", "row"], strict=True))
        kept, removed = 0, 0
        with (args.work / "aln.m8").open() as source, \
                (decontam / "aln.m8").open("w") as sink:
            for line in source:
                shard, row = line.split("\t")[1].split("_")
                if (int(shard), int(row)) in dropped:
                    removed += 1
                    continue
                sink.write(line)
                kept += 1
        print(f"[knn] decontaminated null: kept {kept:,} alignments, "
              f"removed {removed:,} that #225 deleted from the corpus", flush=True)
        run([sys.executable, str(EXP94_DIR / "build_knn_scores.py"),
             "--scratch", str(decontam), "--gt", str(subset), "--ks", str(K)],
            cwd=EXP94_DIR)
        print(f"[knn] decontaminated scores -> "
              f"{decontam / 'scores' / f'k{K}_{MODE}'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
