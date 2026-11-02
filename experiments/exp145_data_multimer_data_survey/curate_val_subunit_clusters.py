# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequence-cluster the val-eligible subunits, to size the structure download.

Phase 1 of ``CURATION_PLAN.md``. The deliverable is one number: how many complexes
we actually have to fetch from EBI in order to define the interface clustering.

The reasoning. PINDER's interface cluster for a dimer is the sorted pair of its two
chains' *structural* community IDs. Two complexes whose chains fall in the same
sequence clusters will almost always land in the same structural community, so one
complex per distinct sequence-cluster pair is enough to define the clustering, and
every other complex in that pair is a download we do not need to make.

The approximation is real and is reported rather than assumed away: near-identical
chains can still present different interface geometry (domain swaps, alternative
binding modes), so this merges some complexes PINDER would have split. For a
validation set that errs toward fewer and more diverse representatives, which is
the safe direction. ``--min-seq-id`` controls how aggressive the merge is, which
is why this sweeps rather than picking one value.

Costs nothing: no downloads, no EBI requests, and the sequences are already on
disk from the split. Usage::

    python3 curate_val_subunit_clusters.py --min-seq-id 0.3 0.5 0.7
"""

import argparse
import collections
import csv
import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path

ASSIGNMENT = "dimer_split_assignment_id30_cov50.csv.gz"
SUBUNITS = "mmseqs_work/subunits.fasta"


def load_val_complexes(path: Path) -> list[tuple[str, str, str, str]]:
    """Return (model_entity_id, kind, uniprot_1, uniprot_2) for the val side."""
    out = []
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "val":
                out.append(
                    (row["model_entity_id"], row["kind"], row["uniprot_1"], row["uniprot_2"])
                )
    return out


def write_subset_fasta(source: Path, wanted: set[str], dest: Path) -> int:
    """Copy the records in ``wanted`` from ``source`` to ``dest``.

    Streams rather than loading 803 MB of FASTA, and fails loudly if the source is
    missing sequences the assignment file references, because a silently short
    query set would make every count below wrong.
    """
    written = 0
    keep = False
    with source.open() as src, dest.open("w") as out:
        for line in src:
            if line.startswith(">"):
                keep = line[1:].split()[0] in wanted
                written += keep
            if keep:
                out.write(line)
    if written != len(wanted):
        raise RuntimeError(
            f"{dest}: wrote {written:,} of {len(wanted):,} requested sequences; "
            f"{source} is missing some val-side subunits"
        )
    return written


def read_clusters(tsv: Path) -> dict[str, str]:
    """Read MMseqs2's ``*_cluster.tsv`` (representative, member) into member -> rep."""
    mapping = {}
    with tsv.open() as handle:
        for line in handle:
            rep, member = line.rstrip("\n").split("\t")[:2]
            mapping[member] = rep
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/curation"))
    parser.add_argument("--min-seq-id", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--mmseqs", default=shutil.which("mmseqs") or "mmseqs")
    args = parser.parse_args()

    work = args.data_dir / "val_cluster_work"
    work.mkdir(parents=True, exist_ok=True)

    complexes = load_val_complexes(args.data_dir / ASSIGNMENT)
    accessions = {a for _, _, a, b in complexes for a in (a, b)}
    print(
        f"{len(complexes):,} val-eligible complexes, "
        f"{len(accessions):,} distinct subunits",
        flush=True,
    )

    fasta = work / "val_subunits.fasta"
    if not fasta.exists() or fasta.stat().st_size == 0:
        print(f"  extracting {len(accessions):,} sequences", flush=True)
        write_subset_fasta(args.data_dir / SUBUNITS, accessions, fasta)
    print(f"  {fasta} ready\n", flush=True)

    results = []
    for min_id in args.min_seq_id:
        tag = f"id{int(round(min_id * 100))}"
        prefix = work / f"clu_{tag}"
        tsv = Path(f"{prefix}_cluster.tsv")
        # The clustering is the expensive step and the pair counting is not, so a
        # rerun that only changes reporting must not redo it.
        if not tsv.exists() or tsv.stat().st_size == 0:
            cmd = [
                args.mmseqs, "easy-cluster", str(fasta), str(prefix), str(work / f"tmp_{tag}"),
                "--min-seq-id", str(min_id),
                "-c", str(args.coverage),
                "--cov-mode", "0",
                "--threads", str(args.threads),
            ]
            print(f"[{tag}] $ {' '.join(cmd)}", flush=True)
            start = time.time()
            log = work / f"mmseqs_{tag}.log"
            with log.open("w") as handle:
                subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
            elapsed = time.time() - start
            print(f"[{tag}] ok, {elapsed / 60:.1f} min (log: {log})", flush=True)
        else:
            elapsed = float("nan")
            print(f"[{tag}] reusing {tsv}", flush=True)

        member_to_rep = read_clusters(tsv)
        missing = len(accessions) - len(member_to_rep)
        if missing:
            raise RuntimeError(f"{tsv}: {missing:,} subunits absent from the clustering")

        # A complex's identity for download purposes is the sorted pair of its two
        # chains' sequence clusters, mirroring PINDER's cluster_{min}_{max}.
        pairs: dict[tuple[str, str], tuple[str, str]] = {}
        per_kind: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
        for model_id, kind, a, b in complexes:
            ra, rb = member_to_rep[a], member_to_rep[b]
            key = (ra, rb) if ra <= rb else (rb, ra)
            per_kind[kind].add(key)
            pairs.setdefault(key, (model_id, kind))

        n_clusters = len(set(member_to_rep.values()))
        summary = {
            "min_seq_id": min_id,
            "coverage": args.coverage,
            "cluster_seconds": round(elapsed, 1),
            "n_complexes": len(complexes),
            "n_subunits": len(accessions),
            "n_sequence_clusters": n_clusters,
            "n_distinct_cluster_pairs": len(pairs),
            "n_pairs_by_kind": {k: len(v) for k, v in sorted(per_kind.items())},
            "download_fraction": round(len(pairs) / len(complexes), 4),
            "wire_gb_at_117kb": round(len(pairs) * 117.0e3 / 1e9, 1),
            "discarded_complexes": len(complexes) - len(pairs),
        }
        print(json.dumps(summary, indent=2), flush=True)
        results.append(summary)

        with gzip.open(args.data_dir / f"val_download_list_{tag}.csv.gz", "wt", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["model_entity_id", "kind", "cluster_1", "cluster_2"])
            for (ra, rb), (model_id, kind) in sorted(pairs.items()):
                writer.writerow([model_id, kind, ra, rb])

    (args.data_dir / "val_cluster_summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
