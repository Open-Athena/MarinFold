# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: run Foldseek-Multimer over the sampled AFDB dimers.

Answers three questions before anyone commits to clustering the ~1.78 M
high-confidence complexes:

1. Does Foldseek read AFDB complex mmCIF at all? The files are ModelCIF written by
   the AFDB pipeline, not PDB depositions, and Foldseek has to recover two chains
   per file rather than one.
2. Does ``easy-multimercluster`` produce a sane clustering on a mixed
   homodimer/heterodimer set?
3. Does ``easy-multimersearch`` find cross-complex interface similarity?

Why the multimer commands and not ``easy-cluster``: ``easy-cluster`` clusters
*chains*, so two complexes built from the same two folds cluster together no matter
how differently the chains are arranged. Redundancy for training is a property of
the interface, which is what ``easy-multimercluster`` scores (``--multimer-tm``,
``--chain-tm``, ``--interface-lddt``).

Run ``fetch_dimer_sample.py`` first. Usage::

    python3 smoke_test_foldseek_multimer.py --sample sample --out data
"""

import argparse
import collections
import csv
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from foldseek_env import ensure_foldseek, foldseek_version

# Complex-level output, written by easy-multimersearch to "<out>_report".
# Tab-separated, no header. The plain "<out>" file is chain-level; the report is
# the one that carries the multimer TM-scores.
REPORT_COLUMNS = [
    "query", "target", "query_chains", "target_chains",
    "qtm", "ttm", "rotation", "translation",
    "col9", "col10", "col11", "col12", "col13", "index",
]

_DB_SIZE = re.compile(r"Query database size:\s+(\d+)")


def run(binary: str, args: list[str], log: Path) -> tuple[float, str]:
    """Run one Foldseek subcommand, tee its output to ``log``.

    Returns (elapsed seconds, combined output). Exceptions propagate.
    """
    print(f"  $ foldseek {' '.join(args)}", flush=True)
    start = time.time()
    proc = subprocess.run([binary, *args], capture_output=True, text=True, check=False)
    elapsed = time.time() - start
    combined = proc.stdout + "\n----- stderr -----\n" + proc.stderr
    log.write_text(combined)
    if proc.returncode != 0:
        # Show the tail rather than the whole log; Foldseek is extremely verbose.
        tail = "\n".join(combined.splitlines()[-25:])
        raise RuntimeError(f"foldseek {args[0]} exited {proc.returncode}. Last lines:\n{tail}")
    print(f"    ok, {elapsed:.1f}s (log: {log})")
    return elapsed, combined


def assert_all_chains_ingested(output: str, n_structures: int, command: str) -> int:
    """Fail if Foldseek read fewer chains than the sample contains.

    Passing the structure files as separate argv entries rather than as a
    directory makes Foldseek treat all but the last one or two as something else,
    and it still exits 0. The result is a clustering of two chains reported as if
    it covered the whole sample, so check the count instead of trusting the exit
    code. Every model in this release is a dimer, hence 2 chains per structure.
    """
    match = _DB_SIZE.search(output)
    if not match:
        raise RuntimeError(f"could not find 'Query database size' in {command} output")
    got = int(match.group(1))
    expected = 2 * n_structures
    if got != expected:
        raise RuntimeError(
            f"{command} ingested {got} chains but the sample has {n_structures} "
            f"dimers ({expected} chains). Pass the structure *directory*, not a "
            f"list of files."
        )
    return got


def load_manifest(sample: Path) -> dict[str, dict[str, str]]:
    with (sample / "manifest.csv").open() as handle:
        return {row["model_entity_id"]: row for row in csv.DictReader(handle)}


def parse_clusters(path: Path) -> dict[str, list[str]]:
    """Read Foldseek's ``<prefix>_cluster.tsv`` (representative, member) pairs."""
    clusters: dict[str, list[str]] = collections.defaultdict(list)
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            clusters[fields[0]].append(fields[1])
    return clusters


def stem(name: str) -> str:
    """Foldseek labels entries by file stem, sometimes with a chain suffix."""
    return name.split("_")[0].removesuffix(".cif").removesuffix(".pdb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("sample"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--multimer-tm", type=float, default=0.5)
    parser.add_argument("--chain-tm", type=float, default=0.0)
    parser.add_argument("--interface-lddt", type=float, default=0.0)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    structures = args.sample / "structures"
    files = sorted(structures.glob("*.cif")) + sorted(structures.glob("*.pdb"))
    if not files:
        raise SystemExit(f"no structures in {structures}; run fetch_dimer_sample.py first")

    manifest = load_manifest(args.sample)
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.sample / "foldseek_work"
    work.mkdir(exist_ok=True)
    logs = work / "logs"
    logs.mkdir(exist_ok=True)

    binary = ensure_foldseek()
    version = foldseek_version()
    print(f"foldseek {version}\n  binary: {binary}\n  {len(files)} structures from {structures}\n")

    timings: dict[str, float] = {}

    print("[1/2] easy-multimercluster")
    cluster_prefix = work / "cluster"
    timings["easy_multimercluster_seconds"], out = run(
        binary,
        [
            "easy-multimercluster", str(structures), str(cluster_prefix),
            str(work / "tmp_cluster"),
            "--multimer-tm-threshold", str(args.multimer_tm),
            "--chain-tm-threshold", str(args.chain_tm),
            "--interface-lddt-threshold", str(args.interface_lddt),
        ],
        logs / "multimercluster.log",
    )
    n_chains = assert_all_chains_ingested(out, len(files), "easy-multimercluster")

    cluster_tsv = Path(f"{cluster_prefix}_cluster.tsv")
    clusters = parse_clusters(cluster_tsv)
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)

    print("\n[2/2] easy-multimersearch (all against all)")
    search_out = work / "multimersearch.tsv"
    timings["easy_multimersearch_seconds"], out = run(
        binary,
        [
            "easy-multimersearch", str(structures), str(structures), str(search_out),
            str(work / "tmp_search"),
        ],
        logs / "multimersearch.log",
    )
    assert_all_chains_ingested(out, len(files), "easy-multimersearch")

    # The complex-level report lists every chain assignment Foldseek tried, so the
    # same complex pair appears once per permutation (A,B vs A,B and A,B vs B,A).
    # Keep the best-scoring assignment per unordered pair.
    best: dict[tuple[str, str], dict[str, Any]] = {}
    report = Path(f"{search_out}_report")
    with report.open() as handle:
        for row in csv.DictReader(handle, fieldnames=REPORT_COLUMNS, delimiter="\t"):
            q, t = row["query"], row["target"]
            if q == t:
                continue
            qtm = float(row["qtm"])
            key = (q, t) if q < t else (t, q)
            if key in best and best[key]["multimer_tm"] >= qtm:
                continue
            best[key] = {
                "query": q,
                "target": t,
                "query_kind": manifest.get(q, {}).get("kind", "?"),
                "target_kind": manifest.get(t, {}).get("kind", "?"),
                "multimer_tm": qtm,
                "target_tm": float(row["ttm"]),
                "query_chains": row["query_chains"],
                "target_chains": row["target_chains"],
                "both_chains_aligned": "," in row["query_chains"] and "," in row["target_chains"],
            }
    hits = sorted(best.values(), key=lambda h: -h["multimer_tm"])

    kinds = collections.Counter(manifest[stem(f.name)]["kind"] for f in files if stem(f.name) in manifest)
    summary = {
        "foldseek_version": version,
        "n_structures": len(files),
        "n_chains_ingested": n_chains,
        "n_heterodimers": kinds.get("heterodimer", 0),
        "n_homodimers": kinds.get("homodimer", 0),
        "n_clusters": len(clusters),
        "reduction_factor": round(len(files) / max(len(clusters), 1), 2),
        "largest_cluster": sizes[0] if sizes else 0,
        "n_multi_member_clusters": sum(1 for s in sizes if s > 1),
        "n_complex_pairs_reported": len(hits),
        "n_pairs_multimer_tm_ge_0.5": sum(1 for h in hits if h["multimer_tm"] >= 0.5),
        "n_pairs_both_chains_aligned": sum(1 for h in hits if h["both_chains_aligned"]),
        "max_multimer_tm": round(max((h["multimer_tm"] for h in hits), default=0.0), 4),
        "thresholds": {
            "multimer_tm": args.multimer_tm,
            "chain_tm": args.chain_tm,
            "interface_lddt": args.interface_lddt,
        },
        **{k: round(v, 1) for k, v in timings.items()},
    }

    (args.out / "foldseek_multimer_smoke_summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out / "foldseek_multimer_smoke_hits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(hits[0].keys()) if hits else ["query"])
        writer.writeheader()
        writer.writerows(hits[:200])
    shutil.copy(cluster_tsv, args.out / "foldseek_multimer_smoke_clusters.tsv")

    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"\ncluster sizes (descending): {sizes[:15]}{' ...' if len(sizes) > 15 else ''}")
    print("\ntop cross-complex pairs by multimer TM-score:")
    for h in hits[:10]:
        print(
            f"  {h['query']} ({h['query_kind'][:6]}) vs {h['target']} ({h['target_kind'][:6]}) "
            f"mTM={h['multimer_tm']:.4f} chains {h['query_chains']}/{h['target_chains']}"
        )
    if not args.keep_work:
        shutil.rmtree(work / "tmp_cluster", ignore_errors=True)
        shutil.rmtree(work / "tmp_search", ignore_errors=True)


if __name__ == "__main__":
    main()
