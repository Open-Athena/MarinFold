"""Reduce sequence hits to auditable, conservative per-complex survival counts."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = [
    "query",
    "target",
    "fident",
    "alnlen",
    "qcov",
    "tcov",
    "evalue",
    "bits",
    "nident",
    "qlen",
    "tlen",
    "qstart",
    "qend",
    "tstart",
    "tend",
]
LOCAL_MARINFOLD_ARMS = ["afdb", "esm_atlas", "afcdb", "pinder"]
MPNN_ARMS = ["mpnn_afdb", "mpnn_esm"]
MARINFOLD_ARMS = LOCAL_MARINFOLD_ARMS + MPNN_ARMS
HELICO_ARM = "helico_finetune"


def contaminates(hit: dict[str, str]) -> bool:
    """Apply the identity and symmetric coverage gates without rounded TSV fractions."""
    identity_ok = int(hit["nident"]) * 10 >= int(hit["alnlen"]) * 3
    query_ok = 2 * (int(hit["qend"]) - int(hit["qstart"]) + 1) >= int(hit["qlen"])
    target_ok = 2 * (int(hit["tend"]) - int(hit["tstart"]) + 1) >= int(hit["tlen"])
    return identity_ok and (query_ok or target_ok)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a table including the union of columns on all rows."""
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(dict.fromkeys(k for r in rows for k in r)),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def fasta_ids(path: Path) -> set[str]:
    """Return identifiers from a FASTA used as an audit query set."""
    return {
        line[1:].split()[0]
        for line in path.read_text().splitlines()
        if line.startswith(">")
    }


def main() -> None:
    """Write stage counts and hit evidence, marking remaining corpora unverified."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=Path("/data/exp350"))
    ap.add_argument("--data", type=Path, default=HERE / "data")
    args = ap.parse_args()
    candidates = list(csv.DictReader((args.data / "candidates.csv").open()))
    membership = list(csv.DictReader((args.data / "query_membership.csv").open()))
    queries: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in membership:
        queries[row["source"], row["target"]].add(row["query"])
    best: dict[tuple[str, str], dict] = {}
    searched: set[str] = set()
    searched_queries: dict[str, set[str]] = defaultdict(set)
    diagnostics = {}
    for family, arms in [
        ("native", ["afdb", "esm_atlas"]),
        ("complex", ["afcdb", "pinder"]),
        ("helico", [HELICO_ARM]),
    ]:
        manifest = args.work / f"{family}_search.json"
        if not manifest.exists():
            continue
        metadata = json.loads(manifest.read_text())
        query_path = Path(metadata["queries"])
        if (
            metadata["queries_sha256"]
            != hashlib.file_digest(query_path.open("rb"), "sha256").hexdigest()
        ):
            raise ValueError(f"Stale {family} search: query manifest hash differs")
        searched.update(arms)
        family_queries = fasta_ids(query_path)
        for arm in arms:
            searched_queries[arm].update(family_queries)
        counts: Counter = Counter()
        for search in metadata["searches"]:
            with Path(search["output"]).open() as fh:
                for fields in csv.reader(fh, delimiter="\t"):
                    hit = dict(zip(FIELDS, fields, strict=True))
                    counts[hit["query"]] += 1
                    if not contaminates(hit):
                        continue
                    target_arm = hit["target"].split("|")[0]
                    arm = HELICO_ARM if target_arm == "helico" else target_arm
                    if arm not in arms:
                        raise ValueError(f"Unexpected training arm {target_arm}")
                    key = (arm, hit["query"])
                    # This is a witness to exclusion, not a nearest-neighbor estimate.
                    if key not in best or float(hit["bits"]) > float(best[key]["bits"]):
                        best[key] = {**hit, "arm": arm}
        diagnostics[family] = {
            "alignment_rows": sum(counts.values()),
            "max_output_alignments_per_query": max(counts.values(), default=0),
            "searches": metadata["searches"],
        }
    rows = []
    for row in candidates:
        qids = queries[row["source"], row["target"]]
        statuses: dict[str, str] = {
            arm: ("hit" if any((arm, q) in best for q in qids) else "no_hit")
            if arm in searched and qids and qids <= searched_queries[arm]
            else "unverified"
            for arm in LOCAL_MARINFOLD_ARMS + [HELICO_ARM]
        }
        statuses.update({arm: "unverified" for arm in MPNN_ARMS})
        remaining = [
            arm for arm in LOCAL_MARINFOLD_ARMS if statuses[arm] == "unverified"
        ]
        marinfold = (
            "rejected"
            if any(statuses[a] == "hit" for a in LOCAL_MARINFOLD_ARMS)
            else "unverified_mpnn"
        )
        if remaining and marinfold != "rejected":
            marinfold = "unverified_native_or_complex_and_mpnn"
        rows.append(
            {
                **row,
                **{f"{a}_status": statuses[a] for a in MARINFOLD_ARMS},
                f"{HELICO_ARM}_status": statuses[HELICO_ARM],
                "marinfold_status": marinfold,
                "helico_pretraining_status": "unverified_complete_inherited_training",
                "end_to_end_status": "rejected_or_exposed"
                if marinfold == "rejected" or statuses[HELICO_ARM] == "hit"
                else "unverified",
            }
        )
    summary = []
    for source in ["foldbench", "pinder"]:
        all_rows = [r for r in rows if r["source"] == source]
        for kind in ["all", "homodimer", "heterodimer"]:
            cohort = [
                r for r in all_rows if kind == "all" or r.get("complex_type") == kind
            ]
            eligible = [r for r in cohort if r["eligibility"] == "candidate"]
            stages = [("source_candidates", cohort), ("quality_and_scope", eligible)]
            remaining = eligible
            for arm in [
                "afcdb",
                "pinder",
                "afdb",
                "esm_atlas",
                *MPNN_ARMS,
                HELICO_ARM,
            ]:
                if arm in searched:
                    remaining = [r for r in remaining if r[f"{arm}_status"] == "no_hit"]
                stages.append((f"after_{arm}", remaining))
            for stage, subset in stages:
                summary.append(
                    {
                        "source": source,
                        "complex_type": kind,
                        "stage": stage,
                        "remaining": len(subset),
                        "pdb_entries": len({r["pdb_id"] for r in subset}),
                        "status": "measured"
                        if not stage.startswith("after_") or stage[6:] in searched
                        else "unverified",
                    }
                )
    write_csv(args.data / "survival.csv", summary)
    write_csv(args.data / "per_complex.csv", rows)
    if best:
        write_csv(args.data / "exclusion_witnesses.csv", list(best.values()))
    (args.data / "search_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    for r in summary:
        if r["complex_type"] == "all":
            print(r)


if __name__ == "__main__":
    main()
