# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 6b — prove the new ground truth is computed the same way #89's was.

:mod:`build_gt_contacts` produces GT records for 23 proteins that will sit
alongside #89's 554 in the same universe. If its path differs from #89's in any
way — a different structure source, a different chain, a different pyconfind
geometry — the 23 would be scored on a subtly different definition of "contact"
and any metric pooled over the 577 would be meaningless.

The check that settles it: run the **new** code path on proteins #89 already
published, and compare records field by field. Same FoldBench-100 stems, fetched
and computed here from scratch, against `gt_universe.jsonl` from the bucket.

This also exercises the chain question directly. #89 passed FoldBench's raw
`chain_id`, which is the mmCIF *label* asym id for some entries, and silently
fell back to the longest polymer chain; this passes the resolved *auth* id. The
comparison confirms the two select the same chain — `5sbj_A` is the worked
example, where #89's manifest says `A` and both paths land on auth `C`.

    uv run --extra gt python validate_gt_against_exp89.py
    uv run --extra gt python validate_gt_against_exp89.py --n 100   # all of them
"""
import argparse
import json
from pathlib import Path

from build_gt_contacts import build_record, fetch_assembly

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: Fields that must agree exactly. `strata` is excluded: it carries the eval
#: manifest's bookkeeping columns, not anything about the contacts.
COMPARED = ("L", "n_resolved", "gt_chain", "gt_align_identity", "resolved", "contacts")


def compare(published: dict, rebuilt: dict) -> list[str]:
    return [
        f"{field}: published={_brief(published[field])} rebuilt={_brief(rebuilt[field])}"
        for field in COMPARED
        if published[field] != rebuilt[field]
    ]


def _brief(value) -> str:
    if isinstance(value, list):
        return f"<list len {len(value)}>"
    return repr(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--published", type=Path,
                    default=Path("/data/exp226_gt/gt_universe_554.jsonl"),
                    help="hf buckets cp hf://buckets/open-athena/MarinFold/"
                         "data/contacts-v1-model-eval-exp89/gt_universe.jsonl")
    ap.add_argument("--targets", type=Path, default=DATA / "foldbench_targets.csv")
    ap.add_argument("--fasta", type=Path, default=DATA / "eval_queries_expanded.fasta")
    ap.add_argument("--cif-cache", type=Path, default=Path("/data/exp226_gt/cif_control"))
    ap.add_argument("--out", type=Path, default=DATA / "gt_validation.json")
    ap.add_argument("--n", type=int, default=15,
                    help="how many FoldBench-100 proteins to re-derive")
    args = ap.parse_args()

    import csv

    by_stem = {r["stem"]: r for r in csv.DictReader(args.targets.open())}
    sequences: dict[str, str] = {}
    header = None
    chunks: list[str] = []
    for line in args.fasta.read_text().splitlines():
        if line.startswith(">"):
            if header:
                sequences[header] = "".join(chunks)
            header, chunks = line[1:].strip(), []
        elif line.strip():
            chunks.append(line.strip())
    if header:
        sequences[header] = "".join(chunks)

    published = {}
    for line in args.published.read_text().splitlines():
        record = json.loads(line)
        if record["dataset"] == "foldbench100":
            published[record["stem"]] = record

    stems = sorted(published)[:args.n]
    print(f"[control] re-deriving {len(stems)} of {len(published)} FoldBench-100 "
          "proteins through the new path", flush=True)

    results, failures = [], []
    for i, stem in enumerate(stems, 1):
        target = by_stem[stem]
        auth = [c for c in target["auth_asym_ids"].split(";") if c]
        cif = fetch_assembly(target["pdb_id"], args.cif_cache)
        rebuilt = build_record({
            "dataset": "foldbench100", "stem": stem,
            "input_seq": sequences[f"foldbench100__{stem}"],
            "prefer_chain": auth[0] if auth else None,
        }, cif)
        differences = compare(published[stem], rebuilt)
        results.append({
            "stem": stem, "match": not differences,
            "n_contacts": len(rebuilt["contacts"]),
            "published_chain": published[stem]["gt_chain"],
            "rebuilt_chain": rebuilt["gt_chain"],
            "foldbench_chain": target["chain_id"],
            "chain_axis": target["chain_match"],
            "differences": differences,
        })
        if differences:
            failures.append(stem)
        print(f"  [{i:2d}/{len(stems)}] {stem}: "
              f"{'MATCH' if not differences else 'DIFFERS -> ' + '; '.join(differences)}"
              f"  (chain {rebuilt['gt_chain']}, {len(rebuilt['contacts'])} contacts)",
              flush=True)

    summary = {
        "n_compared": len(results),
        "n_matching": sum(1 for r in results if r["match"]),
        "compared_fields": list(COMPARED),
        "label_chain_cases": [r["stem"] for r in results if r["chain_axis"] == "label"],
        "results": results,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[control] {summary['n_matching']}/{len(results)} records reproduce "
          f"#89's exactly -> {args.out}", flush=True)
    if failures:
        raise SystemExit(
            f"{len(failures)} records differ from #89's published universe "
            f"({failures}); the new GT is not computed the same way and must not "
            "be concatenated with it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
