# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 5 — build **eval2**: the contact eval set with training homologs removed.

Takes the 776-protein expanded set and drops every protein with a training-set
sequence identity at or above the threshold (default 40 %), leaving **307**.
Each surviving protein is annotated with its nearest training sequence, so a
more stringent cut — 30 %, or any other — is a filter on a committed column
rather than a re-run of the search.

Three things a consumer of this file has to know, all carried as columns rather
than left in prose:

* **``best_identity`` is the annotation that matters.** It is the coverage-gated
  maximum over *both* training arms (exp199 trained on both), so
  ``best_identity < 0.30`` reproduces the 30 % set exactly. ``passes_30`` is
  precomputed for convenience. ``best_identity_ungated`` is the paranoid bound
  — the max identity ignoring the 50 % query-coverage gate — and 18 of the 307
  clear 40 % only because of that gate, so ``passes_40_ungated`` is there for
  anyone who wants the stricter reading.
* **75 % of eval2 is de novo designed protein** (229 of 307). That is not a
  choice made here — it is what survives a homology filter, and it is the exact
  confound #213 flagged. ``designed_any`` splits it; the natural subset is 78 at
  40 % and 61 at 30 %.
* **Every protein is scorable.** 284 come from #89's frozen GT universe; the
  other 23 are #226's net-new FoldBench monomers, whose contacts
  :mod:`build_gt_contacts` computes through a path proven bit-identical to
  #89's on all 100 FoldBench controls. ``has_ground_truth`` is derived from the
  GT files rather than assumed, so it reads 0 until that step has run.

    uv run python build_eval2.py
    uv run python build_eval2.py --threshold 0.30 --out data/eval2_strict.csv
"""
import argparse
import csv
import json
from pathlib import Path

from analyze_survival import DATASET_NEW, GATED, UNGATED, load_rows, survives
from exp213_link import ARMS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: Default exclusion threshold. "Exclude anything at or above 40 % identity" is
#: #213's and #226's published convention and is what the 307 figure means.
#: Exactly one protein sits on the boundary — `6sa6_A` at fident 0.400 — so
#: --boundary is offered rather than left as a silent judgement call.
DEFAULT_THRESHOLD = 0.40

#: The retrospective cuts precomputed as columns. `best_identity` supports any
#: threshold; these are the two the experiment reports on.
RETROSPECTIVE = (0.30,)

COLUMNS = [
    # identity — what makes this eval2
    "dataset", "stem", "input_seq", "length",
    "best_identity", "best_identity_ungated",
    "passes_30", "passes_40_ungated", "passes_30_ungated",
    f"{ARMS[0]}_best_identity", f"{ARMS[1]}_best_identity",
    "best_arm", "best_evalue", "best_target", "n_hits_significant", "stratum",
    # designed vs natural
    "designed", "synthetic", "designed_any", "source_organism",
    # scorability
    "has_ground_truth",
    # orthogonal novelty axes, where they exist
    "neff_tier", "fold_verdict", "seq_leakage", "msa_neff",
]


def read_fasta(path: Path) -> dict[str, str]:
    """``{dataset}__{stem}`` -> sequence."""
    records: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records[header] = "".join(chunks)
            header, chunks = line[1:].strip(), []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        records[header] = "".join(chunks)
    return records


def read_gt_stems(paths: list[Path]) -> set[str]:
    """Stems present in any `gt_universe.jsonl`-shaped file that exists.

    Derived rather than assumed so `has_ground_truth` cannot drift: it is 0
    before :mod:`build_gt_contacts` runs and 1 after, with no edit here.
    """
    stems: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                stems.add(json.loads(line)["stem"])
    return stems


def build(rows: list[dict], sequences: dict[str, str], threshold: float,
          keep_boundary: bool, gt_stems: set[str] | None = None) -> list[dict]:
    """Filter to the homology-free set and annotate every survivor."""
    def keep(row: dict) -> bool:
        if survives(row, threshold):
            return True
        # `survives` is a strict `<`; the boundary case is identity == threshold.
        value = row[GATED]
        return keep_boundary and value != "" and float(value) == threshold

    out = []
    for row in rows:
        if not keep(row):
            continue
        key = f"{row['dataset']}__{row['stem']}"
        entry = {
            "dataset": row["dataset"],
            "stem": row["stem"],
            "input_seq": sequences[key],
            "length": len(sequences[key]),
            "best_identity": row[GATED],
            "best_identity_ungated": row[UNGATED],
            "best_arm": row["best_arm"],
            "best_evalue": row["best_evalue"],
            "best_target": row["best_target"],
            "n_hits_significant": row["n_hits_significant"],
            "stratum": row["stratum"],
            "designed": row["designed"],
            "synthetic": row["synthetic"],
            "designed_any": row["designed_any"],
            "source_organism": row["source_names"],
            # #89's frozen GT universe covers the 554; the net-new monomers are
            # covered only once build_gt_contacts.py has run, so this is read
            # off the GT files rather than inferred from the dataset label.
            "has_ground_truth": int(row["dataset"] != DATASET_NEW
                                    or (gt_stems is not None
                                        and row["stem"] in gt_stems)),
            **{c: row[c] for c in ("neff_tier", "fold_verdict", "seq_leakage",
                                   "msa_neff")},
        }
        for arm in ARMS:
            entry[f"{arm}_best_identity"] = row[f"{arm}_{GATED}"]
        for cut in RETROSPECTIVE:
            entry[f"passes_{cut:.0%}".rstrip("%")] = int(survives(row, cut))
            entry[f"passes_{cut:.0%}_ungated".replace("%_", "_")] = int(
                survives(row, cut, UNGATED))
        entry["passes_40_ungated"] = int(survives(row, threshold, UNGATED))
        out.append(entry)
    return out


def summarize(entries: list[dict], threshold: float) -> dict:
    def count(predicate) -> int:
        return sum(1 for e in entries if predicate(e))

    natural = count(lambda e: not e["designed_any"])
    summary = {
        "threshold": threshold,
        "n": len(entries),
        "n_designed": len(entries) - natural,
        "n_natural": natural,
        "n_with_ground_truth": count(lambda e: e["has_ground_truth"]),
        "n_needing_ground_truth": count(lambda e: not e["has_ground_truth"]),
        "by_dataset": {},
        "retrospective": {},
    }
    for entry in entries:
        summary["by_dataset"][entry["dataset"]] = (
            summary["by_dataset"].get(entry["dataset"], 0) + 1)
    for cut in RETROSPECTIVE:
        key = f"passes_{cut:.0%}".rstrip("%")
        summary["retrospective"][f"<{cut:.0%}"] = {
            "n": count(lambda e, k=key: e[k]),
            "n_natural": count(lambda e, k=key: e[k] and not e["designed_any"]),
            "n_with_ground_truth": count(lambda e, k=key: e[k] and e["has_ground_truth"]),
        }
    summary["retrospective"][f"<{threshold:.0%} ungated"] = {
        "n": count(lambda e: e["passes_40_ungated"]),
        "n_natural": count(lambda e: e["passes_40_ungated"] and not e["designed_any"]),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, default=DATA / "eval_train_identity_expanded.csv")
    ap.add_argument("--targets", type=Path, default=DATA / "foldbench_targets.csv")
    ap.add_argument("--fasta", type=Path, default=DATA / "eval_queries_expanded.fasta")
    ap.add_argument("--out", type=Path, default=DATA / "eval2_manifest.csv")
    ap.add_argument("--out-fasta", type=Path, default=DATA / "eval2.fasta")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--boundary", choices=("exclude", "keep"), default="exclude",
                    help="whether a protein whose identity equals --threshold "
                         "exactly is dropped (default, matching #213/#226's "
                         "published counts) or kept. Affects exactly one "
                         "protein at 0.40: 6sa6_A.")
    ap.add_argument("--new-gt", type=Path, nargs="*",
                    default=[DATA / "gt_universe_eval2_new.jsonl"],
                    help="gt_universe.jsonl files covering proteins outside #89's "
                         "554; missing files are simply not counted")
    args = ap.parse_args()

    rows = load_rows(args.table, args.targets)
    sequences = read_fasta(args.fasta)
    gt_stems = read_gt_stems(args.new_gt)
    entries = build(rows, sequences, args.threshold, args.boundary == "keep",
                    gt_stems=gt_stems)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(entries)
    args.out_fasta.write_text("".join(
        f">{e['dataset']}__{e['stem']}\n{e['input_seq']}\n" for e in entries))

    summary = summarize(entries, args.threshold)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[eval2] {len(entries)}/{len(rows)} proteins at <{args.threshold:.0%} "
          f"identity -> {args.out}", flush=True)
    print(f"[eval2] {summary['n_designed']} designed + {summary['n_natural']} natural "
          f"({summary['n_designed'] / summary['n']:.0%} designed)", flush=True)
    print(f"[eval2] {summary['n_with_ground_truth']} scorable today; "
          f"{summary['n_needing_ground_truth']} need ground-truth contacts", flush=True)
    for label, cut in summary["retrospective"].items():
        print(f"[eval2] retrospective {label}: n={cut['n']}, "
              f"natural={cut['n_natural']}", flush=True)
    print(f"[eval2] sequences -> {args.out_fasta}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
