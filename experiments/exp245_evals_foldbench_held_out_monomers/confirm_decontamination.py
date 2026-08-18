# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 0 -- prove the #232 checkpoints never trained on any of these 334 proteins.

Everything downstream is only interesting if this holds, and "the corpora were
decontaminated" is a claim about a chain of five links, each of which could break
silently. This checks all five and writes the evidence:

1. **The eval proteins are inside the decontamination reference.** All 334
   FoldBench monomer sequences must appear byte-identically in the 1,940-chain
   FASTA #225 searched with. A protein that was never a query cannot have been
   filtered, however good the rule was.
2. **No training row that matches one of them survived.** #225's rule is
   ``fident >= 0.30`` over ``max(qcov, tcov) >= 0.50`` -- coverage of the shorter
   of the two sequences, no E-value arm. Every alignment in #225's own search
   that meets it is checked against the drop list that was actually applied.
   The interesting number is the count of *survivors*, and it must be zero.
3. **The published corpora are the filtered ones.** Row counts and removal
   counts must match #225's ``verify_published.py`` output.
4. **exp232 tokenized those exact prefixes.** Its tokenizer pins the bucket paths
   and requires the parquet row counts to equal the published ones; its sweep
   pins the same document totals for the mixture weights.
5. **The two evaluated runs read only those caches.** Optionally verified live
   against each run's W&B config.

It also prices the one thing the rule does *not* cover. The gate is identity over
half the shorter sequence, so a training protein that matches an eval protein at
high identity over a *short* stretch is kept by design. That residual is
measured per protein rather than described, and lands in
``data/residual_identity.csv``.

    uv run python confirm_decontamination.py            # ~2 min, streams a 1.3 GB m8
    uv run python confirm_decontamination.py --wandb    # + the live W&B config check
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

import upstream as U

DATA = U.DATA
REPORT = DATA / "decontamination_check.json"
RESIDUAL = DATA / "residual_identity.csv"

#: Coverage gates the residual is reported at. 0.50 is the applied rule, so its
#: survivor count is zero by construction and is included as the control; the
#: rest say how much identity survives *below* the gate.
RESIDUAL_COVERAGE_GATES = (0.50, 0.40, 0.30, 0.20, 0.0)

#: #232's two pinned corpus prefixes and the document counts its tokenizer
#: requires before it will write a cache.
EXP232_TOKENIZE = (
    U.EXPERIMENTS / "exp232_sweep_cv1_decontam" / "exp232_tokenize.py"
)
EXP232_SWEEP = U.EXPERIMENTS / "exp232_sweep_cv1_decontam" / "exp232_sweep.py"
EXPECTED_PREFIXES = (
    "data/document_structures/contacts_v1_decontam/train",
    "contacts_v1_esm_atlas_decontam/train",
)

#: The two runs whose final checkpoints PR #244 evaluated.
EXP232_RUNS = (
    "prot-exp232-cw-cv1-decontam-s02-m1-p02-aug",
    "prot-exp232-cw-cv1-decontam-s02-m2-p06-aug",
)
WANDB_ENTITY, WANDB_PROJECT = "open-athena", "MarinFold"


def read_fasta(path: Path) -> dict[str, str]:
    """``{header: sequence}``, headers without the leading ``>``."""
    sequences: dict[str, str] = {}
    name = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].strip()
            sequences[name] = ""
        elif name is not None:
            sequences[name] += line.strip()
    return sequences


def check_reference(sets: pd.DataFrame) -> dict:
    """Link 1 -- every eval protein is a query in #225's search."""
    path = U.exp225_reference()
    if path is None:
        raise SystemExit(
            "#225's reference FASTA is not reachable. It is committed on "
            f"{U.EXP225_BRANCH}; fetch that branch and retry."
        )
    reference = read_fasta(path)
    by_sequence = set(reference.values())
    missing_name, missing_sequence = [], []
    for row in sets.itertuples():
        if f"foldbench_all__{row.pdb_id.upper()}_{row.chain_id}" not in reference:
            missing_name.append(row.stem)
        if row.sequence not in by_sequence:
            missing_sequence.append(row.stem)
    return {
        "reference_fasta": str(path),
        "reference_chains": len(reference),
        "reference_unique_sequences": len(by_sequence),
        "eval_proteins": len(sets),
        "missing_by_name": missing_name,
        "missing_by_sequence": missing_sequence,
        "all_present": not missing_name and not missing_sequence,
    }


def scan_alignments(sets: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Link 2 -- reduce #225's FoldBench search against the applied drop list.

    One streaming pass over the 1.3 GB alignment file. For each eval protein it
    keeps two things: how many training rows meet the applied rule and how many
    of those survived (which must be zero), and the strongest *surviving*
    alignment at each coverage gate, which is the residual-leakage table.
    """
    U.require_pinned(U.FOLDBENCH_ALIGNMENTS, U.FOLDBENCH_ALIGNMENTS_SIZE)
    U.require_pinned(U.DROPLIST_FINAL, U.DROPLIST_FINAL_SIZE)

    droplist = pd.read_parquet(U.DROPLIST_FINAL, columns=["arm", "entry_id"])
    dropped = set(zip(droplist["arm"], droplist["entry_id"]))

    queries = {
        f"foldbench_all__{row.pdb_id.upper()}_{row.chain_id}": row.stem
        for row in sets.itertuples()
    }
    gated = defaultdict(int)          # stem -> rows meeting the applied rule
    survived = defaultdict(int)       # stem -> ... of which still in the corpus
    surviving_hits = defaultdict(int)
    best = defaultdict(lambda: {gate: None for gate in RESIDUAL_COVERAGE_GATES})
    columns = {name: index for index, name in enumerate(U.M8_FIELDS)}

    with U.FOLDBENCH_ALIGNMENTS.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            stem = queries.get(fields[columns["query"]])
            if stem is None:
                continue
            arm, rest = fields[columns["target"]].split("|", 1)
            _shard, _row, entry_id = rest.split("_", 2)
            identity = float(fields[columns["fident"]])
            coverage = max(float(fields[columns["qcov"]]), float(fields[columns["tcov"]]))
            is_dropped = (arm, entry_id) in dropped

            if identity >= U.DECONTAM_MIN_IDENTITY and coverage >= U.DECONTAM_MIN_COVERAGE:
                gated[stem] += 1
                if not is_dropped:
                    survived[stem] += 1
            if is_dropped:
                continue
            surviving_hits[stem] += 1
            for gate in RESIDUAL_COVERAGE_GATES:
                if coverage < gate:
                    continue
                current = best[stem][gate]
                if current is None or identity > current["identity"]:
                    best[stem][gate] = {
                        "identity": identity, "coverage": coverage,
                        "arm": arm, "entry_id": entry_id,
                    }

    rows = []
    for row in sets.itertuples():
        record = {
            "eval_set": row.eval_set, "stem": row.stem, "seq_len": row.seq_len,
            "n_alignments_gated": gated[row.stem],
            "n_alignments_gated_surviving": survived[row.stem],
            "n_alignments_surviving": surviving_hits[row.stem],
        }
        for gate in RESIDUAL_COVERAGE_GATES:
            hit = best[row.stem][gate]
            tag = f"cov{int(gate * 100):02d}"
            record[f"best_surviving_identity_{tag}"] = hit["identity"] if hit else None
            record[f"best_surviving_coverage_{tag}"] = hit["coverage"] if hit else None
            record[f"best_surviving_arm_{tag}"] = hit["arm"] if hit else None
            record[f"best_surviving_entry_{tag}"] = hit["entry_id"] if hit else None
        rows.append(record)
    residual = pd.DataFrame(rows)

    summary = {
        "droplist": str(U.DROPLIST_FINAL),
        "droplist_rows": len(dropped),
        "alignments": str(U.FOLDBENCH_ALIGNMENTS),
        "rule": {
            "min_identity": U.DECONTAM_MIN_IDENTITY,
            "min_coverage_of_shorter": U.DECONTAM_MIN_COVERAGE,
            "evalue_arm": None,
        },
        "alignments_meeting_rule": int(residual["n_alignments_gated"].sum()),
        "alignments_meeting_rule_surviving": int(
            residual["n_alignments_gated_surviving"].sum()),
        "proteins_with_surviving_match": sorted(
            residual.loc[residual["n_alignments_gated_surviving"] > 0, "stem"]),
        "clean": bool(residual["n_alignments_gated_surviving"].sum() == 0),
    }
    return summary, residual


def residual_leakage(residual: pd.DataFrame) -> dict:
    """What the coverage gate lets through, per eval set and per gate."""
    out: dict = {}
    for gate in RESIDUAL_COVERAGE_GATES:
        tag = f"cov{int(gate * 100):02d}"
        column = f"best_surviving_identity_{tag}"
        per_set = {}
        for name, group in residual.groupby("eval_set"):
            values = group[column].dropna()
            per_set[name] = {
                "n_with_surviving_alignment": int(len(values)),
                "n_at_or_above_30": int((values >= 0.30).sum()),
                "n_at_or_above_90": int((values >= 0.90).sum()),
                "median_best_identity": round(float(values.median()), 4) if len(values) else None,
                "max_best_identity": round(float(values.max()), 4) if len(values) else None,
            }
        out[f"coverage>={gate:.2f}"] = per_set
    return out


def check_corpora() -> dict:
    """Links 3 and 4 -- published sizes, and the pins exp232 asserts on them."""
    tokenize_source = EXP232_TOKENIZE.read_text()
    sweep_source = EXP232_SWEEP.read_text()

    def pinned(source: str, pattern: str) -> list[int]:
        return sorted({
            int(match.replace("_", ""))
            for match in re.findall(pattern, source, flags=re.MULTILINE)
        } - {0})  # the smoke-test corpus derives its count at run time (0 here)

    expected_documents = pinned(tokenize_source, r"expected_documents=([\d_]+)")
    sweep_documents = pinned(sweep_source, r"^(?:AFDB|ESM)_DOCUMENTS = ([\d_]+)")
    published = sorted(U.PUBLISHED_CORPUS_ROWS.values())
    return {
        "published_rows": U.PUBLISHED_CORPUS_ROWS,
        "published_removed": U.PUBLISHED_CORPUS_REMOVED,
        "exp232_tokenize_expected_documents": sorted(expected_documents),
        "exp232_sweep_documents": sorted(sweep_documents),
        "exp232_prefixes_present": all(p in tokenize_source for p in EXPECTED_PREFIXES),
        "counts_agree": sorted(expected_documents) == published
        and sorted(sweep_documents) == published,
    }


def check_wandb() -> dict:
    """Link 5 -- the two evaluated runs read only exp232's decontaminated caches."""
    import base64
    import netrc
    import urllib.request

    auth = netrc.netrc().authenticators("api.wandb.ai")
    if auth is None:
        raise SystemExit("no api.wandb.ai credentials in ~/.netrc")
    token = base64.b64encode(f"api:{auth[2]}".encode()).decode()
    query = (
        "query($e:String!,$p:String!,$n:String!)"
        "{project(entityName:$e,name:$p){run(name:$n){state config}}}"
    )
    out = {}
    for name in EXP232_RUNS:
        body = json.dumps({
            "query": query,
            "variables": {"e": WANDB_ENTITY, "p": WANDB_PROJECT, "n": name},
        }).encode()
        request = urllib.request.Request(
            "https://api.wandb.ai/graphql", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Basic {token}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            run = json.loads(response.read())["data"]["project"]["run"]
        config = json.dumps(json.loads(run["config"]))
        train_caches = sorted(set(re.findall(
            r"s3://[\w./-]*exp232_sweep_cv1_decontam/tokenized/[\w./-]+", config)))
        other = sorted({
            uri for uri in re.findall(r"s3://[\w./-]*/tokenized/[\w./-]+", config)
            if "exp232_sweep_cv1_decontam" not in uri
        })
        out[name] = {
            "state": run["state"],
            "training_caches": train_caches,
            "other_caches": other,
            "decontaminated_only": bool(train_caches) and all(
                # The only non-exp232 cache a clean run may reference is the
                # exp154 contacts-v1 *validation* cache, which is loss-only and
                # never trained on.
                "contacts-v1-val" in uri for uri in other
            ),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wandb", action="store_true",
                        help="also verify the two runs' live W&B configs")
    parser.add_argument("--sets", type=Path, default=DATA / "eval_sets.csv")
    args = parser.parse_args()

    sets = pd.read_csv(args.sets)
    report: dict = {"n_proteins": len(sets)}

    report["reference_membership"] = check_reference(sets)
    print(f"[decontam] reference membership: {report['reference_membership']['all_present']}",
          flush=True)

    alignments, residual = scan_alignments(sets)
    report["applied_rule"] = alignments
    residual.to_csv(RESIDUAL, index=False)
    print(f"[decontam] {alignments['alignments_meeting_rule']:,} alignments meet the "
          f"rule; {alignments['alignments_meeting_rule_surviving']} survive", flush=True)

    report["residual_leakage"] = residual_leakage(residual)
    report["corpora"] = check_corpora()
    print(f"[decontam] corpus pins agree: {report['corpora']['counts_agree']}", flush=True)

    if args.wandb:
        report["wandb"] = check_wandb()
        for name, entry in report["wandb"].items():
            print(f"[decontam] {name}: decontaminated_only="
                  f"{entry['decontaminated_only']}", flush=True)

    report["verdict"] = {
        "eval_proteins_in_reference": report["reference_membership"]["all_present"],
        "no_surviving_match_at_rule": alignments["clean"],
        "published_and_pinned_counts_agree": report["corpora"]["counts_agree"],
    }
    report["verdict"]["confirmed"] = all(report["verdict"].values())
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(f"[decontam] -> {REPORT} and {RESIDUAL}", flush=True)
    return 0 if report["verdict"]["confirmed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
