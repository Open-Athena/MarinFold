#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""M3 — which fold did MarinFold's *training data* encode for each fold switcher?

Chakravarty et al. 2024 concluded that AlphaFold's successes on fold switchers
are driven by training-set memorization. They had to infer that; we can read our
own training data, so here it is measured directly.

The contacts-v1 corpora are AFDB and ESM-Atlas — i.e. **predicted** structures,
from AF2 and ESMFold. Whatever conformer those predictors chose is the only one
MarinFold was ever shown for that sequence. So for each fold switcher:

**Stage A (search).** MMseqs2 the 68 reference sequences against exp213's
materialised training DB (70.9M sequences, ``-s 7.5`` — the standing convention
across #65/#94/#213/#225), giving the best training hit and its identity.

**Stage B (label).** For each hit, read the actual corpus row it names, fold its
document back into a contact set, move that set into the pair's reference frame,
and score it against fold1 and fold2. That yields a per-protein **training-fold
label** — the thing to correlate MarinFold's own preference against.

The hit header grammar ``{arm}|{shard:05d}_{row}_{entry_id}`` is what makes
Stage B cheap: a hit names its corpus shard and row directly, so nothing has to
be joined.

**Caveat, recorded in the output.** exp213's DB is exp199's corpus. exp277 trains
on #225's *decontaminated* corpora plus #266's redesigns, so this is a superset
proxy for the native arm — a hit here may have been dropped before exp277 saw
it. Stage B reads the decontaminated shards, so a row that decontamination
removed shows up as ``dropped_by_decontamination``.

    uv run python audit_training_fold.py search
    uv run --with "huggingface_hub>=1.5" python audit_training_fold.py label
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / "_cache"

sys.path.insert(0, str(HERE.parent / "exp74_evals_protenix_pyconfind_contacts"))
from pyconfind_contacts import align_obs_to_ref  # noqa: E402

MMSEQS = Path(os.environ.get("EXP301_MMSEQS", "/home/bizon/exp292_scratch/mmseqs/mmseqs/bin/mmseqs"))
TARGET_DB = Path(os.environ.get("EXP301_TARGET_DB", "/data/exp213_overlap/targetDB"))
SENSITIVITY = os.environ.get("EXP301_MMSEQS_SENSITIVITY", "7.5")
MAX_SEQS = os.environ.get("EXP301_MMSEQS_MAX_SEQS", "20")

BUCKET = "hf://buckets/open-athena/MarinFold/data/document_structures"
#: Which corpus each training arm lives in.
#:
#: These are the **undecontaminated** corpora on purpose. exp213's DB was built
#: from them, so its ``{shard}_{row}`` coordinates address them exactly; #225's
#: decontamination permuted the corpus index, so the same coordinates land on a
#: different document in the ``*_decontam`` shards -- all 68 lookups came back as
#: row mismatches when pointed there. The document itself is unaffected:
#: decontamination removed rows, it did not rewrite them, so the conformer
#: AF2/ESMFold predicted for a surviving sequence is identical in both.
#:
#: What this costs is exposure precision, and only slightly: #225 filtered
#: against the FoldBench eval chains, not against fold switchers, so a row here
#: is missing from exp277's mix only if it happened to be homologous to a
#: FoldBench protein. Read the labels as "what the training data for this
#: sequence encodes", not "this exact row was in exp277's batch".
ARM_PREFIX = {
    "afdb": f"{BUCKET}/contacts_v1/train",
    "esm_atlas": f"{BUCKET}/contacts_v1_esm_atlas/train",
}
#: exp213 built its DB from the *undecontaminated* corpora, and labels the arms
#: with its own names. Map them onto the decontaminated shards Stage B reads.
ARM_ALIASES = {
    "afdb": "afdb", "train_afdb": "afdb", "contacts_v1": "afdb",
    "esm_atlas": "esm_atlas", "train_esm_atlas": "esm_atlas",
    "contacts_v1_esm_atlas": "esm_atlas",
}

HEADER_RE = re.compile(r"^(?P<arm>[^|]+)\|(?P<shard>\d+)_(?P<row>\d+)_(?P<entry>.+)$")

HIT_COLUMNS = [
    "query", "target", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits",
]


def load_universe() -> list[dict]:
    records = [json.loads(line) for line in (DATA / "foldswitch_universe.jsonl").open()]
    passing = {
        row["pair_id"]
        for row in csv.DictReader((DATA / "premise_gate.csv").open())
        if row["passes_gate"] == "True"
    }
    return [r for r in records if r["pair_id"] in passing]


# --------------------------------------------------------------------------
# Stage A — search
# --------------------------------------------------------------------------
def run_search(threads: int) -> Path:
    """MMseqs2 the reference sequences against the training DB."""
    if not MMSEQS.exists():
        raise FileNotFoundError(f"mmseqs not found at {MMSEQS}; set EXP301_MMSEQS")
    if not TARGET_DB.with_suffix(".dbtype").exists():
        raise FileNotFoundError(f"training DB not found at {TARGET_DB}; see exp213")

    work = CACHE / "mmseqs"
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / "queries.fasta"
    records = load_universe()
    with fasta.open("w") as fh:
        for r in records:
            fh.write(f">{r['pair_id']}\n{r['sequence']}\n")
    print(f"[search] {len(records)} query sequences -> {fasta}")

    query_db, result_db = work / "queryDB", work / "resultDB"
    tmp = work / "tmp"
    hits = DATA / "training_hits.tsv"

    def run(*args: str) -> None:
        print(f"[search] mmseqs {' '.join(args[:2])} ...", flush=True)
        subprocess.run([str(MMSEQS), *args], check=True)

    run("createdb", str(fasta), str(query_db))
    run("search", str(query_db), str(TARGET_DB), str(result_db), str(tmp),
        "-s", SENSITIVITY, "--max-seqs", MAX_SEQS, "--threads", str(threads))
    run("convertalis", str(query_db), str(TARGET_DB), str(result_db), str(hits),
        "--format-output", ",".join(HIT_COLUMNS), "--threads", str(threads))
    print(f"[search] wrote {hits}")
    return hits


def best_hits(path: Path) -> dict[str, dict]:
    """Highest-identity training hit per query, keeping ties deterministic."""
    best: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(HIT_COLUMNS):
                continue
            row = dict(zip(HIT_COLUMNS, parts))
            row["fident"] = float(row["fident"])
            row["qcov"] = float(row["qcov"])
            row["bits"] = float(row["bits"])
            key = row["query"]
            incumbent = best.get(key)
            if incumbent is None or (row["fident"], row["bits"]) > (incumbent["fident"], incumbent["bits"]):
                best[key] = row
    return best


# --------------------------------------------------------------------------
# Stage B — label
# --------------------------------------------------------------------------
_SHARD_LISTING: dict[str, list[str]] = {}


def _shards(arm: str) -> list[str]:
    """Shard paths for one arm, listed once per process (2067 entries each)."""
    import fsspec

    if arm not in _SHARD_LISTING:
        fs, root = fsspec.core.url_to_fs(ARM_PREFIX[arm])
        _SHARD_LISTING[arm] = sorted(
            f["name"] for f in fs.ls(root, detail=True) if f["name"].endswith(".parquet"))
    return _SHARD_LISTING[arm]


def corpus_row(arm: str, shard: int, row: int):
    """Read one row of one corpus shard off the public bucket.

    Only the row group containing ``row`` is fetched, not the whole shard --
    parquet footers make that a range request rather than a multi-hundred-MB
    download, which is what keeps 66 lookups over a slow link tolerable.
    """
    import fsspec
    import pyarrow.parquet as pq

    match = [s for s in _shards(arm) if f"-{shard:05d}-of-" in s]
    if not match:
        return None
    with fsspec.open(f"hf://{match[0]}", "rb") as fh:
        parquet = pq.ParquetFile(fh)
        if row >= parquet.metadata.num_rows:
            return None
        # Walk row groups to find the one holding this row, then read just it.
        offset = 0
        for group in range(parquet.num_row_groups):
            n = parquet.metadata.row_group(group).num_rows
            if offset + n > row:
                return parquet.read_row_group(group).slice(row - offset, 1).to_pylist()[0]
            offset += n
    return None


def training_contacts(record: dict, reference: str) -> tuple[set[tuple[int, int]], str, float] | None:
    """A corpus row's contact set, moved into the pair's reference frame."""
    from marinfold.document_structures.contacts_v1.read import live_contacts, sequence_from_document

    document = record["document"]
    seq_len, n_term = int(record["seq_len"]), int(record["n_term_index"])
    sequence = sequence_from_document(document, seq_len, n_term)
    mapping = align_obs_to_ref(sequence, reference)
    matched = sum(1 for k, c in enumerate(mapping) if c is not None and sequence[k] == reference[c])
    identity = matched / len(sequence) if sequence else 0.0

    # Document positions are the wrapped 2000-index clock; undo it to sequence
    # indices, then through the alignment into reference coordinates.
    seq_of_pos = {(n_term + t) % 2000: t for t in range(seq_len)}
    contacts: set[tuple[int, int]] = set()
    for pos_a, pos_b in live_contacts(document):
        sa, sb = seq_of_pos.get(pos_a), seq_of_pos.get(pos_b)
        if sa is None or sb is None:
            continue
        ra = mapping[sa] if sa < len(mapping) else None
        rb = mapping[sb] if sb < len(mapping) else None
        if ra is None or rb is None or ra == rb:
            continue
        contacts.add((min(ra, rb), max(ra, rb)))
    return contacts, sequence, identity


LABEL_COLUMNS = [
    "pair_id", "fold1", "fold2", "tier", "seq_class",
    "hit_target", "hit_arm", "hit_shard", "hit_row", "hit_entry_id",
    "hit_identity", "hit_qcov", "hit_evalue",
    "corpus_status", "uniprot_accession", "train_seq_identity_to_reference",
    "n_train_contacts", "n_hit_a", "n_hit_b", "n_a", "n_b",
    "recall_a", "recall_b", "training_fold",
]


def label_one(record: dict, hit: dict | None) -> dict:
    """One row of the training-fold table."""
    out = {c: None for c in LABEL_COLUMNS}
    out.update({
        "pair_id": record["pair_id"], "fold1": record["fold1"], "fold2": record["fold2"],
        "tier": record["tier"], "seq_class": record["seq_class"],
    })
    set_a = {(i, j) for i, j in record["contacts_fold1"]}
    set_b = {(i, j) for i, j in record["contacts_fold2"]}
    only_a, only_b = set_a - set_b, set_b - set_a
    out["n_a"], out["n_b"] = len(only_a), len(only_b)

    if hit is None:
        out["corpus_status"] = "no_training_hit"
        out["training_fold"] = "none"
        return out

    parsed = HEADER_RE.match(hit["target"])
    if parsed is None:
        out["corpus_status"] = f"unparsable_header:{hit['target'][:40]}"
        return out
    arm = ARM_ALIASES.get(parsed["arm"])
    out.update({
        "hit_target": hit["target"], "hit_arm": arm or parsed["arm"],
        "hit_shard": int(parsed["shard"]), "hit_row": int(parsed["row"]),
        "hit_entry_id": parsed["entry"], "hit_identity": hit["fident"],
        "hit_qcov": hit["qcov"], "hit_evalue": hit["evalue"],
    })
    if arm is None:
        out["corpus_status"] = f"unknown_arm:{parsed['arm']}"
        return out

    row = corpus_row(arm, int(parsed["shard"]), int(parsed["row"]))
    if row is None:
        out["corpus_status"] = "row_out_of_range"
        out["training_fold"] = "unknown"
        return out
    if row["entry_id"] != parsed["entry"]:
        # The coordinates must land on the entry the hit named. Anything else
        # means the corpus being read is not the one exp213 indexed, and scoring
        # it would silently attribute a different protein's fold.
        out["corpus_status"] = f"row_mismatch:{row['entry_id']}"
        out["training_fold"] = "unknown"
        return out

    contacts, _, identity = training_contacts(row, record["sequence"])
    out.update({
        "corpus_status": "ok", "uniprot_accession": row.get("uniprot_accession"),
        "train_seq_identity_to_reference": round(identity, 4),
        "n_train_contacts": len(contacts),
        "n_hit_a": len(contacts & only_a), "n_hit_b": len(contacts & only_b),
    })
    recall_a = len(contacts & only_a) / len(only_a) if only_a else 0.0
    recall_b = len(contacts & only_b) / len(only_b) if only_b else 0.0
    out["recall_a"], out["recall_b"] = round(recall_a, 4), round(recall_b, 4)
    # A label only when one fold is clearly better recovered; the training
    # structure is a prediction and may be neither conformer.
    if max(recall_a, recall_b) < 0.2:
        out["training_fold"] = "neither"
    elif recall_a >= 1.5 * recall_b:
        out["training_fold"] = "fold1"
    elif recall_b >= 1.5 * recall_a:
        out["training_fold"] = "fold2"
    else:
        out["training_fold"] = "ambiguous"
    return out


def run_label() -> Path:
    hits_path = DATA / "training_hits.tsv"
    if not hits_path.exists():
        raise FileNotFoundError(f"{hits_path} missing — run `audit_training_fold.py search` first")
    hits = best_hits(hits_path)
    records = load_universe()
    rows = []
    for n, record in enumerate(records, 1):
        row = label_one(record, hits.get(record["pair_id"]))
        rows.append(row)
        print(f"[label] [{n}/{len(records)}] {row['pair_id']:16s} "
              f"id={row['hit_identity'] if row['hit_identity'] is not None else float('nan'):.3f} "
              f"status={row['corpus_status']} "
              f"recall_A={row['recall_a']} recall_B={row['recall_b']} -> {row['training_fold']}",
              flush=True)

    out = DATA / "training_fold_labels.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    print(f"\ntraining-fold labels: {dict(sorted(Counter(r['training_fold'] for r in rows).items()))}")
    print(f"corpus status:        {dict(sorted(Counter(r['corpus_status'] for r in rows).items()))}")
    ids = [r["hit_identity"] for r in rows if r["hit_identity"] is not None]
    if ids:
        ids.sort()
        print(f"best-hit identity:    median {ids[len(ids) // 2]:.3f}  "
              f"range {ids[0]:.3f}-{ids[-1]:.3f}  |  >=0.9: {sum(i >= 0.9 for i in ids)}/{len(ids)}")
    print(f"wrote {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["search", "label"])
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    if args.stage == "search":
        run_search(args.threads)
    else:
        run_label()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
