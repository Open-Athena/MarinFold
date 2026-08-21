# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 — the eval-decontamination drop list for exp230's protein pool.

The issue's requirement is one sentence: **nothing in the fine-tuning pool may
be >= 30 % sequence-similar to the eval set.**  Operationally that is #225's
Tier A rule (``identity >= 30 % over >= 50 % query coverage``, **or**
``E <= 1e-3``), applied to three corpora and one query set.

Two things make this cheap rather than a multi-hour rebuild:

* **exp225 already searched the 554-protein #89 benchmark** against #213's
  70,889,604-sequence ``targetDB`` (both AFDB and ESM-Atlas), at
  ``--max-seqs 1000000`` so nothing is censored, and left the result at
  ``/data/exp225_decontam/droplist_sequence.parquet``.  That file is consumed
  verbatim.
* **#226's 776-query set is a strict superset of those 554** — verified here,
  key-for-key and sequence-for-sequence, not assumed.  So the only incremental
  search the predicted corpora need is the **222 net-new queries**.

The PDB arm (#222's ``contacts_v1_pdb_deduped_monomers``) has never been
searched against anything, and it is the arm where contamination is most
likely: #222 excluded the eval set's 552 entries *by PDB id*, but still
measured **50.2 % of eval entries with a 40 % homolog** in the corpus.  Id
exclusion is not identity exclusion.  Its 41,661 sequences get a fresh DB and
the full 776.

Why 776 and not the 554 we gate on: it costs one small search, it is a
superset, and it protects the benchmark we are moving to as well as the one we
report.  Decontaminating against proteins we do *not* gate on can only make the
pool cleaner.

    uv run python decontam.py --work /data/exp230_multi
    uv run python decontam.py --work /data/exp230_multi --stage pdb   # just one
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from _decontam_lib import (
    CONVERTALIS_FORMAT,
    PDB_FASTA_TAG,
    REPORT_EVALUE_CEILING,
    SEARCH_SENSITIVITY,
    SEQ_MAX_EVALUE,
    SEQ_MIN_IDENTITY,
    SEQ_MIN_QCOV,
    ensure_mmseqs,
    format_target,
    is_sequence_contaminant,
    parse_target,
    run,
)
from corpus_sources import PDB_MONOMERS, iter_corpus_rows

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: exp225's Tier-A drop list over AFDB + ESM-Atlas, 554 queries.  1,124,983 rows.
EXP225_DROPLIST = Path("/data/exp225_decontam/droplist_sequence.parquet")

#: #213's 70.9M-sequence MMseqs2 DB over both predicted corpora.  Reused, never
#: rebuilt — rebuilding is a 146 GB stream plus several hours.
EXP213_TARGET_DB = Path("/data/exp213_overlap/targetDB")

FIELDS = CONVERTALIS_FORMAT.split(",")

DROPLIST_SCHEMA = pa.schema(
    [
        ("arm", pa.string()),
        ("entry_id", pa.string()),
        ("best_identity", pa.float32()),
        ("best_qcov", pa.float32()),
        ("best_evalue", pa.float64()),
        ("nearest_eval_key", pa.string()),
        ("rule", pa.string()),
        ("source", pa.string()),
    ]
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_fasta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    key = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            key = line[1:]
            out[key] = ""
        elif key is not None:
            out[key] += line
    return out


def write_fasta(path: Path, records: dict[str, str]) -> None:
    with path.open("w") as fh:
        for key, seq in records.items():
            fh.write(f">{key}\n{seq}\n")


# --- Stage 1: the query set -------------------------------------------------


def stage_queries(work: Path, log) -> tuple[dict[str, str], dict[str, str]]:
    """Return (all 776 queries, the 222 net-new ones), after proving the subset."""
    q776 = read_fasta(work / "eval776.fasta")
    q554 = read_fasta(work / "eval554.fasta")
    if len(q776) != 776 or len(q554) != 554:
        raise SystemExit(f"expected 776/554 queries, got {len(q776)}/{len(q554)}")

    # The whole reuse argument rests on this, so it is checked rather than
    # asserted in prose: every #225 query must be present in #226's set with a
    # byte-identical sequence, or the inherited drop list is not the drop list
    # for these queries.
    missing = [k for k in q554 if k not in q776]
    disagree = [k for k in q554 if k in q776 and q554[k] != q776[k]]
    if missing or disagree:
        raise SystemExit(
            f"#226's 776 is not a superset of #225's 554: "
            f"{len(missing)} missing, {len(disagree)} sequence disagreements"
        )
    new = {k: v for k, v in q776.items() if k not in q554}
    log(f"[queries] 776 total; 554 inherited from #225; {len(new)} net-new to search")
    write_fasta(work / "eval222_new.fasta", new)
    return q776, new


# --- mmseqs ------------------------------------------------------------------


def search(
    mmseqs: str,
    query_fasta: Path,
    target_db: Path,
    work: Path,
    tag: str,
    *,
    threads: int,
    max_seqs: int,
    search_evalue: float,
    log,
) -> Path:
    """Search ``query_fasta`` against ``target_db``; return the .m8 path.

    The search runs far looser (``search_evalue``) than the tier reports at
    (:data:`REPORT_EVALUE_CEILING`), exactly as #225 does, so the reporting
    threshold stays a reduce-time decision that can be swept off one search.
    """
    m8 = work / f"aln_{tag}.m8"
    if m8.exists():
        log(f"[mmseqs] reusing {m8} ({m8.stat().st_size / 1e6:.0f} MB)")
        return m8
    qdb = work / f"queryDB_{tag}"
    aln = work / f"alnDB_{tag}"
    tmp = work / f"tmp_{tag}"
    for stale in (aln, tmp):
        # mmseqs refuses to overwrite an existing result DB or search tmp.
        if stale.is_dir():
            shutil.rmtree(stale)
        for sibling in work.glob(stale.name + "*"):
            sibling.unlink() if sibling.is_file() else shutil.rmtree(sibling)
    if not (work / f"queryDB_{tag}.dbtype").exists():
        run([mmseqs, "createdb", query_fasta, qdb], log=log)
    tmp.mkdir(exist_ok=True)
    t0 = time.time()
    run(
        [
            mmseqs, "search", qdb, target_db, aln, tmp,
            "-s", SEARCH_SENSITIVITY,
            "--max-seqs", max_seqs,
            "-e", search_evalue,
            "--threads", threads,
        ],
        log=log,
    )
    log(f"[mmseqs] search {tag} in {time.time() - t0:.0f}s")
    run(
        [mmseqs, "convertalis", qdb, target_db, aln, m8,
         "--format-output", CONVERTALIS_FORMAT, "--threads", threads],
        log=log,
    )
    log(f"[mmseqs] alignments -> {m8} ({m8.stat().st_size / 1e6:.0f} MB)")
    return m8


def reduce_hits(m8: Path, *, source: str, log, tag_to_arm: dict[str, str] | None = None) -> pd.DataFrame:
    """Collapse alignments into one drop row per (arm, entry_id).

    Keeps the *best* alignment per training row for reporting, but membership is
    decided by whether ANY alignment is a contaminant — the two are different
    questions and conflating them under-drops.
    """
    tag_to_arm = tag_to_arm or {}
    best: dict[tuple[str, str], dict] = {}
    n_lines = n_hits = 0
    with m8.open() as fh:
        for line in fh:
            n_lines += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(FIELDS):
                continue
            hit = dict(zip(FIELDS, parts))
            evalue = float(hit["evalue"])
            if evalue > REPORT_EVALUE_CEILING:
                continue
            identity, qcov = float(hit["fident"]), float(hit["qcov"])
            if not is_sequence_contaminant(identity, qcov, evalue):
                continue
            n_hits += 1
            row = parse_target(hit["target"])
            arm = tag_to_arm.get(row.arm, row.arm)
            row = row.__class__(arm=arm, shard=row.shard, row=row.row, entry_id=row.entry_id)
            key = (row.arm, row.entry_id)
            by_evalue = evalue <= SEQ_MAX_EVALUE
            by_identity = identity >= SEQ_MIN_IDENTITY and qcov >= SEQ_MIN_QCOV
            rule = "both" if (by_evalue and by_identity) else ("evalue" if by_evalue else "identity")
            prev = best.get(key)
            if prev is None or identity > prev["best_identity"]:
                best[key] = {
                    "arm": row.arm,
                    "entry_id": row.entry_id,
                    "best_identity": identity,
                    "best_qcov": qcov,
                    "best_evalue": evalue,
                    "nearest_eval_key": hit["query"],
                    "rule": rule,
                    "source": source,
                }
    log(f"[reduce] {m8.name}: {n_lines:,} alignments -> {n_hits:,} contaminating -> {len(best):,} rows")
    return pd.DataFrame(list(best.values()))


# --- Stage 3: the PDB arm ---------------------------------------------------


def build_pdb_fasta(work: Path, log) -> tuple[Path, int]:
    """Materialise the PDB monomer sequences under #213's header grammar.

    Writing ``{tag}|{shard}_{row}_{entry_id}`` means a hit inverts straight to a
    corpus row with no join — the property that makes #213's DB reusable, kept
    for ours so both arms reduce through identical code.

    **No quality filter here, deliberately.**  The drop list is a statement
    about the *corpus*, so its denominator has to be the corpus; filtering first
    would make the reported rate a rate over the pool and not comparable to
    #225's 1.89 % / 1.57 %.  The pool's own quality gates are applied later, in
    ``select_targets.py``.
    """
    fasta = work / "pdb_monomers.fasta"
    count_file = work / "pdb_monomers.count"
    if fasta.exists() and count_file.exists():
        log(f"[pdb] reusing {fasta}")
        return fasta, int(count_file.read_text())
    n = 0
    with fasta.open("w") as fh:
        for rec in iter_corpus_rows(PDB_MONOMERS, work=work, log=log,
                                    max_len=2000, min_contacts=0):
            header = format_target(PDB_FASTA_TAG, rec["shard"], rec["row"], rec["entry_id"])
            fh.write(f">{header}\n{rec['sequence']}\n")
            n += 1
    count_file.write_text(str(n))
    log(f"[pdb] {n:,} sequences -> {fasta}")
    return fasta, n


# --- driver ------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/data/exp230_multi"))
    ap.add_argument("--stage", choices=["all", "predicted", "pdb"], default="all")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--max-seqs", type=int, default=1_000_000,
                    help="prefilter depth; #225 used 1e6 and confirmed no censoring")
    ap.add_argument("--search-evalue", type=float, default=1000.0)
    a = ap.parse_args()

    work: Path = a.work
    work.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(exist_ok=True)
    logf = (work / "decontam.log").open("a")

    def log(*msg):
        line = " ".join(str(m) for m in msg)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    mmseqs = ensure_mmseqs(log=log)
    log(f"[mmseqs] binary: {mmseqs}")
    q776, _ = stage_queries(work, log)

    frames: list[pd.DataFrame] = []
    provenance: dict = {
        "tier": "A",
        "rule": f"identity >= {SEQ_MIN_IDENTITY:.0%} over >= {SEQ_MIN_QCOV:.0%} qcov, or E <= {SEQ_MAX_EVALUE:g}",
        "sensitivity": SEARCH_SENSITIVITY,
        "report_evalue_ceiling": REPORT_EVALUE_CEILING,
        "queries": {"n": len(q776), "sha256": sha256(work / "eval776.fasta")},
    }

    if a.stage in ("all", "predicted"):
        if not EXP225_DROPLIST.exists():
            raise SystemExit(f"missing {EXP225_DROPLIST} — run #225's sequence_droplist.py first")
        inherited = pq.read_table(EXP225_DROPLIST).to_pandas()
        inherited["source"] = "exp225_554"
        keep = ["arm", "entry_id", "best_identity", "best_qcov", "best_evalue",
                "nearest_eval_key", "rule", "source"]
        frames.append(inherited[keep])
        log(f"[predicted] inherited {len(inherited):,} rows from #225 "
            f"({inherited.groupby('arm').size().to_dict()})")
        provenance["exp225_droplist"] = {
            "path": str(EXP225_DROPLIST), "rows": int(len(inherited)),
            "sha256": sha256(EXP225_DROPLIST),
        }

        m8 = search(mmseqs, work / "eval222_new.fasta", EXP213_TARGET_DB, work, "new222",
                    threads=a.threads, max_seqs=a.max_seqs,
                    search_evalue=a.search_evalue, log=log)
        frames.append(reduce_hits(m8, source="exp230_new222", log=log))

    if a.stage in ("all", "pdb"):
        pdb_fasta, n_pdb = build_pdb_fasta(work, log)
        pdb_db = work / "pdbDB"
        if not (work / "pdbDB.dbtype").exists():
            run([mmseqs, "createdb", pdb_fasta, pdb_db], log=log)
        m8 = search(mmseqs, work / "eval776.fasta", pdb_db, work, "pdb",
                    threads=a.threads, max_seqs=a.max_seqs,
                    search_evalue=a.search_evalue, log=log)
        pdb_rows = reduce_hits(m8, source="exp230_pdb776", log=log,
                               tag_to_arm={PDB_FASTA_TAG: "pdb"})
        frames.append(pdb_rows)
        provenance["pdb_corpus_sequences"] = n_pdb
        log(f"[pdb] dropped {len(pdb_rows):,} / {n_pdb:,} "
            f"({100 * len(pdb_rows) / max(n_pdb, 1):.2f}%)")

    if not frames:
        raise SystemExit("nothing to do")

    droplist = pd.concat(frames, ignore_index=True)
    # A row can be dropped by both the inherited 554 and the new 222; keep the
    # strongest evidence so best_identity means "worst-case similarity".
    droplist = (droplist.sort_values("best_identity", ascending=False)
                        .drop_duplicates(["arm", "entry_id"], keep="first")
                        .reset_index(drop=True))
    out = work / "droplist_exp230.parquet"
    pq.write_table(pa.Table.from_pandas(droplist, schema=DROPLIST_SCHEMA,
                                        preserve_index=False), out)
    log(f"[droplist] {len(droplist):,} rows -> {out}")

    summary = (droplist.groupby(["arm", "source"]).size()
               .rename("dropped").reset_index()
               .sort_values(["arm", "source"]))
    summary.to_csv(DATA / "decontam_by_source.csv", index=False)
    log("[summary]\n" + summary.to_string(index=False))

    provenance["dropped_by_arm"] = droplist.groupby("arm").size().to_dict()
    (DATA / "decontam.provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
