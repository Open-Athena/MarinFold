# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3 — the base rate: how often does *any* recent natural PDB protein
escape eval2's filter? (A7)

Everything else in this experiment is conditioned on eval2 membership, and
eval2 is *defined* as "< 40 % identity to the training arms". That makes most
per-protein statistics circular: of course the 78 have no close relative in the
corpus — that is what put them there. The uncircular question is the one the
issue actually asks, and it needs a sample that was never selected on the
outcome:

    Take recent PDB protein chains at random. What fraction of them have **no**
    >= 40 %-identity relative in the 70.9 M sequences we trained on?

If that fraction is ~0 %, eval2-natural is an anomaly and something is wrong
with the pipeline. If it is 10-30 %, eval2-natural is the arithmetic of a
sampled corpus meeting a large protein universe, and needs no other explanation.

The sample is drawn by requesting small pages of RCSB's protein polymer entities
deposited since 2022 at **random offsets** into the (PDB-ID-ordered) result set,
seeded, then reduced to one entity per entry and deduplicated by exact sequence.
The search is exp213's, against exp213's existing ``targetDB`` — same binary,
same ``-s 7.5 --max-seqs 5000 -e 10``, same reduction (``evalue <= 1e-3`` and
``qcov >= 0.50``), so a survival number here is directly comparable to eval2's.

    uv run python measure_base_rate.py --n 1200
    uv run python measure_base_rate.py --skip-search   # re-reduce an existing .m8
"""
import argparse
import csv
import json
import random
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import annotate_rcsb as A
import upstream as U

DATA = U.HERE / "data"
CACHE = DATA / "base_rate_cache.json"
OUT = DATA / "base_rate_per_protein.csv"
SUMMARY = DATA / "base_rate_summary.csv"

#: exp213's published search parameters, mirrored from exp226's ``search_expanded``.
MAX_SEQS = 5000
SENSITIVITY = 7.5
REPORTING_EVALUE = 10.0
TAG = "exp241base"

#: The sampling frame. Deposited from 2022 on — the era CAMEO, FoldBench and the
#: newer PDB eval rows are drawn from, and comfortably after AFDB's UniProt
#: snapshot. Length is bounded to contacts-v1's serializable range's useful part
#: and to what an eval protein looks like.
DEPOSIT_FROM = "2022-01-01"
MIN_LEN, MAX_LEN = 50, 1000

#: Entities per random offset. Small, because consecutive offsets return
#: entities of the *same* entry — a large page would sample entries, not
#: proteins, and would over-weight big complexes.
PAGE = 4
SEED = 241


def search_frame(start: int, rows: int) -> tuple[int, list[str]]:
    nodes = [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.deposit_date",
            "operator": "greater_or_equal", "value": DEPOSIT_FROM}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "entity_poly.rcsb_entity_polymer_type",
            "operator": "exact_match", "value": "Protein"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "entity_poly.rcsb_sample_sequence_length",
            "operator": "range", "value": {"from": MIN_LEN, "to": MAX_LEN}}},
    ]
    payload = {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "polymer_entity",
        "request_options": {"paginate": {"start": start, "rows": rows},
                            "results_content_type": ["experimental"]},
    }
    data = A.post(A.RCSB_SEARCH, payload)
    return (data.get("total_count", 0),
            [r["identifier"] for r in data.get("result_set", [])])


def sample_entities(n: int, cache: dict) -> list[str]:
    """Random-offset sample of entity ids from the frame, seeded and cached."""
    if "entities" in cache and len(cache["entities"]) >= n:
        return cache["entities"][:n]
    total, _ = search_frame(0, 1)
    print(f"[frame] {total:,} protein entities deposited since {DEPOSIT_FROM}",
          flush=True)
    rng = random.Random(SEED)
    seen: list[str] = list(cache.get("entities", []))
    have = set(seen)
    while len(seen) < n:
        offset = rng.randrange(0, max(1, total - PAGE))
        _, ids = search_frame(offset, PAGE)
        for identifier in ids:
            if identifier not in have:
                have.add(identifier)
                seen.append(identifier)
        if len(seen) % 200 < PAGE:
            print(f"[frame] sampled {len(seen)}/{n}", flush=True)
            cache["entities"] = seen
        time.sleep(0.1)
    cache["entities"] = seen
    cache["frame_total"] = total
    return seen[:n]


def build_cohort(entity_ids: list[str], cache: dict) -> list[dict]:
    """Fetch each sampled entity's entry, then annotate it exactly as A2 does."""
    entries = cache.setdefault("entries", {})
    wanted = sorted({e.split("_")[0].lower() for e in entity_ids})
    fresh = [e for e in wanted if e not in entries]
    if fresh:
        entries.update(A.fetch_entries(fresh))

    rng = random.Random(SEED)
    by_entry: dict[str, list[str]] = {}
    for identifier in entity_ids:
        by_entry.setdefault(identifier.split("_")[0].lower(), []).append(identifier)

    rows, seen_seq = [], set()
    for entry_id, identifiers in by_entry.items():
        entry = entries.get(entry_id)
        if entry is None:
            continue
        # One entity per entry, chosen at random among the ones sampled, so a
        # 30-chain ribosome contributes exactly as much as a lysozyme.
        target = rng.choice(sorted(identifiers))
        entity = next((e for e in (entry.get("polymer_entities") or [])
                       if e.get("rcsb_id", "").upper() == target.upper()), None)
        if entity is None:
            continue
        seq = A.entity_seq(entity)
        if not (MIN_LEN <= len(seq) <= MAX_LEN) or seq in seen_seq:
            continue
        seen_seq.add(seq)

        organisms = entity.get("rcsb_entity_source_organism") or []
        taxids = [str(o.get("ncbi_taxonomy_id") or "") for o in organisms]
        lineage = {str(t.get("name")) for o in organisms
                   for t in (o.get("taxonomy_lineage") or [])}
        names = "|".join(str(o.get("ncbi_scientific_name") or "") for o in organisms)
        keywords = entry.get("struct_keywords") or {}
        kw_text = (f"{keywords.get('pdbx_keywords') or ''} "
                   f"{keywords.get('text') or ''}").upper()
        xrefs = (entity.get("rcsb_polymer_entity_container_identifiers")
                 or {}).get("reference_sequence_identifiers") or []
        rows.append({
            "entity_id": target, "entry_id": entry_id, "length": len(seq),
            "sequence": seq,
            "deposit_date": ((entry.get("rcsb_accession_info") or {})
                             .get("deposit_date") or "")[:10],
            "source_organisms": names,
            "kingdom": A.kingdom_of(lineage, names),
            "is_synthetic_taxon": int(bool(organisms)
                                      and all(t == A.SYNTHETIC_TAXID for t in taxids)
                                      or not organisms),
            "has_denovo_keyword": int(A.DENOVO_KEYWORD in kw_text),
            "has_uniprot_xref": int(any(
                (x.get("database_name") or "").upper() == "UNIPROT" for x in xrefs)),
            "title": (entry.get("struct") or {}).get("title", "") or "",
        })
    for row in rows:
        row["designed_signal"] = int(row["is_synthetic_taxon"]
                                     or row["has_denovo_keyword"])
    return rows


def run_search(work: Path, cohort: list[dict], threads: int,
               split_memory_limit: str) -> Path:
    target_db = work / "targetDB"
    if not target_db.exists():
        raise SystemExit(
            f"{target_db} is gone; exp213's 70.9 M-sequence target database is "
            "the input to this step and rebuilding it is out of scope here.")
    mmseqs = U.ensure_mmseqs()
    query_fasta = work / f"query_{TAG}.fasta"
    query_fasta.write_text(
        "".join(f">base__{r['entity_id']}\n{r['sequence']}\n" for r in cohort))
    query_db = work / f"queryDB_{TAG}"
    aln_db = work / f"alnDB_{TAG}"
    tmp = work / f"mmseqs_tmp_{TAG}"
    for stale in list(work.glob(f"alnDB_{TAG}*")) + list(work.glob(f"queryDB_{TAG}*")):
        stale.unlink()
    U.run([mmseqs, "createdb", query_fasta, query_db])
    t0 = time.time()
    U.run([mmseqs, "search", query_db, target_db, aln_db, tmp,
           "-s", SENSITIVITY, "--max-seqs", MAX_SEQS, "-e", REPORTING_EVALUE,
           "--threads", threads, "--split-memory-limit", split_memory_limit])
    print(f"[mmseqs] search in {time.time() - t0:.0f}s", flush=True)
    m8 = work / f"aln_{TAG}.m8"
    U.run([mmseqs, "convertalis", query_db, target_db, aln_db, m8,
           "--format-output", U.FORMAT, "--threads", threads])
    return m8


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1200,
                    help="entity ids to sample before per-entry reduction")
    ap.add_argument("--work", type=Path, default=U.ARM_FASTA_DIR)
    ap.add_argument("--threads", type=int, default=max(1, (subprocess.os.cpu_count() or 8) - 2))
    ap.add_argument("--split-memory-limit", default="64G")
    ap.add_argument("--skip-search", action="store_true")
    args = ap.parse_args(argv)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    entity_ids = sample_entities(args.n, cache)
    CACHE.write_text(json.dumps(cache))
    cohort = build_cohort(entity_ids, cache)
    CACHE.write_text(json.dumps(cache))
    print(f"[cohort] {len(cohort)} unique sequences from {len(entity_ids)} entities "
          f"({sum(r['designed_signal'] for r in cohort)} carry a designed signal)",
          flush=True)

    m8 = args.work / f"aln_{TAG}.m8"
    if not args.skip_search:
        m8 = run_search(args.work, cohort, args.threads, args.split_memory_limit)
    if not m8.exists():
        raise SystemExit(f"{m8} does not exist; drop --skip-search")

    # exp213's reducer keys on the FASTA header and reads `dataset`/`stem`/
    # `query_len` back out of this map, so the reduction that produced eval2's
    # identities is the one that produces these — no re-implementation.
    meta = {f"base__{r['entity_id']}": {"dataset": "base_rate",
                                        "stem": r["entity_id"],
                                        "query_len": r["length"],
                                        # Foldseek/Neff strata are eval-set
                                        # columns this cohort has no analogue
                                        # for; blanks keep the reducer's schema.
                                        **{c: "" for c in U.MANIFEST_STRATA}}
            for r in cohort}
    reduced = {r["stem"]: r for r in U.reduce_alignments(m8, meta, MAX_SEQS)}

    for row in cohort:
        rec = reduced.get(row["entity_id"], {})
        gated = rec.get("best_identity_covered")
        row["best_identity"] = "" if gated in (None, "") else f"{float(gated):.3f}"
        row["n_hits"] = rec.get("n_hits", 0)
        row["n_hits_significant"] = rec.get("n_hits_significant", 0)
        row["afdb_best_identity"] = rec.get("afdb_best_identity_covered", "")
        row["esm_atlas_best_identity"] = rec.get("esm_atlas_best_identity_covered", "")
        row["passes_40"] = int(gated in (None, "") or float(gated) < 0.40)
        row["passes_30"] = int(gated in (None, "") or float(gated) < 0.30)
        row.pop("sequence")

    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cohort[0]))
        writer.writeheader()
        writer.writerows(cohort)

    def rate(rows, key):
        return (len(rows), sum(r[key] for r in rows),
                sum(r[key] for r in rows) / len(rows) if rows else float("nan"))

    natural = [r for r in cohort if not r["designed_signal"]]
    designed = [r for r in cohort if r["designed_signal"]]
    summary = []
    for label, subset in (("all", cohort), ("natural", natural),
                          ("designed_signal", designed)):
        for threshold in ("passes_40", "passes_30"):
            n, k, p = rate(subset, threshold)
            summary.append({"subset": label, "filter": threshold, "n": n,
                            "survivors": k, "rate": f"{p:.4f}"})
    for kingdom in sorted({r["kingdom"] for r in natural}):
        subset = [r for r in natural if r["kingdom"] == kingdom]
        n, k, p = rate(subset, "passes_40")
        summary.append({"subset": f"natural/{kingdom}", "filter": "passes_40",
                        "n": n, "survivors": k, "rate": f"{p:.4f}"})
    with SUMMARY.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["subset", "filter", "n",
                                                "survivors", "rate"])
        writer.writeheader()
        writer.writerows(summary)

    for row in summary:
        print(f"  {row['subset']:22s} {row['filter']:10s} "
              f"{row['survivors']:4d}/{row['n']:<5d} = {float(row['rate']):.1%}")
    print(f"[out] {OUT}\n[out] {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
