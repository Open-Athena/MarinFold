# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1 — resolve every eval protein to its RCSB entity and annotate it.

This is §A2 of issue #241: the designed/natural label audit. exp226 resolved
RCSB source organisms **only for the 334 FoldBench monomers**, so the 24
``cameo_hard`` and 19 ``casp_fm`` rows inside eval2-natural carry
``designed_any = 0`` as a *default* — nothing ever looked. This step looks, for
every protein in the eval universe, and records four independent designed
signals so the verdict does not hang on any one field:

1. ``rcsb_entity_source_organism.ncbi_scientific_name == "synthetic construct"``
   (taxon 32630) — the exp226 proxy, extended to the rows it skipped.
2. ``struct_keywords.pdbx_keywords`` containing ``DE NOVO PROTEIN`` — the PDB's
   own structural-genomics class, and what exp65's ``fetch_denovo_pdb`` selected
   the de novo set on in the first place.
3. Absence of any ``reference_sequence_identifiers`` UniProt cross-reference —
   a natural protein's entity almost always has one; a design has none.
4. The entry title, kept verbatim for adjudication (design papers say so).

It also collects what §A3 and §A6 need: the UniProt accession(s) per entity, the
deposit and initial-release dates, and the full NCBI taxonomy lineage (for §A4's
kingdom cross-tab).

**Resolving CASP domains.** A CASP stem (``T1027-D1``) names no PDB entry.
exp65's committed fallback map covers 6 of the 19; the rest came out of
predictioncenter tarballs and were never mapped. Those resolve here through
RCSB's sequence-search service at >= SEARCH_IDENTITY, which finds the deposited
entry the domain was clipped from. Every resolution records *how* it was made
(``entry_source``) and its identity/coverage, so a weak match is visible rather
than laundered.

    uv run python annotate_rcsb.py                 # all 776
    uv run python annotate_rcsb.py --cohort natural # just the 78 (fast)
    uv run python annotate_rcsb.py --offline        # re-derive from the cache
"""
import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import upstream as U

DATA = U.HERE / "data"
CACHE = DATA / "rcsb_cache.json"
OUT = DATA / "rcsb_annotation.csv"

RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
BATCH_SIZE = 40

ENTRY_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    rcsb_accession_info { deposit_date initial_release_date }
    struct_keywords { pdbx_keywords text }
    polymer_entities {
      rcsb_id
      entity_poly { type pdbx_seq_one_letter_code_can rcsb_mutation_count }
      rcsb_polymer_entity_container_identifiers {
        auth_asym_ids
        asym_ids
        reference_sequence_identifiers { database_accession database_name }
      }
      rcsb_entity_source_organism {
        ncbi_taxonomy_id
        ncbi_scientific_name
        taxonomy_lineage { id name }
      }
      rcsb_polymer_entity { pdbx_description }
    }
  }
}
"""

#: NCBI taxon the PDB assigns an entity with no natural source organism.
SYNTHETIC_TAXID = "32630"
DENOVO_KEYWORD = "DE NOVO PROTEIN"

#: Sequence-search settings for the CASP domains. Deliberately loose on
#: *coverage* (a domain covers only part of the deposited entity) and tight on
#: identity, which is the right way round: we want the entry this exact sequence
#: came from, not a homolog. 0.80 rather than 0.90 because a CASP target
#: sequence and the entry finally deposited for it are not always byte-identical
#: — T1104-D1 matches its entry (7ROA, EntV from *Enterococcus faecalis*) at
#: 0.886 and was the one stem a 0.90 gate left unresolved. Every resolution
#: records its identity in ``search_identity`` so a weak match stays visible.
SEARCH_IDENTITY = 0.80
SEARCH_EVALUE = 0.1

#: Kingdom buckets for §A4, tested against the NCBI lineage in order. "Viruses"
#: must precede the rest because a virus lineage contains no cellular clade.
KINGDOMS = (
    ("synthetic", ("artificial sequences",)),
    ("virus", ("Viruses",)),
    ("archaea", ("Archaea",)),
    ("bacteria", ("Bacteria",)),
    ("eukaryote", ("Eukaryota",)),
)


def post(url: str, payload: dict, *, retries: int = 5) -> dict:
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                if resp.status == 204:  # search API: no hits
                    return {}
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 204:
                return {}
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"unreachable: {url}")


def fetch_entries(pdb_ids: list[str]) -> dict[str, dict]:
    """GraphQL, batched. Keyed by lowercase entry id."""
    out: dict[str, dict] = {}
    ids = sorted({p.upper() for p in pdb_ids})
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i:i + BATCH_SIZE]
        data = post(RCSB_GRAPHQL, {"query": ENTRY_QUERY, "variables": {"ids": batch}})
        if "errors" in data:
            raise SystemExit(f"RCSB GraphQL errors: {data['errors']}")
        for entry in data["data"]["entries"] or []:
            if entry:
                out[entry["rcsb_id"].lower()] = entry
        print(f"[rcsb] entries {i + len(batch)}/{len(ids)}", flush=True)
        time.sleep(0.2)
    missing = set(p.lower() for p in pdb_ids) - set(out)
    if missing:
        print(f"[warn] RCSB returned nothing for {sorted(missing)}")
    return out


def search_by_sequence(sequence: str) -> list[dict]:
    """RCSB sequence search → ranked polymer-entity hits for one sequence."""
    payload = {
        "query": {
            "type": "terminal", "service": "sequence",
            "parameters": {
                "evalue_cutoff": SEARCH_EVALUE,
                "identity_cutoff": SEARCH_IDENTITY,
                "sequence_type": "protein",
                "value": sequence,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": 5},
            "results_content_type": ["experimental"],
            # ``verbose`` is not optional: it is the only verbosity that returns
            # ``match_context`` (the identity we gate on), and the service
            # answers 500 — not 400 — for the compact form of this query.
            "results_verbosity": "verbose",
        },
    }
    data = post(RCSB_SEARCH, payload)
    return data.get("result_set", []) or []


def is_subsequence(query: str, target: str) -> bool:
    """Do ``query``'s residues appear in ``target`` in order, gaps allowed?

    The right containment test here. An eval protein's sequence is the residues
    *observed* in the deposited structure (exp65 counted them off the coordinate
    file; a CASP stem is a clipped domain), while RCSB's
    ``pdbx_seq_one_letter_code_can`` is the full construct. Disordered loops and
    unmodelled termini mean the query is usually a gapped subset of the entity
    sequence, not a contiguous substring — so a substring test reports a
    "mismatch" for chains that are in fact the right chain.
    """
    it = iter(target)
    return all(residue in it for residue in query)


def entity_seq(entity: dict) -> str:
    return ((entity.get("entity_poly") or {})
            .get("pdbx_seq_one_letter_code_can") or "")


def select_entity(entry: dict, chain: str | None, sequence: str) -> dict | None:
    """Pick the polymer entity a stem names.

    Chain first (auth ids, then label asym ids — exp226's gotcha), and when the
    stem carries no chain (CASP) or the chain does not resolve, fall back to the
    entity whose canonical sequence contains the query — as a substring, then as
    a gapped subsequence — and only then to the longest entity.
    """
    entities = entry.get("polymer_entities") or []
    if not entities:
        return None
    if chain:
        for key in ("auth_asym_ids", "asym_ids"):
            for ent in entities:
                ids = (ent.get("rcsb_polymer_entity_container_identifiers")
                       or {}).get(key) or []
                if chain in ids:
                    return ent
    if sequence:
        for test in (lambda q, t: q in t, is_subsequence):
            for ent in entities:
                if test(sequence, entity_seq(ent)):
                    return ent
    return max(entities, key=lambda e: len(entity_seq(e)))


def kingdom_of(lineage_names: set[str], organism: str) -> str:
    for label, needles in KINGDOMS:
        if any(n in lineage_names for n in needles):
            return label
    if organism:
        return "unclassified"
    return "unknown"


def annotate(protein: U.Protein, entry: dict | None, entity: dict | None,
             entry_source: str, search_identity: float | None) -> dict:
    """One output row: the four designed signals + what §A3/§A4/§A6 need."""
    row = {
        "dataset": protein.dataset, "stem": protein.stem,
        "length": protein.length, "in_eval2": int(protein.in_eval2),
        "designed_any_published": int(protein.designed_any),
        "entry_id": (entry or {}).get("rcsb_id", "").lower(),
        "entity_id": (entity or {}).get("rcsb_id", ""),
        "entry_source": entry_source,
        "search_identity": "" if search_identity is None else f"{search_identity:.3f}",
        "title": "", "pdbx_keywords": "", "struct_keywords_text": "",
        "entity_description": "", "deposit_date": "", "initial_release_date": "",
        "source_organisms": "", "source_taxids": "", "kingdom": "unknown",
        "uniprot_accessions": "", "n_uniprot_xrefs": 0,
        "mutation_count": "", "entity_seq_len": "", "seq_match": "",
        "is_synthetic_taxon": 0, "has_denovo_keyword": 0, "has_uniprot_xref": 0,
    }
    if entry is None:
        return row
    row["title"] = (entry.get("struct") or {}).get("title", "") or ""
    kw = entry.get("struct_keywords") or {}
    row["pdbx_keywords"] = kw.get("pdbx_keywords") or ""
    row["struct_keywords_text"] = kw.get("text") or ""
    acc = entry.get("rcsb_accession_info") or {}
    row["deposit_date"] = (acc.get("deposit_date") or "")[:10]
    row["initial_release_date"] = (acc.get("initial_release_date") or "")[:10]
    row["has_denovo_keyword"] = int(
        DENOVO_KEYWORD in row["pdbx_keywords"].upper()
        or DENOVO_KEYWORD in row["struct_keywords_text"].upper()
    )
    if entity is None:
        return row

    organisms = entity.get("rcsb_entity_source_organism") or []
    row["source_organisms"] = "|".join(
        str(o.get("ncbi_scientific_name") or "") for o in organisms)
    taxids = [str(o.get("ncbi_taxonomy_id") or "") for o in organisms]
    row["source_taxids"] = "|".join(taxids)
    lineage = {
        str(t.get("name"))
        for o in organisms for t in (o.get("taxonomy_lineage") or [])
    }
    row["kingdom"] = kingdom_of(lineage, row["source_organisms"])
    # exp226's proxy: no *natural* source organism. An entity with no organism
    # record at all also fails it — that is the conservative direction and is
    # kept, so this column is comparable to exp226's `synthetic`.
    row["is_synthetic_taxon"] = int(
        bool(organisms) and all(t == SYNTHETIC_TAXID for t in taxids)
        or not organisms
    )

    ids = entity.get("rcsb_polymer_entity_container_identifiers") or {}
    xrefs = ids.get("reference_sequence_identifiers") or []
    accessions = [
        x["database_accession"] for x in xrefs
        if (x.get("database_name") or "").upper() == "UNIPROT"
        and x.get("database_accession")
    ]
    row["uniprot_accessions"] = "|".join(dict.fromkeys(accessions))
    row["n_uniprot_xrefs"] = len(accessions)
    row["has_uniprot_xref"] = int(bool(accessions))

    poly = entity.get("entity_poly") or {}
    canonical = poly.get("pdbx_seq_one_letter_code_can") or ""
    row["mutation_count"] = poly.get("rcsb_mutation_count")
    row["entity_seq_len"] = len(canonical)
    # The control on entity resolution. Every designed/natural verdict below is
    # a property of *this* entity, so it is only the right verdict if this entity
    # is the one the eval protein's sequence came from. ``exact``, ``substring``
    # (a construct tag trimmed off) and ``subsequence`` (unmodelled loops, or a
    # clipped CASP domain) all confirm the right chain; ``mismatch`` means the
    # resolution failed and the row's labels must not be trusted.
    if not protein.sequence:
        row["seq_match"] = "no_query_seq"
    elif protein.sequence == canonical:
        row["seq_match"] = "exact"
    elif protein.sequence in canonical:
        row["seq_match"] = "substring"
    elif is_subsequence(protein.sequence, canonical):
        row["seq_match"] = "subsequence"
    else:
        row["seq_match"] = "mismatch"
    row["entity_description"] = (
        (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "")
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["natural", "eval2", "all"], default="all",
                    help="which proteins to annotate (default: the whole universe)")
    ap.add_argument("--offline", action="store_true",
                    help="re-derive the CSV from data/rcsb_cache.json, no network")
    args = ap.parse_args(argv)

    proteins = U.read_proteins()
    U.eval2_natural(proteins)  # assert the 78 before doing any work
    if args.cohort == "natural":
        cohort = [p for p in proteins if p.in_eval2 and not p.designed_any]
    elif args.cohort == "eval2":
        cohort = [p for p in proteins if p.in_eval2]
    else:
        cohort = proteins
    print(f"[cohort] {len(cohort)} proteins ({args.cohort})", flush=True)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {
        "entries": {}, "searches": {}}
    casp_map = U.read_casp_pdb_map()

    # --- resolve each stem to an entry id ---------------------------------
    resolved: dict[str, tuple[str | None, str | None, str, float | None]] = {}
    need_search: list[U.Protein] = []
    for p in cohort:
        if p.pdb_id:
            resolved[p.stem] = (p.pdb_id, p.chain, "stem", None)
        elif p.stem in casp_map:
            pdb, chain = casp_map[p.stem]
            resolved[p.stem] = (pdb, chain or None, "exp65_fallback", None)
        else:
            need_search.append(p)

    if need_search and not args.offline:
        print(f"[rcsb] sequence-searching {len(need_search)} unmapped stems",
              flush=True)
    for p in need_search:
        hits = cache["searches"].get(p.stem)
        if hits is None:
            if args.offline:
                resolved[p.stem] = (None, None, "unresolved_offline", None)
                continue
            hits = search_by_sequence(p.sequence)
            cache["searches"][p.stem] = hits
            time.sleep(0.3)
        if not hits:
            resolved[p.stem] = (None, None, "no_sequence_hit", None)
            continue
        best = hits[0]
        # ``identifier`` is ``<ENTRY>_<entity>``; the match metrics live in
        # services[].nodes[].match_context[].
        entry_id = best["identifier"].split("_")[0].lower()
        ident = None
        for svc in best.get("services", []):
            for node in svc.get("nodes", []):
                for ctx in node.get("match_context", []):
                    if ctx.get("sequence_identity") is not None:
                        ident = float(ctx["sequence_identity"])
        resolved[p.stem] = (entry_id, None, "rcsb_sequence_search", ident)

    # --- fetch the entries -------------------------------------------------
    wanted = sorted({e for e, _, _, _ in resolved.values() if e})
    fresh = [e for e in wanted if e not in cache["entries"]]
    if fresh and not args.offline:
        cache["entries"].update(fetch_entries(fresh))
    elif fresh:
        print(f"[warn] {len(fresh)} entries missing from the cache (offline)")

    CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))

    # --- annotate ----------------------------------------------------------
    rows = []
    for p in cohort:
        entry_id, chain, source, ident = resolved[p.stem]
        entry = cache["entries"].get(entry_id) if entry_id else None
        entity = select_entity(entry, chain, p.sequence) if entry else None
        rows.append(annotate(p, entry, entity, source, ident))

    DATA.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[out] {OUT} ({len(rows)} rows)")

    unresolved = [r["stem"] for r in rows if not r["entry_id"]]
    if unresolved:
        print(f"[warn] {len(unresolved)} stems unresolved: {unresolved[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
