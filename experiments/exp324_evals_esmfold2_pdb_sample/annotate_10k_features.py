# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Annotate the exp324 10k PDB-deduped proteins with coarse biology features.

The output is intentionally small and analysis-facing: two lineage columns, one
GO-slim tag column, CATH structural tags, and explicit status/synthetic fields.
Raw provenance columns are retained so the coarse labels can be audited later.

Network use is batched and cacheable:

* RCSB GraphQL resolves PDB chain -> polymer entity, taxonomy, source organism,
  UniProt xrefs, keywords, and title.
* UniProt REST is queried in accession batches for GO/EC/protein names.
* GO and CATH are downloaded as versioned bulk text files and cached locally.
"""

import argparse
import csv
import gzip
import io
import json
import re
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
GO_BASIC_OBO = "https://current.geneontology.org/ontology/go-basic.obo"
GO_SLIM_GENERIC_OBO = "https://current.geneontology.org/ontology/subsets/goslim_generic.obo"
CATH_DOMAIN_LIST = (
    "https://download.cathdb.info/cath/releases/latest-release/"
    "cath-classification-data/cath-domain-list.txt"
)
NCBI_TAXDUMP = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"

BATCH_SIZE_RCSB = 60
BATCH_SIZE_UNIPROT = 40
SYNTHETIC_TAXID = "32630"

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
      rcsb_polymer_entity { pdbx_description }
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
    }
  }
}
"""

CELLULAR_LEVEL_1 = {"Bacteria", "Archaea", "Eukaryota"}
SYNTHETIC_MARKERS = {"artificial sequences"}
TITLE_SYNTHETIC_RE = re.compile(
    r"\b(de novo|designed|synthetic|engineered|computational(?:ly)? designed)\b",
    re.IGNORECASE,
)

EC_TOP_CLASS = {
    "1": "oxidoreductase",
    "2": "transferase",
    "3": "hydrolase",
    "4": "lyase",
    "5": "isomerase",
    "6": "ligase",
    "7": "translocase",
}

CATH_CLASS = {
    "1": "mainly_alpha",
    "2": "mainly_beta",
    "3": "alpha_beta",
    "4": "few_secondary_structures",
}


@dataclass(frozen=True)
class GoTerm:
    go_id: str
    name: str
    namespace: str
    parents: tuple[str, ...]
    subset: tuple[str, ...]


def post_json(url: str, payload: dict[str, Any], *, retries: int = 5) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(2**attempt)
    raise RuntimeError(f"unreachable: {url}")


def download_bytes(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {path}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "MarinFold exp324 annotation/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read()
    path.write_bytes(raw)
    return raw


def download_text(url: str, path: Path) -> str:
    if path.exists():
        return path.read_text()
    raw = download_bytes(url, path)
    text = gzip.decompress(raw).decode() if url.endswith(".gz") else raw.decode()
    path.write_text(text)
    return text


def split_multi(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in re.split(r";\s*", value) if x.strip()]


def join_sorted(values: set[str] | list[str] | tuple[str, ...]) -> str:
    return ";".join(sorted({v for v in values if v}))


def fetch_rcsb_entries(pdb_ids: list[str], cache_path: Path) -> dict[str, dict[str, Any]]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    out: dict[str, dict[str, Any]] = {}
    ids = sorted({p.upper() for p in pdb_ids})
    for start in range(0, len(ids), BATCH_SIZE_RCSB):
        batch = ids[start:start + BATCH_SIZE_RCSB]
        data = post_json(RCSB_GRAPHQL, {"query": ENTRY_QUERY, "variables": {"ids": batch}})
        if data.get("errors"):
            raise RuntimeError(f"RCSB GraphQL errors: {data['errors']}")
        for entry in (data.get("data", {}).get("entries") or []):
            if entry:
                out[entry["rcsb_id"].lower()] = entry
        print(f"[rcsb] {min(start + len(batch), len(ids))}/{len(ids)}", flush=True)
        time.sleep(0.2)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def select_entity(entry: dict[str, Any] | None, chain: str, sequence: str) -> tuple[dict[str, Any] | None, str]:
    if not entry:
        return None, "no_rcsb_entry"
    entities = entry.get("polymer_entities") or []
    if not entities:
        return None, "no_polymer_entity"
    chain_matches = []
    for ent in entities:
        ids = ent.get("rcsb_polymer_entity_container_identifiers") or {}
        auth = set(ids.get("auth_asym_ids") or [])
        asym = set(ids.get("asym_ids") or [])
        if chain in auth or chain in asym:
            chain_matches.append(ent)
    if len(chain_matches) == 1:
        return chain_matches[0], "matched"
    if len(chain_matches) > 1:
        return chain_matches[0], "ambiguous_chain"

    # Fallback for rare auth/asym mismatches: exact or subsequence match.
    for ent in entities:
        ent_seq = ((ent.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can") or "")
        if sequence and sequence in ent_seq:
            return ent, "matched_by_sequence"
    return None, "no_chain_match"


def load_ncbi_taxonomy(cache_dir: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Load NCBI taxdump parent/rank/scientific-name maps from a cached tarball."""

    raw = download_bytes(NCBI_TAXDUMP, cache_dir / "taxdump.tar.gz")
    parent: dict[str, str] = {}
    rank: dict[str, str] = {}
    name: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        nodes = tar.extractfile("nodes.dmp")
        if nodes is None:
            raise RuntimeError("taxdump missing nodes.dmp")
        for raw_line in nodes:
            fields = [x.strip() for x in raw_line.decode().split("|")]
            tax_id, parent_id, node_rank = fields[:3]
            parent[tax_id] = parent_id
            rank[tax_id] = node_rank
        names = tar.extractfile("names.dmp")
        if names is None:
            raise RuntimeError("taxdump missing names.dmp")
        for raw_line in names:
            fields = [x.strip() for x in raw_line.decode().split("|")]
            tax_id, tax_name, _, name_class = fields[:4]
            if name_class == "scientific name":
                name[tax_id] = tax_name
    return parent, rank, name


def ranked_lineage(tax_id: str, parent: dict[str, str], rank: dict[str, str]) -> list[str]:
    if not tax_id or tax_id not in parent:
        return []
    out = [tax_id]
    current = tax_id
    seen = {tax_id}
    while current in parent and parent[current] != current and parent[current] not in seen:
        current = parent[current]
        out.append(current)
        seen.add(current)
    return list(reversed(out))


def lineage_levels(
    orgs: list[dict[str, Any]],
    tax_parent: dict[str, str],
    tax_rank: dict[str, str],
    tax_name: dict[str, str],
) -> tuple[str, str, str, str, str]:
    if not orgs:
        return "unknown", "unknown", "", "", ""
    org = orgs[0]
    tax_id = str(org.get("ncbi_taxonomy_id") or "")
    organism = org.get("ncbi_scientific_name") or ""
    rcsb_lineage = org.get("taxonomy_lineage") or []
    rcsb_names = [x.get("name") for x in rcsb_lineage if x.get("name")]
    rcsb_name_set = set(rcsb_names)

    if tax_id == SYNTHETIC_TAXID or rcsb_name_set & SYNTHETIC_MARKERS or organism == "synthetic construct":
        return "synthetic", "synthetic_construct", organism, tax_id, join_sorted(rcsb_names)

    lineage = ranked_lineage(tax_id, tax_parent, tax_rank)
    by_rank = {tax_rank.get(t): tax_name.get(t, t) for t in lineage}
    lineage_names = [tax_name.get(t, t) for t in lineage]
    lineage_set = set(lineage_names)
    for domain in ("Bacteria", "Archaea", "Eukaryota"):
        if domain in lineage_set:
            level_2 = by_rank.get("phylum") or by_rank.get("class") or "unknown"
            return domain, level_2, organism, tax_id, join_sorted(rcsb_names)
    if "Viruses" in lineage_set:
        level_2 = by_rank.get("realm") or by_rank.get("kingdom") or by_rank.get("phylum") or "unknown"
        return "Viruses", level_2, organism, tax_id, join_sorted(rcsb_names)
    level_1 = by_rank.get("domain") or by_rank.get("superkingdom") or "unknown"
    return level_1, by_rank.get("phylum") or by_rank.get("class") or "unknown", organism, tax_id, join_sorted(rcsb_names)


def uniprot_accessions(ent: dict[str, Any] | None) -> list[str]:
    if not ent:
        return []
    ids = ent.get("rcsb_polymer_entity_container_identifiers") or {}
    refs = ids.get("reference_sequence_identifiers") or []
    return sorted({r.get("database_accession") for r in refs if r.get("database_name") == "UniProt" and r.get("database_accession")})


def fetch_uniprot(accessions: list[str], cache_path: Path) -> dict[str, dict[str, str]]:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    out: dict[str, dict[str, str]] = {}
    accs = sorted(set(accessions))
    fields = "accession,protein_name,go_id,go_f,go_p,go_c,ec,keyword"
    for start in range(0, len(accs), BATCH_SIZE_UNIPROT):
        batch = accs[start:start + BATCH_SIZE_UNIPROT]
        query = " OR ".join(f"accession:{acc}" for acc in batch)
        params = urllib.parse.urlencode({"query": query, "format": "tsv", "fields": fields})
        with urllib.request.urlopen(f"{UNIPROT_STREAM}?{params}", timeout=180) as resp:
            text = resp.read().decode()
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        for row in reader:
            acc = row.pop("Entry")
            out[acc] = row
        print(f"[uniprot] {min(start + len(batch), len(accs))}/{len(accs)}", flush=True)
        time.sleep(0.2)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def parse_obo(text: str) -> dict[str, GoTerm]:
    terms: dict[str, GoTerm] = {}
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line == "[Term]":
            if current and current.get("id") and not current.get("is_obsolete"):
                terms[current["id"]] = GoTerm(
                    current["id"], current.get("name", ""), current.get("namespace", ""),
                    tuple(current.get("parents", [])), tuple(current.get("subset", [])),
                )
            current = {"parents": [], "subset": []}
            continue
        if line.startswith("["):
            if current and current.get("id") and not current.get("is_obsolete"):
                terms[current["id"]] = GoTerm(
                    current["id"], current.get("name", ""), current.get("namespace", ""),
                    tuple(current.get("parents", [])), tuple(current.get("subset", [])),
                )
            current = None
            continue
        if current is None or not line or line.startswith("!"):
            continue
        if line.startswith("id: "):
            current["id"] = line.split("id: ", 1)[1]
        elif line.startswith("name: "):
            current["name"] = line.split("name: ", 1)[1]
        elif line.startswith("namespace: "):
            current["namespace"] = line.split("namespace: ", 1)[1]
        elif line.startswith("is_a: "):
            current["parents"].append(line.split()[1])
        elif line.startswith("relationship: part_of "):
            current["parents"].append(line.split()[2])
        elif line.startswith("subset: "):
            current["subset"].append(line.split("subset: ", 1)[1])
        elif line == "is_obsolete: true":
            current["is_obsolete"] = True
    if current and current.get("id") and not current.get("is_obsolete"):
        terms[current["id"]] = GoTerm(
            current["id"], current.get("name", ""), current.get("namespace", ""),
            tuple(current.get("parents", [])), tuple(current.get("subset", [])),
        )
    return terms


def ancestors(go_id: str, terms: dict[str, GoTerm], memo: dict[str, set[str]]) -> set[str]:
    if go_id in memo:
        return memo[go_id]
    term = terms.get(go_id)
    if not term:
        memo[go_id] = set()
        return set()
    out = set(term.parents)
    for parent in term.parents:
        out.update(ancestors(parent, terms, memo))
    memo[go_id] = out
    return out


def load_go_slim(cache_dir: Path) -> tuple[dict[str, GoTerm], set[str]]:
    terms = parse_obo(download_text(GO_BASIC_OBO, cache_dir / "go-basic.obo"))
    slim_terms = parse_obo(download_text(GO_SLIM_GENERIC_OBO, cache_dir / "goslim_generic.obo"))
    slim_ids = set(slim_terms) | {go_id for go_id, term in terms.items() if "goslim_generic" in term.subset}
    return terms, slim_ids


def map_go_to_slim(go_ids: list[str], terms: dict[str, GoTerm], slim_ids: set[str]) -> tuple[str, str]:
    memo: dict[str, set[str]] = {}
    mapped: set[str] = set()
    for go_id in go_ids:
        if go_id in slim_ids:
            mapped.add(go_id)
        mapped.update(ancestors(go_id, terms, memo) & slim_ids)
    names = {terms[x].name for x in mapped if x in terms}
    return join_sorted(mapped), join_sorted(names)


def parse_cath_domain_list(text: str) -> dict[tuple[str, str], dict[str, set[str]]]:
    by_chain: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        domain = fields[0]
        pdb_id = domain[:4].lower()
        chain = domain[4]
        class_id, arch_id, topology_id, superfamily_id = fields[1:5]
        key = (pdb_id, chain)
        by_chain[key]["domain_ids"].add(domain)
        by_chain[key]["classes"].add(CATH_CLASS.get(class_id, class_id))
        by_chain[key]["class_ids"].add(class_id)
        by_chain[key]["architecture_ids"].add(f"{class_id}.{arch_id}")
        by_chain[key]["topology_ids"].add(f"{class_id}.{arch_id}.{topology_id}")
        by_chain[key]["superfamily_ids"].add(f"{class_id}.{arch_id}.{topology_id}.{superfamily_id}")
    return by_chain


def synthetic_label(entry: dict[str, Any] | None, ent: dict[str, Any] | None, lineage_level_1: str, accessions: list[str]) -> tuple[bool, str, str]:
    evidence: set[str] = set()
    if lineage_level_1 == "synthetic":
        evidence.add("synthetic_taxonomy")
    title = ((entry or {}).get("struct") or {}).get("title") or ""
    keywords = ((entry or {}).get("struct_keywords") or {}).get("pdbx_keywords") or ""
    keyword_text = ((entry or {}).get("struct_keywords") or {}).get("text") or ""
    if "DE NOVO PROTEIN" in keywords.upper() or "DE NOVO PROTEIN" in keyword_text.upper():
        evidence.add("pdb_keyword_de_novo_protein")
    if TITLE_SYNTHETIC_RE.search(title):
        evidence.add("title_keyword_design")
    if ent and not accessions:
        evidence.add("no_uniprot_xref")
    is_synth = bool(evidence & {"synthetic_taxonomy", "pdb_keyword_de_novo_protein", "title_keyword_design"})
    if "pdb_keyword_de_novo_protein" in evidence or "title_keyword_design" in evidence:
        synth_type = "de_novo_or_designed"
    elif "synthetic_taxonomy" in evidence:
        synth_type = "synthetic_construct"
    elif is_synth:
        synth_type = "synthetic"
    else:
        synth_type = "natural_or_unknown"
    return is_synth, synth_type, join_sorted(evidence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_biology_features.parquet"))
    parser.add_argument("--csv-out", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/sample_10000_biology_features.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/exp324_evals_esmfold2_pdb_sample/data/annotation_cache"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    entries = fetch_rcsb_entries(manifest["pdb_id"].astype(str).tolist(), args.cache_dir / "rcsb_entries.json")
    tax_parent, tax_rank, tax_name = load_ncbi_taxonomy(args.cache_dir)

    prelim: list[dict[str, Any]] = []
    all_accessions: set[str] = set()
    for row in manifest.itertuples(index=False):
        entry = entries.get(str(row.pdb_id).lower())
        ent, lineage_status = select_entity(entry, str(row.chain_id), str(row.sequence))
        accessions = uniprot_accessions(ent)
        all_accessions.update(accessions)
        orgs = (ent or {}).get("rcsb_entity_source_organism") or []
        l1, l2, organism, tax_id, lineage_names = lineage_levels(orgs, tax_parent, tax_rank, tax_name)
        is_synth, synth_type, synth_evidence = synthetic_label(entry, ent, l1, accessions)
        prelim.append({
            "stem": row.stem,
            "pdb_id": str(row.pdb_id).lower(),
            "chain_id": row.chain_id,
            "entry_id": row.entry_id,
            "lineage_level_1": l1,
            "lineage_level_2": l2,
            "lineage_status": lineage_status if orgs else "no_taxonomy",
            "organism_name": organism,
            "ncbi_tax_id": tax_id,
            "lineage_names": lineage_names,
            "uniprot_accessions": join_sorted(accessions),
            "is_synthetic": is_synth,
            "synthetic_type": synth_type,
            "synthetic_evidence": synth_evidence,
            "pdb_title": ((entry or {}).get("struct") or {}).get("title") or "",
            "pdb_keywords": ((entry or {}).get("struct_keywords") or {}).get("pdbx_keywords") or "",
        })

    uniprot = fetch_uniprot(sorted(all_accessions), args.cache_dir / "uniprot.json") if all_accessions else {}
    go_terms, slim_ids = load_go_slim(args.cache_dir)
    cath = parse_cath_domain_list(download_text(CATH_DOMAIN_LIST, args.cache_dir / "cath-domain-list.txt"))

    out_rows: list[dict[str, Any]] = []
    for rec in prelim:
        accs = split_multi(rec["uniprot_accessions"])
        protein_names: set[str] = set()
        go_ids: set[str] = set()
        ec_numbers: set[str] = set()
        keywords: set[str] = set()
        for acc in accs:
            data = uniprot.get(acc, {})
            if data.get("Protein names"):
                protein_names.add(data["Protein names"])
            go_ids.update(split_multi(data.get("Gene Ontology IDs")))
            ec_numbers.update(split_multi(data.get("EC number")))
            keywords.update(split_multi(data.get("Keywords")))
        go_slim_ids, go_slim_names = map_go_to_slim(sorted(go_ids), go_terms, slim_ids)
        function_status = "matched_go" if go_ids else ("matched_uniprot_no_go" if accs else "no_uniprot")
        ec_top_classes = {EC_TOP_CLASS.get(ec.split(".", 1)[0], ec.split(".", 1)[0]) for ec in ec_numbers if ec}

        cath_rec = cath.get((rec["pdb_id"], str(rec["chain_id"])))
        if cath_rec:
            n_domains = len(cath_rec["domain_ids"])
            structure_status = "matched_cath"
            cath_classes = join_sorted(cath_rec["classes"])
            cath_topologies = join_sorted(cath_rec["topology_ids"])
            cath_superfamilies = join_sorted(cath_rec["superfamily_ids"])
        else:
            n_domains = 0
            structure_status = "no_cath_domain"
            cath_classes = ""
            cath_topologies = ""
            cath_superfamilies = ""

        out_rows.append({
            **rec,
            "protein_names": join_sorted(protein_names),
            "go_ids": join_sorted(go_ids),
            "go_slim_ids": go_slim_ids,
            "go_slim_tags": go_slim_names,
            "function_status": function_status,
            "ec_numbers": join_sorted(ec_numbers),
            "ec_top_classes": join_sorted(ec_top_classes),
            "uniprot_keywords": join_sorted(keywords),
            "cath_class_tags": cath_classes,
            "cath_topology_tags": cath_topologies,
            "cath_superfamily_tags": cath_superfamilies,
            "n_cath_domains": n_domains,
            "is_multi_domain": n_domains > 1,
            "structure_status": structure_status,
        })

    frame = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    frame.to_csv(args.csv_out, index=False)
    print(f"[features] wrote {len(frame)} rows -> {args.out} and {args.csv_out}", flush=True)
    print("[features] lineage_status", frame["lineage_status"].value_counts(dropna=False).to_dict(), flush=True)
    print("[features] function_status", frame["function_status"].value_counts(dropna=False).to_dict(), flush=True)
    print("[features] structure_status", frame["structure_status"].value_counts(dropna=False).to_dict(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
