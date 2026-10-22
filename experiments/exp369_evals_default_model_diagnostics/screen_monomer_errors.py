"""Screen retained AFDB training rows against later biological monomers.

An isolated chain cut out of a complex is not an eligible monomer. Every
deposited biological assembly must contain exactly one polymer instance, which
must be protein; the entry must contain only one polymer entity. Multiple
crystallographic copies are allowed when their biological assemblies are each
monomeric. Final cases still need a literature/context audit and exact AFDB v4
coordinate and contact verification.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from screen_training_errors import screen

HERE = Path(__file__).resolve().parent
FIELDS = """rcsb_id struct{title} rcsb_accession_info{initial_release_date}
rcsb_entry_container_identifiers{assembly_ids}
assemblies{rcsb_id rcsb_assembly_info{polymer_entity_instance_count polymer_entity_instance_count_protein}
  pdbx_struct_assembly{details oligomeric_details oligomeric_count}
  pdbx_struct_assembly_gen{asym_id_list oper_expression}}
polymer_entities{rcsb_id rcsb_polymer_entity{pdbx_description}
  entity_poly{type pdbx_seq_one_letter_code_can}
  rcsb_polymer_entity_container_identifiers{auth_asym_ids asym_ids uniprot_ids}
  polymer_entity_instances{rcsb_polymer_entity_instance_container_identifiers{asym_id auth_asym_id}}}
"""


def fetch_entries(ids: list[str], destination: Path) -> dict:
    """Cache complete assembly and chain metadata in small GraphQL batches."""
    saved = json.loads(destination.read_text()) if destination.exists() else {}
    missing = sorted(set(ids) - set(saved))
    for start in range(0, len(missing), 30):
        batch = missing[start:start + 30]
        query = "{entries(entry_ids:" + json.dumps([p.upper() for p in batch]) + "){" + FIELDS + "}}"
        response = requests.post("https://data.rcsb.org/graphql", json={"query": query}, timeout=90)
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise ValueError(payload["errors"])
        entries = payload["data"]["entries"]
        if len(entries) != len(batch) or any(entry is None for entry in entries):
            raise ValueError(f"Missing entry metadata: {batch}")
        saved.update({entry["rcsb_id"].lower(): entry for entry in entries})
        destination.write_text(json.dumps(saved, indent=2) + "\n")
        print(f"Assembly metadata: {len(saved)}/{len(ids)}", flush=True)
    return saved


def monomer_eligibility(entry: dict, author_chain: str) -> dict:
    """Require unambiguous one-protein biological assemblies and map the chain."""
    assemblies = entry["assemblies"] or []
    counts = [assembly["rcsb_assembly_info"]["polymer_entity_instance_count"] for assembly in assemblies]
    result = {"title": entry["struct"]["title"], "release_date": entry["rcsb_accession_info"]["initial_release_date"],
              "assembly_polymer_counts": json.dumps(counts)}
    if not assemblies or len(assemblies) != len(entry["rcsb_entry_container_identifiers"]["assembly_ids"]):
        return result | {"monomer_status": "missing_assembly_metadata"}
    if any(assembly["rcsb_assembly_info"]["polymer_entity_instance_count"] != 1 or
           assembly["rcsb_assembly_info"]["polymer_entity_instance_count_protein"] != 1 for assembly in assemblies):
        return result | {"monomer_status": "nonmonomer_or_ambiguous_assemblies"}
    entities = entry["polymer_entities"]
    if len(entities) != 1 or entities[0]["entity_poly"]["type"] != "polypeptide(L)":
        return result | {"monomer_status": "additional_polymer_entities"}
    entity = entities[0]
    label_chains = [instance["rcsb_polymer_entity_instance_container_identifiers"]["asym_id"]
                    for instance in entity["polymer_entity_instances"]
                    if instance["rcsb_polymer_entity_instance_container_identifiers"]["auth_asym_id"] == author_chain]
    matching = [assembly["rcsb_id"] for assembly in assemblies
                if any(set(label_chains) & set(group["asym_id_list"]) for group in assembly["pdbx_struct_assembly_gen"])]
    if len(label_chains) != 1 or not matching:
        return result | {"monomer_status": "chain_not_in_monomer_assembly"}
    return result | {"monomer_status": "passes_assembly_metadata", "label_chain": label_chains[0],
                     "assembly_ids": json.dumps(matching), "protein_name": entity["rcsb_polymer_entity"]["pdbx_description"]}


def main() -> None:
    """Freeze the monomer search, exact retained training rows, and geometry screen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/data/exp369"))
    parser.add_argument("--corpus", type=Path, default=Path("/data/exp225_decontam/contacts_v1_decontam"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    work = args.work / "monomers"
    work.mkdir(exist_ok=True)
    query_path = work / "rcsb_monomer_query.json"
    search_path = work / "rcsb_monomer_search.json"
    if not search_path.exists():
        nodes = [{"type": "terminal", "service": "text", "parameters": {"attribute": attr, "operator": op, "value": value}}
                 for attr, op, value in (("rcsb_assembly_info.polymer_entity_instance_count", "equals", 1),
                                        ("rcsb_assembly_info.polymer_entity_instance_count_protein", "equals", 1),
                                        ("rcsb_accession_info.initial_release_date", "greater_or_equal", "2022-12-01"))]
        query = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes}, "return_type": "assembly",
                 "request_options": {"return_all_hits": True, "results_content_type": ["experimental"]}}
        response = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=90)
        response.raise_for_status()
        query_path.write_text(json.dumps(query, indent=2) + "\n")
        search_path.write_text(json.dumps(response.json(), indent=2) + "\n")
    assembly_hits = json.loads(search_path.read_text())["result_set"]
    entries = {hit["identifier"].rsplit("-", 1)[0].lower() for hit in assembly_hits}
    sifts = pd.read_csv(args.work / "pdb_chain_uniprot.csv.gz", comment="#")
    index = pd.read_parquet(args.corpus.parent / "index_afdb.parquet")
    index["SP_PRIMARY"] = index.entry_id.str.removeprefix("AF-").str.removesuffix("-F1")
    candidates = sifts.loc[sifts.PDB.isin(entries)].merge(index, on="SP_PRIMARY", validate="many_to_one")
    drop = pd.read_parquet(args.corpus.parent / "droplist_final.parquet").query("arm == 'afdb'")
    candidates = candidates.loc[~candidates.entry_id.isin(drop.entry_id)].drop_duplicates(["PDB", "CHAIN", "SP_PRIMARY"])
    metadata = fetch_entries(sorted(set(candidates.PDB)), work / "entry_assemblies.json")
    candidates = pd.DataFrame([row | monomer_eligibility(metadata[row["PDB"]], row["CHAIN"])
                               for row in candidates.to_dict("records")])
    candidates.to_csv(HERE / "data/monomer_candidate_eligibility.csv", index=False)
    eligible = candidates.loc[candidates.monomer_status == "passes_assembly_metadata"]
    records = []
    for shard, group in eligible.groupby("shard"):
        path = args.corpus / f"contacts_v1-{shard:05d}-of-02067.parquet"
        table = pq.read_table(path, columns=["entry_id", "document", "seq_len", "n_term_index", "global_plddt", "truncated"])
        selected = {row["entry_id"]: row for row in table.filter(pc.is_in(table["entry_id"], value_set=pc.unique(group.entry_id.to_numpy()))).to_pylist()}
        for row in group.to_dict("records"):
            records.append((row, selected[row["entry_id"]]))
    (work / "screen_inputs.json").write_text(json.dumps(records) + "\n")
    print(f"Eligible chains: {len(records)}; unique training proteins: {eligible.SP_PRIMARY.nunique()}", flush=True)
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(screen, row, training, args.work / "pdb_chains", 1600): row for row, training in records}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = row | future.result()
                if result["status"] == "screened" and (result["identity"] < .98 or result["coverage"] < .90):
                    result["status"] = "ineligible_strict_monomer_mapping"
            except (requests.RequestException, RuntimeError, ValueError) as error:
                result = row | {"status": "error", "error": f"{type(error).__name__}: {error}"}
                failures.append(result)
            results.append(result)
            if len(results) % 25 == 0 or len(results) == len(records):
                pd.DataFrame(results).to_csv(HERE / "data/monomer_candidate_screen.csv", index=False)
                print(f"Screened {len(results)}/{len(records)}; errors={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} failures recorded in monomer_candidate_screen.csv")


if __name__ == "__main__":
    main()
