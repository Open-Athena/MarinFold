"""Census exact FoldBench targets and PINDER test dimers.

This is a feasibility manifest, not the frozen benchmark. In particular, design
annotations and nonpolymer dependencies need review on the surviving shortlist.
Both full deposited sequences and resolved chain sequences are searched, so a
match to a crystallized fragment cannot disappear behind unresolved residues.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import gemmi

HERE = Path(__file__).resolve().parent
DESIGN = re.compile(
    r"de novo|de-novo|designed protein|designed hetero|designed homo|synthetic protein",
    re.IGNORECASE,
)
ANTIBODY = re.compile(
    r"antibody|nanobody|immunoglobulin|single.domain antibody", re.IGNORECASE
)


def foldbench_members(rows: list[dict[str, str]]) -> list[str]:
    """Return scored chain labels in first-seen order, retaining symmetry copies."""
    return list(
        dict.fromkeys(
            chain
            for row in rows
            for chain in (
                row["interface_chain_id_1"],
                row["interface_chain_id_2"],
            )
        )
    )


def read_entry(path: Path, pdb: str) -> dict | None:
    """Read canonical sequence, assembly and resolved-residue metadata from one entry."""
    if not path.exists():
        return None
    block = gemmi.cif.read(str(path)).sole_block()
    structure = gemmi.make_structure_from_block(block)
    polymers = block.get_mmcif_category("_entity_poly.")
    entities = {}
    for i, entity in enumerate(polymers["entity_id"]):
        entities[entity] = {
            "type": polymers["type"][i],
            "sequence": "".join(polymers["pdbx_seq_one_letter_code_can"][i].split()),
        }
    asym = block.get_mmcif_category("_struct_asym.")
    asym_entities = dict(zip(asym["id"], asym["entity_id"], strict=True))
    atoms = block.get_mmcif_category("_atom_site.")
    resolved: dict[str, set[int]] = defaultdict(set)
    for chain, resid, atom, model in zip(
        atoms["label_asym_id"],
        atoms["label_seq_id"],
        atoms["label_atom_id"],
        atoms["pdbx_PDB_model_num"],
        strict=True,
    ):
        if atom == "CA" and resid and model == "1":
            resolved[chain].add(int(resid))
    chains = {}
    for chain, entity in asym_entities.items():
        if entity not in entities:
            continue
        data = entities[entity]
        sequence = data["sequence"]
        if not sequence.isalpha():
            raise ValueError(f"Noncanonical sequence notation: {pdb} {entity}")
        chains[chain] = {
            **data,
            "entity": entity,
            "resolved_sequence": "".join(
                sequence[i - 1] for i in sorted(resolved[chain])
            ),
        }
    assemblies = {}
    for assembly in structure.assemblies:
        expanded = []
        for generator in assembly.generators:
            for operator in generator.operators:
                expanded.extend(
                    (chain, operator.name)
                    for chain in generator.subchains
                    if chain in chains
                )
        assemblies[assembly.name] = expanded
    title = gemmi.cif.as_string(block.find_value("_struct.title") or "")
    return {
        "pdb": pdb,
        "path": str(path),
        "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        "title": title,
        "design_flag": bool(DESIGN.search(title)),
        "antibody_flag": bool(ANTIBODY.search(title)),
        "chains": chains,
        "assemblies": assemblies,
        "method": ";".join(
            gemmi.cif.as_string(x) for x in block.find_values("_exptl.method")
        ),
        "release_date": next(
            iter(block.find_values("_pdbx_audit_revision_history.revision_date")), ""
        ),
        "resolution": min(
            (
                float(x)
                for x in block.find_values("_refine.ls_d_res_high")
                if x not in (".", "?")
            ),
            default=structure.resolution,
        ),
        "nonpolymer_entities": len(asym_entities) - len(chains),
    }


def main() -> None:
    """Write all candidates, constituent-chain queries and a source census."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cif-root", type=Path, default=Path("/data/tim/helico-data/raw/mmCIF")
    )
    ap.add_argument(
        "--foldbench",
        type=Path,
        default=Path("/home/bizon/git/FoldBench/targets/interface_protein_protein.csv"),
    )
    ap.add_argument(
        "--foldbench-ground-truths",
        type=Path,
        default=Path(
            "/data/tim/helico-data/benchmarks/FoldBench/examples/ground_truths"
        ),
    )
    ap.add_argument("--pinder-root", type=Path, default=Path("/data/exp294_stageE"))
    ap.add_argument("--out", type=Path, default=HERE / "data")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fb: dict[str, list[dict]] = defaultdict(list)
    for row in csv.DictReader(args.foldbench.open()):
        fb[row["pdb_id"]].append(row)
    con = duckdb.connect()
    index_path, metadata_path = (
        str(args.pinder_root / "index.parquet"),
        str(args.pinder_root / "metadata.parquet"),
    )
    result = con.execute(
        "SELECT i.*, m.asym_id_1, m.asym_id_2, m.oligomeric_count, m.resolution, "
        "m.release_date, m.interface_atom_gaps_4A, m.missing_interface_residues_4A "
        "FROM read_parquet(?) i JOIN read_parquet(?) m USING(id) WHERE i.split = 'test'",
        [index_path, metadata_path],
    )
    columns = [d[0] for d in result.description]
    pinder = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
    pdbs = sorted({a.split("-")[0] for a in fb} | {r["pdb_id"] for r in pinder})
    with ThreadPoolExecutor(16) as pool:
        entries = dict(
            zip(
                pdbs,
                pool.map(
                    lambda p: read_entry(args.cif_root / p[1:3] / f"{p}.cif.gz", p),
                    pdbs,
                ),
                strict=True,
            )
        )
        foldbench_entries = dict(
            zip(
                sorted(fb),
                pool.map(
                    lambda assembly: read_entry(
                        args.foldbench_ground_truths / f"{assembly}.cif",
                        assembly.split("-assembly")[0],
                    ),
                    sorted(fb),
                ),
                strict=True,
            )
        )
    candidates, query_rows = [], []
    sequences: dict[str, str] = {}

    def add(
        source: str,
        target: str,
        pdb: str,
        members: list[str],
        entry: dict | None,
        metadata_entry: dict | None = None,
        **extra: object,
    ) -> None:
        row = {"source": source, "target": target, "pdb_id": pdb, **extra}
        if entry is None:
            candidates.append({**row, "eligibility": "missing_snapshot_entry"})
            return
        metadata = metadata_entry or entry
        row.update(
            {
                k: metadata[k]
                for k in [
                    "title",
                    "design_flag",
                    "method",
                    "release_date",
                    "resolution",
                    "nonpolymer_entities",
                ]
            }
        )
        row["antibody_flag"] = bool(
            row.get("antibody_flag", False)
            or entry["antibody_flag"]
            or metadata["antibody_flag"]
        )
        row["design_flag"] = bool(entry["design_flag"] or metadata["design_flag"])
        missing_members = [c for c in members if c not in entry["chains"]]
        chains = [entry["chains"][c] for c in members if c in entry["chains"]]
        proteins = [c for c in chains if c["type"] == "polypeptide(L)"]
        row.update(
            n_chains=len(chains),
            n_protein_chains=len(proteins),
            total_residues=sum(len(c["sequence"]) for c in proteins),
            complex_type="homodimer"
            if len(proteins) == 2 and proteins[0]["sequence"] == proteins[1]["sequence"]
            else "heterodimer",
        )
        reasons = []
        if not members:
            reasons.append("missing_assembly")
        if missing_members:
            reasons.append("missing_chain_labels")
        if len(chains) != 2:
            reasons.append("not_dimer")
        if len(proteins) != len(chains):
            reasons.append("unsupported_polymer")
        if any(len(c["sequence"]) < 40 for c in proteins):
            reasons.append("short_chain_under_40")
        if row["total_residues"] > 1998:
            reasons.append("ring_over_1998")
        if row["design_flag"]:
            reasons.append("design_annotation")
        if row["antibody_flag"]:
            reasons.append("antibody")
        if row["resolution"] <= 0 or row["resolution"] > 3.5:
            reasons.append("resolution_above_3.5_or_unknown")
        if row["method"] not in ("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"):
            reasons.append("experimental_method")
        row["eligibility"] = ";".join(reasons) or "candidate"
        candidates.append(row)
        for index, c in enumerate(proteins):
            for variant in ("sequence", "resolved_sequence"):
                seq = c[variant]
                if not seq:
                    continue
                key = "q" + hashlib.sha256(seq.encode()).hexdigest()[:24]
                if key in sequences and sequences[key] != seq:
                    raise ValueError("Query hash collision")
                sequences[key] = seq
                query_rows.append(
                    {
                        "source": source,
                        "target": target,
                        "chain_index": index,
                        "asym_id": members[index],
                        "variant": variant,
                        "query": key,
                        "length": len(seq),
                        "sequence": seq,
                    }
                )

    for assembly, rows in sorted(fb.items()):
        pdb = assembly.split("-assembly")[0]
        members = foldbench_members(rows)
        add(
            "foldbench",
            assembly,
            pdb,
            members,
            foldbench_entries[assembly],
            entries[pdb],
            n_interfaces=len(rows),
            cluster_id="",
        )
    for r in pinder:
        add(
            "pinder",
            r["id"],
            r["pdb_id"],
            [r["asym_id_1"], r["asym_id_2"]],
            entries[r["pdb_id"]],
            n_interfaces=1,
            cluster_id=r["cluster_id"],
            antibody_flag=r["contains_antibody"],
            interface_gaps=r["interface_atom_gaps_4A"],
            missing_interface_residues=r["missing_interface_residues_4A"],
        )
    for name, rows in [("candidates", candidates), ("query_membership", query_rows)]:
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with (args.out / f"{name}.csv").open("w") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    with (args.out / "queries.fasta").open("w") as fh:
        for key, seq in sorted(sequences.items()):
            fh.write(f">{key}\n{seq}\n")
    provenance = {
        "cif_root": str(args.cif_root),
        "n_entries": len(entries),
        "missing_entries": [p for p, e in entries.items() if e is None],
        "entries": [
            {k: e[k] for k in ("pdb", "path", "sha256")} for e in entries.values() if e
        ],
        "foldbench_ground_truths": str(args.foldbench_ground_truths),
        "foldbench_entries": [
            {"target": target, **{k: entry[k] for k in ("pdb", "path", "sha256")}}
            for target, entry in foldbench_entries.items()
            if entry
        ],
        "policy": "natural candidate dimers; L-polypeptide only; each chain >=40 aa; total <=1998; X-ray/EM <=3.5 A; design/antibody title flags excluded provisionally",
        "source_hashes": {
            str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            for p in [args.foldbench, Path(index_path), Path(metadata_path)]
        },
    }
    (args.out / "candidate_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(Counter((r["source"], r["eligibility"]) for r in candidates))
    print("unique query sequences", len(sequences), flush=True)


if __name__ == "__main__":
    main()
