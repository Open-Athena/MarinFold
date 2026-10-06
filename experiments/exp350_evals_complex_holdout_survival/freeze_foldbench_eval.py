"""Freeze the FoldBench-only pair-held-out complex evaluation set."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import gemmi
import pyarrow as pa
import pyarrow.parquet as pq
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    analyze_structure,
    build_document,
    residues_from_sequence,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FOLDBENCH = Path("/home/bizon/git/FoldBench")
FOLDBENCH_TARGETS = FOLDBENCH / "targets/interface_protein_protein.csv"
FOLDBENCH_GT = Path(
    "/data/tim/helico-data/benchmarks/FoldBench/examples/ground_truths"
)
MMSEQS = Path.home() / ".cache/marinfold/mmseqs/mmseqs/bin/mmseqs"
FOLDBENCH_REVISION = "90a6033fed4fcb74d4f1304e932fd93b52d48a6c"
FORMAT = (
    "query,target,fident,alnlen,qcov,tcov,evalue,bits,nident,qlen,tlen,"
    "qstart,qend,tstart,tend"
)
HELICO_EXCLUSIONS = {
    "7txy-assembly1",
    "8s0u-assembly1",
    "8z2m-assembly1",
}
DESIGN_EXCLUSIONS = {
    "8g4k-assembly1": (
        "de_novo_binder; citation: Computational Design of High Affinity "
        "Binders to Convex Protein Target Sites"
    ),
    "9bk6-assembly1": (
        "de_novo_binder; citation: De novo designed proteins neutralize "
        "lethal snake venom toxins"
    ),
}
TARGET_COLUMNS = [
    "pdb_id",
    "interface_chain_id_1",
    "interface_chain_id_2",
    "interface_chain_type_1",
    "interface_chain_type_2",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dictionaries."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write deterministic CSV rows."""
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(command: list[str | Path], log_path: Path | None = None) -> None:
    """Run a subprocess and optionally preserve its combined output."""
    argv = list(map(str, command))
    print(" ".join(argv), flush=True)
    if log_path is None:
        subprocess.run(argv, check=True)
        return
    with log_path.open("w") as handle:
        subprocess.run(argv, check=True, stdout=handle, stderr=subprocess.STDOUT)


def remove_prefix(prefix: Path) -> None:
    """Remove files and directories sharing an MMseqs database prefix."""
    for path in prefix.parent.glob(f"{prefix.name}*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def select_targets() -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    """Apply the reviewed FoldBench-only freeze policy."""
    audits = read_csv(DATA / "pair_per_complex.csv")
    interfaces: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(FOLDBENCH_TARGETS):
        interfaces[row["pdb_id"]].append(row)

    audit_rows = []
    selected = []
    for row in audits:
        if (
            row["source"] != "foldbench"
            or row["eligibility"] != "candidate"
            or row["complex_pair_status"] != "pair_clean"
        ):
            continue
        target = row["target"]
        if row["helico_finetune_pair_status"] == "hit":
            status = "excluded_helico_finetune_pair_homolog"
            reason = (
                "both partners have >=30% identity over >=50% shorter-side "
                "coverage to distinct chains in one Helico fine-tuning PDB"
            )
            if target not in HELICO_EXCLUSIONS:
                raise ValueError(f"Unreviewed Helico exclusion: {target}")
        elif target in DESIGN_EXCLUSIONS:
            status = "excluded_designed_binder"
            reason = DESIGN_EXCLUSIONS[target]
        else:
            status = "included"
            reason = "natural pair-held-out FoldBench dimer"
            selected.append(row)
        audit_rows.append(
            {
                "target_id": target,
                "pdb_id": row["pdb_id"],
                "complex_type": row["complex_type"],
                "release_date": row["release_date"],
                "resolution": row["resolution"],
                "title": row["title"],
                "marinfold_pair_status": row["complex_pair_status"],
                "helico_finetune_pair_status": row[
                    "helico_finetune_pair_status"
                ],
                "freeze_status": status,
                "freeze_reason": reason,
            }
        )
    if len(audit_rows) != 35 or len(selected) != 30:
        raise ValueError(
            f"Expected 35 audited and 30 selected targets, got "
            f"{len(audit_rows)} and {len(selected)}"
        )
    selected.sort(key=lambda item: item["target"])
    return selected, audit_rows, interfaces


def load_chains(selected: list[dict]) -> list[dict]:
    """Load the two canonical FoldBench chain sequences for each target."""
    selected_ids = {row["target"] for row in selected}
    rows = [
        row
        for row in read_csv(DATA / "query_membership.csv")
        if row["source"] == "foldbench"
        and row["target"] in selected_ids
        and row["variant"] == "sequence"
    ]
    rows.sort(key=lambda item: (item["target"], int(item["chain_index"])))
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["target"]] += 1
        row["chain_key"] = (
            f"{row['target']}|{row['chain_index']}|{row['asym_id']}"
        )
    wrong = {target: count for target, count in counts.items() if count != 2}
    if len(rows) != 60 or wrong:
        raise ValueError(f"Expected 60 chains in pairs; bad counts: {wrong}")
    return rows


def write_fasta(path: Path, chains: list[dict]) -> None:
    """Write selected chains with unique, traceable identifiers."""
    with path.open("w") as handle:
        for row in chains:
            handle.write(f">{row['chain_key']}\n{row['sequence']}\n")


def self_search(fasta: Path, work: Path, threads: int) -> dict:
    """Run symmetric shorter-side MMseqs searches for homology grouping."""
    work.mkdir(parents=True, exist_ok=True)
    query_db = work / "chainsDB"
    remove_prefix(query_db)
    run([MMSEQS, "createdb", fasta, query_db, "--shuffle", "0"])
    started = time.monotonic()
    searches = []
    outputs = []
    for coverage, mode in (("query", "2"), ("target", "1")):
        result = work / f"self_{coverage}_alnDB"
        temporary = work / f"self_{coverage}_tmp"
        output = DATA / f"foldbench_self_{coverage}_alignments.tsv"
        log = DATA / f"foldbench_self_{coverage}_search.log"
        remove_prefix(result)
        remove_prefix(temporary)
        output.unlink(missing_ok=True)
        command = [
            MMSEQS,
            "search",
            query_db,
            query_db,
            result,
            temporary,
            "-s",
            "7.5",
            "-e",
            "1000",
            "--max-seqs",
            "10000",
            "--min-seq-id",
            "0.30",
            "--alignment-mode",
            "3",
            "--seq-id-mode",
            "0",
            "-a",
            "1",
            "-c",
            "0.50",
            "--cov-mode",
            mode,
            "--threads",
            str(threads),
        ]
        run(command, log)
        run(
            [
                MMSEQS,
                "convertalis",
                query_db,
                query_db,
                result,
                output,
                "--format-output",
                FORMAT,
                "--threads",
                str(threads),
            ]
        )
        searches.append(
            {
                "coverage": coverage,
                "coverage_mode": int(mode),
                "command": list(map(str, command)),
                "output": output.name,
                "output_sha256": sha256(output),
                "log": log.name,
                "log_sha256": sha256(log),
            }
        )
        outputs.append(output)
    return {
        "mmseqs_version": subprocess.check_output(
            [str(MMSEQS), "version"], text=True
        ).strip(),
        "queries": fasta.name,
        "queries_sha256": sha256(fasta),
        "format": FORMAT,
        "rule": (
            "exact nident/alnlen >=0.30 and query or target aligned span "
            ">=0.50; two cov-mode searches implement shorter-side coverage"
        ),
        "elapsed_seconds": time.monotonic() - started,
        "searches": searches,
        "outputs": outputs,
    }


class UnionFind:
    """Small deterministic disjoint-set implementation."""

    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        """Return a component representative with path compression."""
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        """Join two components with a deterministic representative."""
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        keep, drop = sorted((root_left, root_right))
        self.parent[drop] = keep


def qualifying_alignment(row: dict[str, str]) -> bool:
    """Apply the exact 30% identity and 50% shorter-side coverage rule."""
    nident = int(row["nident"])
    alnlen = int(row["alnlen"])
    query_span = int(row["qend"]) - int(row["qstart"]) + 1
    target_span = int(row["tend"]) - int(row["tstart"]) + 1
    return nident * 10 >= alnlen * 3 and (
        query_span * 2 >= int(row["qlen"])
        or target_span * 2 >= int(row["tlen"])
    )


def homology_groups(
    chains: list[dict], outputs: list[Path]
) -> tuple[dict[str, str], list[dict]]:
    """Cluster complexes by connected chain-level homology edges."""
    chain_to_target = {row["chain_key"]: row["target"] for row in chains}
    targets = sorted(set(chain_to_target.values()))
    union_find = UnionFind(targets)
    best: dict[tuple[str, str, str, str], dict] = {}
    for path in outputs:
        with path.open() as handle:
            reader = csv.DictReader(handle, fieldnames=FORMAT.split(","), delimiter="\t")
            for row in reader:
                if not qualifying_alignment(row):
                    continue
                left_target = chain_to_target[row["query"]]
                right_target = chain_to_target[row["target"]]
                if left_target == right_target:
                    continue
                union_find.union(left_target, right_target)
                target_pair = tuple(sorted((left_target, right_target)))
                chain_pair = tuple(sorted((row["query"], row["target"])))
                key = (*target_pair, *chain_pair)
                record = {
                    "target_1": target_pair[0],
                    "target_2": target_pair[1],
                    "chain_1": chain_pair[0],
                    "chain_2": chain_pair[1],
                    "nident": int(row["nident"]),
                    "alnlen": int(row["alnlen"]),
                    "identity": int(row["nident"]) / int(row["alnlen"]),
                    "qlen": int(row["qlen"]),
                    "tlen": int(row["tlen"]),
                    "qspan": int(row["qend"]) - int(row["qstart"]) + 1,
                    "tspan": int(row["tend"]) - int(row["tstart"]) + 1,
                }
                previous = best.get(key)
                if previous is None or record["nident"] > previous["nident"]:
                    best[key] = record
    components: dict[str, list[str]] = defaultdict(list)
    for target in targets:
        components[union_find.find(target)].append(target)
    target_to_group = {}
    for members in components.values():
        members.sort()
        digest = hashlib.sha256("\n".join(members).encode()).hexdigest()[:10]
        group_id = f"g-{digest}"
        for target in members:
            target_to_group[target] = group_id
    return target_to_group, sorted(
        best.values(), key=lambda row: (row["target_1"], row["target_2"], row["chain_1"])
    )


def choose_dev_groups(
    selected: list[dict], chains: list[dict], target_to_group: dict[str, str]
) -> set[str]:
    """Choose about 25% development targets using metadata only."""
    lengths: dict[str, int] = defaultdict(int)
    for chain in chains:
        lengths[chain["target"]] += len(chain["sequence"])
    rows = {row["target"]: row for row in selected}
    grouped: dict[str, list[str]] = defaultdict(list)
    for target, group in target_to_group.items():
        grouped[group].append(target)
    items = []
    for group, targets in grouped.items():
        items.append(
            (
                group,
                len(targets),
                sum(rows[target]["complex_type"] == "homodimer" for target in targets),
                sum(lengths[target] for target in targets),
            )
        )
    items.sort()
    total_count = len(selected)
    total_homo = sum(row["complex_type"] == "homodimer" for row in selected)
    total_length = sum(lengths.values())
    target_count = round(total_count * 0.25)
    best_score = None
    best_groups: tuple[str, ...] | None = None

    def visit(
        index: int,
        count: int,
        homo: int,
        length: int,
        chosen: tuple[str, ...],
    ) -> None:
        nonlocal best_score, best_groups
        if count == target_count:
            score = (
                abs(homo * total_count - count * total_homo),
                abs(length * total_count - count * total_length),
                chosen,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_groups = chosen
            return
        if index == len(items) or count > target_count:
            return
        group, item_count, item_homo, item_length = items[index]
        visit(index + 1, count, homo, length, chosen)
        visit(
            index + 1,
            count + item_count,
            homo + item_homo,
            length + item_length,
            chosen + (group,),
        )

    visit(0, 0, 0, 0, ())
    if best_groups is None:
        raise ValueError(f"Cannot assign exactly {target_count} targets to development")
    return set(best_groups)


def atom_site_maps(
    path: Path, label_chains: list[str]
) -> tuple[list[str], dict[tuple[str, int], int]]:
    """Map FoldBench label chains and author residue numbers to SEQRES indices."""
    block = gemmi.cif.read_file(str(path)).sole_block()
    table = block.find(
        [
            "_atom_site.label_asym_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_seq_id",
            "_atom_site.label_seq_id",
        ]
    )
    label_to_auth: dict[str, set[str]] = defaultdict(set)
    residue_map: dict[tuple[str, int], int] = {}
    wanted = set(label_chains)
    for label, auth, auth_seq, label_seq in table:
        if label not in wanted or auth_seq in {".", "?"} or label_seq in {".", "?"}:
            continue
        label_to_auth[label].add(auth)
        key = (auth, int(auth_seq))
        value = int(label_seq) - 1
        previous = residue_map.setdefault(key, value)
        if previous != value:
            raise ValueError(f"Ambiguous residue mapping in {path}: {key}")
    auth_chains = []
    for label in label_chains:
        values = label_to_auth[label]
        if len(values) != 1:
            raise ValueError(f"{path}: label chain {label} maps to {sorted(values)}")
        auth_chains.append(next(iter(values)))
    if len(set(auth_chains)) != 2:
        raise ValueError(f"{path}: interface does not map to two author chains")
    return auth_chains, residue_map


def structure_positions_by_chain(path: Path, label_chains: list[str]) -> list[list[int]]:
    """Return canonical coordinates for residues Helico reads from the mmCIF."""
    block = gemmi.cif.read_file(str(path)).sole_block()
    table = block.find(
        [
            "_atom_site.label_asym_id",
            "_atom_site.label_seq_id",
        ]
    )
    positions = [[] for _ in label_chains]
    seen = [set() for _ in label_chains]
    label_to_index = {label: index for index, label in enumerate(label_chains)}
    for label, label_seq in table:
        if label not in label_to_index or label_seq in {".", "?"}:
            continue
        chain_index = label_to_index[label]
        local_index = int(label_seq) - 1
        if local_index not in seen[chain_index]:
            positions[chain_index].append(local_index)
            seen[chain_index].add(local_index)
    if any(not chain for chain in positions):
        raise ValueError(f"{path}: no atom-site positions for {label_chains}")
    return positions


def selected_structure(path: Path, auth_chains: list[str]) -> gemmi.Structure:
    """Clone a structure and retain only the scored FoldBench interface chains."""
    structure = gemmi.read_structure(str(path)).clone()
    for model in structure:
        names = [chain.name for chain in model]
        missing = set(auth_chains) - set(names)
        if missing:
            raise ValueError(f"{path}: missing author chains {sorted(missing)}")
        for name in names:
            if name not in auth_chains:
                model.remove_chain(name)
    structure.remove_ligands_and_waters()
    return structure


def ground_truth_row(
    target: dict,
    interface: dict,
    chain_rows: list[dict],
    group_id: str,
    split: str,
) -> dict:
    """Build full-sequence coordinates and resolved inter-chain truth."""
    target_id = target["target"]
    path = FOLDBENCH_GT / f"{target_id}.cif"
    label_chains = [
        interface["interface_chain_id_1"],
        interface["interface_chain_id_2"],
    ]
    if [row["asym_id"] for row in chain_rows] != label_chains:
        raise ValueError(f"{target_id}: interface and membership chain order differ")
    auth_chains, residue_map = atom_site_maps(path, label_chains)
    structure_local_positions = structure_positions_by_chain(path, label_chains)
    analyzed = analyze_structure(
        selected_structure(path, auth_chains),
        entry_id=target_id,
        max_chains=2,
    )
    offsets = [0, len(chain_rows[0]["sequence"])]
    auth_to_index = {auth: index for index, auth in enumerate(auth_chains)}
    resolved = []
    local_to_full = {}
    for residue in analyzed.residues:
        chain_index = auth_to_index[residue.chain]
        key = (residue.chain, residue.resnum)
        if key not in residue_map:
            raise ValueError(f"{target_id}: no SEQRES coordinate for {key}")
        local_index = residue_map[key]
        if local_index >= len(chain_rows[chain_index]["sequence"]):
            raise ValueError(f"{target_id}: mapped residue exceeds canonical sequence")
        observed = gemmi.find_tabulated_residue(residue.resname).one_letter_code
        expected = chain_rows[chain_index]["sequence"][local_index]
        if expected != "X" and observed != expected:
            raise ValueError(
                f"{target_id}: residue mismatch for {key}: "
                f"structure={observed}, canonical={expected}"
            )
        full_index = offsets[chain_index] + local_index
        local_to_full[residue.seq_index] = full_index
        resolved.append(full_index)
    contacts = []
    for contact in analyzed.contacts:
        if contact.degree < 0.001:
            continue
        left = analyzed.residues[contact.seq_i]
        right = analyzed.residues[contact.seq_j]
        if left.chain == right.chain:
            continue
        i = local_to_full[contact.seq_i]
        j = local_to_full[contact.seq_j]
        contacts.append(
            {"i": min(i, j), "j": max(i, j), "degree": contact.degree}
        )
    contacts.sort(key=lambda row: (row["i"], row["j"]))
    if not contacts:
        raise ValueError(f"{target_id}: no inter-chain contacts")
    residues = []
    for chain_index, chain in enumerate(chain_rows):
        for residue in residues_from_sequence(
            chain["sequence"], chain=label_chains[chain_index]
        ):
            residues.append(replace(residue, seq_index=len(residues)))
    prompt = build_document(
        target_id,
        tuple(residues),
        (),
        config=GenerationConfig(max_chains=2),
    )
    if prompt is None:
        raise ValueError(f"{target_id}: sequence does not fit contacts-v1 context")
    resolved_by_chain = [[], []]
    boundary = offsets[1]
    for position in sorted(set(resolved)):
        resolved_by_chain[int(position >= boundary)].append(position)
    structure_positions = [
        [offsets[index] + position for position in chain]
        for index, chain in enumerate(structure_local_positions)
    ]
    length = sum(len(row["sequence"]) for row in chain_rows)
    sequence = "".join(row["sequence"] for row in chain_rows)
    return {
        "target_id": target_id,
        "dataset": "foldbench_complex_pair_holdout_v1",
        "stem": target_id,
        "split": split,
        "group_id": group_id,
        "complex_type": target["complex_type"],
        "release_date": target["release_date"],
        "resolution": float(target["resolution"]),
        "title": target["title"],
        "L": length,
        "length": length,
        "sequence": sequence,
        "input_seq": sequence,
        "chain_ids": label_chains,
        "author_chain_ids": auth_chains,
        "chain_sequences": [row["sequence"] for row in chain_rows],
        "chain_lengths": [len(row["sequence"]) for row in chain_rows],
        "chain_offsets": offsets,
        "resolved_positions": sorted(set(resolved)),
        "resolved": sorted(set(resolved)),
        "n_resolved": len(set(resolved)),
        "resolved_positions_by_chain": resolved_by_chain,
        "structure_positions_by_chain": structure_positions,
        "n_resolved_pairs": len(resolved_by_chain[0]) * len(resolved_by_chain[1]),
        "n_gt": len(contacts),
        "gt_contacts": [[row["i"], row["j"]] for row in contacts],
        "contacts": [[row["i"], row["j"], row["degree"]] for row in contacts],
        "gt_i": [row["i"] for row in contacts],
        "gt_j": [row["j"] for row in contacts],
        "gt_degree": [row["degree"] for row in contacts],
        "sequence_prompt": prompt.document,
        "ground_truth_cif_sha256": sha256(path),
    }


def build_bundle(
    bundle: Path,
    selected: list[dict],
    interfaces: dict[str, list[dict]],
    eval_rows: list[dict],
) -> list[dict]:
    """Create a FoldBench-native directory plus compact scoring artifacts."""
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "targets").mkdir(parents=True)
    (bundle / "examples/ground_truths").mkdir(parents=True)
    (bundle / "data").mkdir(parents=True)
    selected_ids = {row["target"] for row in selected}
    target_rows = []
    for target_id in sorted(selected_ids):
        rows = interfaces[target_id]
        if len(rows) != 1:
            raise ValueError(f"{target_id}: expected one scored interface, got {len(rows)}")
        target_rows.append({key: rows[0][key] for key in TARGET_COLUMNS})
    write_csv(bundle / "targets/interface_protein_protein.csv", target_rows)
    af3_inputs = []
    for row in eval_rows:
        source = FOLDBENCH_GT / f"{row['target_id']}.cif"
        destination = bundle / "examples/ground_truths" / source.name
        selected_structure(source, row["author_chain_ids"]).make_mmcif_document().write_file(
            str(destination)
        )
        af3_inputs.append(
            {
                "dialect": "alphafold3",
                "version": 2,
                "name": row["target_id"],
                "sequences": [
                    {
                        "protein": {
                            "id": chain_id,
                            "sequence": sequence,
                            "modifications": [],
                            "unpairedMsa": None,
                            "pairedMsa": None,
                            "templates": None,
                        }
                    }
                    for chain_id, sequence in zip(
                        row["chain_ids"], row["chain_sequences"], strict=True
                    )
                ],
                "modelSeeds": ["42", "66", "101", "2024", "8888"],
                "userCCD": None,
            }
        )
    (bundle / "examples/alphafold3_inputs.json").write_text(
        json.dumps(af3_inputs, indent=2) + "\n"
    )
    for name in [
        "foldbench_freeze_audit.csv",
        "foldbench_eval_chains.csv",
        "foldbench_eval_chains.fasta",
        "foldbench_homology_edges.csv",
        "foldbench_homology_groups.csv",
        "foldbench_complex_eval.csv",
        "foldbench_complex_gt_universe.jsonl",
        "foldbench_complex_eval_targets.parquet",
        "foldbench_self_search.json",
        "foldbench_self_query_alignments.tsv",
        "foldbench_self_target_alignments.tsv",
        "foldbench_self_query_search.log",
        "foldbench_self_target_search.log",
    ]:
        shutil.copy2(DATA / name, bundle / "data" / name)
    shutil.copy2(FOLDBENCH / "LICENSE", bundle / "LICENSE.foldbench")
    (bundle / "README.md").write_text(
        """# FoldBench complex pair-holdout evaluation v1

This frozen subset contains 30 natural protein dimers released after the
FoldBench cutoff. Each candidate is pair-held-out from the MarinFold exp343
complex training corpus: no observed training complex contains homologs of
both partners at >=30% identity over >=50% of the shorter sequence. Three
targets with the same paired-homology exposure in Helico fine-tuning and two
manually identified de novo binders were excluded.

`targets/` and `examples/` form a FoldBench-compatible structural benchmark.
Each reference mmCIF is an exact coordinate-preserving subset containing only
the two chains named by the scored interface; this prevents extra biological-
assembly copies from entering Helico's input or DockQ calculation.
`data/foldbench_complex_eval_targets.parquet` contains full canonical sequences,
resolved-residue masks, and inter-chain contacts in zero-based concatenated
sequence coordinates for R-precision. Development/test assignments keep every
connected 30% homology group intact. Ground truth uses contacts-v1 pyconfind
native-only contacts with degree >=0.001.
"""
    )
    files = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(bundle)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return files


def main() -> None:
    """Freeze, cluster, split, extract contacts, and package the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work", type=Path, default=Path("/data/exp350_foldbench_eval")
    )
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    selected, audit_rows, interfaces = select_targets()
    write_csv(DATA / "foldbench_freeze_audit.csv", audit_rows)
    chains = load_chains(selected)
    write_csv(DATA / "foldbench_eval_chains.csv", chains)
    fasta = DATA / "foldbench_eval_chains.fasta"
    write_fasta(fasta, chains)
    search = self_search(fasta, args.work / "mmseqs", args.threads)
    outputs = search.pop("outputs")
    (DATA / "foldbench_self_search.json").write_text(
        json.dumps(search, indent=2) + "\n"
    )
    target_to_group, edges = homology_groups(chains, outputs)
    if edges:
        write_csv(DATA / "foldbench_homology_edges.csv", edges)
    else:
        (DATA / "foldbench_homology_edges.csv").write_text(
            "target_1,target_2,chain_1,chain_2,nident,alnlen,identity,qlen,"
            "tlen,qspan,tspan\n"
        )
    dev_groups = choose_dev_groups(selected, chains, target_to_group)
    groups: dict[str, list[str]] = defaultdict(list)
    for target, group in target_to_group.items():
        groups[group].append(target)
    group_rows = []
    for group, targets in sorted(groups.items()):
        targets.sort()
        group_rows.append(
            {
                "group_id": group,
                "split": "dev" if group in dev_groups else "test",
                "n_targets": len(targets),
                "targets": ";".join(targets),
            }
        )
    write_csv(DATA / "foldbench_homology_groups.csv", group_rows)
    chains_by_target: dict[str, list[dict]] = defaultdict(list)
    for chain in chains:
        chains_by_target[chain["target"]].append(chain)
    eval_rows = []
    for target in selected:
        target_id = target["target"]
        interface_rows = interfaces[target_id]
        if len(interface_rows) != 1:
            raise ValueError(
                f"{target_id}: expected one interface, got {len(interface_rows)}"
            )
        group = target_to_group[target_id]
        eval_rows.append(
            ground_truth_row(
                target,
                interface_rows[0],
                chains_by_target[target_id],
                group,
                "dev" if group in dev_groups else "test",
            )
        )
    compact = [
        {
            "target_id": row["target_id"],
            "split": row["split"],
            "group_id": row["group_id"],
            "complex_type": row["complex_type"],
            "length": row["length"],
            "chain_1_length": row["chain_lengths"][0],
            "chain_2_length": row["chain_lengths"][1],
            "n_resolved_chain_1": len(row["resolved_positions_by_chain"][0]),
            "n_resolved_chain_2": len(row["resolved_positions_by_chain"][1]),
            "n_resolved_pairs": row["n_resolved_pairs"],
            "n_gt": row["n_gt"],
            "release_date": row["release_date"],
            "resolution": row["resolution"],
        }
        for row in eval_rows
    ]
    write_csv(DATA / "foldbench_complex_eval.csv", compact)
    with (DATA / "foldbench_complex_gt_universe.jsonl").open("w") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    table = pa.Table.from_pylist(eval_rows)
    pq.write_table(
        table,
        DATA / "foldbench_complex_eval_targets.parquet",
        compression="zstd",
    )
    bundle = args.work / "bundle"
    files = build_bundle(bundle, selected, interfaces, eval_rows)
    manifest = {
        "dataset": "foldbench_complex_pair_holdout_v1",
        "public_prefix": (
            "hf://buckets/open-athena/MarinFold/data/evals/"
            "exp350_foldbench_pair_holdout/v1"
        ),
        "foldbench_revision": FOLDBENCH_REVISION,
        "foldbench_targets_sha256": sha256(FOLDBENCH_TARGETS),
        "selection": {
            "pair_clean_candidates": len(audit_rows),
            "helico_finetune_pair_exclusions": len(HELICO_EXCLUSIONS),
            "manual_design_exclusions": len(DESIGN_EXCLUSIONS),
            "included": len(eval_rows),
            "dev": sum(row["split"] == "dev" for row in eval_rows),
            "test": sum(row["split"] == "test" for row in eval_rows),
            "homology_groups": len(groups),
        },
        "coordinate_convention": (
            "zero-based full canonical chain-1 || chain-2 sequence coordinates; "
            "candidate universe is Cartesian product of resolved positions in "
            "different chains"
        ),
        "contact_definition": (
            "contacts-v1 pyconfind native-only; degree >=0.001; inter-chain only"
        ),
        "files": files,
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(manifest_path, DATA / "foldbench_bundle_manifest.json")
    print(json.dumps(manifest["selection"], indent=2))


if __name__ == "__main__":
    main()
