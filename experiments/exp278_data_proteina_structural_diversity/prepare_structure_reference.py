"""Stage legacy and full FoldBench-monomer structures for near-duplicate exclusion.

Uses existing local ground-truth caches; no evaluation predictions or scores are
read. Reference coverage and coordinate hashes are saved alongside the staged CA
traces. Sequence decontamination separately covers all FoldBench protein chains.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np

from analyze_screen import ca_structure, write_csv


def stage_chain(chain: gemmi.Chain, key: str, source: Path, output: Path) -> dict:
    """Extract resolved protein C-alpha positions and keep explicit coverage counts."""
    residues = [
        residue
        for residue in chain
        if gemmi.find_tabulated_residue(residue.name).is_amino_acid()
        and "CA" in residue
    ]
    sequence = "".join(
        gemmi.find_tabulated_residue(residue.name).one_letter_code.upper()
        for residue in residues
    )
    coordinates = np.asarray([list(residue["CA"][0].pos) for residue in residues])
    if len(sequence) < 3 or not np.isfinite(coordinates).all():
        raise ValueError(f"Invalid reference chain: {key}")
    target = output / (key + ".pdb")
    ca_structure(sequence, coordinates).write_pdb(str(target))
    return {
        "key": key,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "chain": chain.name,
        "resolved_residues": len(sequence),
        "ca_trace_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy", type=Path, default=Path("/data/exp225_decontam/eval_structures")
    )
    parser.add_argument(
        "--foldbench-targets",
        type=Path,
        default=Path("/home/bizon/git/FoldBench/targets/monomer_protein.csv"),
    )
    parser.add_argument("--foldbench-cif", type=Path, default=Path("/data/exp245/cif"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = sorted(args.legacy.glob("*.cif"))
    if len(sources) != 554:
        raise ValueError(f"Expected 554 frozen legacy structures, found {len(sources)}")
    for source in sources:
        structure = gemmi.read_structure(str(source))
        if len(structure) != 1 or len(structure[0]) != 1:
            raise ValueError(
                f"Legacy structure is not a single resolved chain: {source}"
            )
        rows.append(
            stage_chain(structure[0][0], "legacy__" + source.stem, source, args.output)
        )
    with args.foldbench_targets.open() as handle:
        targets = list(csv.DictReader(handle))
    ground_truth_path = (
        Path(__file__).resolve().parent.parent
        / "exp245_evals_foldbench_held_out_monomers/data/gt_universe_foldbench_monomers.jsonl"
    )
    with ground_truth_path.open() as handle:
        chain_resolution = {
            row["stem"]: row["gt_chain"] for row in map(json.loads, handle)
        }
    for row in targets:
        source = args.foldbench_cif / (row["pdb_id"] + ".cif")
        structure = gemmi.read_structure(str(source))
        stem = row["pdb_id"].split("-")[0] + "_" + row["chain_id"]
        chain = structure[0][chain_resolution[stem]]
        rows.append(
            stage_chain(
                chain,
                f"foldbench__{row['pdb_id']}__{row['chain_id']}",
                source,
                args.output,
            )
        )
    write_csv(args.report / "structure-reference.csv", rows)
    (args.report / "structure-reference.json").write_text(
        json.dumps(
            {
                "legacy": len(sources),
                "foldbench_monomers": len(targets),
                "reference_structures": len(rows),
                "scope": "legacy 554 plus full FoldBench monomer list, including all eval245 partitions; resolved CA traces",
                "foldbench_targets_sha256": hashlib.sha256(
                    args.foldbench_targets.read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
