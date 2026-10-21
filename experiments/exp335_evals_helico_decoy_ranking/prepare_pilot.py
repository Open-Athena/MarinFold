"""Extract and validate candidate contact maps for the bounded Helico pilot."""

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import gemmi
from pyconfind import analyze, cached_rotamer_library, load_library

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
MIN_CONTACT_DEGREE = 0.001
MIN_SEQUENCE_SEPARATION = 6
PYCONFIND_KWARGS = {
    "native_only": True,
    "contact_distance": 3.0,
    "dcut": 25.0,
    "clash_distance": 2.0,
    "assembly": None,
}


def load_candidates(path: Path) -> list[dict[str, str]]:
    """Load the pilot candidate manifest."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} has no candidates")
    keys = [(row["target"], row["decoy_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("pilot candidate manifest has duplicate keys")
    return rows


def candidate_path(decoy_dir: Path, target: str, decoy_id: str) -> Path:
    """Resolve a pilot identifier to its source PDB."""
    if decoy_id == "native":
        return decoy_dir / "natives" / f"{target}.pdb"
    return decoy_dir / target / decoy_id


def model_residues(
    structure: gemmi.Structure,
) -> tuple[str, list[tuple[str, int, str]]]:
    """Return the single protein chain's sequence and residue identifiers."""
    if len(structure) != 1:
        raise ValueError(f"expected one model, found {len(structure)}")
    chains: list[tuple[str, list[tuple[str, int, str]]]] = []
    for chain in structure[0]:
        residues: list[tuple[str, int, str]] = []
        for residue in chain:
            name = residue.name.strip().upper()
            if name not in THREE_TO_ONE:
                continue
            residues.append((name, residue.seqid.num, residue.seqid.icode.strip()))
        if residues:
            chains.append((chain.name, residues))
    if len(chains) != 1:
        raise ValueError(
            f"expected one protein chain, found {[chain for chain, _ in chains]}"
        )
    _, residues = chains[0]
    sequence = "".join(THREE_TO_ONE[name] for name, _, _ in residues)
    return sequence, residues


def contact_pairs(
    structure: gemmi.Structure, rotamer_library
) -> tuple[str, list[list[int]]]:
    """Derive the Helico full-map PRESENT entries from one candidate."""
    sequence, residues = model_residues(structure)
    analysis = analyze(structure, rotamer_library=rotamer_library, **PYCONFIND_KWARGS)
    if len(analysis.positions) != len(residues):
        raise ValueError(
            f"pyconfind returned {len(analysis.positions)} positions for {len(residues)} residues"
        )
    observed = "".join(
        THREE_TO_ONE.get(item.position.resname.strip().upper(), "X")
        for item in analysis.positions
    )
    if observed != sequence:
        raise ValueError(
            "pyconfind residue order does not match the parsed PDB sequence"
        )

    pairs = sorted(
        [min(contact.pos_i, contact.pos_j), max(contact.pos_i, contact.pos_j)]
        for contact in analysis.report.contacts
        if contact.degree >= MIN_CONTACT_DEGREE
        and abs(contact.pos_i - contact.pos_j) >= MIN_SEQUENCE_SEPARATION
    )
    if len(pairs) != len({tuple(pair) for pair in pairs}):
        raise ValueError("pyconfind emitted duplicate contact pairs")
    return sequence, pairs


def ca_coordinates(structure: gemmi.Structure) -> list[list[float]]:
    """Return one C-alpha coordinate per standard residue in sequence order."""
    coordinates: list[list[float]] = []
    for chain in structure[0]:
        for residue in chain:
            if residue.name.strip().upper() not in THREE_TO_ONE:
                continue
            ca_atoms = [atom for atom in residue if atom.name.strip() == "CA"]
            if len(ca_atoms) != 1:
                raise ValueError(
                    f"residue {chain.name}:{residue.seqid} has {len(ca_atoms)} C-alpha atoms"
                )
            position = ca_atoms[0].pos
            coordinates.append(
                [float(position.x), float(position.y), float(position.z)]
            )
    return coordinates


def map_digest(sequence: str, pairs: list[list[int]]) -> str:
    """Hash the exact full-map input represented by sequence plus PRESENT pairs."""
    payload = json.dumps(
        {"sequence": sequence, "present_pairs": pairs},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries to CSV with stable columns."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/pilot_candidates.csv")
    )
    parser.add_argument("--decoy-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, default=Path("scratch"))
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("data/pilot_contact_map_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    """Extract every pilot map, validate indexing, and report duplicates."""
    args = parse_args()
    candidates = load_candidates(args.candidates)
    rotamer_library = load_library(cached_rotamer_library())
    target_sequences: dict[str, str] = {}
    maps: list[dict] = []
    summaries: list[dict] = []

    for index, candidate in enumerate(candidates, start=1):
        target = candidate["target"]
        decoy_id = candidate["decoy_id"]
        path = candidate_path(args.decoy_dir, target, decoy_id)
        if not path.is_file():
            raise FileNotFoundError(path)

        started = time.monotonic()
        structure = gemmi.read_structure(str(path))
        sequence, pairs = contact_pairs(structure, rotamer_library)
        candidate_ca = ca_coordinates(structure)
        if len(candidate_ca) != len(sequence):
            raise ValueError(
                f"{target}/{decoy_id}: {len(candidate_ca)} C-alpha atoms for "
                f"{len(sequence)} residues"
            )
        elapsed = time.monotonic() - started
        if target in target_sequences and sequence != target_sequences[target]:
            raise ValueError(
                f"{target}/{decoy_id}: sequence differs from the target native "
                f"({len(sequence)} vs {len(target_sequences[target])} residues)"
            )
        target_sequences.setdefault(target, sequence)
        digest = map_digest(sequence, pairs)
        maps.append(
            {
                "target": target,
                "decoy_id": decoy_id,
                "sequence": sequence,
                "present_pairs": pairs,
                "candidate_ca": candidate_ca,
                "contact_map_sha256": digest,
            }
        )
        summaries.append(
            {
                "target": target,
                "decoy_id": decoy_id,
                "candidate_kind": candidate["candidate_kind"],
                "n_residues": len(sequence),
                "n_pairs": len(sequence) * (len(sequence) - 1) // 2,
                "n_present_contacts": len(pairs),
                "contacts_per_residue": len(pairs) / len(sequence),
                "contact_map_sha256": digest,
                "extraction_seconds": round(elapsed, 6),
            }
        )
        if index % 25 == 0 or index == len(candidates):
            print(f"extracted {index}/{len(candidates)} contact maps")

    duplicate_counts = Counter(
        (row["target"], row["contact_map_sha256"]) for row in maps
    )
    for row in summaries:
        multiplicity = duplicate_counts[(row["target"], row["contact_map_sha256"])]
        row["same_target_map_multiplicity"] = multiplicity
        row["is_duplicate_map"] = int(multiplicity > 1)

    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    map_path = args.scratch_dir / "pilot_contact_maps.jsonl.gz"
    with gzip.open(map_path, "wt") as stream:
        for row in maps:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.summary_path, summaries)

    n_unique = len(duplicate_counts)
    print(
        f"validated {len(maps)} maps across {len(target_sequences)} targets; "
        f"{n_unique} unique within target ({len(maps) - n_unique} duplicates); "
        f"wrote {map_path}"
    )


if __name__ == "__main__":
    main()
