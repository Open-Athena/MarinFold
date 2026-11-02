"""Create compact chain-specific experimental backbones for every eval protein."""

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gemmi


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/eval.json"
STRUCTURES = HERE / "structures"
BACKBONE = ("N", "CA", "C", "O")


def chain_pdb(protein: dict) -> tuple[str, str]:
    """Return one experimental chain's backbone as a browser-friendly PDB."""
    url = f"https://files.rcsb.org/download/{protein['pdbId'].upper()}.cif"
    with urllib.request.urlopen(url, timeout=60) as response:
        cif = response.read().decode()
    structure = gemmi.read_structure_string(cif, format=gemmi.CoorFormat.Mmcif)
    structure.remove_alternative_conformations()
    chain_name = protein.get("sourceChain", protein["viewerChain"].split(";")[0])
    chains = [chain for chain in structure[0] if chain.name == chain_name]
    if len(chains) != 1:
        raise ValueError(
            f"{protein['id']}: expected chain {chain_name}, found {len(chains)}"
        )
    chain = chains[0]
    lines = []
    serial = 1
    residue_index = 0
    ca_count = 0
    for residue in chain:
        if residue.entity_type != gemmi.EntityType.Polymer:
            continue
        atoms = {atom.name: atom for atom in residue if atom.name in BACKBONE}
        if "CA" not in atoms:
            continue
        residue_index += 1
        ca_count += 1
        for name in BACKBONE:
            if name not in atoms:
                continue
            atom = atoms[name]
            pos = atom.pos
            lines.append(
                f"ATOM  {serial:5d} {name:4s} {residue.name:3s} A{residue_index:4d}    "
                f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}{1.0:6.2f}{atom.b_iso:6.2f}"
                f"          {name[0]:>2s}\n"
            )
            serial += 1
    if ca_count < 2:
        raise ValueError(
            f"{protein['id']}: chain {chain_name} has only {ca_count} Cα atoms"
        )
    if ca_count > protein["length"]:
        raise ValueError(
            f"{protein['id']}: {ca_count} Cα atoms exceed {protein['length']} sequence residues"
        )
    lines.extend(("TER\n", "END\n"))
    return protein["id"], "".join(lines)


def main() -> None:
    """Save all eval backbones locally and update the evaluation snapshot."""
    snapshot = json.loads(DATA.read_text())
    proteins = snapshot["proteins"]
    STRUCTURES.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (protein_id, pdb) in enumerate(pool.map(chain_pdb, proteins), 1):
            protein = proteins[index - 1]
            if protein["id"] != protein_id:
                raise ValueError(f"Unexpected extraction order: {protein_id}")
            filename = hashlib.sha256(protein_id.encode()).hexdigest()[:20] + ".pdb"
            (STRUCTURES / filename).write_text(pdb)
            protein["structureUrl"] = f"structures/{filename}"
            protein["structureFormat"] = "pdb"
            protein["pdbFallbackUrl"] = f"structures/{filename}"
            protein["sourceChain"] = protein.get(
                "sourceChain", protein["viewerChain"].split(";")[0]
            )
            protein["structureNote"] = (
                f"Experimental PDB chain {protein['sourceChain']} backbone"
            )
            protein["viewerChain"] = "A"
            if index % 20 == 0 or index == len(proteins):
                print(f"Extracted {index}/{len(proteins)} eval chains", flush=True)
    DATA.write_text(json.dumps(snapshot, separators=(",", ":")))


if __name__ == "__main__":
    main()
