# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch the ground-truth structures for the low-MSA-depth set, in eval indices.

The dashboard draws contacts as lines between residues, so a Cα coordinate has
to be addressable by the *evaluation* residue index — the same index the
contact records and the vote matrices use. Getting there means fetching the
deposited entry, finding the chain the evaluation unit came from, and mapping
its modelled residues onto the evaluation sequence.

Every atom of the mapped chain is kept, not just Cα: a Cα-only model renders as
spaghetti, and the point of the viewer is to see the fold. The emitted PDB is
renumbered into evaluation indices and carries HELIX/SHEET records derived from
biotite's P-SEA annotation, so the viewer draws a real cartoon rather than a
tube. A separate Cα array indexed the same way is what the contact lines
attach to.

Chain selection does not trust an identifier. #226 already ran into RCSB's
auth-versus-label chain naming, and the CASP domains are a sub-range of a chain
under a third numbering. So every polymer chain in the entry is aligned against
the evaluation sequence and the best match wins, with the resulting coverage
reported per protein — a bad mapping shows up as low coverage rather than as a
plausible-looking wrong picture.

    uv run python dashboard/build_structures.py            # fetch + map
    uv run python dashboard/build_structures.py --refresh  # re-download the mmCIFs
"""

import argparse
import json
import re
import tarfile
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from biotite.structure import annotate_sse, array as structure_array, Atom

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DATA = EXPERIMENT / "data"
CACHE = EXPERIMENT / "scratch" / "mmcif"
RCSB_CIF = "https://files.rcsb.org/download/{pdb}.cif"

#: Two CASP free-modeling domains have no released PDB entry to fall back on
#: (#65's ``casp_fm_pdb_fallback.csv`` covers the rest), so their coordinates
#: come from predictioncenter's public domain tarballs, the same source #65
#: used to build the set. The filenames carry a release date, so the directory
#: index is matched by prefix rather than pinned.
#: CASP14 splits its release into a domain-level tarball that holds only the
#: multi-domain targets and a full-target tarball for everything else, so both
#: are tried and a single-domain target is matched by its target name.
CASP_TARBALLS = {
    "T1043-D1": (
        "https://predictioncenter.org/download_area/CASP14/targets/",
        ("casp14.targets.T-dom.public", "casp14.targets.T.public"),
    ),
    "T1123-D1": (
        "https://predictioncenter.org/download_area/CASP15/targets/",
        ("casp15.targets.TS-domains.public", ),
    ),
}

#: Below this fraction of the evaluation sequence mapped to coordinates, the
#: structure is reported as unusable rather than drawn.
MIN_COVERAGE = 0.5


def structure_sources() -> pd.DataFrame:
    """Resolve each low-MSA-depth protein to an RCSB entry and chain, if any."""

    low = pd.read_csv(DATA / "low_msa_depth_set.csv")
    exp65 = EXPERIMENT.parent / "exp65_evals_low_msa_depth_proteins" / "data"
    cameo = pd.read_csv(exp65 / "cameo_hard_manifest.csv")[["stem", "pdb_id", "chain"]]
    casp = pd.read_csv(exp65 / "casp_fm_pdb_fallback.csv")
    casp = casp[casp.status == "pdb_fallback"].rename(columns={"domain": "stem"})
    casp = casp[["stem", "pdb_id", "chain", "casp_range"]]
    foldbench = pd.read_csv(DATA / "foldbench_chains.csv")[["stem", "pdb_id", "chain"]]

    sources = pd.concat([cameo, casp, foldbench], ignore_index=True)
    merged = low[["dataset", "stem", "L"]].merge(sources, on="stem", how="left")
    return merged


def fetch_cif(pdb_id: str, *, refresh: bool = False) -> Path:
    """Download one mmCIF into the scratch cache."""

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{pdb_id}.cif"
    if path.exists() and path.stat().st_size and not refresh:
        return path
    with urllib.request.urlopen(RCSB_CIF.format(pdb=pdb_id)) as response:
        path.write_bytes(response.read())
    return path


def fetch_casp_domain(domain: str, *, refresh: bool = False) -> Path | None:
    """Extract one CASP domain PDB from predictioncenter's public tarball."""

    if domain not in CASP_TARBALLS:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / f"{domain}.pdb"
    if target.exists() and target.stat().st_size and not refresh:
        return target
    base, prefixes = CASP_TARBALLS[domain]
    with urllib.request.urlopen(base) as response:
        index = response.read().decode("utf-8", "replace")
    wanted = {f"{domain}.pdb", f"{domain.split('-')[0]}.pdb"}
    for prefix in prefixes:
        names = sorted(
            set(re.findall(rf'href="({re.escape(prefix)}[^"]*?\.tar\.gz)"', index))
        )
        if not names:
            continue
        archive = CACHE / names[-1]
        if not archive.exists() or not archive.stat().st_size:
            with urllib.request.urlopen(base + names[-1]) as response:
                archive.write_bytes(response.read())
        with tarfile.open(archive) as handle:
            member = next(
                (m for m in handle.getmembers() if Path(m.name).name in wanted), None
            )
            if member is None:
                continue
            target.write_bytes(handle.extractfile(member).read())
            return target
    raise RuntimeError(f"none of {sorted(wanted)} found under {base}")


def chain_traces(path: Path) -> list[dict]:
    """Return one record per polymer chain: sequence, Cα, and every atom.

    Only the first model is read, and only residues that carry a Cα — an
    unmodelled residue has no coordinate to draw. Hydrogens are dropped (they
    are absent from most of these entries anyway and double the atom count),
    and alternate locations collapse to the first.
    """

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    traces = []
    for chain in structure[0]:
        letters, coordinates, residues = [], [], []
        for residue in chain:
            atom = residue.find_atom("CA", "*")
            info = gemmi.find_tabulated_residue(residue.name)
            if atom is None or info is None or not info.is_amino_acid():
                continue
            letters.append(info.one_letter_code.upper())
            coordinates.append([round(atom.pos.x, 2), round(atom.pos.y, 2), round(atom.pos.z, 2)])
            residues.append(
                {
                    "name": residue.name,
                    "atoms": [
                        (a.name, a.element.name, round(a.pos.x, 3), round(a.pos.y, 3), round(a.pos.z, 3))
                        for a in residue
                    ],
                }
            )
        if len(letters) >= 10:
            traces.append(
                {
                    "chain": chain.name,
                    "sequence": "".join(letters),
                    "coordinates": coordinates,
                    "residues": residues,
                }
            )
    return traces


def secondary_structure(residues: list[dict]) -> list[str]:
    """P-SEA secondary structure per residue: ``a`` helix, ``b`` strand, ``c`` coil."""

    atoms = []
    for index, residue in enumerate(residues):
        if residue is None:
            continue
        for name, element, x, y, z in residue["atoms"]:
            atoms.append(
                Atom(
                    [x, y, z], chain_id="A", res_id=index + 1, res_name=residue["name"],
                    atom_name=name, element=element or name[0], hetero=False,
                )
            )
    if not atoms:
        return []
    annotation = annotate_sse(structure_array(atoms))
    order = [index for index, residue in enumerate(residues) if residue is not None]
    per_residue = ["c"] * len(residues)
    for position, code in zip(order, annotation, strict=False):
        per_residue[position] = str(code)
    return per_residue


def pdb_text(residues: list[dict], sse: list[str]) -> str:
    """Render the mapped chain as PDB, renumbered into evaluation indices.

    HELIX/SHEET records come from the P-SEA annotation so the viewer's cartoon
    shows the actual fold; without them 3Dmol falls back to a featureless tube.
    """

    lines, serial = [], 1
    runs: list[tuple[str, int, int]] = []
    for index, code in enumerate(sse):
        if code in ("a", "b") and runs and runs[-1][0] == code and runs[-1][2] == index - 1:
            runs[-1] = (code, runs[-1][1], index)
        elif code in ("a", "b"):
            runs.append((code, index, index))
    helix = sheet = 0
    for code, start, end in runs:
        if end - start < 2:
            continue
        if code == "a":
            helix += 1
            lines.append(
                f"HELIX  {helix:3d} {helix:3d} ALA A {start + 1:4d}  ALA A {end + 1:4d}  1"
            )
        else:
            sheet += 1
            lines.append(
                f"SHEET  {sheet:3d} S{sheet:2d} 1 ALA A {start + 1:4d}  ALA A {end + 1:4d}  0"
            )
    for index, residue in enumerate(residues):
        if residue is None:
            continue
        for name, element, x, y, z in residue["atoms"]:
            atom_name = f" {name:<3}" if len(name) < 4 else name
            lines.append(
                f"ATOM  {serial:5d} {atom_name} {residue['name']:>3} A{index + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {(element or name[0]):>2}"
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines)


def map_to_eval(trace: dict, eval_sequence: str) -> tuple[list, float, int]:
    """Align one chain trace onto the evaluation sequence.

    Returns per-eval-index coordinates (``None`` where unmodelled), the fraction
    of the evaluation sequence covered, and the number of mismatched residues in
    the aligned blocks.
    """

    matcher = SequenceMatcher(None, trace["sequence"], eval_sequence, autojunk=False)
    coordinates: list = [None] * len(eval_sequence)
    residues: list = [None] * len(eval_sequence)
    covered = 0
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            coordinates[block.b + offset] = trace["coordinates"][block.a + offset]
            residues[block.b + offset] = trace["residues"][block.a + offset]
            covered += 1
    return (coordinates, residues), covered / len(eval_sequence), len(eval_sequence) - covered


def build(*, refresh: bool) -> dict:
    """Fetch and map every structure we can resolve."""

    sources = structure_sources()
    targets = pd.read_csv(DATA / "low_depth_sequences.csv").set_index("stem")
    out: dict[str, dict] = {}
    for record in sources.itertuples(index=False):
        key = f"{record.dataset}__{record.stem}"
        eval_sequence = targets.loc[record.stem, "input_seq"]
        if not isinstance(record.pdb_id, str) or not record.pdb_id:
            path = fetch_casp_domain(record.stem, refresh=refresh)
            if path is None:
                out[key] = {"available": False, "reason": "no released PDB entry"}
                print(f"[structure] {key}: no entry", flush=True)
                continue
            source = f"CASP domain tarball ({record.stem}.pdb)"
        else:
            path = fetch_cif(record.pdb_id, refresh=refresh)
            source = record.pdb_id
        traces = chain_traces(path)
        best, best_coverage, best_mismatch = None, -1.0, None
        for trace in traces:
            (coordinates, residues), coverage, mismatch = map_to_eval(trace, eval_sequence)
            if coverage > best_coverage:
                best, best_coverage, best_mismatch = (
                    {"chain": trace["chain"], "coordinates": coordinates, "residues": residues},
                    coverage,
                    mismatch,
                )
        if best is None or best_coverage < MIN_COVERAGE:
            out[key] = {
                "available": False,
                "reason": f"best chain covers only {best_coverage:.2f} of the sequence",
                "pdb_id": source,
            }
            print(f"[structure] {key}: coverage {best_coverage:.2f} — dropped", flush=True)
            continue
        sse = secondary_structure(best["residues"])
        out[key] = {
            "available": True,
            "pdb_id": source,
            "chain": best["chain"],
            "expected_chain": record.chain if isinstance(record.chain, str) else None,
            "coverage": round(best_coverage, 4),
            "unmapped_residues": best_mismatch,
            "n_atoms": sum(
                len(residue["atoms"]) for residue in best["residues"] if residue
            ),
            "coordinates": best["coordinates"],
            "pdb": pdb_text(best["residues"], sse),
        }
        print(
            f"[structure] {key}: {source} chain {best['chain']} "
            f"coverage {best_coverage:.2f}",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    structures = build(refresh=args.refresh)
    destination = DATA / "low_depth_structures.json"
    destination.write_text(json.dumps(structures, sort_keys=True))
    available = sum(1 for record in structures.values() if record["available"])
    print(
        json.dumps(
            {
                "proteins": len(structures),
                "with_coordinates": available,
                "missing": [
                    key for key, record in structures.items() if not record["available"]
                ],
                "bytes": destination.stat().st_size,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
