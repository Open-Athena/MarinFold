"""Count interface residues on both sides of each predicted dimer.

Phase 5 step 2 of CURATION_PLAN.md. PINDER's `GraphConfig.min_interface_length`
drops systems whose two chains barely touch, 7 residues on both sides. PINDER does
not publish the contact radius alongside that constant, so the radius is a flag
here rather than an inherited value: a residue is an interface residue when any of
its heavy atoms lies within `--contact-angstroms` of any heavy atom of the partner
chain, and the default 5.0 A is the CAPRI/DockQ heavy-atom contact convention.
`--sensitivity` recomputes at several radii at once so the cost of that choice is
measurable rather than assumed.

Distances are blocked, not all-pairs: each chain is first cut to the atoms inside
the partner's bounding box grown by the radius, which on these structures removes
most of the work, and the surviving block is chunked so peak memory stays flat
regardless of chain length.
"""

import argparse
import csv
import gzip
import time
from pathlib import Path

import numpy as np

# Positional fields of the AFDB _atom_site loop, which is written without a
# header per row. Verified against sample/structures/AF-0000000066921499.cif.
GROUP, TYPE_SYMBOL, ASYM, SEQ_ID, X, Y, Z = 0, 2, 6, 8, 10, 11, 12
CHUNK = 4096


def parse_atoms(text: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Heavy-atom coordinates and residue numbers, per chain.

    Hydrogens are excluded so the radius means what the DockQ convention means.
    AFDB predictions carry none, but the filter costs nothing and makes the
    function safe on any mmCIF.
    """
    coords: dict[str, list[tuple[float, float, float]]] = {}
    residues: dict[str, list[int]] = {}
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            continue
        field = line.split()
        if field[TYPE_SYMBOL] == "H":
            continue
        chain = field[ASYM]
        coords.setdefault(chain, []).append(
            (float(field[X]), float(field[Y]), float(field[Z])))
        residues.setdefault(chain, []).append(int(field[SEQ_ID]))
    return {chain: (np.array(values, dtype=np.float32),
                    np.array(residues[chain], dtype=np.int32))
            for chain, values in coords.items()}


def residue_blocks(coords: np.ndarray, residues: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-residue centroid, enclosing radius, and the residue index of each atom.

    The radius is measured, not assumed: it is the largest distance from a
    residue's centroid to one of its own heavy atoms, so the bound below is exact
    for any residue type.
    """
    _, index = np.unique(residues, return_inverse=True)
    count = np.bincount(index).astype(np.float32)
    centroid = np.zeros((len(count), 3), dtype=np.float32)
    np.add.at(centroid, index, coords)
    centroid /= count[:, None]
    spread = np.linalg.norm(coords - centroid[index], axis=1)
    radius = np.zeros(len(count), dtype=np.float32)
    np.maximum.at(radius, index, spread)
    return centroid, radius, index


def interface_residues(first: np.ndarray, second: np.ndarray,
                       first_res: np.ndarray, second_res: np.ndarray,
                       radius: float) -> tuple[int, int]:
    """Count residues on each side with a heavy atom within `radius` of the other.

    Two stages. A residue pair can only touch if the distance between their
    centroids is within the sum of their enclosing radii plus the contact radius,
    so the centroid pass (a few hundred rows per chain) discards most residues
    exactly, without approximating. Only the atoms of the survivors reach the
    blocked atom-against-atom pass, which is where the time would otherwise go.
    """
    left_centroid, left_radius, left_index = residue_blocks(first, first_res)
    right_centroid, right_radius, right_index = residue_blocks(second, second_res)

    gap = np.linalg.norm(left_centroid[:, None, :] - right_centroid[None, :, :], axis=2)
    reachable = gap <= (left_radius[:, None] + right_radius[None, :] + radius)
    left_live = reachable.any(axis=1)
    right_live = reachable.any(axis=0)
    if not left_live.any() or not right_live.any():
        return 0, 0

    left_keep = left_live[left_index]
    right_keep = right_live[right_index]
    left, left_owner = first[left_keep], left_index[left_keep]
    right, right_owner = second[right_keep], right_index[right_keep]

    hit_left = np.zeros(len(left_centroid), dtype=bool)
    hit_right = np.zeros(len(right_centroid), dtype=bool)
    squared = radius * radius
    for start in range(0, len(left), CHUNK):
        block = left[start:start + CHUNK]
        close = ((block[:, None, :] - right[None, :, :]) ** 2).sum(axis=2) <= squared
        hit_left[left_owner[start:start + CHUNK][close.any(axis=1)]] = True
        hit_right[right_owner[close.any(axis=0)]] = True
    return int(hit_left.sum()), int(hit_right.sum())


def read_text(path: Path) -> str:
    """One structure, gzipped or not."""
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode()
    return path.read_text()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, required=True,
                        help="directory of .cif or .cif.gz dimers")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contact-angstroms", type=float, default=5.0)
    parser.add_argument("--sensitivity", type=float, nargs="*", default=[],
                        help="extra radii to count at, for comparison")
    parser.add_argument("--min-interface", type=int, default=7,
                        help="PINDER's min_interface_length, required on both sides")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(p for p in args.structures.iterdir()
                   if p.name.endswith((".cif", ".cif.gz")))[:args.limit]
    radii = [args.contact_angstroms] + [r for r in args.sensitivity
                                        if r != args.contact_angstroms]
    fields = (["model_entity_id", "n_atoms", "n_chains", "parse_seconds",
               "contact_seconds", "interface_a", "interface_b", "min_side", "passes"]
              + [f"min_side_{radius:g}A" for radius in radii[1:]])

    kept = 0
    started = time.time()
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            mark = time.time()
            chains = parse_atoms(read_text(path))
            parsed = time.time() - mark
            name = path.name.removesuffix(".gz").removesuffix(".cif")
            if len(chains) != 2:
                raise SystemExit(f"{name} has {len(chains)} chains, expected 2")
            (first, first_res), (second, second_res) = (chains[key] for key in sorted(chains))

            mark = time.time()
            counts = [interface_residues(first, second, first_res, second_res, radius)
                      for radius in radii]
            contact = time.time() - mark

            left, right = counts[0]
            row = {
                "model_entity_id": name,
                "n_atoms": len(first) + len(second),
                "n_chains": len(chains),
                "parse_seconds": round(parsed, 4),
                "contact_seconds": round(contact, 4),
                "interface_a": left,
                "interface_b": right,
                "min_side": min(left, right),
                "passes": int(min(left, right) >= args.min_interface),
            }
            for radius, (one, two) in zip(radii[1:], counts[1:]):
                row[f"min_side_{radius:g}A"] = min(one, two)
            writer.writerow(row)
            kept += row["passes"]

    elapsed = time.time() - started
    print(f"{len(paths):,} structures, {kept:,} pass "
          f"(>= {args.min_interface} residues both sides at "
          f"{args.contact_angstroms:g} A), {elapsed:.1f} s total, "
          f"{elapsed / max(len(paths), 1) * 1000:.1f} ms each")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
