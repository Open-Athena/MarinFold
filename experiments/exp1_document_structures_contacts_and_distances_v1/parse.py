# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Gemmi-based PDB / mmCIF parsing for the contacts-and-distances-v1 experiment.

Local to this experiment dir. Imported by ``generate.py`` and
``inference.py``. Reads PDB / mmCIF (+ ``.gz``), filters atoms to
the v1 vocab, maps non-canonical residues to UNK, and reads pLDDT
from B-factors.
"""

import math
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import fsspec

from vocab import AMINO_ACIDS, ATOM_NAMES

# Canonical 20 amino-acid set — non-canonical residues are mapped to
# UNK at parse time (matching contactdoc's
# canonical_residue_policy = "map_to_unk").
_CANONICAL_20 = frozenset(AMINO_ACIDS)

# Recognised file extensions when ``iter_structure_paths`` is given
# a directory. Gemmi auto-detects format from extension via
# ``gemmi.read_structure(path)``.
_STRUCTURE_EXTS = frozenset({
    ".cif", ".cif.gz", ".mmcif", ".mmcif.gz",
    ".pdb", ".pdb.gz", ".ent", ".ent.gz",
})


@dataclass(frozen=True)
class Residue:
    """A single residue extracted from a polymer chain."""

    index: int  # 1-based position in the chain
    name: str   # 3-letter canonical name, or "UNK"
    plddt: float
    # (atom_name, x, y, z) tuples for all non-hydrogen, in-vocab atoms.
    atoms: tuple[tuple[str, float, float, float], ...]


@dataclass(frozen=True)
class ParsedStructure:
    """One parsed polymer chain, ready for doc generation or evaluation.

    Records yielded by ``iter_structure_paths`` + ``parse_structure``.
    """

    entry_id: str  # the file stem (e.g. "AF-P00767-F1-model_v4")
    residues: tuple[Residue, ...]
    source_path: str  # fully-qualified URL or local path, for provenance

    @property
    def sequence(self) -> list[str]:
        return [r.name for r in self.residues]

    @property
    def global_plddt(self) -> float:
        """Mean pLDDT across all residues."""
        if not self.residues:
            return float("nan")
        return sum(r.plddt for r in self.residues) / len(self.residues)


def _vocab_safe_atoms(gemmi_residue) -> tuple[tuple[str, float, float, float], ...]:
    """Heavy atoms whose names are present in the v1 atom vocab.

    Drops hydrogens and any atom whose name is outside ``ATOM_NAMES``
    (e.g. non-canonical residue atoms, alt-loc artifacts).
    """
    valid_names = set(ATOM_NAMES)
    out: list[tuple[str, float, float, float]] = []
    for atom in gemmi_residue:
        if atom.is_hydrogen():
            continue
        name = atom.name.strip()
        if name not in valid_names:
            continue
        out.append((name, atom.pos.x, atom.pos.y, atom.pos.z))
    return tuple(out)


def _residue_plddt(gemmi_residue) -> float:
    """Per-residue pLDDT = mean B-factor of heavy atoms.

    Matches contactdoc's ``_residue_plddt`` exactly. For non-AFDB
    structures the B-factor isn't pLDDT but the algorithm doesn't
    care — it just uses the value as a confidence proxy.
    """
    b_values = []
    for atom in gemmi_residue:
        if not atom.is_hydrogen():
            b_values.append(atom.b_iso)
    if not b_values:
        return float("-inf")
    return sum(b_values) / len(b_values)


def _strip_gz(name: str) -> str:
    return name[:-3] if name.lower().endswith(".gz") else name


def _coor_format(url: str):
    """Map a (possibly gzipped) URL to the gemmi format enum.

    ``read_structure_string`` can't sniff format from a filename the way
    ``read_structure`` does, so we infer it from the extension and fall
    back to content detection.
    """
    import gemmi

    name = _strip_gz(url).lower()
    if name.endswith((".pdb", ".ent")):
        return gemmi.CoorFormat.Pdb
    if name.endswith((".cif", ".mmcif")):
        return gemmi.CoorFormat.Mmcif
    return gemmi.CoorFormat.Detect


def _entry_id_from_url(url: str) -> str:
    """Derive the entry id from a URL/path basename (drops .gz + format ext)."""
    base = _strip_gz(url.rstrip("/").rsplit("/", 1)[-1])
    for ext in (".cif", ".mmcif", ".pdb", ".ent"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return base


def parse_structure(
    path,
    *,
    require_single_chain: bool = True,
) -> ParsedStructure:
    """Parse one structure file (mmCIF or PDB, optionally gzipped).

    ``path`` may be a local path or any fsspec-supported URL (``gs://``,
    ``s3://`, ...); bytes are read through fsspec so gemmi never has to
    touch a remote filesystem directly.

    Raises:
        ValueError: if the structure has no polymer chains, requires
            single-chain and has multiple, or has no residues.
    """
    import gemmi

    url = str(path)
    with fsspec.open(url, "rb", compression="infer") as f:
        data = f.read()
    structure = gemmi.read_structure_string(data, format=_coor_format(url))
    # Populate entity / polymer metadata. AFDB mmCIFs ship with it
    # baked in but hand-rolled PDBs (and bare PDBs without TER records)
    # need this call before chain.get_polymer() returns anything.
    structure.setup_entities()
    if len(structure) == 0:
        raise ValueError(f"{path}: no models in structure")
    model = structure[0]
    polymer_chains = [ch for ch in model if ch.get_polymer()]
    if not polymer_chains:
        raise ValueError(f"{path}: no polymer chain")
    if require_single_chain and len(polymer_chains) != 1:
        raise ValueError(
            f"{path}: expected single polymer chain, found {len(polymer_chains)}. "
            "Pass require_single_chain=False to take the first."
        )
    chain = polymer_chains[0]
    polymer = chain.get_polymer()
    residues: list[Residue] = []
    for idx, res in enumerate(polymer, start=1):
        name = res.name.strip()
        if name not in _CANONICAL_20:
            name = "UNK"
        residues.append(Residue(
            index=idx,
            name=name,
            plddt=_residue_plddt(res),
            atoms=_vocab_safe_atoms(res),
        ))
    if not residues:
        raise ValueError(f"{url}: no residues parsed")
    # read_structure_string leaves structure.name empty (no filename to
    # derive it from), so fall back to the URL basename.
    entry_id = structure.name or _entry_id_from_url(url)
    return ParsedStructure(
        entry_id=entry_id,
        residues=tuple(residues),
        source_path=url,
    )


# Characters that make a path a glob pattern (incl. brace expansion, which
# zephyr's from_files supports). If any are present we pass the spec straight
# through to the globber rather than trying to stat it as a literal path.
_GLOB_MAGIC = frozenset("*?[{")


def input_glob(path) -> str:
    """Turn an ``--input`` spec into a glob pattern for ``Dataset.from_files``.

    Globs (and brace patterns) pass through verbatim. A bare directory is
    expanded to a recursive structure-file pattern (so the worker-side glob
    only matches parseable extensions); a single file is returned unchanged —
    ``from_files`` globs a concrete path to itself.
    """
    spec = str(path)
    if any(c in spec for c in _GLOB_MAGIC):
        return spec
    fs, root = fsspec.core.url_to_fs(spec)
    if fs.isdir(root):
        exts = ",".join(sorted(e.lstrip(".") for e in _STRUCTURE_EXTS))
        return f"{spec.rstrip('/')}/**/*.{{{exts}}}"
    return spec


def try_parse_structure(path) -> "ParsedStructure | None":
    """Parse one structure, returning ``None`` (with a warning) on failure.

    The ``.map``-friendly counterpart to :func:`iter_parsed_structures`'s
    warn-and-skip behaviour — raising inside a Zephyr map stage would kill
    the worker, so unparseable files are dropped instead.
    """
    try:
        return parse_structure(path)
    except ValueError as exc:
        warnings.warn(f"skipping {path}: {exc}", stacklevel=2)
        return None


def iter_structure_paths(path) -> Iterator[str]:
    """Yield structure-file URLs under ``path`` (file or directory, recursive).

    ``path`` may be a local path or any fsspec-supported URL (``gs://``,
    ``s3://``, ...). Yields fully-qualified URLs (protocol preserved) in
    sorted order so downstream readers can re-open them on any filesystem
    and so iteration order is deterministic.
    """
    url = str(path)
    fs, root = fsspec.core.url_to_fs(url)
    if fs.isfile(root):
        candidates = [root]
    elif fs.isdir(root):
        # find() is recursive and returns files only (no directory entries).
        candidates = sorted(fs.find(root))
    else:
        raise FileNotFoundError(url)
    for p in candidates:
        if any(p.lower().endswith(ext) for ext in _STRUCTURE_EXTS):
            yield fs.unstrip_protocol(p)


def iter_parsed_structures(path) -> Iterator[ParsedStructure]:
    """Convenience: parse every structure file under ``path``.

    Files that fail to parse are skipped with a warning rather than
    aborting iteration.
    """
    for p in iter_structure_paths(path):
        parsed = try_parse_structure(p)
        if parsed is not None:
            yield parsed


# --------------------------------------------------------------------------
# Atom-position helpers used by both generation and evaluation
# --------------------------------------------------------------------------


def atom_position(residue: Residue, atom_name: str) -> tuple[float, float, float] | None:
    """Return the (x, y, z) of the named atom on ``residue``, or None if absent."""
    for name, x, y, z in residue.atoms:
        if name == atom_name:
            return (x, y, z)
    return None


def cb_or_ca_position(residue: Residue) -> tuple[float, float, float] | None:
    """CB position for ``residue`` (or CA for GLY / missing CB).

    Matches contactdoc's ``_get_cb_or_ca``. Returns ``None`` if
    neither CB nor CA is present.
    """
    target = "CA" if residue.name == "GLY" else "CB"
    fallback_ca: tuple[float, float, float] | None = None
    for name, x, y, z in residue.atoms:
        if name == target:
            return (x, y, z)
        if name == "CA":
            fallback_ca = (x, y, z)
    return fallback_ca


def euclidean(p1, p2) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)
