# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v1 DocumentStructure.

The canonical MarinFold protein document format. A document looks
like::

    <contacts-and-distances-v1>
    <begin_sequence> <AA_1> ... <AA_n>
    <begin_statements>
    <long-range-contact> <p_i> <p_j>
    <medium-range-contact> <p_i> <p_j>
    <distance> <p_i> <p_j> <atom_i> <atom_j> <d_value>
    <short-range-contact> <p_i> <p_j>
    <plddt_80_85>
    <end>

See ``README.md`` and the `HF dataset page
<https://huggingface.co/datasets/timodonnell/protein-docs>`_ for the
full format spec.

The doc-generation algorithm is faithfully ported from the original
generator at
``timodonnell/contactdoc/contactdoc/generators/contacts_and_distances_v1.py``.
That implementation produced the ``timodonnell/protein-docs`` HF
dataset.

This file is intentionally one module — vocab, parsing, generation,
and evaluation are all here. We can split later if it grows past
~1500 lines, but co-location keeps the format definition reviewable
in one place.

NOTE: ``evaluate`` is still a ``NotImplementedError`` stub —
vllm-backed rollouts come in a subsequent commit (issue #1 task #23).
"""

import hashlib
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marinfold_document_structures import DocumentStructure, EvalResult


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
#
# Order is load-bearing. Token IDs derived from this list must stay
# stable for every checkpoint trained against the v1 vocab. Append-
# only — never reorder. (Reordering is a v2 event: new structure,
# new tokenizer, new experiment.)
#
# Source of truth ported from
# experiments/exp0_models_protein_docs_initial_port/create_protein_tokenizer.py
# and timodonnell/LlamaFold-experiments/.../exp6_contact_prediction/src/data.py.

CONTROL_TOKENS = [
    "<contacts-and-distances-v1>",
    "<begin_sequence>",
    "<begin_statements>",
    "<end>",
]

CONTACT_TYPES = [
    # CASP-standard separation ranges. Defined by CB-CB <= 8 Å.
    "<long-range-contact>",     # sequence separation >= 24
    "<medium-range-contact>",   # 12 .. 24
    "<short-range-contact>",    # 6 .. 12
]

DISTANCE_MARKER = ["<distance>"]

# 64 bins at 0.5 Å resolution: <d0.5>, <d1.0>, ..., <d32.0>.
DISTANCE_BINS = [f"<d{i * 0.5:.1f}>" for i in range(1, 65)]

PLDDT_BINS = [
    "<plddt_lt70>",
    "<plddt_70_75>",
    "<plddt_75_80>",
    "<plddt_80_85>",
    "<plddt_85_90>",
    "<plddt_90_95>",
    "<plddt_95_100>",
]

AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]

ATOM_NAMES = [
    # Order is load-bearing: matches
    # exp0_models_protein_docs_initial_port/create_protein_tokenizer.py
    # and LlamaFold-experiments/.../exp6_contact_prediction/src/data.py.
    # Reordering would change token IDs and break every existing v1
    # checkpoint. Backbone atoms (N, CA, C, O, OXT) are interleaved
    # alphabetically — do NOT group them.
    "C", "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3",
    "CG", "CG1", "CG2", "CH2", "CZ", "CZ2", "CZ3",
    "N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ",
    "O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
    "SD", "SG", "OXT",
]

# Position tokens <p0> through <p2700>. The upper bound caps the
# longest chain we tokenize; longer chains are dropped at generation
# time (PROBABLY shouldn't appear in training data anyway).
MAX_POSITION = 2700
POSITION_TOKENS = [f"<p{i}>" for i in range(MAX_POSITION + 1)]

UNK_TOKEN = ["<UNK>"]


# Backbone-set used by the standard residue → valid atoms lookup
# below. Used at generation time to discard atom records that the
# vocab doesn't have a token for.
_BACKBONE = {"N", "CA", "C", "O", "OXT"}

# Per-residue valid atom set. Atom records outside this set get
# dropped during doc generation (they're either nonstandard or
# alt-loc artifacts the format doesn't represent).
VALID_ATOMS: dict[str, set[str]] = {
    "ALA": _BACKBONE | {"CB"},
    "ARG": _BACKBONE | {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": _BACKBONE | {"CB", "CG", "OD1", "ND2"},
    "ASP": _BACKBONE | {"CB", "CG", "OD1", "OD2"},
    "CYS": _BACKBONE | {"CB", "SG"},
    "GLN": _BACKBONE | {"CB", "CG", "CD", "OE1", "NE2"},
    "GLU": _BACKBONE | {"CB", "CG", "CD", "OE1", "OE2"},
    "GLY": _BACKBONE,
    "HIS": _BACKBONE | {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": _BACKBONE | {"CB", "CG1", "CG2", "CD1"},
    "LEU": _BACKBONE | {"CB", "CG", "CD1", "CD2"},
    "LYS": _BACKBONE | {"CB", "CG", "CD", "CE", "NZ"},
    "MET": _BACKBONE | {"CB", "CG", "SD", "CE"},
    "PHE": _BACKBONE | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": _BACKBONE | {"CB", "CG", "CD"},
    "SER": _BACKBONE | {"CB", "OG"},
    "THR": _BACKBONE | {"CB", "OG1", "CG2"},
    "TRP": _BACKBONE | {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": _BACKBONE | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": _BACKBONE | {"CB", "CG1", "CG2"},
}


def _all_domain_tokens() -> list[str]:
    """Canonical 2838-token domain vocabulary in deterministic order.

    The published tokenizer prepends ``<pad>`` and ``<eos>`` (specials
    at ids 0 and 1), giving 2840 total. ``build_tokenizer(structure)``
    handles the prepend; this function returns the domain tokens
    only.

    The category-by-category order here (control → contact-types →
    distance-marker → distance-bins → plddt-bins → AAs → atoms →
    positions → UNK) and the within-category order are both
    load-bearing. Any reordering breaks compatibility with every
    checkpoint trained against the v1 vocab.
    """
    out: list[str] = []
    out += CONTROL_TOKENS
    out += CONTACT_TYPES
    out += DISTANCE_MARKER
    out += DISTANCE_BINS
    out += PLDDT_BINS
    out += [f"<{aa}>" for aa in AMINO_ACIDS]
    out += [f"<{atom}>" for atom in ATOM_NAMES]
    out += POSITION_TOKENS
    out += UNK_TOKEN
    return out


# --------------------------------------------------------------------------
# Structure parsing (gemmi-based; mmCIF for AFDB, also PDB)
# --------------------------------------------------------------------------

# Canonical 20 amino-acid set — non-canonical residues are mapped to UNK
# at parse time (matching contactdoc's `canonical_residue_policy =
# "map_to_unk"`).
_CANONICAL_20 = frozenset(AMINO_ACIDS)

# File extensions iter_inputs / iter_ground_truth recognize when given a
# directory. Gemmi auto-detects the format from extension via
# `read_structure(path)`.
_STRUCTURE_EXTS = frozenset({
    ".cif", ".cif.gz", ".mmcif", ".mmcif.gz",
    ".pdb", ".pdb.gz", ".ent", ".ent.gz",
})


@dataclass(frozen=True)
class Residue:
    """A single residue extracted from a polymer chain."""

    index: int  # 1-based residue position in the chain
    name: str   # 3-letter canonical name, or "UNK"
    plddt: float
    # (atom_name, x, y, z) tuples for all non-hydrogen atoms in this
    # residue. Atom names are filtered to ``ATOM_NAMES`` (vocab-safe);
    # atoms not in the v1 atom vocab are dropped so distance statements
    # can't emit out-of-vocab tokens.
    atoms: tuple[tuple[str, float, float, float], ...]


@dataclass(frozen=True)
class ParsedStructure:
    """One parsed polymer chain, ready for doc generation or evaluation.

    Records yielded by ``iter_inputs`` / ``iter_ground_truth``.
    """

    entry_id: str  # the file stem (e.g. "AF-P00767-F1-model_v4")
    residues: tuple[Residue, ...]
    source_path: Path  # for error messages / provenance

    @property
    def sequence(self) -> list[str]:
        return [r.name for r in self.residues]

    @property
    def global_plddt(self) -> float:
        """Mean pLDDT across all residues (the value the v1 generator bins)."""
        if not self.residues:
            return float("nan")
        return sum(r.plddt for r in self.residues) / len(self.residues)


def _vocab_safe_atoms(
    gemmi_residue, residue_name: str
) -> tuple[tuple[str, float, float, float], ...]:
    """Heavy atoms whose names are present in the v1 atom vocab.

    Drops hydrogens and any atom whose name is outside ``ATOM_NAMES``
    (e.g. non-canonical residue atoms, alt-loc artifacts). For UNK
    residues we still keep whatever standard-named atoms happen to be
    present.
    """
    import gemmi  # noqa: F401 — assert availability at call time

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


def parse_structure(
    path: Path,
    *,
    require_single_chain: bool = True,
) -> ParsedStructure:
    """Parse one structure file (mmCIF or PDB, optionally gzipped).

    Raises:
        ValueError: if the structure has no polymer chains, requires
            single-chain and has multiple, has no residues, or contains
            atoms that can't be loaded by gemmi.
    """
    import gemmi

    structure = gemmi.read_structure(str(path))
    # Populate entity / polymer metadata. AFDB mmCIFs ship with it
    # baked in but hand-rolled PDBs (and bare PDBs without TER records)
    # need this call before `chain.get_polymer()` returns anything.
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
            # match contactdoc's "map_to_unk" policy
            name = "UNK"
        residues.append(Residue(
            index=idx,
            name=name,
            plddt=_residue_plddt(res),
            atoms=_vocab_safe_atoms(res, name),
        ))
    if not residues:
        raise ValueError(f"{path}: no residues parsed")
    entry_id = structure.name or path.stem
    # gemmi may leave a .pdb / .cif extension on structure.name for some
    # readers — normalise.
    for ext in (".cif", ".mmcif", ".pdb", ".ent"):
        if entry_id.endswith(ext):
            entry_id = entry_id[: -len(ext)]
            break
    return ParsedStructure(
        entry_id=entry_id,
        residues=tuple(residues),
        source_path=path,
    )


def _iter_structure_paths(path: Path) -> Iterator[Path]:
    """Yield structure files under ``path`` (file or directory).

    Recursive; respects ``_STRUCTURE_EXTS``. Symlinks are not followed
    into the same directory tree more than once.
    """
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        # Check the full suffix (so .pdb.gz matches), then the single-
        # suffix form.
        name = p.name.lower()
        if any(name.endswith(ext) for ext in _STRUCTURE_EXTS):
            yield p


# --------------------------------------------------------------------------
# Document generation
# --------------------------------------------------------------------------
#
# Faithfully ports
# timodonnell/contactdoc/contactdoc/generators/contacts_and_distances_v1.py.
#
# One doc per parsed structure. Deterministic given the structure's
# entry_id (the seed for the per-doc rng is sha1(entry_id)). The `-5x`
# HF dataset subset takes up to 5 AFDB entries per Foldseek structural
# cluster — that's a cluster-selection concern handled by the data
# pipeline that wraps generate_documents, not by the structure itself.


CONTACT_TOKENS_PER_STATEMENT = 3      # <mode> <p_i> <p_j>
DISTANCE_TOKENS_PER_STATEMENT = 6     # <distance> <p_i> <p_j> <atom_i> <atom_j> <d_value>


@dataclass(frozen=True)
class GenerationConfig:
    """Hyperparameters for ``generate_documents``.

    Defaults match the contactdoc ``ContactsAndDistancesV1Config`` that
    produced the published HF dataset. Changing any of these produces
    a different distribution of documents — bump the structure to v2
    rather than tweaking these silently.
    """

    contact_cutoff_angstrom: float = 8.0
    long_range_sep: int = 24
    medium_range_sep: int = 12
    short_range_sep: int = 6
    # Range over which each per-mode fraction is sampled. The lower
    # bound is intentionally negative so a fraction of zero is
    # ~33% likely (clamps to 0 before scaling). Matches contactdoc.
    contact_f_range: tuple[float, float] = (-0.1, 0.2)
    contact_rank_mean: float = 2.0
    distance_rank_mean: float = 0.0
    rank_std: float = 1.0
    # Residues below this pLDDT are excluded from contact eligibility
    # but still appear in the sequence and in distance statements.
    residue_plddt_min: float = 70.0
    # Right-open bin edges; 100 is the implicit ceiling.
    plddt_bin_edges: tuple[float, ...] = (70.0, 75.0, 80.0, 85.0, 90.0, 95.0)


def _plddt_bin_token(global_plddt: float, bin_edges: tuple[float, ...]) -> str:
    """Map a per-structure mean pLDDT to the matching ``plddt_*`` token.

    Replicates the contactdoc helper exactly: the first edge whose
    value exceeds ``global_plddt`` defines the bin; values above the
    last edge fall into ``plddt_<last>_100``.
    """
    for i, edge in enumerate(bin_edges):
        if global_plddt < edge:
            if i == 0:
                return f"plddt_lt{int(edge)}"
            return f"plddt_{int(bin_edges[i-1])}_{int(edge)}"
    return f"plddt_{int(bin_edges[-1])}_100"


def _distance_token(dist_angstrom: float) -> str:
    """Map a distance to a ``<d_X.X>`` bin token.

    64 bins at 0.5 Å resolution. ``<d0.5>`` covers ``[0, 0.5]``,
    ``<d1.0>`` covers ``(0.5, 1.0]``, …, ``<d32.0>`` covers
    ``(31.5, 32.0]``. Distances above 32.0 Å clamp to the top bin.
    Matches contactdoc's ``_distance_token``.
    """
    bin_idx = int(math.ceil(dist_angstrom / 0.5))
    if bin_idx < 1:
        bin_idx = 1
    if bin_idx > 64:
        bin_idx = 64
    return f"d{bin_idx * 0.5:.1f}"


def _cb_or_ca_position(
    residue: Residue,
) -> tuple[float, float, float] | None:
    """Return the CB position for ``residue`` (or CA for GLY / missing CB).

    Matches contactdoc's ``_get_cb_or_ca``. Returns ``None`` if neither
    CB nor CA is present (e.g. a residue with only sidechain atoms).
    """
    target = "CA" if residue.name == "GLY" else "CB"
    fallback_ca: tuple[float, float, float] | None = None
    for name, x, y, z in residue.atoms:
        if name == target:
            return (x, y, z)
        if name == "CA":
            fallback_ca = (x, y, z)
    return fallback_ca


def _euclidean(p1, p2) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(frozen=True)
class _Statement:
    rank: float
    tokens: tuple[str, ...]


def _generate_one(
    structure: ParsedStructure,
    *,
    context_length: int,
    cfg: GenerationConfig,
    structure_name: str,
) -> str | None:
    """Build a single document string for ``structure``, or None if it doesn't fit.

    The output omits the trailing newline; callers that want jsonl
    semantics should add it themselves. ``None`` is returned when the
    sequence alone exceeds the token budget (matches the
    ``sequence_too_long_for_budget`` failure mode in contactdoc).
    """
    residues = structure.residues
    num_residues = len(residues)
    if num_residues < 2:
        return None

    # Deterministic seed per entry — keeps generation reproducible
    # across re-runs and lets a downstream pipeline pick "the N-th
    # variant" by varying the entry_id suffix if it ever wants to.
    seed = int(hashlib.sha1(structure.entry_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # --- Contact eligibility (pLDDT-filtered CB/CA positions) ---
    cb_positions: dict[int, tuple[float, float, float]] = {}
    for r in residues:
        if r.plddt < cfg.residue_plddt_min:
            continue
        pos = _cb_or_ca_position(r)
        if pos is not None:
            cb_positions[r.index] = pos

    contacts_long: list[tuple[int, int]] = []
    contacts_medium: list[tuple[int, int]] = []
    contacts_short: list[tuple[int, int]] = []

    cb_indices = sorted(cb_positions)
    for ii in range(len(cb_indices)):
        for jj in range(ii + 1, len(cb_indices)):
            i = cb_indices[ii]
            j = cb_indices[jj]
            sep = j - i
            if sep <= 1 or sep < cfg.short_range_sep:
                continue
            d = _euclidean(cb_positions[i], cb_positions[j])
            if d > cfg.contact_cutoff_angstrom:
                continue
            if sep >= cfg.long_range_sep:
                contacts_long.append((i, j))
            elif sep >= cfg.medium_range_sep:
                contacts_medium.append((i, j))
            else:
                contacts_short.append((i, j))

    # --- Budgeting (overhead per the v1 layout) ---
    # 5 = <task> + <begin_sequence> + <begin_statements> + <end> + <plddt_bin>
    fixed_overhead = 5 + num_residues
    available_tokens = context_length - fixed_overhead
    if available_tokens <= 0:
        return None

    f_long = max(0.0, rng.uniform(*cfg.contact_f_range))
    f_medium = max(0.0, rng.uniform(*cfg.contact_f_range))
    f_short = max(0.0, rng.uniform(*cfg.contact_f_range))

    tokens_long = int(available_tokens * f_long)
    tokens_medium = int(available_tokens * f_medium)
    tokens_short = int(available_tokens * f_short)

    n_long = min(tokens_long // CONTACT_TOKENS_PER_STATEMENT, len(contacts_long))
    n_medium = min(tokens_medium // CONTACT_TOKENS_PER_STATEMENT, len(contacts_medium))
    n_short = min(tokens_short // CONTACT_TOKENS_PER_STATEMENT, len(contacts_short))

    contact_tokens_used = (n_long + n_medium + n_short) * CONTACT_TOKENS_PER_STATEMENT
    remaining = available_tokens - contact_tokens_used
    n_distance = remaining // DISTANCE_TOKENS_PER_STATEMENT

    # --- Sample which specific pairs to emit ---
    selected_long = rng.sample(contacts_long, n_long) if n_long > 0 else []
    selected_medium = rng.sample(contacts_medium, n_medium) if n_medium > 0 else []
    selected_short = rng.sample(contacts_short, n_short) if n_short > 0 else []

    # --- Distance statements: uniformly sampled residue pairs (|i-j|>1),
    # uniformly sampled atoms within each. Residues with zero in-vocab
    # heavy atoms are skipped.
    distance_indices = [r.index for r in residues if r.atoms]
    distance_atoms = {r.index: r.atoms for r in residues if r.atoms}

    distance_statements: list[tuple[int, int, str, str, str]] = []
    if len(distance_indices) >= 2:
        for _ in range(n_distance):
            a, b = rng.sample(distance_indices, 2)
            i, j = (a, b) if a < b else (b, a)
            if j - i <= 1:
                # Retry once (contactdoc retries up to 10x).
                ok = False
                for _retry in range(10):
                    a, b = rng.sample(distance_indices, 2)
                    i, j = (a, b) if a < b else (b, a)
                    if j - i > 1:
                        ok = True
                        break
                if not ok:
                    continue
            ai_name, ax, ay, az = rng.choice(distance_atoms[i])
            aj_name, bx, by, bz = rng.choice(distance_atoms[j])
            d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
            distance_statements.append((i, j, ai_name, aj_name, _distance_token(d)))

    # --- Rank-order statements (contacts skew earlier; matches contactdoc) ---
    statements: list[_Statement] = []
    for i, j in selected_long:
        statements.append(_Statement(
            rank=rng.gauss(cfg.contact_rank_mean, cfg.rank_std),
            tokens=("<long-range-contact>", f"<p{i}>", f"<p{j}>"),
        ))
    for i, j in selected_medium:
        statements.append(_Statement(
            rank=rng.gauss(cfg.contact_rank_mean, cfg.rank_std),
            tokens=("<medium-range-contact>", f"<p{i}>", f"<p{j}>"),
        ))
    for i, j in selected_short:
        statements.append(_Statement(
            rank=rng.gauss(cfg.contact_rank_mean, cfg.rank_std),
            tokens=("<short-range-contact>", f"<p{i}>", f"<p{j}>"),
        ))
    for i, j, ai, aj, dist_tok in distance_statements:
        statements.append(_Statement(
            rank=rng.gauss(cfg.distance_rank_mean, cfg.rank_std),
            tokens=("<distance>", f"<p{i}>", f"<p{j}>", f"<{ai}>", f"<{aj}>", f"<{dist_tok}>"),
        ))
    statements.sort(key=lambda s: -s.rank)

    # --- pLDDT bin token: 50/50 between mid-statement insertion and at-end ---
    plddt_token = _plddt_bin_token(structure.global_plddt, cfg.plddt_bin_edges)
    plddt_at_end = rng.random() < 0.5
    plddt_insert_idx = None if plddt_at_end else rng.randint(0, len(statements))

    # --- Serialize ---
    out: list[str] = []
    out.append(f"<{structure_name}>")
    out.append("<begin_sequence>")
    for r in residues:
        out.append(f"<{r.name}>")
    out.append("<begin_statements>")
    for idx, stmt in enumerate(statements):
        if plddt_insert_idx is not None and idx == plddt_insert_idx:
            out.append(f"<{plddt_token}>")
        out.extend(stmt.tokens)
    if plddt_insert_idx is not None and plddt_insert_idx >= len(statements):
        out.append(f"<{plddt_token}>")
    if plddt_at_end:
        out.append(f"<{plddt_token}>")
    out.append("<end>")
    return " ".join(out)


# --------------------------------------------------------------------------
# DocumentStructure implementation
# --------------------------------------------------------------------------


class ContactsAndDistancesV1:
    """The contacts-and-distances-v1 document structure.

    Stateless apart from the cached token list and the generation
    config — safe to construct eagerly via :func:`get_structure`.
    """

    name = "contacts-and-distances-v1"
    context_length = 8192

    def __init__(self, generation_config: GenerationConfig | None = None) -> None:
        self._tokens = _all_domain_tokens()
        self.generation_config = generation_config or GenerationConfig()

    def tokens(self) -> list[str]:
        # Return a copy so callers can't accidentally mutate the
        # canonical list (which would corrupt every subsequent
        # build_tokenizer call).
        return list(self._tokens)

    # ---- generate side --------------------------------------------------

    def iter_inputs(self, path: Path) -> Iterator[ParsedStructure]:
        """Parse PDB / mmCIF (optionally gzipped) files at ``path``.

        ``path`` may be a single file or a directory walked recursively.
        Yields one ``ParsedStructure`` per file. Files that fail to
        parse (no residues, multi-chain in single-chain mode, etc.)
        are skipped with a warning rather than aborting the whole
        iteration.
        """
        import warnings
        for p in _iter_structure_paths(Path(path)):
            try:
                yield parse_structure(p)
            except ValueError as exc:
                warnings.warn(f"skipping {p}: {exc}", stacklevel=2)
                continue

    def generate_documents(
        self,
        input_records: Iterator[ParsedStructure],
        *,
        context_length: int | None = None,
        num_docs: int | None = None,
    ) -> Iterator[str]:
        """Emit one document per input structure.

        Skips inputs that don't fit the token budget (sequence alone
        exceeds ``context_length``) — those are emitted as no-op
        rather than raising, matching contactdoc's
        ``sequence_too_long_for_budget`` failure mode.
        """
        ctx = context_length if context_length is not None else self.context_length
        produced = 0
        for record in input_records:
            doc = _generate_one(
                record,
                context_length=ctx,
                cfg=self.generation_config,
                structure_name=self.name,
            )
            if doc is None:
                continue
            yield doc
            produced += 1
            if num_docs is not None and produced >= num_docs:
                return

    # ---- evaluate side --------------------------------------------------

    def iter_ground_truth(self, path: Path) -> Iterator[ParsedStructure]:
        """Same parser as ``iter_inputs`` — eval-time records are the same shape."""
        return self.iter_inputs(path)

    def evaluate(
        self,
        *,
        model_path: str,
        ground_truth_records: Iterator[Any],
    ) -> EvalResult:
        raise NotImplementedError(
            "evaluate is not yet implemented — coming in the next commit "
            "(vllm-backed rollout eval)."
        )


def get_structure() -> DocumentStructure:
    """Entry point read by the marinfold-document-structure CLI."""
    return ContactsAndDistancesV1()
