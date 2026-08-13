# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Protenix/AF3-style curation of a PDB entry into contacts-v1 inputs.

The library half of stage 1. ``curate_and_generate.py`` drives it over the
whole mirror; the filters live here so they can be unit-tested and read
without the multiprocessing scaffolding around them.

Two structures come out of one entry, and they are deliberately different:

* the **asymmetric unit**, from which each protein chain is pulled out and
  analyzed *on its own* -- that is what a monomer document describes, and it
  matches how the AFDB training corpus and the exp74/exp89 eval ground truth
  are built (an isolated chain, ``assembly=None``);
* **biological assembly 1**, kept whole -- that is what a multimer document
  describes, and its interfaces are the entire reason the subset exists.

Filters follow Protenix's ``prepare_training_data.md``, which follows the
AF3 supplement:

======================================  =========================================
Waters / hydrogens / element X          removed
Non-protein molecules                   removed (contacts-v1 has no vocabulary
                                        for ligands, nucleic acids or glycans)
Chains of entirely unknown residues     removed
Chains with no resolved residues        removed
Adjacent-numbered CA-CA > 10 A          chain removed
>= 1/3 of heavy atoms clashing (<1.7 A) chain removed
======================================  =========================================

Entry-level filters (release date, resolution) are applied upstream from the
header scan -- they need no coordinates.

Every rejection returns a named reason. Nothing is dropped silently: the
driver tallies the reasons into a ledger, so the funnel from 195,858 entries
to the final corpora is auditable line by line.
"""

from dataclasses import dataclass, field

import gemmi


# AF3 / Protenix numbers, all in Angstrom.
CLASH_DISTANCE = 1.7
CLASH_FRACTION = 1.0 / 3.0
MAX_ADJACENT_CA_DISTANCE = 10.0

# gemmi's polymer type for an L-peptide chain; the only thing contacts-v1 can
# serialize. Derived by gemmi from ``_entity_poly.type``, so it agrees with
# the header scan's PROTEIN_ENTITY_TYPES.
PROTEIN_POLYMER_TYPE = gemmi.PolymerType.PeptideL

# Residue names contacts-v1 understands, mirroring
# ``marinfold...contacts_v1.parse._RESNAME_TO_CANONICAL``. A chain none of
# whose residues are in here carries no information the format can express.
_KNOWN_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HSD", "HSE", "HSC", "HSP", "HIP", "MSE", "CSO", "SEC", "SEP", "TPO",
    "PTR",
})


@dataclass(frozen=True)
class CuratedChain:
    """One protein chain that survived curation, plus its provenance."""

    chain_id: str          # chain name in the structure being analyzed
    asu_chain_id: str      # author chain id in the asymmetric unit
    entity_id: str         # mmCIF ``_entity.id``, for cluster lookup
    n_residues: int

    def entity_key(self, pdb_id: str) -> str:
        """RCSB cluster key, ``<PDBID>_<entity>`` uppercased."""
        return f"{pdb_id.upper()}_{self.entity_id}"


@dataclass
class ChainLedger:
    """Why each chain was kept or dropped, tallied per entry."""

    kept: list[CuratedChain] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)

    def drop(self, chain_id: str, reason: str) -> None:
        self.dropped[chain_id] = reason


def clean_structure(path: str) -> gemmi.Structure:
    """Read an mmCIF and strip everything contacts-v1 cannot represent.

    Removes alternative conformations (keeping the first), hydrogens, waters
    and every non-polymer molecule, then any chain left empty. Atoms whose
    element is unknown (mmCIF element ``X``) are dropped too -- Protenix
    removes these explicitly, and pyconfind would otherwise try to place a
    rotamer around an atom with no chemistry.

    Non-protein *polymers* (DNA, RNA, D-peptides) survive this call and are
    filtered by entity type in :func:`protein_subchains`, which needs the
    entity table this function leaves intact.
    """
    structure = gemmi.read_structure(path)
    structure.setup_entities()
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    structure.remove_ligands_and_waters()
    _remove_unknown_elements(structure)
    structure.remove_empty_chains()
    return structure


def _remove_unknown_elements(structure: gemmi.Structure) -> None:
    unknown = gemmi.Element("X")
    for model in structure:
        for chain in model:
            for residue in chain:
                atoms = [a for a in residue if a.element != unknown]
                if len(atoms) != len(residue):
                    del residue[:]
                    for atom in atoms:
                        residue.add_atom(atom)


def protein_subchains(structure: gemmi.Structure) -> dict[str, str]:
    """Map ``label_asym_id`` -> ``entity_id`` for every L-peptide subchain."""
    out: dict[str, str] = {}
    for entity in structure.entities:
        if entity.polymer_type != PROTEIN_POLYMER_TYPE:
            continue
        for subchain in entity.subchains:
            out[subchain] = entity.name
    return out


def _chain_protein_residues(
    chain: gemmi.Chain, protein_subchain_ids: dict[str, str]
) -> tuple[list[gemmi.Residue], str | None]:
    """The chain's protein residues, and the entity they belong to."""
    residues = [r for r in chain if r.subchain in protein_subchain_ids]
    if not residues:
        return [], None
    entity_ids = {protein_subchain_ids[r.subchain] for r in residues}
    # A chain spanning two protein entities is a chimera we have no sensible
    # entity id for; take the first by subchain order and let the caller keep
    # it -- the entity id only drives cluster lookup, not correctness.
    entity_id = sorted(entity_ids)[0]
    return residues, entity_id


def _has_adjacent_ca_break(residues: list[gemmi.Residue]) -> bool:
    """True if two *consecutively numbered* residues have CA atoms > 10 A apart.

    AF3's filter. Only consecutive author numbering counts: a numbering gap
    means the intervening residues were unresolved, and the resulting jump in
    space is expected rather than a modelling error.
    """
    previous_num: int | None = None
    previous_ca: gemmi.Position | None = None
    for residue in residues:
        ca = residue.find_atom("CA", "*")
        num = residue.seqid.num
        if ca is not None and previous_ca is not None and previous_num == num - 1:
            if previous_ca.dist(ca.pos) > MAX_ADJACENT_CA_DISTANCE:
                return True
        if ca is not None:
            previous_ca = ca.pos
            previous_num = num
        else:
            previous_ca = None
            previous_num = None
    return False


def clashing_atom_counts(
    structure: gemmi.Structure, model_index: int = 0
) -> dict[str, int]:
    """Per chain, how many of its atoms sit within 1.7 A of another chain's.

    Uses gemmi's C++ contact search restricted to different chains, and
    ignores crystal-symmetry images (``image_idx != 0``) -- a neighbour in the
    next unit cell is not a modelling error, and for the assembly pass the
    symmetry copies we care about are already explicit chains.
    """
    search = gemmi.NeighborSearch(structure, 5.0, model_index).populate()
    contacts = gemmi.ContactSearch(CLASH_DISTANCE)
    contacts.ignore = gemmi.ContactSearch.Ignore.SameChain
    # An atom can clash with several partners; the filter is about how much of
    # a chain is buried in another, so each atom counts once. Identify it by
    # its coordinates in the structure rather than by object identity -- gemmi
    # hands back short-lived proxy objects, so ``id()`` is not a stable key.
    counts: dict[str, set[tuple[str, int, str, str, str]]] = {}
    for result in contacts.find_contacts(search):
        if result.image_idx != 0:
            continue
        for partner in (result.partner1, result.partner2):
            key = (
                partner.chain.name,
                partner.residue.seqid.num,
                partner.residue.seqid.icode,
                partner.residue.name,
                partner.atom.name,
            )
            counts.setdefault(partner.chain.name, set()).add(key)
    return {name: len(atoms) for name, atoms in counts.items()}


def _heavy_atom_count(residues: list[gemmi.Residue]) -> int:
    return sum(len(r) for r in residues)


def curate_chains(
    structure: gemmi.Structure,
    protein_subchain_ids: dict[str, str],
    *,
    min_residues: int = 2,
    max_residues: int = 2000,
) -> ChainLedger:
    """Apply every chain-level filter to one already-cleaned structure.

    ``structure`` is either the asymmetric unit or a built assembly; the
    filters are identical, but the clash test is inherently contextual and so
    sees whichever neighbours that structure actually contains.

    ``protein_subchain_ids`` maps ``label_asym_id`` to ``entity_id`` and is
    passed in rather than read off ``structure.entities``: a built assembly's
    entity table is regenerated from scratch by gemmi and renumbered from 1
    in chain order, which would silently mis-key the RCSB cluster lookup
    whenever an entry's protein entities are not 1..n (i.e. whenever a ligand
    entity comes first). See :func:`assembly_subchain_entities`.
    """
    ledger = ChainLedger()
    clashes = clashing_atom_counts(structure)
    model = structure[0]

    for chain in model:
        residues, entity_id = _chain_protein_residues(chain, protein_subchain_ids)
        if not residues:
            ledger.drop(chain.name, "not_protein")
            continue
        if all(r.name.strip().upper() not in _KNOWN_RESIDUES for r in residues):
            ledger.drop(chain.name, "all_unknown_residues")
            continue
        resolved = [r for r in residues if len(r) > 0]
        if not resolved:
            ledger.drop(chain.name, "no_resolved_residues")
            continue
        if len(resolved) < min_residues:
            ledger.drop(chain.name, "too_short")
            continue
        if len(resolved) > max_residues:
            ledger.drop(chain.name, "too_long")
            continue
        if _has_adjacent_ca_break(resolved):
            ledger.drop(chain.name, "ca_break")
            continue
        heavy = _heavy_atom_count(resolved)
        if heavy and clashes.get(chain.name, 0) >= CLASH_FRACTION * heavy:
            ledger.drop(chain.name, "clashing")
            continue
        ledger.kept.append(CuratedChain(
            chain_id=chain.name,
            asu_chain_id=chain.name,
            entity_id=entity_id or "",
            n_residues=len(resolved),
        ))
    return ledger


def assembly_subchain_entities(
    asu_protein_subchains: dict[str, str], assembly: gemmi.Structure
) -> dict[str, str]:
    """Map a built assembly's subchains back to the ASU's entity ids.

    ``gemmi.make_assembly(..., HowToNameCopiedChain.AddNumber)`` names each
    copy ``<original><copy-number>``, so this inverts that: for every subchain
    in the assembly, find the longest ASU protein subchain that is a prefix
    with an all-digit remainder. Copies of non-protein subchains simply find
    no match and are left out, which is what filters them from the assembly.

    The longest-prefix rule matters for entries with two-character asym ids
    (``A`` and ``A1`` can both exist), where a plain "strip trailing digits"
    would pick the wrong entity.
    """
    out: dict[str, str] = {}
    candidates = sorted(asu_protein_subchains, key=len, reverse=True)
    seen: set[str] = set()
    for model in assembly:
        for chain in model:
            for residue in chain:
                name = residue.subchain
                if name in seen:
                    continue
                seen.add(name)
                for original in candidates:
                    if name.startswith(original) and name[len(original):].isdigit():
                        out[name] = asu_protein_subchains[original]
                        break
    return out


def single_chain_structure(
    structure: gemmi.Structure, chain_id: str
) -> gemmi.Structure:
    """A copy of ``structure`` holding only ``chain_id``.

    This is what makes a monomer document a *monomer*: the chain is analyzed
    with no partner present, so its surface is genuinely exposed, exactly as
    it is for an AFDB prediction or for the eval set's ground truth. Analyzing
    the chain in place would instead bury the interface and change which
    residues pyconfind calls solvent-accessible.
    """
    copy = structure.clone()
    for model in copy:
        for chain in list(model):
            if chain.name != chain_id:
                model.remove_chain(chain.name)
    copy.setup_entities()
    return copy


def build_assembly(
    structure: gemmi.Structure, assembly_id: str = "1"
) -> gemmi.Structure | None:
    """Expand ``structure`` to a biological assembly, as its own Structure.

    Returns ``None`` when the entry declares no such assembly (a handful of
    old entries have no ``_pdbx_struct_assembly`` at all), which the caller
    records as a named rejection rather than silently falling back to the
    asymmetric unit -- the ASU of a crystal is not a biological complex.
    """
    assembly = next((a for a in structure.assemblies if a.name == assembly_id), None)
    if assembly is None:
        return None
    model = gemmi.make_assembly(
        assembly, structure[0], gemmi.HowToNameCopiedChain.AddNumber
    )
    built = gemmi.Structure()
    built.name = structure.name
    built.cell = structure.cell
    built.spacegroup_hm = structure.spacegroup_hm
    built.add_model(model)
    built.setup_entities()
    return built


def load_clusters(path: str) -> dict[str, int]:
    """Read RCSB ``clusters-by-entity-40.txt`` into ``<PDBID>_<entity> -> id``.

    One line per cluster, space-separated members. The integer is the line
    number, i.e. an arbitrary but stable cluster id in the file's own order.
    """
    clusters: dict[str, int] = {}
    with open(path) as handle:
        for cluster_id, line in enumerate(handle):
            for member in line.split():
                clusters[member.strip().upper()] = cluster_id
    return clusters
