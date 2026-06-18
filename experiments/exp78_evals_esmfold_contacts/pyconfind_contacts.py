# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth + predicted contacts via pyconfind, indexed to the model's sequence.

This is the contact-prediction analogue of exp12's distance scoring. For
issue #74 we define a "contact" exactly the way the ``contacts_v1``
document-generation pipeline does — pyconfind side-chain contact degree,
run in ``native_only=True`` mode with the C++ confind geometry defaults
(``contact_distance=3.0``, ``dcut=25.0``, ``clash_distance=2.0``) — and a
"true" contact as one with degree ``>= 0.001`` (``GenerationConfig``'s
``min_contact_degree``) and primary-sequence separation ``>= 6``.

The single non-trivial bit is **index alignment**. pyconfind numbers its
contacts by 0-based position in the *resolved* residue list of the chain
it analyzed. The Protenix distogram (and the predicted structure) are
indexed by the *input sequence* fed to the model. A ground-truth crystal
structure can have unresolved residues, so its resolved-residue indexing
is a (gappy) subsequence of the input-sequence indexing. We align the two
with :func:`align_obs_to_ref` (difflib, decoupled from any
label_seq/numbering assumption) and remap every contact into
input-sequence coordinates, so ground-truth contacts, the distogram, and
the predicted-structure contacts all live in one ``[0, L)`` space.

Single entry point: :func:`compute_contacts`. The pyconfind run is also
where the "save all contacts" deliverable comes from — :class:`ContactResult`
carries every degree>0 contact (no degree/separation filter); thresholding
happens at eval time in ``contact_eval.py``.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

import gemmi

from marinfold.document_structures.contacts_v1 import analyze_structure

# pyconfind geometry knobs — MUST match
# marinfold.document_structures.contacts_v1.GenerationConfig defaults
# (verified: native_only=True, contact_distance=3.0, dcut=25.0,
# clash_distance=2.0, assembly=None). The issue's "we used a cd threshold
# of 0.001" is GenerationConfig.min_contact_degree, applied at eval time
# (see contact_eval.py), not here — we keep *all* contacts pyconfind emits.
PYCONFIND_KWARGS = dict(
    native_only=True,
    contact_distance=3.0,
    dcut=25.0,
    clash_distance=2.0,
    assembly=None,
)

# Canonical 3-letter -> 1-letter. analyze_structure already canonicalizes
# residue names (HIS variants, MSE->MET, modified->parent, else "UNK"), so
# we only map the standard 20; anything else (incl. "UNK") -> "X".
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _one_letter(canonical_resname: str) -> str:
    return _THREE_TO_ONE.get(canonical_resname, "X")


@dataclass(frozen=True)
class ContactResult:
    """All pyconfind contacts for one chain, remapped to input-seq coords.

    Attributes:
        stem: protein identifier.
        chain: author chain id that was analyzed.
        n_input_residues: ``L`` = length of the input sequence (the
            distogram / predicted-structure index space).
        n_resolved_residues: residues pyconfind actually placed (chain
            length seen in the structure).
        n_mapped_residues: resolved residues that aligned to an input-seq
            position (= the eligible-residue set for contact scoring).
        alignment_identity: fraction of resolved residues whose amino acid
            matches the input sequence at the aligned position. ~1.0 means
            a clean alignment; a low value flags a wrong chain / sequence
            mismatch worth investigating.
        resolved_positions: sorted tuple of input-seq indices that are
            resolved in this structure (the candidate-pair universe).
        contacts: every contact with degree > 0, as ``(i, j, degree)`` with
            ``i < j`` in input-seq coordinates. No degree / separation
            filter — that is applied downstream.
    """

    stem: str
    chain: str
    n_input_residues: int
    n_resolved_residues: int
    n_mapped_residues: int
    alignment_identity: float
    resolved_positions: tuple[int, ...]
    contacts: tuple[tuple[int, int, float], ...]


def extract_single_chain(
    structure, *, prefer_chain: str | None = None
) -> tuple[gemmi.Structure, str]:
    """Return a single-protein-chain ``gemmi.Structure`` + the chosen chain id.

    pyconfind / contacts_v1 only handle one protein chain. Ground-truth
    biological-assembly mmCIFs can hold several (homo-oligomer copies or
    other entities); predicted Protenix CIFs are monomers. We keep the
    first model, pick the requested chain if present and polymeric, else
    the longest polymer-peptide chain, drop everything else, and strip
    ligands / waters. Intra-chain contacts of the kept copy are exactly
    the single-chain contact problem we want.
    """
    st = structure.clone() if isinstance(structure, gemmi.Structure) else gemmi.read_structure(str(structure))
    st.setup_entities()
    while len(st) > 1:
        del st[1]
    model = st[0]

    def pep_len(chain: gemmi.Chain) -> int:
        poly = chain.get_polymer()
        try:
            return len(poly)
        except Exception:  # noqa: BLE001
            return 0

    candidates = [(c.name, pep_len(c)) for c in model]
    candidates = [(name, n) for name, n in candidates if n > 0]
    if not candidates:
        raise ValueError("no polymer (peptide) chain found in structure")

    chosen: str | None = None
    if prefer_chain is not None:
        for name, _ in candidates:
            if name == prefer_chain:
                chosen = name
                break
    if chosen is None:
        chosen = max(candidates, key=lambda t: t[1])[0]

    for name in [c.name for c in list(model)]:
        if name != chosen:
            model.remove_chain(name)
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    return st, chosen


def align_obs_to_ref(obs: str, ref: str) -> list[int | None]:
    """Map each observed-residue index to a reference (input-seq) index.

    ``obs`` is the one-letter sequence of resolved residues (in chain
    order); ``ref`` is the input sequence. Returns a list the length of
    ``obs`` whose ``k``-th entry is the ``ref`` index that ``obs[k]``
    aligns to, or ``None`` if it aligns to no reference position. Uses
    difflib opcodes (so it makes no assumption about residue numbering),
    with an identity fast-path for the common fully-resolved case.
    """
    if obs == ref:
        return list(range(len(obs)))
    sm = difflib.SequenceMatcher(a=obs, b=ref, autojunk=False)
    mapping: list[int | None] = [None] * len(obs)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
        elif tag == "replace":
            # Mismatched run (e.g. a modified residue rendered differently):
            # map the positional overlap so the contact still lands somewhere
            # sane; surplus obs positions stay None.
            for k in range(min(i2 - i1, j2 - j1)):
                mapping[i1 + k] = j1 + k
        # 'delete' (obs-only, no ref) -> None; 'insert' (ref-only,
        # unresolved in the structure) -> nothing to map.
    return mapping


def compute_contacts(
    structure,
    input_seq: str,
    *,
    stem: str,
    prefer_chain: str | None = None,
) -> ContactResult:
    """Run pyconfind on one structure and return contacts in input-seq coords.

    ``structure`` is a path or ``gemmi.Structure``; ``input_seq`` is the
    exact sequence handed to Protenix (the distogram index space).
    """
    st, chain = extract_single_chain(structure, prefer_chain=prefer_chain)
    analyzed = analyze_structure(st, entry_id=stem, **PYCONFIND_KWARGS)

    obs = "".join(_one_letter(r.resname) for r in analyzed.residues)
    mapping = align_obs_to_ref(obs, input_seq)

    matched = sum(
        1 for k, c in enumerate(mapping)
        if c is not None and obs[k] == input_seq[c]
    )
    identity = matched / len(obs) if obs else 0.0
    resolved = sorted({c for c in mapping if c is not None})

    contacts: list[tuple[int, int, float]] = []
    for c in analyzed.contacts:
        ci = mapping[c.seq_i] if c.seq_i < len(mapping) else None
        cj = mapping[c.seq_j] if c.seq_j < len(mapping) else None
        if ci is None or cj is None or ci == cj:
            continue
        lo, hi = (ci, cj) if ci < cj else (cj, ci)
        contacts.append((lo, hi, float(c.degree)))
    contacts.sort()

    return ContactResult(
        stem=stem,
        chain=chain,
        n_input_residues=len(input_seq),
        n_resolved_residues=len(analyzed.residues),
        n_mapped_residues=sum(1 for c in mapping if c is not None),
        alignment_identity=identity,
        resolved_positions=tuple(resolved),
        contacts=tuple(contacts),
    )
