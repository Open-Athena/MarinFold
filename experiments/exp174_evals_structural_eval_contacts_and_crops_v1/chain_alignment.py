# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-chain extraction + observed-to-input-sequence alignment.

Copied verbatim (docstrings included) from
``exp89_evals_contacts_v1_model_on_eval_set/pyconfind_contacts.py``, which in
turn copied it from exp78. Both of those modules carry the full
contact-scoring stack; this experiment needs only the two structure-handling
helpers, so it takes those and nothing else.

Per the repo's kind-library rule these helpers have now earned promotion
(three experiments use them), but that is a separate refactor — an ``evals``
kind library does not exist yet, and creating one is not this issue's job.
"""

import difflib

import gemmi

# Canonical 3-letter -> 1-letter. contacts-v1's parse layer already
# canonicalizes residue names (HIS variants, MSE->MET, modified->parent, else
# "UNK"), so we only map the standard 20; anything else (incl. "UNK") -> "X".
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def one_letter(canonical_resname: str) -> str:
    """One-letter code of a canonical residue name; ``X`` for anything else."""
    return _THREE_TO_ONE.get(canonical_resname, "X")


def extract_single_chain(
    structure, *, prefer_chain: str | None = None
) -> tuple[gemmi.Structure, str]:
    """Return a single-protein-chain ``gemmi.Structure`` + the chosen chain id.

    pyconfind / contacts_v1 only handle one protein chain. Ground-truth
    biological-assembly mmCIFs can hold several (homo-oligomer copies or
    other entities). We keep the first model, pick the requested chain if
    present and polymeric, else the longest polymer-peptide chain, drop
    everything else, and strip ligands / waters. Intra-chain contacts of the
    kept copy are exactly the single-chain problem we want.
    """
    st = (
        structure.clone()
        if isinstance(structure, gemmi.Structure)
        else gemmi.read_structure(str(structure))
    )
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

    ``obs`` is the one-letter sequence of resolved residues (in chain order);
    ``ref`` is the input sequence. Returns a list the length of ``obs`` whose
    ``k``-th entry is the ``ref`` index that ``obs[k]`` aligns to, or ``None``
    if it aligns to no reference position. Uses difflib opcodes (so it makes
    no assumption about residue numbering), with an identity fast-path for the
    common fully-resolved case.
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
            # map the positional overlap so the residue still lands somewhere
            # sane; surplus obs positions stay None.
            for k in range(min(i2 - i1, j2 - j1)):
                mapping[i1 + k] = j1 + k
        # 'delete' (obs-only, no ref) -> None; 'insert' (ref-only,
        # unresolved in the structure) -> nothing to map.
    return mapping
