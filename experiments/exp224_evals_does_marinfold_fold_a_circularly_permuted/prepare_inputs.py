#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step A — build the exp224 eval units: sequences, pyconfind ground truth, CP<->WT map.

Four units, all of them *E. coli* DsbA (UniProt ``P0AEG4``, mature chain =
residues 20-208 of the precursor, renumbered 1-189 here):

* ``cp_1un2``  — the circular permutant CPDsbA-Q100T99, PDB ``1UN2`` (2.4 A).
* ``wt_1fvk``  — wild-type oxidised DsbA, PDB ``1FVK`` (1.7 A). The primary WT.
* ``wt_1dsb``  — wild-type, PDB ``1DSB`` (2.0 A). Ground-truth replicate.
* ``wt_1a2j``  — wild-type oxidised, crystal form II, PDB ``1A2J`` (2.0 A). Replicate.

The two WT replicates exist to put a *noise floor* under the CP-vs-WT
comparison: they are the same molecule solved three times, so whatever
spread they show is the spread we must beat before calling a CP-WT gap real.

Ground truth is pyconfind side-chain contact degree, run with exactly the
geometry ``contacts_v1``'s ``GenerationConfig`` uses, and remapped into
input-sequence coordinates -- we import exp74's ``pyconfind_contacts`` module
verbatim rather than reimplementing it, so the contact definition here is
byte-identical to the one every other MarinFold contact number uses.

The CP<->WT bijection is the point of the experiment. The 1UN2 construct is

    [WT 100..189] + GGGTG + [WT 1..99] + LIK        (90 + 5 + 99 + 3 = 197)

so every CP residue except the 5-residue linker and the 3-residue cloning tail
has a WT counterpart. We *derive* the segmentation by alignment rather than
hard-coding it, then assert it matches the construct above.

Writes ``data/units.json`` (sequences + GT contacts + resolved sets) and
``data/cp_wt_map.json`` (the bijection). Both are small enough to commit.

    uv run python prepare_inputs.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gemmi
import requests

HERE = Path(__file__).resolve().parent
# exp74 owns the ground-truth contact definition; import it rather than fork it.
sys.path.insert(0, str(HERE.parent / "exp74_evals_protenix_pyconfind_contacts"))
from pyconfind_contacts import compute_contacts  # noqa: E402

# Local RCSB mmCIF mirror (~2022-09 snapshot); all four entries predate it.
MIRROR = Path("/data/tim/af3-db/mmcif_files")
DATA = HERE / "data"

# UniProt P0AEG4 residues 20-208 -- the mature DsbA chain after the 19-residue
# signal peptide is cleaved. This is "WT numbering" everywhere in exp224:
# 0-based index w <-> mature residue w+1.
MATURE_WT = (
    "AQYEDGKQYTTLEKPVAGAPQVLEFFSFFCPHCYQFEEVLHISDNVKKKLPEGVKMTKYHVNFMGGDLGKDL"
    "TQAWAVAMALGVEDKVTVPLFEGVQKTQTIRSASDIRDVFINAGIKGEEYDAAWNSFVVKSLVAQQEKAAAD"
    "VQLRGVPAMFVNGKYQLNPQGMDTSNMDVFVQQYADTVKYLSEKK"
)

UNITS = [
    dict(unit="cp_1un2", pdb="1UN2", chain="A", role="cp",
         label="CPDsbA-Q100T99 (1UN2)", resolution=2.4),
    dict(unit="wt_1fvk", pdb="1FVK", chain="A", role="wt",
         label="wild-type DsbA (1FVK)", resolution=1.7),
    dict(unit="wt_1dsb", pdb="1DSB", chain="A", role="wt_replicate",
         label="wild-type DsbA (1DSB)", resolution=2.0),
    dict(unit="wt_1a2j", pdb="1A2J", chain="A", role="wt_replicate",
         label="wild-type DsbA (1A2J)", resolution=2.0),
]


def structure_path(pdb: str) -> Path:
    """Prefer the local mirror; fall back to RCSB and cache into data/cif/."""
    local = MIRROR / f"{pdb.lower()}.cif"
    if local.exists():
        return local
    cache = DATA / "cif" / f"{pdb.lower()}.cif"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"https://files.rcsb.org/download/{pdb.upper()}.cif", timeout=60)
        r.raise_for_status()
        cache.write_text(r.text)
    return cache


def seqres(pdb: str, chain: str) -> str:
    """Deposited one-letter SEQRES for the entity containing ``chain``.

    This -- not the resolved-residue sequence -- is the model input: it is the
    molecule that was crystallised, including residues the crystal did not
    resolve. Matches the exp74/exp89 convention where ``input_seq`` is the full
    construct and unresolved positions simply drop out of the candidate-pair
    universe.
    """
    r = requests.get(
        f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb.upper()}/1", timeout=60
    )
    r.raise_for_status()
    d = r.json()
    ids = d["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]
    if chain not in ids:
        raise ValueError(f"{pdb}: chain {chain} not in entity 1 ({ids})")
    return d["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", "").strip()


def build_cp_wt_map(cp_seq: str, wt_seq: str) -> dict:
    """Derive the CP->WT residue bijection, then assert the construct is as named.

    Returns a dict with ``cp_to_wt`` (list of length len(cp_seq), entries are
    0-based WT indices or None) plus the derived segmentation.
    """
    # Segment A = the longest prefix of cp_seq that is a suffix of wt_seq.
    best_a = 0
    for n in range(len(cp_seq), 0, -1):
        if wt_seq.endswith(cp_seq[:n]):
            best_a = n
            break
    # Segment B = the longest run of cp_seq, after the linker, that is a prefix
    # of wt_seq. Search forward from the end of segment A.
    best_b, best_b_start = 0, None
    for start in range(best_a, len(cp_seq)):
        n = 0
        while (start + n < len(cp_seq) and n < len(wt_seq)
               and cp_seq[start + n] == wt_seq[n]):
            n += 1
        if n > best_b:
            best_b, best_b_start = n, start
    if not best_a or not best_b:
        raise ValueError("could not decompose the permutant into two WT segments")

    cp_to_wt: list[int | None] = [None] * len(cp_seq)
    wt_off = len(wt_seq) - best_a            # segment A starts here in WT
    for k in range(best_a):
        cp_to_wt[k] = wt_off + k
    for k in range(best_b):
        cp_to_wt[best_b_start + k] = k

    linker = cp_seq[best_a:best_b_start]
    tail = cp_seq[best_b_start + best_b:]
    # The entry is named CPDSBA_Q100T99: new N-term = WT residue 100, new
    # C-term = WT residue 99 (1-based mature numbering). Verify, don't trust.
    new_n_term = wt_off + 1
    new_c_term = best_b
    assert new_n_term == 100, f"new N-terminus is WT {new_n_term}, expected 100"
    assert new_c_term == 99, f"new C-terminus is WT {new_c_term}, expected 99"
    assert wt_seq[wt_off] == "Q" and wt_seq[best_b - 1] == "T", "expected Q100 / T99"

    mapped = sum(c is not None for c in cp_to_wt)
    for c, w in enumerate(cp_to_wt):
        if w is not None:
            assert cp_seq[c] == wt_seq[w], f"residue mismatch at CP {c} / WT {w}"
    return dict(
        cp_to_wt=cp_to_wt,
        seg_a=dict(cp_start=0, cp_end=best_a, wt_start=wt_off, wt_end=len(wt_seq)),
        seg_b=dict(cp_start=best_b_start, cp_end=best_b_start + best_b,
                   wt_start=0, wt_end=best_b),
        linker=linker, tail=tail,
        n_mapped=mapped, n_cp=len(cp_seq), n_wt=len(wt_seq),
        new_n_term_wt_resnum=new_n_term, new_c_term_wt_resnum=new_c_term,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DATA)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    units = {}
    for spec in UNITS:
        pdb, chain = spec["pdb"], spec["chain"]
        path = structure_path(pdb)
        seq = seqres(pdb, chain)
        res = compute_contacts(path, seq, stem=pdb.lower(), prefer_chain=chain)
        print(f"{spec['unit']:10s} {pdb} chain {res.chain}  L={len(seq)}  "
              f"resolved={res.n_resolved_residues}  mapped={res.n_mapped_residues}  "
              f"identity={res.alignment_identity:.3f}  contacts={len(res.contacts)}")
        if res.alignment_identity < 0.95:
            raise SystemExit(f"{spec['unit']}: alignment identity too low; wrong chain?")
        units[spec["unit"]] = dict(
            **spec,
            input_seq=seq,
            L=len(seq),
            structure=str(path),
            analyzed_chain=res.chain,
            n_resolved=res.n_resolved_residues,
            n_mapped=res.n_mapped_residues,
            alignment_identity=res.alignment_identity,
            resolved_positions=list(res.resolved_positions),
            contacts=[[i, j, d] for i, j, d in res.contacts],
        )

    # WT units must all be the same molecule; check before using them as replicates.
    wt_seqs = {u: units[u]["input_seq"] for u in units if units[u]["role"].startswith("wt")}
    ref = units["wt_1fvk"]["input_seq"]
    for u, s in wt_seqs.items():
        if s != ref:
            print(f"  note: {u} SEQRES differs from 1FVK ({len(s)} vs {len(ref)} aa)")
    if ref != MATURE_WT:
        print(f"  note: 1FVK SEQRES != UniProt mature chain "
              f"({len(ref)} vs {len(MATURE_WT)} aa)")

    cpmap = build_cp_wt_map(units["cp_1un2"]["input_seq"], ref)

    # --- Length/composition control: the permutation, minus the permuting. ---
    # The CP construct is 8 residues longer than WT (a GGGTG linker + a LIK
    # cloning tail) and those residues have no structure of their own. Appending
    # exactly them to the *unpermuted* WT sequence gives a molecule of the same
    # length, with the same non-native residues, in the original order — so
    # whatever this control loses relative to wt_1fvk is the cost of the extra
    # residues, and whatever cp_1un2 loses beyond it is the cost of re-ordering.
    wt = units["wt_1fvk"]
    ctrl_seq = ref + cpmap["linker"] + cpmap["tail"]
    assert len(ctrl_seq) == units["cp_1un2"]["L"], "control must match the CP length"
    units["ctrl_identity"] = dict(
        unit="ctrl_identity", pdb="1FVK", chain="A", role="control",
        label="WT + linker + tail, unpermuted", resolution=1.7,
        input_seq=ctrl_seq, L=len(ctrl_seq), structure=wt["structure"],
        analyzed_chain=wt["analyzed_chain"], n_resolved=wt["n_resolved"],
        n_mapped=wt["n_mapped"], alignment_identity=wt["alignment_identity"],
        # WT residues keep indices 0..188 in this construct, so its ground truth
        # is wt_1fvk's, unchanged.
        resolved_positions=list(wt["resolved_positions"]),
        contacts=[list(c) for c in wt["contacts"]],
    )
    print(f"\ncontrol: WT + {cpmap['linker']} + {cpmap['tail']} "
          f"= {len(ctrl_seq)} aa (CP is {units['cp_1un2']['L']} aa), order unchanged")
    print(f"\nCP<->WT map: {cpmap['n_mapped']}/{cpmap['n_cp']} CP residues have a WT "
          f"counterpart; linker={cpmap['linker']!r} tail={cpmap['tail']!r}")
    print(f"  segment A: CP[{cpmap['seg_a']['cp_start']}:{cpmap['seg_a']['cp_end']}] "
          f"= WT[{cpmap['seg_a']['wt_start']}:{cpmap['seg_a']['wt_end']}] "
          f"(mature {cpmap['seg_a']['wt_start'] + 1}-{cpmap['seg_a']['wt_end']})")
    print(f"  segment B: CP[{cpmap['seg_b']['cp_start']}:{cpmap['seg_b']['cp_end']}] "
          f"= WT[{cpmap['seg_b']['wt_start']}:{cpmap['seg_b']['wt_end']}] "
          f"(mature {cpmap['seg_b']['wt_start'] + 1}-{cpmap['seg_b']['wt_end']})")

    (args.out_dir / "units.json").write_text(json.dumps(units, indent=1))
    (args.out_dir / "cp_wt_map.json").write_text(json.dumps(cpmap, indent=1))
    print(f"\nwrote {args.out_dir/'units.json'} and {args.out_dir/'cp_wt_map.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
