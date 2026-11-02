# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Read helpers for the ESM Atlas (the "ESMFold2 Atlas").

The Atlas is published as a public AWS Open Data bucket,
``s3://esm-protein-atlas/`` (region ``us-west-2``), readable anonymously.
The predicted structures live in a Lance dataset
``v1/folds/folds_1B.lance`` (1,095,530,880 rows, one **monomer** per row)
with schema::

    header, protein_hash, ptm:double, mean_plddt:double,
    per_residue_plddt:binary, pae:binary, structure_blob:binary,
    sequence:large_string

The three ``binary`` columns are NOT raw text:

- ``structure_blob`` is **brotli-compressed msgpack** (msgpack-numpy
  encoded). Decoded, it is a dict in AlphaFold/ESM **atom37** layout:
  ``sequence``, ``atom37_positions`` (the *present* atoms, flattened to
  ``[n_present, 3]`` float16), ``atom37_mask`` ``[n_res, 37]`` bool,
  ``confidence`` (per-residue pLDDT), ``residue_index``, ``chain_id``,
  ``chain_boundaries`` (``[[0, n_res]]`` for the single chain), etc.
- ``per_residue_plddt`` and ``pae`` are ZIP archives each holding a single
  ``arr.npy`` (float16 pLDDT vector, uint8 PAE matrix respectively).

This module isolates those decode quirks so the analysis scripts stay
readable. Nothing here mutates remote state; all reads are anonymous and
streamed (no bulk download).
"""

import io
import zipfile

import brotli
import gemmi
import lance
import msgpack
import msgpack_numpy as mnp
import numpy as np

ATLAS_BUCKET = "esm-protein-atlas"
ATLAS_REGION = "us-west-2"
# Lance reads through the Rust ``object_store`` crate; this is how it does an
# unsigned (public) S3 request. (pyarrow's S3FileSystem uses anonymous=True.)
#
# ``timeout``/``connect_timeout`` are CRITICAL for the high-fan-out materialize:
# without a per-request timeout, object_store blocks FOREVER on a dropped or
# throttled S3 connection. Observed failure (exp91 materialize, 96 workers/box):
# after ~8-10 min of sustained high-request-rate reads, S3 started dropping
# connections, NetworkIn fell to ~0, and every worker wedged in ``ds.take`` with no
# timeout to break it (CPU -> ~1%). A bounded timeout turns that hang into a
# retriable error; object_store then retries with backoff (default max_retries).
ANON_STORAGE = {
    "aws_skip_signature": "true",
    "region": ATLAS_REGION,
    "timeout": "120s",          # per-request cap; ranged GETs are small so this is generous
    "connect_timeout": "20s",   # cap TCP connect so a black-holed endpoint fails fast
}

FOLDS_1B_URI = f"s3://{ATLAS_BUCKET}/v1/folds/folds_1B.lance"
FOLDS_ATLAS_URI = f"s3://{ATLAS_BUCKET}/v1/folds/folds_atlas.lance"
N_FOLDS_1B = 1_095_530_880  # verified by Lance count_rows()

# AlphaFold residue_constants.atom_types — the 37 atom-name slots, in order.
# atom37_mask[res, i] is True iff slot i is present for that residue; the
# present atoms are concatenated in this order into atom37_positions.
ATOM37_NAMES: tuple[str, ...] = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
    "CZ3", "NZ", "OXT",
)

# Three-letter codes for the 20 standard amino acids (plus X→UNK).
_AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}


def open_folds(uri: str = FOLDS_1B_URI) -> lance.LanceDataset:
    """Open an Atlas fold dataset for anonymous, streamed reads."""
    return lance.dataset(uri, storage_options=ANON_STORAGE)


def decode_structure_blob(blob: bytes) -> dict:
    """Decode a ``structure_blob`` to its atom37 msgpack dict.

    Returns numpy arrays for the array-valued fields (``atom37_positions``,
    ``atom37_mask``, ``confidence``, ``residue_index``, ``chain_id``, …).
    """
    raw = brotli.decompress(blob)
    return msgpack.unpackb(
        raw, raw=False, strict_map_key=False, object_hook=mnp.decode
    )


def decode_npy_zip(blob: bytes) -> np.ndarray:
    """Decode a ``per_residue_plddt`` / ``pae`` ZIP(``arr.npy``) blob."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return np.load(io.BytesIO(zf.read("arr.npy")))


def is_monomer(struct: dict) -> bool:
    """True iff the decoded structure has a single chain."""
    return len(struct.get("chain_boundaries", [])) == 1


def atom37_to_structure(struct: dict, name: str = "ATLAS") -> gemmi.Structure:
    """Build a single-chain ``gemmi.Structure`` from a decoded atom37 dict.

    ``atom37_positions`` holds only the *present* atoms (mask-selected),
    flattened in ``ATOM37_NAMES`` order; we walk the mask to re-expand them.
    We build through gemmi (rather than hand-formatting PDB columns) and call
    ``setup_entities()`` so the result is recognised as a polymer — Foldseek's
    structure parser and ``gemmi.read_structure`` both need that, otherwise the
    chain reads as zero polymer residues and every search returns no hits.
    """
    seq = struct["sequence"]
    mask = np.asarray(struct["atom37_mask"], dtype=bool)
    pos = np.asarray(struct["atom37_positions"], dtype=np.float64)
    conf = np.asarray(struct.get("confidence", []), dtype=np.float64)

    st = gemmi.Structure()
    st.name = name[:4] if name else "ATLS"
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    cursor = 0  # index into the flat present-atom positions
    for res_i in range(mask.shape[0]):
        res = gemmi.Residue()
        res.name = _AA1_TO_AA3.get(seq[res_i], "UNK")
        res.seqid = gemmi.SeqId(res_i + 1, " ")
        bfac = float(conf[res_i] * 100.0) if res_i < len(conf) else 0.0
        for slot in range(37):
            if not mask[res_i, slot]:
                continue
            x, y, z = pos[cursor]
            cursor += 1
            atom = gemmi.Atom()
            atom.name = ATOM37_NAMES[slot]
            atom.element = gemmi.Element(ATOM37_NAMES[slot][0])
            atom.pos = gemmi.Position(x, y, z)
            atom.occ = 1.0
            atom.b_iso = bfac
            res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    return st


def write_structure_cif(struct: dict, path, name: str = "ATLAS") -> None:
    """Decode-to-atom37 helper: write a mmCIF Foldseek/gemmi can read."""
    atom37_to_structure(struct, name=name).make_mmcif_document().write_file(str(path))
