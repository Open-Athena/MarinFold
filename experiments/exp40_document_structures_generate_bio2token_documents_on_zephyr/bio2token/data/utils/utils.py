# SPDX-FileCopyrightText: 2025 flagshippioneering
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List, Optional
import torch
import pandas as pd
import torch.nn.functional as F
import torch
import numpy as np
from Bio.PDB import PDBParser
import os

from bio2token.data.utils.tokens import AA_TO_TOKEN, RNA_TO_TOKEN, BB_CLASS, C_REF_CLASS, SC_CLASS, PAD_CLASS
from bio2token.data.utils.molecule_conventions import (
    SC_ATOMS_AA,
    SC_ATOMS_RNA,
    AMINO_ACID_ABBRS,
    AA_ABRV_REVERSED,
    RNA_ABRV_REVERSED,
    BB_ATOMS_AA,
    BB_ATOMS_RNA,
    AA_C_REF,
    RNA_C_REF,
    AA_BB_REF,
    RNA_BB_REF,
)


def filter_on_length(
    block: pd.DataFrame, length_key: str = "natoms", max_length: int = None, min_length: int = None
) -> pd.DataFrame:
    """
    Filter a DataFrame based on sequence length constraints.

    This function filters a DataFrame to only include rows that meet specified minimum and maximum
    length criteria for a given length attribute. It is useful for selecting sequences of desired
    lengths, thereby excluding excessively short or long entries.

    Args:
        block (pd.DataFrame): The DataFrame containing the data to be filtered, with sequences characterized
                              by a specific length-related attribute.
        length_key (str): The key in the DataFrame used to determine the length of each entry.
                          Defaults to "natoms", representing the number of atoms or similar units.
        max_length (int, optional): The maximum allowable length for any given sequence. Entries exceeding
                                    this length will be filtered out. If None, no maximum length filtering
                                    is applied.
        min_length (int, optional): The minimum allowable length for any given sequence. Entries shorter
                                    than this length will be filtered out. If None, no minimum length filtering
                                    is applied.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the rows that satisfy the specified length
                      conditions.

    Notes:
        - The function checks row-wise against the specified length criteria and returns entries within the bounds.
        - It handles cases where either or both length constraints (min/max) are set to None, which means no filtering
          will be applied for those respective constraints.
    """
    # Filter out rows with lengths less than the specified minimum, if given.
    if min_length is not None:
        block = block[block[length_key] >= min_length]

    # Filter out rows with lengths greater than the specified maximum, if given.
    if max_length is not None:
        block = block[block[length_key] <= max_length]

    # Return the filtered DataFrame.
    return block


def pad_and_stack_tensors(sequences: List[torch.Tensor], pad_token: int, multiple_of: int = 1, left_pad: bool = False):
    """
    Pads and stacks tensors in a list to a uniform length (a multiple of N) and stacks them into a single tensor.

    Args:
        sequences (List[torch.Tensor]): A list of 1D tensors to be padded.
        pad_token (int): The value to use for padding.
        multiple_of (int): The multiple to pad the tensors to. Default is 1 (no specific multiple).
        left_pad (bool): Whether to pad on the left (True) or right (False). Default is False.

    Returns:
        torch.Tensor: A 2D tensor where all sequences are padded to the same length.
    """
    # Find the maximum length in the sequences
    max_len = max(seq.size(0) for seq in sequences)

    # Adjust max_len to be a multiple of `multiple_of`
    target_len = ((max_len + multiple_of - 1) // multiple_of) * multiple_of

    padded_sequences = []
    for seq in sequences:
        padding_size = target_len - seq.size(0)
        if seq.dim() == 1:
            # For 1D tensors, pad according to left_pad parameter
            pad_left = padding_size if left_pad else 0
            pad_right = 0 if left_pad else padding_size
            padded_seq = F.pad(seq, (pad_left, pad_right), value=pad_token)
        elif seq.dim() == 2:
            # For 2D tensors, pad according to left_pad parameter (dim=0)
            pad_left = padding_size if left_pad else 0
            pad_right = 0 if left_pad else padding_size
            padded_seq = F.pad(seq, (0, 0, pad_left, pad_right), value=pad_token)
        else:
            # For 3D tensors, pad according to left_pad parameter (dim=0)
            pad_left = padding_size if left_pad else 0
            pad_right = 0 if left_pad else padding_size
            padded_seq = F.pad(seq, (0, 0, 0, 0, pad_left, pad_right), value=pad_token)
        padded_sequences.append(padded_seq)

    # Stack the padded sequences into a single tensor
    return torch.stack(padded_sequences, dim=0)


def pad_and_stack_batch(
    batch: List[Dict[str, torch.Tensor]],
    sequences_to_pad: Dict[str, Any],
    pad_to_multiple_of: int = 1,
):
    # assert all([k in batch[0] for k in sequences_to_pad]), f"All sequences to pad must be in the batch: {sequences_to_pad}"

    # Convert list of dicts to dict of lists
    batch = {key: [sample[key] for sample in batch] for key in batch[0].keys()}
    token_keys = {k: v for k, v in sequences_to_pad.items()}

    processed_batch = {}
    for key, values in batch.items():
        if key in token_keys:
            # Pad and stack tensors that need padding
            if token_keys[key] == "no_pad":
                processed_batch[key] = values
            else:
                processed_batch[key] = pad_and_stack_tensors(values, token_keys[key], pad_to_multiple_of)
        else:
            # Directly stack other tensors
            processed_batch[key] = torch.stack(values, dim=0)

    return processed_batch


def filter_batch(batch: List[Dict[str, torch.Tensor]], filter_keys: List[str]):
    return [{k: v for k, v in sample.items() if k in filter_keys} for sample in batch]


def compute_masks(
    sample: Dict,
    structure_track: Optional[bool] = False,
):
    """
    Compute usefull masks for language modeling.
    """
    ### Padding masks ###
    # Mask of positions that are end of sequence padding
    sample["eos_pad_mask"] = torch.isin(sample["token_class"], torch.tensor([PAD_CLASS]))

    ### Metastructure masks ###
    # Mask of positions that are reference carbon atoms
    sample["cref_mask"] = torch.isin(sample["token_class"], torch.tensor([C_REF_CLASS]))
    # Mask of positions that are backbone atoms
    sample["bb_atom_mask"] = torch.isin(sample["token_class"], torch.tensor([BB_CLASS, C_REF_CLASS]))
    # Mask of positions that are sidechain atoms
    sample["sc_atom_mask"] = torch.isin(sample["token_class"], torch.tensor([SC_CLASS]))
    # Mask of positions that are either backbone or sidechain atoms
    sample["all_atom_mask"] = sample["bb_atom_mask"] + sample["sc_atom_mask"]

    ### Track loss masks ###
    # Structure
    if structure_track:
        # Atoms with known structure that are not padding
        sample["structure_known_all_atom_mask"] = sample["all_atom_mask"] * (~sample["unknown_structure"])
        # BB Atoms with known structure that are not padding
        sample["bb_atom_known_structure_mask"] = sample["bb_atom_mask"] * (~sample["unknown_structure"])
        # SC Atoms with known structure that are not padding
        sample["sc_atom_known_structure_mask"] = sample["sc_atom_mask"] * (~sample["unknown_structure"])
        # Cref Atoms with known structure that are not padding
        sample["cref_atom_known_structure_mask"] = sample["cref_mask"] * (~sample["unknown_structure"])

    return sample


def uniform_dataframe(seq, res_types, atom_coords, atom_names, res_atom_start, res_atom_end):
    """
    Standardize and extract backbone and sidechain coordinates for each amino acid residue in a sequence.

    This method processes sequences of amino acids and their associated atomic data to extract and align
    both backbone and sidechain coordinates with canonical atom names.
    It then generates a structure track by combining backbone (BB) atoms and sidechain (SC) atoms coordinates.

    This function takes the backbone and sidechain coordinates of a molecule and concatenates them to form a complete
    structure representation. The backbone and sidechain coordinates are expected to be in a format compatible with
    np.stack, and are reshaped into a (-1, 3) array format to represent the 3D points of the structure.

    Args:
        seq (str): A string representing the residue sequence, where each character is a single-letter code.
        res_types (list): A list of residue types, indicating rna or aa.
        atom_coords (list): A list of coordinates for atoms, structured as a list of lists [[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn]]
        atom_names (list): A list of atom names corresponding to the positions in `atom_coords`.
        res_atom_start (list): Starting indices for the atom entries corresponding to each residue within `atom_coords`.
        res_atom_end (list): Ending indices for the atom entries corresponding to each residue within `atom_coords`.

    Returns:
        np.ndarray: A numpy array representing the concatenated structure of BB and SC coordinates, reshaped
                    into a 2D array with each row representing a point in 3D space.

    Raises:
        AssertionError: If there are multiple occurrences of a canonical atom name within a residue's atom names.
                        This behavior helps ensure that each expected atom is accounted for without duplication.
    """
    # Convert atom coordinates to a numpy array and split into lists based on residue boundaries.
    atom_coords = np.array(atom_coords)
    atom_coords = [atom_coords[res_atom_start[i] : res_atom_end[i]].tolist() for i in range(len(res_atom_start))]

    # Convert atom names to a numpy array and split into lists based on residue boundaries.
    atom_names = np.array(atom_names)
    atom_names = [atom_names[res_atom_start[i] : res_atom_end[i]].tolist() for i in range(len(res_atom_start))]

    # Prepare a list of canonical sidechain atom names for each amino acid in the sequence.
    canonical_sc_atoms = []
    res_types_per_atom = []
    residue_name = []
    mask_bb = []
    mask_c_ref = []
    mask_sc = []
    atom_names_reordered = []
    for id_k, k in enumerate(seq):
        if res_types[id_k] == "aa":
            canonical_sc_atoms.append(SC_ATOMS_AA[AMINO_ACID_ABBRS[k]])
            res_types_per_atom.append(res_types[id_k])
            residue_name.append(AA_TO_TOKEN[k])
            mask_bb.append(AA_BB_REF + len(SC_ATOMS_AA[AMINO_ACID_ABBRS[k]]) * [False])
            mask_c_ref.append(AA_C_REF + len(SC_ATOMS_AA[AMINO_ACID_ABBRS[k]]) * [False])
            mask_sc.append(len(AA_BB_REF) * [False] + len(SC_ATOMS_AA[AMINO_ACID_ABBRS[k]]) * [True])
        else:
            canonical_sc_atoms.append(SC_ATOMS_RNA[RNA_ABRV_REVERSED[k]])
            res_types_per_atom.append(res_types[id_k])
            residue_name.append(RNA_TO_TOKEN[k])
            mask_bb.append(RNA_BB_REF + len(SC_ATOMS_RNA[RNA_ABRV_REVERSED[k]]) * [False])
            mask_c_ref.append(RNA_C_REF + len(SC_ATOMS_RNA[RNA_ABRV_REVERSED[k]]) * [False])
            mask_sc.append(len(RNA_BB_REF) * [False] + len(SC_ATOMS_RNA[RNA_ABRV_REVERSED[k]]) * [True])

    sc_coords = []  # List to store sidechain coordinates.
    bb_coords = []  # List to store backbone coordinates.
    # Iterate over each residue to fill in backbone and sidechain coordinates.
    for i in range(len(atom_coords)):
        # Initialize placeholder lists for backbone coordinates using canonical names.
        if res_types_per_atom[i] == "aa":
            GT_BB_FULL = BB_ATOMS_AA
        else:
            GT_BB_FULL = BB_ATOMS_RNA
        atom_names_reordered.append(GT_BB_FULL + canonical_sc_atoms[i])

        bb_coords.append([None] * len(GT_BB_FULL))
        for j in range(len(GT_BB_FULL)):
            idx = np.arange(len(atom_names[i]))[np.array(atom_names[i]) == GT_BB_FULL[j]]
            # Check to ensure no duplicate backbone atom names.
            assert len(idx) <= 1, f"Warning: {GT_BB_FULL[j]} is present in {atom_names[i]} {len(idx)} times."
            # Assign coordinates or NaN if missing.
            bb_coords[i][j] = atom_coords[i][idx[0]] if len(idx) == 1 else [np.nan, np.nan, np.nan]

        # Initialize placeholder lists for sidechain coordinates based on canonical sidechain atoms.
        sc_coords.append([None] * len(canonical_sc_atoms[i]))
        for j in range(len(canonical_sc_atoms[i])):
            idx = np.arange(len(atom_names[i]))[np.array(atom_names[i]) == canonical_sc_atoms[i][j]]
            # Check to ensure no duplicate sidechain atom names.
            assert len(idx) <= 1, f"Warning: {canonical_sc_atoms[i][j]} is present in {atom_names[i]} {len(idx)} times."
            # Assign coordinates or NaN if missing.
            sc_coords[i][j] = atom_coords[i][idx[0]] if len(idx) == 1 else [np.nan, np.nan, np.nan]
    residue_size = [len(bb) + len(sc) for bb, sc in zip(bb_coords, sc_coords)]
    structure = [bb + sc for bb, sc in zip(bb_coords, sc_coords)]
    structure = np.array([item for sublist in structure for item in sublist])
    unknown_structure = np.isnan(structure).sum(axis=-1) > 0

    residue_name = [residue_size[i] * [residue_name[i]] for i in range(len(seq))]
    residue_name = [item for sublist in residue_name for item in sublist]
    residue_ids = [residue_size[i] * [i] for i in range(len(seq))]
    residue_ids = np.array([item for sublist in residue_ids for item in sublist])

    atom_names_reordered = [item for sublist in atom_names_reordered for item in sublist]

    # Backbone mask: True for backbone atoms, False for sidechain and padding tokens.
    mask_bb = np.array([item for sublist in mask_bb for item in sublist])

    # C-alpha reference mask: True for C-alpha atom, False for sidechain and padding tokens.
    mask_c_ref = np.array([item for sublist in mask_c_ref for item in sublist])

    # Sidechain mask: True for sidechain atoms, False for backbone and padding tokens.
    mask_sc = np.array([item for sublist in mask_sc for item in sublist])

    # Token classes: Assign integer codes to each class based on the masks.
    token_class = np.zeros(mask_bb.shape).astype(int)
    token_class[mask_bb] = BB_CLASS
    token_class[mask_c_ref] = C_REF_CLASS
    token_class[mask_sc] = SC_CLASS

    # Calculate the barycenter of the structure.
    barycenter = np.mean(structure[~unknown_structure], axis=0)
    structure[~unknown_structure] = structure[~unknown_structure] - barycenter

    # Return the structured coordinates as numpy arrays.
    return structure, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered


def pdb_2_dict(pdb_path: str, chains: List[str] = None):
    parser = PDBParser()
    pdb_ids = []
    pdb_id = os.path.basename(pdb_path).split(".")[0]
    structure = parser.get_structure(pdb_id, pdb_path)
    seq = ""
    coords = []
    atom_names = []
    atom_types = []
    res_names = []
    chain_ids = []
    res_ids = []
    res_types = []
    continuous_res_ids = []
    res_atom_start = []
    res_atom_end = []
    full_res_names = []
    # if no chains are provided we read in everything
    if chains is None:
        chains = [chain.id for chain in structure.get_chains()]
    print(f"PDB loading. Chains: {chains}")
    continuous_res_id = 0
    continuous_atom_id = 0
    for chain in structure.get_chains():
        if chain.id not in chains:
            pass
        else:
            res_id = 0
            for residue in chain:
                resname = residue.get_resname()
                if len(resname) == 3:
                    ABBRS_REVERSED = AA_ABRV_REVERSED
                    res_type = "aa"
                else:
                    ABBRS_REVERSED = RNA_ABRV_REVERSED
                    res_type = "rna"
                if resname not in ABBRS_REVERSED:
                    pass
                else:
                    res = ABBRS_REVERSED[resname]
                    seq += res
                    res_atom_start.append(continuous_atom_id)
                    res_types.append(res_type)
                    for atom in residue:
                        atom_name = atom.get_name()
                        if atom_name[0] == "H":
                            pass
                        else:
                            atom_names.append(atom.get_name())
                            atom_types.append(atom.get_name()[0])
                            chain_ids.append(chain.id)
                            pdb_ids.append(pdb_id)
                            coords.append(np.array(atom.get_coord()))
                            res_names.append(res)
                            full_res_names.append(resname)
                            res_ids.append(res_id)
                            continuous_res_ids.append(continuous_res_id)

                            continuous_atom_id += 1
                            res_id += 1
                    res_atom_end.append(continuous_atom_id)
                    continuous_res_id += 1
    coords = np.vstack(coords)
    print(f"PDB loading. N residues: {len(res_atom_start)}")
    print(f"PDB loading. N atoms: {len(atom_names)}")
    # Create DataFrame
    pdb_dict = {
        "pdb_id": pdb_ids,
        "seq": seq,
        "res_names": res_names,
        "coords_groundtruth": coords,
        "atom_names": atom_names,
        "atom_types": atom_types,
        "seq_length": len(seq),
        "atom_length": len(atom_names),
        "chains": chain_ids,
        "res_ids": res_ids,
        "continuous_res_ids": continuous_res_ids,
        "res_types": res_types,
        "res_atom_start": res_atom_start,
        "res_atom_end": res_atom_end,
        "full_res_names": full_res_names,
    }

    return pdb_dict


def write_pdb(coords: np.ndarray, atom_types: List[str], residue_types: List[str], residue_ids: List[int], output_path: str):
    """
    Write protein coordinates to PDB format.

    Args:
        coords: numpy array of shape (N, 3) containing atomic coordinates
        atom_types: list of length N containing atom names (e.g., 'N', 'CA', 'C', 'O')
        residue_types: list of length N containing residue names (e.g., 'ALA', 'GLY')
        residue_ids: list of length N containing residue ids
        output_path: path to save the PDB file
    """
    assert len(coords) == len(atom_types) == len(residue_types), "Length mismatch in inputs"
    if os.path.exists(output_path):
        os.remove(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        atom_num = 1
        current_res_num = 1
        prev_res_id = residue_ids[0]

        for coord, atom, res, res_id in zip(coords, atom_types, residue_types, residue_ids):
            # Increment residue number when we see a new residue
            if res_id != prev_res_id:
                current_res_num += 1
                prev_res_id = res_id

            # PDB format specification
            line = (
                f"ATOM  {atom_num:5d}  {atom:<3s} {res:3s} A{current_res_num:4d}"
                f"    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           {atom[0]:>2s}\n"
            )
            f.write(line)
            atom_num += 1

        # Add END marker
        f.write("END\n")
