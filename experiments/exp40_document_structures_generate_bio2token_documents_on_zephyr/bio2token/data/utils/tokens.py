# SPDX-FileCopyrightText: 2025 flagshippioneering
#
# SPDX-License-Identifier: MIT

#################################################################
#               Token definitions
#################################################################


"""
Residue token definitions.
"""
AA_TO_TOKEN = {
    "A": "AA_A",
    "R": "AA_R",
    "N": "AA_N",
    "D": "AA_D",
    "C": "AA_C",
    "Q": "AA_Q",
    "E": "AA_E",
    "G": "AA_G",
    "H": "AA_H",
    "I": "AA_I",
    "L": "AA_L",
    "K": "AA_K",
    "M": "AA_M",
    "F": "AA_F",
    "P": "AA_P",
    "S": "AA_S",
    "T": "AA_T",
    "W": "AA_W",
    "Y": "AA_Y",
    "V": "AA_V",
    "X": "AA_X",
    "U": "AA_U",
}

RNA_TO_TOKEN = {
    "A": "RNA_A",
    "G": "RNA_G",
    "C": "RNA_C",
    "U": "RNA_U",
    "T": "RNA_T",
}

INDEXED_RESIDUES = {
    "AA_A": 0,
    "AA_R": 1,
    "AA_N": 2,
    "AA_D": 3,
    "AA_C": 4,
    "AA_Q": 5,
    "AA_E": 6,
    "AA_G": 7,
    "AA_H": 8,
    "AA_I": 9,
    "AA_L": 10,
    "AA_K": 11,
    "AA_M": 12,
    "AA_F": 13,
    "AA_P": 14,
    "AA_S": 15,
    "AA_T": 16,
    "AA_W": 17,
    "AA_Y": 18,
    "AA_V": 19,
    "AA_X": 20,
    "AA_U": 21,
    "RNA_A": 22,
    "RNA_G": 23,
    "RNA_C": 24,
    "RNA_U": 25,
    "RNA_T": 26,
}
INDEXED_RESIDUES_REVERSED = {v: k for k, v in INDEXED_RESIDUES.items()}

SPECIAL_TOKENS_RESIDUES = {
    "<pad>": 27,
    "<cls>": 28,
    "<eos>": 29,
    "<mask>": 30,
    "<unk>": 31,
    "<na>": 32,
}
SPECIAL_TOKENS_RESIDUES_REVERSED = {v: k for k, v in SPECIAL_TOKENS_RESIDUES.items()}

# Extra residue tokens.
PAD_RES = SPECIAL_TOKENS_RESIDUES["<pad>"]  # Outside residue padding token
CLS_RES = SPECIAL_TOKENS_RESIDUES["<cls>"]  # Start of sequence token
EOS_RES = SPECIAL_TOKENS_RESIDUES["<eos>"]  # End of sequence token
MASKED_RES = SPECIAL_TOKENS_RESIDUES["<mask>"]  # Masked token
MISSING_RES = SPECIAL_TOKENS_RESIDUES["<unk>"]  # Missing token
NA_RES = SPECIAL_TOKENS_RESIDUES["<na>"]  # Not applicable token (i.e. when a dataset does not have such information)


"""
Atom token definitions.
"""
INDEXED_ATOMS = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "B": 5,
    "Al": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Br": 12,
    "I": 13,
}
SPECIAL_TOKENS_ATOMS = {
    "<pad>": 14,
    "<cls>": 15,
    "<eos>": 16,
    "<mask>": 17,
    "<unk>": 18,
    "<na>": 19,
}
# Extra atom tokens.
PAD_ATOM = SPECIAL_TOKENS_ATOMS["<pad>"]  # Outside residue padding token
CLS_ATOM = SPECIAL_TOKENS_ATOMS["<cls>"]  # Start of sequence token
EOS_ATOM = SPECIAL_TOKENS_ATOMS["<eos>"]  # End of sequence token
MASKED_ATOM = SPECIAL_TOKENS_ATOMS["<mask>"]  # Masked token
MISSING_ATOM = SPECIAL_TOKENS_ATOMS["<unk>"]  # Missing token
NA_ATOM = SPECIAL_TOKENS_ATOMS["<na>"]  # Not applicable token (i.e. when a dataset does not have such information)

"""
Metastructure token definitions. Use to define the type the token belongs to, i,e
- backbone
- sidechain
- reference carbon (c-alpha)
- padding
"""
BB_CLASS = 0  # Backbone class
C_REF_CLASS = 1  # Reference carbon (c-alpha) class
SC_CLASS = 2  # Sidechain class
PAD_CLASS = 3  # Outside residue padding class
