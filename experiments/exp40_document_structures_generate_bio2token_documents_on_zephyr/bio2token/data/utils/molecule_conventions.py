# SPDX-FileCopyrightText: 2025 flagshippioneering
#
# SPDX-License-Identifier: MIT

# Annreviations and conventions for amio acids and nucleic acids
# Amino acid abbreviations
AMINO_ACID_ABBRS = {
    "A": "ALA",
    "B": "ASX",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "U": "SEC",
    "V": "VAL",
    "W": "TRP",
    "X": "XAA",
    "Y": "TYR",
    "Z": "GLX",
}

RNA_ABBRS = {
    "A": "A",
    "G": "G",
    "C": "C",
    "U": "U",
    "T": "T",
}

ABBRS = {"AA": AMINO_ACID_ABBRS, "RNA": RNA_ABBRS}

# Backbone convention
BB_ATOMS_AA = ["N", "CA", "C", "O"]
BB_ATOMS_AA_TYPES = [a[0] for a in BB_ATOMS_AA]
BB_ATOMS_RNA = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
BB_ATOMS_RNA_TYPES = [a[0] for a in BB_ATOMS_RNA]

# Sidechain convention
SC_ATOMS_AA = {
    "ALA": ["CB"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "CYS": ["CB", "SG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "GLY": [],
    "HIS": ["CB", "CG", "ND1", "CE1", "NE2", "CD2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "MET": ["CB", "CG", "SD", "CE"],
    "PHE": ["CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
    "PRO": ["CB", "CG", "CD"],
    "SEC": ["CB", "SE"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1", "CG2"],
    "TRP": ["CB", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2"],
    "TYR": ["CB", "CG", "CD1", "CE1", "CZ", "OH", "CE2", "CD2"],
    "VAL": ["CB", "CG1", "CG2"],
    "XAA": [],
}
SC_ATOMS_AA_TYPES = {k: [a[0] for a in v] for k, v in SC_ATOMS_AA.items()}

SC_ATOMS_RNA = {
    "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C": ["N1", "C2", "O2", "C6", "C5", "N4", "N3", "C4"],
    "U": ["N1", "C2", "O2", "C6", "C5", "N3", "C4", "O4"],
    "T": ["N1", "C2", "O2", "C6", "C5", "C7", "O4", "N3", "C4"],  # Thymine for DNA
}
SC_ATOMS_RNA_TYPES = {k: [a[0] for a in v] for k, v in SC_ATOMS_RNA.items()}

# reversed dict
AA_ABRV_REVERSED = {v: k for k, v in AMINO_ACID_ABBRS.items()}
RNA_ABRV_REVERSED = {v: k for k, v in RNA_ABBRS.items()}

# Reference masks
AA_C_REF = [False, True, False, False]
AA_BB_REF = [True, False, True, True]
RNA_C_REF = [False, False, False, False, False, False, False, True, False, False, False, False]
RNA_BB_REF = [True, True, True, True, True, True, True, False, True, True, True, True]
