"""Pinned experiment contract and lossless contacts-v1 document conversion."""

import hashlib
import re
from dataclasses import dataclass

MODELS = {
    "0.8B": ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
    "2B": ("Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"),
    "4B": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
}
FORMATS = ("contacts_v1", "prompted")
ROOT = "s3://marin-us-east-02a/MarinFold/exp347_qwen_base_contacts"
SOURCE = "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/data/afdb"
MAX_LENGTH = 16384
TOKEN_BUDGET = 1_000_000_000
SEED = 347
AA = dict(
    zip(
        [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
            "UNK",
        ],
        "ARNDCQEGHILKMFPSTWYVX",
        strict=True,
    )
)
POSITION = re.compile(r"<p(\d+)>")
PROMPT = (
    "Predict residue contacts for this protein. Residues are numbered from 1 in "
    "the order of the amino-acid sequence. Write one pair of residue numbers per "
    "line, separated by a space. Each pair denotes a side-chain contact with "
    "pyconfind contact degree at least 0.001 and sequence separation at least 6. "
    "List each pair once, in any order. Finish with END.\n\nSequence:\n"
)


@dataclass(frozen=True)
class Document:
    """Two renderings with identical sequence and contact information."""

    sequence: str
    contacts: tuple[tuple[int, int], ...]
    raw_prefix: str
    raw_completion: str
    prompted_prefix: str
    prompted_completion: str


def position(token: str) -> int:
    """Parse a position on the contacts-v1 2000-index ring."""
    match = POSITION.fullmatch(token)
    if match is None or not 0 <= int(match[1]) < 2000:
        raise ValueError(f"Invalid position: {token}")
    return int(match[1])


def convert_document(text: str) -> Document:
    """Decode disjoint chain runs without changing the contact set or order.

    Each chain occupies a contiguous interval on the 2000-position ring.
    Number chains by their N-terminal ring index, then concatenate their residue
    indices for the prompted rendering. Monomer rendering remains unchanged.
    """
    prefix, separator, completion = text.partition("<begin_statements>")
    if not separator or not prefix.startswith("<contacts-v1> <begin_sequence> "):
        raise ValueError("Expected a contacts-v1 document with both sections")
    fields = prefix.split()[2:]
    if len(fields) % 2:
        raise ValueError("Unpaired sequence statement")
    residues: dict[int, str] = {}
    starts: list[int] = []
    ends: list[int] = []
    for a, b in zip(fields[::2], fields[1::2], strict=True):
        if a == "<n-term>":
            starts.append(position(b))
        elif a == "<c-term>":
            ends.append(position(b))
        else:
            index = position(a)
            if index in residues or b[1:-1] not in AA:
                raise ValueError(f"Invalid or duplicate amino acid statement: {a} {b}")
            residues[index] = AA[b[1:-1]]
    if (
        not starts
        or len(starts) != len(ends)
        or len(set(starts)) != len(starts)
        or len(set(ends)) != len(ends)
        or not 2 <= len(residues) <= 2000
    ):
        raise ValueError("Expected unique paired nonempty protein termini")
    chains: list[list[int]] = []
    for start in sorted(starts):
        chain = []
        for offset in range(2000):
            index = (start + offset) % 2000
            if index not in residues or (offset and index in starts):
                raise ValueError("Broken or overlapping chain interval")
            chain.append(index)
            if index in ends:
                break
        else:
            raise ValueError("Chain has no C terminus")
        chains.append(chain)
    ordered = [index for chain in chains for index in chain]
    if len(ordered) != len(residues) or set(ordered) != residues.keys():
        raise ValueError("Sequence positions or termini are inconsistent")
    one_based = {p: i + 1 for i, p in enumerate(ordered)}
    chain_ids = {one_based[p]: i for i, chain in enumerate(chains) for p in chain}
    sequence = "".join(residues[p] for p in ordered)
    tokens = completion.split()
    if not tokens or tokens[-1] != "<end>" or (len(tokens) - 1) % 3:
        raise ValueError("Malformed contact section")
    contacts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(0, len(tokens) - 1, 3):
        if tokens[i] != "<contact>":
            raise ValueError(f"Unsupported structure statement: {tokens[i]}")
        a, b = (one_based[position(t)] for t in tokens[i + 1 : i + 3])
        pair = tuple(sorted((a, b)))
        if (chain_ids[a] == chain_ids[b] and abs(a - b) < 6) or pair in seen:
            raise ValueError(f"Duplicate or too-close contact: {pair}")
        seen.add(pair)
        contacts.append((a, b))
    prompted_prefix = PROMPT + sequence + "\n\nContacts:\n"
    if len(chains) > 1:
        prompted_prefix = (
            "Predict residue contacts for this protein complex. Residues are numbered "
            "consecutively across the chains below; chain boundaries are explicit. "
            "Write one pair of residue numbers per line, separated by a space. "
            "Each pair denotes a side-chain contact with pyconfind contact degree "
            "at least 0.001. Within a chain, sequence separation must be at least 6; "
            "between different chains, any separation is allowed. "
            "List each pair once, in any order. Finish with END.\n\n"
            + "\n".join(
                f"Chain {i + 1} (residues {one_based[chain[0]]}-{one_based[chain[-1]]}):\n"
                + "".join(residues[p] for p in chain)
                for i, chain in enumerate(chains)
            )
            + "\n\nContacts:\n"
        )
    return Document(
        sequence,
        tuple(contacts),
        prefix + separator,
        completion,
        prompted_prefix,
        "".join(f"{a} {b}\n" for a, b in contacts) + "END\n",
    )


def split_key(sequence: str, sequence_cluster: str | None) -> str:
    """Keep every source sequence cluster in exactly one split."""
    return hashlib.sha256((sequence_cluster or sequence).encode()).hexdigest()


def run_name(size: str, document_format: str, smoke: bool = False) -> str:
    """Return a placement-independent run identity."""
    return f"exp347-qwen35-{size.lower().replace('.', 'p')}-{document_format}-{'smoke' if smoke else '1bt'}"
