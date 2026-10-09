"""Frozen cohort and decoding rules for the training/validation diagnostic."""

import hashlib
import re

import numpy as np

SEED = 357
BEGIN = "<begin_statements>"
AA = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK".split(),
        "ARNDCQEGHILKMFPSTWYVX",
        strict=True,
    )
)
RESIDUE = re.compile(r"<p(\d+)>\s+<([A-Z]{3})>")
CONTACT = re.compile(r"<(contact|retract)>\s+<p(\d+)>\s+<p(\d+)>")


def stable_seed(*parts: object) -> int:
    """Return a seed independent of shard assignment and batch boundaries."""
    value = ":".join(map(str, (SEED, *parts))).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") % (2**31 - 1)


def live_contacts(text: str, nterm: int, length: int) -> list[list[int]]:
    """Decode the final live set in canonical sequence coordinates."""
    live: set[tuple[int, int]] = set()
    for operation, left, right in CONTACT.findall(text):
        i, j = sorted(((int(left) - nterm) % 2000, (int(right) - nterm) % 2000))
        if j >= length or j - i < 6:
            continue
        if operation == "retract":
            live.discard((i, j))
        else:
            live.add((i, j))
    return [list(pair) for pair in sorted(live)]


def parse_training_document(row: dict) -> tuple[str, list[list[int]]]:
    """Decode a complete original training label without inferring geometry."""
    prefix, suffix = row["document"].split(BEGIN)
    length, nterm = int(row["seq_len"]), int(row["n_term_index"])
    residues = {
        (int(position) - nterm) % 2000: AA[name]
        for position, name in RESIDUE.findall(prefix)
    }
    if set(residues) != set(range(length)):
        raise ValueError(f"Incomplete sequence in {row['entry_id']}")
    contacts = live_contacts(suffix, nterm, length)
    if len(contacts) != row["contacts_emitted"]:
        raise ValueError(f"Contact round trip failed for {row['entry_id']}")
    if (
        row["truncated"]
        or row["contacts_emitted"] != row["contacts_passing_min_degree"]
    ):
        raise ValueError(f"Incomplete contact label: {row['entry_id']}")
    return "".join(residues[i] for i in range(length)), contacts


def oracle_pairs(record: dict) -> list[list[int]]:
    """Freeze a random half of resolved true contacts for an oracle prefix."""
    resolved = set(record["resolved"])
    contacts = sorted(
        [int(i), int(j)]
        for i, j, degree in record["contacts"]
        if degree >= 0.001 and j - i >= 6 and i in resolved and j in resolved
    )
    rng = np.random.default_rng(
        stable_seed(record["dataset"], record["stem"], "oracle")
    )
    chosen = sorted(
        rng.choice(len(contacts), len(contacts) // 2, replace=False).tolist()
    )
    return [contacts[index] for index in chosen]


def vote_matrix(rollouts: list[dict], length: int) -> np.ndarray:
    """Count each terminated rollout's live contact set once."""
    votes = np.zeros((length, length), dtype=np.float64)
    for rollout in rollouts:
        if rollout["finish_reason"] != "stop":
            continue
        for i, j in rollout["contacts"]:
            votes[i, j] += 1
            votes[j, i] += 1
    return votes
