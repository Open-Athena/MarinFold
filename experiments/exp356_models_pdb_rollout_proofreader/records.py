"""Preserve rollout evidence while deriving prefix-local classification targets.

Labels never enter the model's input. Cropping slices the actual token sequence
before encoding; later contact tokens and true completion status are unavailable
to an earlier prefix. Precision and recall count unique, eligible contact pairs.
"""

import hashlib
import random
from dataclasses import dataclass

import numpy as np

AA = dict(zip('ARNDCQEGHILKMFPSTWYV', 'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(), strict=True))
AA['X'] = 'UNK'
CONTEXT = 8192


def stable_seed(value: str) -> int:
    """Return a reproducible sampler seed."""
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'big') & 0x7FFFFFFF


def make_prompt(sequence: str, identity: str) -> tuple[str, int]:
    """Resample contacts-v1 sequence order and its circular residue numbering."""
    if not 1 <= len(sequence) <= 2000:
        raise ValueError('Sequence is outside the contacts-v1 residue ring')
    rng = random.Random(stable_seed(identity))
    offset = rng.randrange(2000)
    statements = [f'<p{(offset+i)%2000}> <{AA[a]}>' for i, a in enumerate(sequence)]
    statements += [f'<n-term> <p{offset}>', f'<c-term> <p{(offset+len(sequence)-1)%2000}>']
    rng.shuffle(statements)
    return '<contacts-v1> <begin_sequence> ' + ' '.join(statements) + ' <begin_statements>', offset


def parse_contacts(tokens: list[str], offset: int, length: int, ground_truth: list[list[int]]) -> dict:
    """Parse every emitted triple, retaining diagnostics and duplicate identity."""
    positions, labels, unique, pairs = [], [], [], []
    gt = {tuple(pair) for pair in ground_truth}
    seen = set()
    malformed = 0
    invalid = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {'<end>', '<think>'}:
            index += 1
            continue
        if token != '<contact>' or index + 2 >= len(tokens):
            malformed += 1
            index += 1
            continue
        endpoints = tokens[index+1:index+3]
        if not all(t.startswith('<p') and t.endswith('>') and t[2:-1].isdigit() for t in endpoints):
            malformed += 1
            index += 1
            continue
        a, b = sorted((int(t[2:-1]) - offset) % 2000 for t in endpoints)
        eligible = 0 <= a < b < length and b - a >= 6
        invalid += int(not eligible)
        pair = (a, b)
        positions.append(index + 2)
        labels.append(float(eligible and pair in gt))
        unique.append(pair not in seen)
        pairs.append([a, b])
        seen.add(pair)
        index += 3
    return dict(contact_ends=positions, labels=labels, unique=unique, pairs=pairs,
                malformed=malformed, invalid_contacts=invalid)


def prefix_length(n_contacts: int, identity: str, epoch: int) -> int:
    """Mix complete, short and broadly distributed prefixes deterministically."""
    if n_contacts < 1:
        raise ValueError('A proofreading example needs at least one contact')
    rng = random.Random(stable_seed(f'{identity}:epoch:{epoch}'))
    draw = rng.random()
    if draw < 0.25:
        return n_contacts
    if draw < 0.50:
        return rng.choice([k for k in (1, 2, 4, 8) if k <= n_contacts])
    return rng.randint(1, n_contacts)


@dataclass(frozen=True)
class Example:
    """One physically cropped input and its separate supervised targets."""

    input_ids: list[int]
    contact_positions: list[int]
    labels: list[float]
    unique: list[bool]
    recall: float
    precision: float
    identity: str


def make_example(row: dict, k: int, assessment_id: int) -> Example:
    """Crop at a complete contact boundary; expose EOS only if genuinely seen."""
    count = len(row['contact_ends'])
    if not 1 <= k <= count:
        raise ValueError(f'Prefix {k} outside 1..{count}')
    end = row['contact_ends'][k-1] + 1
    completion = row['completion_ids'][:end]
    if k == count and row['finished']:
        completion = row['completion_ids']
    prompt = row['prompt_ids']
    ids = prompt + completion + [assessment_id]
    if len(ids) > CONTEXT:
        raise ValueError('Proofreader input exceeds the verified context limit')
    labels = row['labels'][:k]
    unique = row['unique'][:k]
    tp = sum(y for y, use in zip(labels, unique, strict=True) if use)
    predicted = sum(unique)
    return Example(ids, [len(prompt)+p for p in row['contact_ends'][:k]], labels, unique,
                   tp / row['gt_count'], tp / predicted, row['identity'])


def collate(examples: list[Example], pad_id: int) -> dict[str, np.ndarray]:
    """Right-pad inputs and readout indices; masks exclude all padded labels."""
    batch = len(examples)
    width = max(len(e.input_ids) for e in examples)
    contacts = max(len(e.labels) for e in examples)
    ids = np.full((batch, width), pad_id, dtype=np.int64)
    mask = np.zeros((batch, width), dtype=bool)
    positions = np.zeros((batch, contacts), dtype=np.int64)
    labels = np.zeros((batch, contacts), dtype=np.float32)
    valid = np.zeros((batch, contacts), dtype=bool)
    unique = np.zeros((batch, contacts), dtype=bool)
    for i, example in enumerate(examples):
        n, k = len(example.input_ids), len(example.labels)
        ids[i, :n] = example.input_ids
        mask[i, :n] = True
        positions[i, :k] = example.contact_positions
        labels[i, :k] = example.labels
        valid[i, :k] = True
        unique[i, :k] = example.unique
    return dict(input_ids=ids, token_mask=mask, contact_positions=positions,
                labels=labels, contact_mask=valid, unique_mask=unique,
                recall=np.array([e.recall for e in examples], dtype=np.float32),
                precision=np.array([e.precision for e in examples], dtype=np.float32))
