"""Frozen eval-val inputs and tokenizer-aware rollout scoring contract."""

import csv
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow.parquet as pq
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    live_contacts,
    residues_from_sequence,
)
from transformers import PreTrainedTokenizerBase

from common import PROMPT

INPUT_HASHES = {
    "eval_sets.csv": "b13d060a091240921bc8466acecc9fa6ccbb45a56de4efda02cd903f6abf9861",
    "gt_universe_scored.jsonl": "f30c23e3d2fbab245755fc01548388b41730ddfa45da87325539698cadb153e5",
    "eval_targets_foldbench_monomers.parquet": "2eb4f1fee148fe2d6601bd171ef6e9431b96f38c82eaed1ad119a069a13f1fb8",
}
N_ROLLOUTS = 100


def freeze_eval_val(source: Path, destination: Path) -> list[dict]:
    """Filter the pinned published universe before sending inputs to workers."""
    for name, expected in INPUT_HASHES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Evaluation input digest changed: {name}")
    with (source / "eval_sets.csv").open() as handle:
        membership = {
            row["stem"]: row
            for row in csv.DictReader(handle)
            if row["eval_set"] == "eval-val"
        }
    targets = {
        row["stem"]: row
        for row in pq.read_table(
            source / "eval_targets_foldbench_monomers.parquet"
        ).to_pylist()
        if row["stem"] in membership
    }
    records = []
    with (source / "gt_universe_scored.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["stem"] not in membership:
                continue
            target = targets[row["stem"]]
            if row["L"] != target["L"] or len(target["input_seq"]) != row["L"]:
                raise ValueError(
                    f"Sequence/ground-truth length mismatch: {row['stem']}"
                )
            records.append(
                {
                    **row,
                    "input_seq": target["input_seq"],
                    "eval_set": "eval-val",
                    "is_viral": membership[row["stem"]]["is_viral"],
                }
            )
    if len(records) != 97 or len({r["stem"] for r in records}) != 97:
        raise ValueError("Expected exactly 97 unique eval-val proteins")
    records.sort(key=lambda row: (row["L"], row["dataset"], row["stem"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row) + "\n" for row in records))
    return records


def rollout_prompt(record: dict, document_format: str, rollout: int) -> tuple[str, int]:
    """Build a fresh canonical realization, or the fixed ordinary-sequence prompt."""
    if document_format == "prompted":
        return PROMPT + record["input_seq"] + "\n\nContacts:\n", 0
    document = build_document(
        f"{record['stem']}:r{rollout}",
        residues_from_sequence(record["input_seq"]),
        [],
        config=GenerationConfig(),
    )
    if document is None:
        raise ValueError(f"Unrepresentable eval sequence: {record['stem']}")
    prefix, separator, _ = document.document.partition("<begin_statements>")
    if not separator:
        raise ValueError("Document builder omitted structure boundary")
    return prefix + separator, document.n_term_index


def generation_budget(
    tokenizer: PreTrainedTokenizerBase, length: int, document_format: str
) -> int:
    """Translate the 6L+128 domain-token allowance to the checkpoint vocabulary.

    The original protein tokenizer uses one token per domain symbol. Qwen splits
    those symbols and uses several tokens per numeric pair. Budget enough native
    tokens for the same number of three-symbol contact statements, with a small
    ending allowance. This is a declared tokenizer adaptation, not an assertion
    that numeric prompts have the original resampling semantics.
    """
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    logical = 6 * length + 128
    if document_format == "contacts_v1":
        if all(len(encode(t)) == 1 for t in ["<contact>", "<p0>", "<p1999>", "<end>"]):
            return logical
        position_max = max(len(encode(f" <p{i}>")) for i in range(2000))
        statement_max = len(encode(" <contact>")) + 2 * position_max
    else:
        number_max = max(len(encode(str(i))) for i in range(1, length + 1))
        statement_max = 2 * number_max + len(encode(" ")) + len(encode("\n"))
    return math.ceil(logical / 3) * statement_max + 64


def contact_votes(
    completions: list[str], starts: list[int], length: int, document_format: str
) -> np.ndarray:
    """Vote once per live, valid pair in each complete rollout, in sequence coordinates."""
    votes = np.zeros((length, length), dtype=np.int32)
    for text, start in zip(completions, starts, strict=True):
        if document_format == "contacts_v1":
            pairs = {
                tuple(sorted(((a - start) % 2000, (b - start) % 2000)))
                for a, b in live_contacts(text)
                if 0 <= a < 2000 and 0 <= b < 2000
            }
        else:
            pairs = {
                tuple(sorted((int(a) - 1, int(b) - 1)))
                for a, b in re.findall(
                    r"^\s*(\d+)[ \t]+(\d+)[ \t]*$", text, re.MULTILINE
                )
            }
        for a, b in pairs:
            if 0 <= a < b < length and b - a >= 6:
                votes[a, b] += 1
                votes[b, a] += 1
    return votes


def load_metric_reference(path: Path) -> ModuleType:
    """Load the unchanged exp89 measurement implementation supplied by the launcher."""
    spec = importlib.util.spec_from_file_location("exp89_metric_reference", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load metric reference: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_votes(votes: np.ndarray, record: dict, reference: ModuleType) -> list[dict]:
    """Score the frozen resolved-residue universe with exp89, without reimplementation."""
    length = record["L"]
    if votes.shape != (length, length) or not np.array_equal(votes, votes.T):
        raise ValueError("Vote matrix must be symmetric and match the input sequence")
    pi, pj, psep = reference.resolved_pairs(
        np.asarray(record["resolved"], dtype=np.int64)
    )
    truth = reference.true_matrix(length, record["contacts"])
    return reference.metric_rows(
        votes, truth, pi, pj, psep, length, with_precision=True
    )
