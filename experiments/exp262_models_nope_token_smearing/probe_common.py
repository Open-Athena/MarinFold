# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the issue #262 Phase 0 probes.

Builds *ground-truth* contacts-v1 documents (real sequence, real contacts) for
the exp245 FoldBench monomers, and loads a checkpoint for teacher-forced
analysis. Ground truth rather than rollouts on purpose: Phase 0 asks what the
model does while reading a correct document, not how well it writes one.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from marinfold.document_structures.contacts_v1.generate import GenerationConfig, build_document
from marinfold.document_structures.contacts_v1.inference import residues_from_sequence
from marinfold.document_structures.contacts_v1.parse import RawContact
from marinfold.inference._tokenizer import load_tokenizer, model_source_path
from marinfold.registry import resolve_model
from transformers import AutoModelForCausalLM

EXP245_DATA = Path(__file__).resolve().parent.parent / "exp245_evals_foldbench_held_out_monomers" / "data"
BEGIN_STATEMENTS = "<begin_statements>"


@dataclass(frozen=True)
class ProbeDocument:
    """One teacher-forced contacts-v1 document with its section boundary."""

    stem: str
    residue_count: int
    tokens: list[str]
    token_ids: list[int]
    statements_start: int  # index of <begin_statements> in token_ids
    contact_statements: int


def load_targets() -> pd.DataFrame:
    """The exp245 monomer universe joined to its ground-truth contacts."""
    targets = pd.read_parquet(EXP245_DATA / "eval_targets_foldbench_monomers.parquet")
    ground_truth = {}
    with (EXP245_DATA / "gt_universe_foldbench_monomers.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            ground_truth[record["stem"]] = record
    targets["contacts"] = targets["stem"].map(lambda stem: ground_truth[stem]["contacts"])
    return targets


def build_probe_document(stem: str, sequence: str, contacts: list, tokenizer, seed_suffix: str = "") -> ProbeDocument:
    """Serialize one ground-truth document and tokenize it.

    ``contacts`` arrive from ``gt_universe`` as 1-based ``[i, j, degree]``
    triples over the input sequence; ``RawContact`` wants 0-based indices.
    """
    residues = residues_from_sequence(sequence)
    raw = [RawContact(first - 1, second - 1, degree) for first, second, degree in contacts]
    result = build_document(f"{stem}{seed_suffix}", residues, raw, config=GenerationConfig())
    if result is None:
        raise ValueError(f"{stem}: build_document rejected the ground-truth structure")
    tokens = result.document.split()
    token_ids = tokenizer(result.document, add_special_tokens=False).input_ids
    if len(token_ids) != len(tokens):
        raise ValueError(
            f"{stem}: tokenizer produced {len(token_ids)} ids for {len(tokens)} whitespace tokens; "
            "the contacts-v1 vocabulary should be one id per token"
        )
    statements_start = tokens.index(BEGIN_STATEMENTS)
    return ProbeDocument(
        stem=stem,
        residue_count=len(residues),
        tokens=tokens,
        token_ids=token_ids,
        statements_start=statements_start,
        contact_statements=(len(tokens) - statements_start - 2) // 3,
    )


def load_model(model_spec: str | None, *, attn_implementation: str, device: str = "cuda"):
    """Resolve a MODELS.yaml nickname (or ``None`` for the default) and load it.

    Goes through ``model_source_path`` / ``load_tokenizer`` rather than plain
    ``AutoModelForCausalLM.from_pretrained``: the published exports were written
    by transformers 5.x, whose ``rope_parameters`` block the pinned 4.x silently
    ignores (loading rope_theta 10000 for a model trained at 500000) and whose
    ``tokenizer_class: TokenizersBackend`` 4.x rejects outright. Both are
    repaired in an overlay directory by the marinfold inference helpers.
    """
    directory = Path(resolve_model(model_spec))
    effective = Path(model_source_path(directory))
    tokenizer = load_tokenizer(effective)
    model = AutoModelForCausalLM.from_pretrained(
        effective,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    model.eval().to(device)
    rope = model.config.rope_scaling or {}
    if model.config.rope_theta != 500_000 or rope.get("rope_type") != "llama3":
        raise ValueError(
            f"rope did not survive loading: theta={model.config.rope_theta} scaling={rope}"
        )
    return effective, tokenizer, model
