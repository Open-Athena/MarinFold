# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare strict CE vs greedy set loss on a trained contacts-v1 model.

This is a Colab/CoreWeave smoke script, not the final Levanter training path. It
loads a recent MarinFold contacts-v1 HF export, runs a few teacher-forced toy
contacts-v1 documents, and checks that relaxing contact order/orientation lowers
(or ties) the contact-block loss on the same logits.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from marinfold.document_structures.contacts_v1.greedy_set_loss import (
    greedy_matched_contact_block_loss,
)
from marinfold.registry import resolve_model


ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


DOCS = [
    " ".join(
        [
            "<contacts-v1>",
            "<begin_sequence>",
            "<p0>", "<ALA>",
            "<p1>", "<GLY>",
            "<p2>", "<SER>",
            "<p3>", "<VAL>",
            "<p4>", "<THR>",
            "<p5>", "<LEU>",
            "<p6>", "<ASP>",
            "<p7>", "<LYS>",
            "<begin_statements>",
            "<contact>", "<p0>", "<p6>",
            "<contact>", "<p1>", "<p7>",
            "<end>",
            "<eos>",
        ]
    ),
    " ".join(
        [
            "<contacts-v1>",
            "<begin_sequence>",
            "<p10>", "<PHE>",
            "<p11>", "<ALA>",
            "<p12>", "<GLU>",
            "<p13>", "<TYR>",
            "<p14>", "<ASN>",
            "<p15>", "<ARG>",
            "<p16>", "<GLY>",
            "<begin_statements>",
            "<contact>", "<p10>", "<p16>",
            "<contact>", "<p11>", "<p15>",
            "<end>",
            "<eos>",
        ]
    ),
]


def _doc_from_sequence_and_contacts(
    sequence: str,
    contacts: list[list[int]],
    *,
    max_contacts: int | None,
) -> str:
    tokens = ["<contacts-v1>", "<begin_sequence>"]
    for index, code in enumerate(sequence.strip().upper()):
        tokens.extend([f"<p{index}>", f"<{ONE_TO_THREE.get(code, 'UNK')}>"])
    tokens.append("<begin_statements>")
    selected_contacts = contacts if max_contacts is None else contacts[:max_contacts]
    for left, right in selected_contacts:
        tokens.extend(["<contact>", f"<p{int(left)}>", f"<p{int(right)}>"])
    tokens.extend(["<end>", "<eos>"])
    return " ".join(tokens)


def _load_foldbench_jsonl(path: Path, *, max_contacts: int | None, limit: int) -> list[str]:
    docs: list[str] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            docs.append(
                _doc_from_sequence_and_contacts(
                    row["input_seq"],
                    row["contacts"],
                    max_contacts=max_contacts,
                )
            )
            if len(docs) >= limit:
                break
    if not docs:
        raise ValueError(f"no docs loaded from {path}")
    return docs


def _token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"token {token!r} is absent from tokenizer")
    return int(token_id)


def _encode_batch(docs: list[str], tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = _token_id(tokenizer, "<pad>")
    rows: list[list[int]] = []
    for doc in docs:
        row = [_token_id(tokenizer, token) for token in doc.split()]
        rows.append(row)
    width = max(len(row) for row in rows)
    input_ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(rows), width), dtype=np.int64)
    for row_index, row in enumerate(rows):
        input_ids[row_index, : len(row)] = row
        attention_mask[row_index, : len(row)] = 1
    return torch.as_tensor(input_ids), torch.as_tensor(attention_mask)


def _finite_log_softmax(logits: torch.Tensor, *, logit_clip: float | None) -> torch.Tensor:
    logits = logits.float()
    if logit_clip is not None:
        logits = torch.clamp(logits, min=-logit_clip, max=logit_clip)
    if not torch.isfinite(logits).all():
        bad = int((~torch.isfinite(logits)).sum().item())
        raise RuntimeError(f"model produced {bad} non-finite logits before log_softmax")
    log_probs = torch.log_softmax(logits, dim=-1)
    if not torch.isfinite(log_probs).all():
        bad = int((~torch.isfinite(log_probs)).sum().item())
        raise RuntimeError(f"log_softmax produced {bad} non-finite values")
    return log_probs


def _strict_loss(log_probs: np.ndarray, token_ids: np.ndarray, *, end_id: int) -> float:
    end_positions = np.flatnonzero(token_ids == end_id)
    if end_positions.size != 1:
        raise ValueError("expected exactly one <end>")
    end_pos = int(end_positions[0])
    positions = np.arange(end_pos, dtype=np.int64)
    targets = token_ids[1 : end_pos + 1]
    return -float(np.sum(log_probs[positions, targets]))


def _strict_loss_torch(
    logits: torch.Tensor,
    token_ids: np.ndarray,
    *,
    end_id: int,
    logit_clip: float | None,
) -> torch.Tensor:
    """Strict next-token CE through ``<end>`` from Torch logits."""
    log_probs = _finite_log_softmax(logits, logit_clip=logit_clip)
    row_losses: list[torch.Tensor] = []
    for row in range(token_ids.shape[0]):
        end_positions = np.flatnonzero(token_ids[row] == end_id)
        if end_positions.size != 1:
            raise ValueError("expected exactly one <end>")
        end_pos = int(end_positions[0])
        row_losses.append(-log_probs[row, np.arange(end_pos), token_ids[row, 1 : end_pos + 1]].sum())
    return torch.stack(row_losses).mean()


def _greedy_loss_torch(
    logits: torch.Tensor,
    token_ids: np.ndarray,
    tokenizer,
    *,
    logit_clip: float | None,
) -> torch.Tensor:
    """Compute hard-assigned greedy loss from Torch logits.

    Greedy assignments are chosen from detached host log-probs, but the returned
    loss gathers from Torch ``log_softmax(logits)``, so ``loss.backward()`` gives
    gradients for the selected token logits.
    """
    log_probs = _finite_log_softmax(logits, logit_clip=logit_clip)
    log_probs_np = log_probs.detach().cpu().numpy()
    position_token_ids = np.asarray([_token_id(tokenizer, f"<p{index}>") for index in range(2000)], dtype=np.int64)
    begin_id = _token_id(tokenizer, "<begin_statements>")
    contact_id = _token_id(tokenizer, "<contact>")
    end_id = _token_id(tokenizer, "<end>")

    row_losses: list[torch.Tensor] = []
    for row in range(token_ids.shape[0]):
        matched = greedy_matched_contact_block_loss(
            log_probs_np[row],
            token_ids[row],
            begin_statements_token_id=begin_id,
            contact_token_id=contact_id,
            end_token_id=end_id,
            position_token_ids=position_token_ids,
        )
        end_positions = np.flatnonzero(token_ids[row] == end_id)
        begin_positions = np.flatnonzero(token_ids[row] == begin_id)
        if end_positions.size != 1 or begin_positions.size != 1:
            raise ValueError("expected exactly one <begin_statements> and one <end>")
        begin_pos = int(begin_positions[0])
        end_pos = int(end_positions[0])

        row_loss = -log_probs[row, np.arange(begin_pos), token_ids[row, 1 : begin_pos + 1]].sum()
        for choice in matched.choices:
            slot_start = begin_pos + 1 + 3 * choice.slot_index
            left_token, right_token = choice.oriented_tokens
            row_loss = row_loss - log_probs[row, slot_start - 1, contact_id]
            row_loss = row_loss - log_probs[row, slot_start, left_token]
            row_loss = row_loss - log_probs[row, slot_start + 1, right_token]
        row_loss = row_loss - log_probs[row, end_pos - 1, end_id]
        row_losses.append(row_loss)
    return torch.stack(row_losses).mean()


def _logit_grad_stats(loss_fn, logits: torch.Tensor) -> tuple[float, float, float, int]:
    grad_logits = logits.detach().clone().requires_grad_(True)
    loss = loss_fn(grad_logits)
    loss.backward()
    grad = grad_logits.grad
    if grad is None:
        raise RuntimeError("gradient check failed: logits.grad is None")
    return (
        float(loss.detach().cpu()),
        float(grad.norm().detach().cpu()),
        float(grad.abs().mean().detach().cpu()),
        int((grad.abs() > 0).sum().item()),
    )


def _check_logits_gradient(
    logits: torch.Tensor,
    token_ids: np.ndarray,
    tokenizer,
    *,
    end_id: int,
    logit_clip: float | None,
) -> None:
    strict_stats = _logit_grad_stats(
        lambda grad_logits: _strict_loss_torch(
            grad_logits,
            token_ids,
            end_id=end_id,
            logit_clip=logit_clip,
        ),
        logits,
    )
    greedy_stats = _logit_grad_stats(
        lambda grad_logits: _greedy_loss_torch(
            grad_logits,
            token_ids,
            tokenizer,
            logit_clip=logit_clip,
        ),
        logits,
    )
    print("gradient check:")
    print(
        "  strict:     "
        f"loss={strict_stats[0]:.4f} grad_norm={strict_stats[1]:.6f} "
        f"mean_abs_grad={strict_stats[2]:.8f} nonzero_logit_grads={strict_stats[3]}"
    )
    print(
        "  greedy_set: "
        f"loss={greedy_stats[0]:.4f} grad_norm={greedy_stats[1]:.6f} "
        f"mean_abs_grad={greedy_stats[2]:.8f} nonzero_logit_grads={greedy_stats[3]}"
    )
    print(
        "  delta:      "
        f"loss={greedy_stats[0] - strict_stats[0]:.4f} "
        f"grad_norm={greedy_stats[1] - strict_stats[1]:.6f}"
    )


def compare_once(args: argparse.Namespace) -> None:
    model_path = Path(args.model) if Path(args.model).is_dir() else resolve_model(args.model)
    print(f"loading model from {model_path}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json"),
        unk_token="<UNK>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    dtype_by_name = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_by_name[args.dtype],
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    model.eval()

    docs = DOCS[: args.num_examples]
    if args.foldbench_jsonl is not None:
        docs = _load_foldbench_jsonl(
            Path(args.foldbench_jsonl),
            max_contacts=args.max_contacts,
            limit=args.num_examples,
        )
        print(
            f"loaded {len(docs)} real docs from {args.foldbench_jsonl} "
            f"with max_contacts={args.max_contacts}"
        )
    input_ids, attention_mask = _encode_batch(docs, tokenizer)
    print(f"batch shape: {tuple(input_ids.shape)}")
    first_device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(first_device),
            attention_mask=attention_mask.to(first_device),
            use_cache=False,
        )
        logits = outputs.logits
        log_probs = _finite_log_softmax(logits, logit_clip=args.logit_clip).cpu().numpy()
    token_ids = input_ids.numpy()

    position_token_ids = np.asarray([_token_id(tokenizer, f"<p{index}>") for index in range(2000)], dtype=np.int64)
    begin_id = _token_id(tokenizer, "<begin_statements>")
    contact_id = _token_id(tokenizer, "<contact>")
    end_id = _token_id(tokenizer, "<end>")

    strict_losses = []
    greedy_losses = []
    for row in range(token_ids.shape[0]):
        strict = _strict_loss(log_probs[row], token_ids[row], end_id=end_id)
        greedy = greedy_matched_contact_block_loss(
            log_probs[row],
            token_ids[row],
            begin_statements_token_id=begin_id,
            contact_token_id=contact_id,
            end_token_id=end_id,
            position_token_ids=position_token_ids,
        ).loss
        strict_losses.append(strict)
        greedy_losses.append(greedy)
        print(f"doc {row}: strict={strict:.4f} greedy_set={greedy:.4f} delta={greedy - strict:.4f}")
    strict_mean = float(np.mean(strict_losses))
    greedy_mean = float(np.mean(greedy_losses))
    print(f"mean: strict={strict_mean:.4f} greedy_set={greedy_mean:.4f} delta={greedy_mean - strict_mean:.4f}")
    if args.check_gradient:
        _check_logits_gradient(
            logits,
            token_ids,
            tokenizer,
            end_id=end_id,
            logit_clip=args.logit_clip,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="contacts-v1-exp120-1.5B",
        help="MODELS.yaml nickname or local HF model directory.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Use eager attention by default for smoke-test numerical stability.",
    )
    parser.add_argument("--num-examples", type=int, default=len(DOCS))
    parser.add_argument(
        "--foldbench-jsonl",
        help="Optional JSONL with input_seq and contacts columns, e.g. exp82 foldbench_dev.jsonl.",
    )
    parser.add_argument(
        "--max-contacts",
        type=int,
        default=None,
        help="Use only the first N contacts from each JSONL row to keep T4 smoke memory bounded.",
    )
    parser.add_argument(
        "--logit-clip",
        type=float,
        default=50.0,
        help="Clamp model logits before log_softmax for fp16 smoke stability. Use 'nan' to disable.",
    )
    parser.add_argument(
        "--check-gradient",
        action="store_true",
        help="Backprop through the Torch gather loss to the logits. This avoids full model backprop/OOM.",
    )
    args = parser.parse_args()
    if math.isnan(args.logit_clip):
        args.logit_clip = None
    compare_once(args)


if __name__ == "__main__":
    main()
