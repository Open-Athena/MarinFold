# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate the eval on the prepared checkpoints actually being loadable and comparable.

Three checkpoints are scored against one fixed set of contacts-v1 prompts, so a
silent tokenizer or config discrepancy between them would not fail loudly — it
would just produce wrong, confidently-reported R-precision. This checks the
things that would do that, against the reference checkpoint whose numbers are
already published (the #117 final export used in exp82's `where_we_stand` run):

* **vocabulary identity** — every token id must map to the same token string in
  every checkpoint. contacts-v1 documents address residues by `<pN>` tokens; a
  one-id shift silently renames every position.
* **special tokens** — `<end>` (the rollout stop token), eos and pad ids.
* **config compatibility** — the rope scaling and dtype-relevant fields the
  transformers-5 -> 4.57 downgrade touches.
* **weights** — a real forward pass on a real contacts-v1 prompt, checking the
  logits are finite and that the model puts most of its mass on the token
  classes the format allows at that position.

    uv run python verify_prepared_exports.py --model label=dir [--model ...]
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from marinfold.document_structures.contacts_v1 import (
    GenerationConfig, build_document, residues_from_sequence,
)
from marinfold.inference._tokenizer import load_tokenizer

BEGIN = "<begin_statements>"
# A real eval-set sequence (FoldBench 1UBQ, ubiquitin) — short enough to run a
# forward pass on CPU, real enough that the prompt exercises the actual format.
PROBE_SEQ = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG")


def check_tokenizer(label: str, path: Path, reference: dict | None) -> dict:
    tokenizer = load_tokenizer(path)
    vocab = tokenizer.get_vocab()
    end_id = tokenizer.convert_tokens_to_ids("<end>")
    if end_id is None or end_id < 0:
        raise SystemExit(f"{label}: no <end> token — the rollout stop token is missing")
    facts = dict(size=len(vocab), end_id=end_id,
                 eos_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id,
                 vocab=vocab)
    print(f"  tokenizer: {len(vocab)} tokens, <end>={end_id}, "
          f"eos={facts['eos_id']}, pad={facts['pad_id']}")
    if reference is not None and vocab != reference["vocab"]:
        differing = [t for t in set(vocab) | set(reference["vocab"])
                     if vocab.get(t) != reference["vocab"].get(t)]
        raise SystemExit(f"{label}: vocabulary differs from the reference at "
                         f"{len(differing)} token(s), e.g. {differing[:5]}")
    return facts


def check_config(label: str, path: Path) -> None:
    config = AutoConfig.from_pretrained(str(path))
    scaling = config.rope_scaling
    if not scaling or scaling.get("rope_type") != "llama3":
        raise SystemExit(f"{label}: rope_scaling is {scaling!r}, expected the llama3 "
                         f"config the model was trained with")
    raw = json.loads((path / "config.json").read_text())
    if "rope_parameters" in raw:
        raise SystemExit(f"{label}: config.json still carries the transformers-5 "
                         f"`rope_parameters` block; vLLM 0.9.2 would ignore rope scaling")
    print(f"  config: {config.num_hidden_layers}L hidden={config.hidden_size} "
          f"heads={config.num_attention_heads}/{config.num_key_value_heads} "
          f"vocab={config.vocab_size} rope_theta={config.rope_theta} "
          f"rope_type={scaling['rope_type']}")


def check_forward(label: str, path: Path) -> None:
    """Run one real contacts-v1 prompt through the weights on CPU."""
    tokenizer = load_tokenizer(path)
    residues = residues_from_sequence(PROBE_SEQ)
    document = build_document("probe", residues, [], config=GenerationConfig())
    prompt = document.document[: document.document.index(BEGIN) + len(BEGIN)]
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    if not torch.isfinite(logits).all():
        raise SystemExit(f"{label}: non-finite logits from the forward pass")

    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, 5)
    names = tokenizer.convert_ids_to_tokens(top.indices.tolist())
    # After <begin_statements> the format admits only <contact> or <end>; a
    # correctly-loaded checkpoint puts essentially all its mass there.
    allowed = tokenizer.convert_tokens_to_ids(["<contact>", "<end>"])
    mass = float(probs[allowed].sum())
    print(f"  forward: prompt={ids.shape[1]} tokens, top-5 = "
          + ", ".join(f"{n}:{p:.3f}" for n, p in zip(names, top.values.tolist()))
          + f" | P(<contact>|<end>) = {mass:.4f}")
    if mass < 0.9:
        raise SystemExit(f"{label}: only {mass:.3f} of the next-token mass is on the "
                         f"format-legal tokens — the weights or tokenizer are mismatched")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=DIR",
                    help="repeatable; the FIRST one is the reference")
    a = ap.parse_args()

    reference = None
    for spec in a.model:
        label, _, directory = spec.partition("=")
        path = Path(directory)
        print(f"== {label}  ({path})")
        facts = check_tokenizer(label, path, reference)
        check_config(label, path)
        check_forward(label, path)
        if reference is None:
            reference = facts
            print("  (reference for vocabulary comparison)")
        elif (facts["end_id"], facts["eos_id"], facts["pad_id"]) != (
              reference["end_id"], reference["eos_id"], reference["pad_id"]):
            raise SystemExit(f"{label}: special-token ids differ from the reference")

    print("\nAll checkpoints share one vocabulary and load with llama3 rope. OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
