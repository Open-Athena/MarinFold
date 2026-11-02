# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Assert a prepared export is what the vLLM eval workers will actually load (#160).

``prepare_eval_model.py`` runs in a marin venv (transformers 5.x, no torch) and
rewrites the config as plain JSON, so nothing there proves a **transformers
4.57** runtime — the one the eval workers use — reads it the way we intend.
This script is that proof, and it is deliberately a separate file because it
must run in a separate environment (exp159's venv, transformers 4.57.6).

The checks are the four failure modes that have actually cost runs:

1. **rope silently defaulting** — a 5.x ``rope_parameters`` block that a 4.x
   config ignores leaves ``rope_scaling=None``, which degrades predictions
   *worse as the protein gets longer*: the shape most easily mistaken for a
   real finding about long proteins.
2. **unresolvable tokenizer class** — ``TokenizersBackend`` is a levanter
   export name ``AutoTokenizer`` cannot import.
3. **fp32 on disk** — the TPU ragged-paged-attention kernel throws on fp32
   values rather than casting.
4. **vocab drift** — the whole append-only-vocab argument of #158 rests on the
   embedding table being the size the tokenizer says, with ``<retract>`` last.

    <exp159>/.venv/bin/python verify_eval_model.py --dir <prepared dir> \\
        --expect-vocab 3849 --expect-retract
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_ROPE_TYPE = "llama3"


def check_config(d: Path, *, expect_vocab: int | None) -> dict:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(d))
    problems = []
    scaling = getattr(cfg, "rope_scaling", None)
    if not scaling:
        problems.append("rope_scaling is empty — the llama3 rope was lost")
    elif scaling.get("rope_type") != EXPECTED_ROPE_TYPE:
        problems.append(f"rope_type={scaling.get('rope_type')!r} != {EXPECTED_ROPE_TYPE!r}")
    if not getattr(cfg, "rope_theta", None):
        problems.append("rope_theta missing")
    if expect_vocab is not None and cfg.vocab_size != expect_vocab:
        problems.append(f"config vocab_size={cfg.vocab_size} != {expect_vocab}")
    print(f"[verify] config: {cfg.model_type} layers={cfg.num_hidden_layers} "
          f"hidden={cfg.hidden_size} vocab={cfg.vocab_size} "
          f"rope_theta={getattr(cfg, 'rope_theta', None)} rope_scaling={scaling}")
    return {"cfg": cfg, "problems": problems}


def check_tokenizer(d: Path, *, vocab_size: int, expect_retract: bool) -> list[str]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(d))
    problems = []
    # An unknown token does not come back as None — it comes back as the UNK id,
    # so `is not None` would report every pre-retract tokenizer as having
    # `<retract>`. Compare against unk_token_id instead.
    unk = tok.unk_token_id

    def resolve(token: str) -> int | None:
        tid = tok.convert_tokens_to_ids(token)
        return None if tid is None or tid < 0 or tid == unk else tid

    end_id = resolve("<end>")
    if end_id is None:
        problems.append("<end> does not resolve — rollouts would never stop")
    retract_id = resolve("<retract>")
    has_retract = retract_id is not None
    if expect_retract and not has_retract:
        problems.append("<retract> does not resolve in a retraction-trained model's tokenizer")
    if len(tok) > vocab_size:
        problems.append(f"tokenizer len={len(tok)} exceeds embedding rows {vocab_size}")
    print(f"[verify] tokenizer: {type(tok).__name__} len={len(tok)} "
          f"<end>={end_id} <retract>={retract_id if has_retract else 'absent'}")
    return problems


def check_weights(d: Path) -> list[str]:
    from safetensors import safe_open

    index = json.loads((d / "model.safetensors.index.json").read_text())
    problems, dtypes, total = [], set(), 0
    for shard in sorted(set(index["weight_map"].values())):
        path = d / shard
        if not path.exists():
            problems.append(f"missing shard {shard}")
            continue
        total += path.stat().st_size
        with safe_open(str(path), framework="np") as fh:
            for name in fh.keys():                                  # noqa: SIM118
                dtypes.add(fh.get_slice(name).get_dtype())
    floating = {t for t in dtypes if t not in {"I8", "I16", "I32", "I64", "U8", "BOOL"}}
    if floating - {"BF16"}:
        problems.append(f"non-bf16 floating tensors on disk: {sorted(floating)}")
    declared = index.get("metadata", {}).get("total_size")
    if declared is not None and declared != total:
        problems.append(f"index total_size={declared} != on-disk {total}")
    print(f"[verify] weights: {len(index['weight_map'])} tensors, dtypes={sorted(dtypes)}, "
          f"{total / 2**30:.2f} GiB")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--expect-vocab", type=int, default=None)
    ap.add_argument("--expect-retract", action="store_true")
    a = ap.parse_args()

    result = check_config(a.dir, expect_vocab=a.expect_vocab)
    problems = list(result["problems"])
    problems += check_tokenizer(a.dir, vocab_size=result["cfg"].vocab_size,
                                expect_retract=a.expect_retract)
    problems += check_weights(a.dir)

    if problems:
        print(f"[verify] FAILED ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"[verify] OK — {a.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
