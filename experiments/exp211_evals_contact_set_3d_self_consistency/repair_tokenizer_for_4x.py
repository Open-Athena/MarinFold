# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Make the #199 export's tokenizer loadable by transformers 4.x (issue #211).

#199's published ``tokenizer_config.json`` declares
``"tokenizer_class": "TokenizersBackend"``, a transformers **5** class. Under
transformers 4.x the load fails hard (unlike the rope bug in the same export,
which failed *silently* until #180 caught it). That matters here because the
only vLLM new enough to understand transformers 5 ships a torch that needs
CUDA driver >= 12.8, and this workstation runs 12.2 — so the local run has to
be vLLM 0.9.2 + transformers 4.x, and the tokenizer has to meet it there.

The fix is a config rewrite, not a retokenization. The vocabulary and merges
live in ``tokenizer.json``, which is a plain ``tokenizers`` library file and is
backend-agnostic; only the *class name* the config asks transformers to
instantiate is version-specific. Rewriting it to ``PreTrainedTokenizerFast``
(and dropping the transformers-5-only ``backend`` / ``is_local`` /
``local_files_only`` keys) leaves every token id untouched.

``--verify`` proves that rather than asserting it: it tokenizes a real
contacts-v1 document under the repaired config and compares the ids against a
reference id list produced by the transformers-5 load, failing if a single
token moved. Nothing downstream is trustworthy if the ids shift — a contacts-v1
document is almost entirely special tokens, so a one-id offset would silently
turn every position token into its neighbour.

    uv run python repair_tokenizer_for_4x.py --model-dir _scratch/model_exp199
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# transformers-5-only keys that 4.x does not understand.
DROP_KEYS = ("backend", "is_local", "local_files_only")


def repair(model_dir: Path) -> dict:
    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("tokenizer_class") != "TokenizersBackend":
        print(f"[repair] {cfg_path} already declares "
              f"{cfg.get('tokenizer_class')!r}; nothing to do")
        return cfg

    backup = model_dir / "tokenizer_config.transformers5.json"
    if not backup.exists():
        shutil.copy2(cfg_path, backup)

    out = {k: v for k, v in cfg.items() if k not in DROP_KEYS}
    out["tokenizer_class"] = "PreTrainedTokenizerFast"
    # model_max_length in the export is the int64 sentinel; keep the real
    # context length so nothing downstream silently truncates.
    out["model_max_length"] = 8192
    cfg_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[repair] rewrote {cfg_path}: TokenizersBackend -> "
          f"PreTrainedTokenizerFast (original kept at {backup.name})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("_scratch/model_exp199"))
    ap.add_argument("--verify-against", type=Path, default=None,
                    help="JSON list of reference token ids from the transformers-5 load")
    ap.add_argument("--emit-reference", type=Path, default=None,
                    help="tokenize the probe document and dump ids here (run under "
                         "transformers 5, before repairing)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig, build_document, residues_from_sequence,
    )

    # A real contacts-v1 document: doc-type token, section markers, position
    # tokens, amino acids, contact statements. Almost all special tokens, which
    # is exactly where an id shift would hide.
    probe = build_document(
        "exp211-tokenizer-probe",
        residues_from_sequence("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"),
        [], config=GenerationConfig(),
    ).document

    if args.emit_reference:
        tok = AutoTokenizer.from_pretrained(str(args.model_dir))
        ids = tok(probe, add_special_tokens=False).input_ids
        args.emit_reference.write_text(json.dumps(
            {"ids": ids, "end_id": tok.convert_tokens_to_ids("<end>"),
             "vocab_size": len(tok)}))
        print(f"[repair] reference: {len(ids)} ids, <end>="
              f"{tok.convert_tokens_to_ids('<end>')}, vocab {len(tok)} "
              f"-> {args.emit_reference}")
        return 0

    repair(args.model_dir)

    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    ids = tok(probe, add_special_tokens=False).input_ids
    end_id = tok.convert_tokens_to_ids("<end>")
    print(f"[repair] loaded under transformers 4.x: {len(ids)} ids, "
          f"<end>={end_id}, vocab {len(tok)}")
    assert end_id is not None and end_id >= 0, "<end> missing after repair"

    if args.verify_against:
        ref = json.loads(args.verify_against.read_text())
        if ids != ref["ids"]:
            n_diff = sum(a != b for a, b in zip(ids, ref["ids"]))
            raise SystemExit(
                f"TOKEN IDS MOVED: {n_diff}/{len(ids)} differ from the "
                f"transformers-5 reference. The repair is not id-preserving; "
                f"do not use this checkpoint."
            )
        if end_id != ref["end_id"] or len(tok) != ref["vocab_size"]:
            raise SystemExit("<end> id or vocab size changed after repair")
        print(f"[repair] VERIFIED id-preserving: all {len(ids)} ids match the "
              f"transformers-5 reference, <end> and vocab size unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
