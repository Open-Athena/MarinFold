# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Make #230's published checkpoint loadable by SkyRL — issue #237.

The RL warm start is #230's export. Three things have to be true of it before a
training run reads it, and none of them is true of the raw directory:

1. **rope must be readable by whatever reads it.** levanter's HF export writes
   rope in the transformers-5 form only — ``rope_parameters`` present, top-level
   ``rope_theta`` and ``rope_scaling`` absent. A transformers-4.x reader silently
   falls back to a default rope and the model loses 0.76 nats/token with no error
   anywhere; that bug already forced one retraction in #163. SkyRL's venv is
   transformers 5.8 and reads the new form, but its vLLM engines are a separate
   code path, and "probably fine" is not a thing to discover 40 minutes into a
   run. This restores the 4.x keys *from* ``rope_parameters`` and refuses to
   proceed if it cannot. #230's own publisher does exactly this; the logic is
   reproduced rather than imported because the two experiments must not share a
   mutable dependency.

2. **The tokenizer must carry a pass-through chat template.** SkyRL's
   `PromptDataset` templates through the *tokenizer*, not through
   `generator.chat_template`, so a checkpoint with no template renders an empty
   prompt — and the max-prompt-length filter *passes* those rows precisely
   because they tokenize to zero. #208 lost an hour to it. The template emits
   message content verbatim; `--verify` asserts the render is token-identical to
   the raw string, because a template that prepends anything shifts every ``<pN>``
   off the position it was trained at and produces fluent nonsense rather than an
   error.

3. **bf16.** #230's export is fp32 (5.89 GB). The engines run bf16 regardless and
   the policy is trained in bf16, so the cast costs nothing and halves every load.

Also verified, because getting it wrong means multi mode silently means something
else: vocab is 2,845 and **token id 7 is ``<contacts-v1.multi>``**.

    python prepare_model.py --src ~/exp230_data/checkpoints/hf/step-1988 \\
        --out ~/exp237_data/model/exp230_step1988_bf16 --verify
"""

import argparse
import json
import shutil
from pathlib import Path

MULTI_TOKEN, MULTI_ID, VOCAB = "<contacts-v1.multi>", 7, 2845
TEMPLATE_NAME = "contacts_v1_passthrough.jinja"
PROBES = [
    "<contacts-v1.multi> <begin_sequence> <p17> <ALA> <p18> <GLY> <begin_statements>",
    "<contacts-v1.multi> <n-term> <p1382> <begin_statements> <contact> <p17> <p25> <end>",
]


def repair_rope(cfg: dict) -> list[str]:
    """Restore the transformers-4.x rope keys from ``rope_parameters``, in place.

    Idempotent: a config that already carries the 4.x keys is left untouched.
    """
    notes: list[str] = []
    rp = cfg.get("rope_parameters")
    if not rp:
        return notes
    if cfg.get("rope_theta") is None and "rope_theta" in rp:
        cfg["rope_theta"] = rp["rope_theta"]
        notes.append(f"rope_theta <- {rp['rope_theta']}")
    # rope_scaling is the 4.x spelling of the same dict WITHOUT rope_theta in it.
    if cfg.get("rope_scaling") is None and rp.get("rope_type", "default") != "default":
        cfg["rope_scaling"] = {k: v for k, v in rp.items() if k != "rope_theta"}
        notes.append(f"rope_scaling <- {cfg['rope_scaling']}")
    return notes


def check_tokenizer(src: Path) -> None:
    """Refuse a tokenizer whose id 7 is not the multi marker, or whose vocab moved."""
    tj = src / "tokenizer.json"
    if not tj.exists():
        raise SystemExit(f"FATAL: no tokenizer.json in {src}")
    t = json.loads(tj.read_text())
    added = {a["id"]: a["content"] for a in t.get("added_tokens", [])}
    vocab = t["model"]["vocab"]
    inv = {i: s for s, i in vocab.items()} if isinstance(vocab, dict) else {}
    got = added.get(MULTI_ID, inv.get(MULTI_ID))
    if got != MULTI_TOKEN:
        raise SystemExit(
            f"FATAL: tokenizer id {MULTI_ID} is {got!r}, expected {MULTI_TOKEN!r} -- this is not "
            f"the renamed tokenizer and multi mode would mean something else entirely")
    if len(vocab) != VOCAB:
        raise SystemExit(f"FATAL: vocab is {len(vocab)}, expected {VOCAB}")
    print(f"[prepare] tokenizer ok: id {MULTI_ID} = {MULTI_TOKEN}, vocab {len(vocab)}")


def verify_template(out: Path) -> None:
    """Assert the pass-through template produces byte-identical token ids."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(out))
    if not getattr(tok, "chat_template", None):
        raise SystemExit("FATAL: the written tokenizer still has no chat_template")
    ok = True
    for probe in PROBES:
        rendered = tok.apply_chat_template([{"role": "user", "content": probe}], tokenize=False)
        same = tok.encode(rendered, add_special_tokens=False) == tok.encode(probe, add_special_tokens=False)
        ok &= same
        print(f"  {'OK  ' if same else 'DIFF'} {probe[:56]}...")
    if not ok:
        raise SystemExit("FATAL: the chat template ALTERS the prompt; every <pN> would shift")
    print("[prepare] pass-through template verified token-identical")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--template", default=str(Path(__file__).resolve().parent / TEMPLATE_NAME))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    src, out = Path(a.src).expanduser(), Path(a.out).expanduser()
    if not src.is_dir():
        raise SystemExit(f"FATAL: {src} is not a directory")
    check_tokenizer(src)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM

    print(f"[prepare] loading {src} as {a.dtype} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(src), dtype=getattr(torch, a.dtype))
    model.save_pretrained(str(out))

    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)

    template = Path(a.template).read_text()
    # BOTH spellings on purpose. transformers 5 prefers the file; 4.x and several
    # downstream readers only look in tokenizer_config.json. Writing one is a
    # coin flip on which reader gets there first, and the failure is an empty
    # prompt rather than an exception.
    (out / "chat_template.jinja").write_text(template)
    tcfg = json.loads((out / "tokenizer_config.json").read_text())
    tcfg["chat_template"] = template
    (out / "tokenizer_config.json").write_text(json.dumps(tcfg, indent=2) + "\n")

    cfg = json.loads((out / "config.json").read_text())
    notes = repair_rope(cfg)
    if cfg.get("rope_theta") is None:
        raise SystemExit("FATAL: config.json still has no top-level rope_theta after repair")
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[prepare] rope: {notes or 'already 4.x-readable'} "
          f"(theta={cfg['rope_theta']} scaling={cfg.get('rope_scaling')})")

    total = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"[prepare] wrote {out} ({total / 1e9:.2f} GB)")
    if a.verify:
        verify_template(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
