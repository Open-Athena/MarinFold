# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""P(mode | <begin_statements>) — training-data prior vs 1B model logits.

Exp27 found that the v1 1B model gives ~99% medium-range mass on
the first contact-mode token after ``<begin_statements>``. This
probe answers two questions:

A) What's the empirical training-data prior? Stream ``contacts-
   and-distances-v1-5x`` (the published training subset) and, for
   each doc, look at the token *immediately after*
   ``<begin_statements>``: tabulate how often it's
   ``<long-range-contact>`` / ``<medium-range-contact>`` /
   ``<short-range-contact>`` / ``<distance>``.

B) What does the 1B model assign at that same position? Build a
   prompt ``<contacts-and-distances-v1> <begin_sequence> <AA>* <begin_statements>``
   for each protein in a small input set, run a single forward pass,
   read the softmax mass on the four candidate tokens.

The gap between (A) and (B) is the bias.

Two stages run independently. Defaults stream 100k docs from HF for
(A) and probe the 100 FoldBench monomer sequences for (B).
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent

sys.path.insert(0, str(EXP_ROOT))

from vocab import NAME


# Tokens of interest at the position right after <begin_statements>.
_MODE_TOKENS = [
    "<long-range-contact>",
    "<medium-range-contact>",
    "<short-range-contact>",
    "<distance>",
]


# Default input dirs / paths.
_DEFAULT_PROTEIN_DIR = (
    EXP_ROOT.parent
    / "exp20_evals_marinfold_1b_foldbench"
    / "protenix_data" / "data" / "protenix-foldbench-monomers" / "gt"
)
_DEFAULT_DOCS_CACHE = EXP_ROOT / "data" / "begin_statements_training_prior.csv"
_DEFAULT_MODEL_OUT = EXP_ROOT / "data" / "begin_statements_model_prior.csv"


# ---------------------------------------------------------------------------
# Stage A — training-data prior
# ---------------------------------------------------------------------------


def _iter_v1_5x_docs(num_docs: int) -> Iterator[str]:
    """Stream up to ``num_docs`` from the published v1-5x subset.

    Streams (not full-download) — the published subset is ~5.4M
    docs and we only need a sample. Yields the raw document string.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "timodonnell/protein-docs",
        "contacts-and-distances-v1-5x",
        split="train",
        streaming=True,
    )
    for i, row in enumerate(ds):
        if i >= num_docs:
            return
        # Public dataset has ``text`` as the doc column.
        doc = row.get("text") or row.get("document")
        if doc is None:
            raise RuntimeError(f"unknown doc column; row keys: {list(row.keys())}")
        yield doc


def _first_statement_token(doc: str) -> str | None:
    """Token immediately after ``<begin_statements>`` in ``doc``."""
    parts = doc.split()
    try:
        idx = parts.index("<begin_statements>")
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def stage_training_prior(num_docs: int, out_csv: Path) -> None:
    """Count first-statement-token frequency over a streamed sample."""
    counts: dict[str, int] = collections.Counter()
    other = collections.Counter()
    n_processed = 0
    n_skipped = 0
    print(f"streaming {num_docs} docs from timodonnell/protein-docs/contacts-and-distances-v1-5x ...")
    for doc in _iter_v1_5x_docs(num_docs):
        n_processed += 1
        tok = _first_statement_token(doc)
        if tok is None:
            n_skipped += 1
            continue
        if tok in _MODE_TOKENS:
            counts[tok] += 1
        else:
            other[tok] += 1
        if n_processed % 10_000 == 0:
            print(f"  ... {n_processed} docs")

    total = sum(counts.values()) + sum(other.values())
    print(f"processed {n_processed} docs ({n_skipped} had no <begin_statements>)")
    print(f"first-token distribution (n_total = {total}):")
    rows = []
    for tok in _MODE_TOKENS:
        c = counts[tok]
        rows.append({"token": tok, "count": c, "fraction": c / total if total else 0.0})
        print(f"  {tok:<27s} {c:>8d}  ({100*c/total:.2f}%)")
    # Surface anything else above 0.1% so we don't quietly hide drift.
    for tok, c in other.most_common():
        if c / total >= 0.001:
            rows.append({"token": tok, "count": c, "fraction": c / total})
            print(f"  {tok:<27s} {c:>8d}  ({100*c/total:.2f}%)   [other]")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "count", "fraction"])
        w.writeheader()
        for r in rows:
            r = dict(r); r["fraction"] = f"{r['fraction']:.6f}"
            w.writerow(r)
    print(f"\nwrote {out_csv}")


# ---------------------------------------------------------------------------
# Stage B — 1B model softmax at the <begin_statements> position
# ---------------------------------------------------------------------------


def _build_prompt_token_ids(tokenizer, sequence: list[str]) -> list[int]:
    """Build prompt tokens up to and including <begin_statements>."""
    tokens = (
        [f"<{NAME[:-1].replace('-v2', '-v1')}>"]  # but we want the v1 prompt — see below
    )
    # Use v1 marker explicitly so we're probing the v1 model in its
    # native format. NAME is "contacts-and-distances-v2" but the 1B
    # model was trained on v1 docs.
    tokens = ["<contacts-and-distances-v1>", "<begin_sequence>"]
    for aa in sequence:
        tokens.append(f"<{aa}>")
    tokens.append("<begin_statements>")
    return tokenizer.encode(" ".join(tokens), add_special_tokens=False)


def stage_model_prior(
    model_dir: Path, protein_dir: Path, out_csv: Path, num_proteins: int,
) -> None:
    """Run forward passes on the 1B model; record softmax mass per token."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(EXP_ROOT.parent / "exp1_document_structures_contacts_and_distances_v1"))
    from parse import iter_parsed_structures, parse_structure  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading 1B model from {model_dir} (device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16,
    ).to(device).eval()

    mode_ids = {t: tokenizer.encode(t, add_special_tokens=False)[0] for t in _MODE_TOKENS}
    print("token ids for probe:", mode_ids)

    rows = []
    for i, structure in enumerate(iter_parsed_structures(protein_dir)):
        if i >= num_proteins:
            break
        seq = structure.sequence
        prompt_ids = _build_prompt_token_ids(tokenizer, seq)
        ids = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            out = model(ids)
            logits = out.logits[0, -1].float().cpu()
        probs = torch.softmax(logits, dim=-1)
        # Mass on the four candidates of interest:
        per_token = {t: float(probs[mode_ids[t]]) for t in _MODE_TOKENS}
        # Argmax across the entire vocab — to see if the model puts
        # almost everything on one bucket regardless of input.
        top_id = int(torch.argmax(probs))
        top_tok = tokenizer.decode([top_id]).strip()
        row = {
            "entry_id": structure.source_path.stem,
            "n_residues": len(seq),
            "argmax_token": top_tok,
            "argmax_prob": float(probs[top_id]),
        }
        row.update({f"p_{t}": v for t, v in per_token.items()})
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1} proteins")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["entry_id", "n_residues", "argmax_token", "argmax_prob"] + [
        f"p_{t}" for t in _MODE_TOKENS
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r = dict(r)
            for k in r:
                if isinstance(r[k], float):
                    r[k] = f"{r[k]:.6f}"
            w.writerow(r)

    # Summary across proteins.
    print(f"\nProbed {len(rows)} proteins. Mean softmax mass at <begin_statements>:")
    import statistics
    for t in _MODE_TOKENS:
        vals = [float(r[f"p_{t}"]) for r in rows]
        print(f"  {t:<27s} mean={statistics.mean(vals):.4f}, "
              f"median={statistics.median(vals):.4f}")
    print(f"argmax distribution: {collections.Counter(r['argmax_token'] for r in rows)}")
    print(f"\nwrote {out_csv}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("training-prior",
                       help="Count first-statement tokens in v1-5x.")
    a.add_argument("--num-docs", type=int, default=100_000,
                   help="How many docs to stream.")
    a.add_argument("--out", type=Path, default=_DEFAULT_DOCS_CACHE)

    b = sub.add_parser("model-prior",
                       help="Probe 1B model softmax at <begin_statements>.")
    b.add_argument("--model", default="1B",
                   help="Model spec — local dir or MODELS.yaml nickname (default '1B').")
    b.add_argument("--protein-dir", type=Path, default=_DEFAULT_PROTEIN_DIR)
    b.add_argument("--num-proteins", type=int, default=20)
    b.add_argument("--out", type=Path, default=_DEFAULT_MODEL_OUT)

    args = ap.parse_args(argv)
    if args.cmd == "training-prior":
        stage_training_prior(args.num_docs, args.out)
    elif args.cmd == "model-prior":
        from marinfold.registry import resolve_model
        model_dir = resolve_model(args.model)
        print(f"resolved {args.model!r} -> {model_dir}")
        stage_model_prior(model_dir, args.protein_dir, args.out, args.num_proteins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
