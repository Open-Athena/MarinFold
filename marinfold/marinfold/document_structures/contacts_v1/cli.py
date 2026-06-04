# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-v1 driver — generate / view / tokenizer.

Usage::

    # Write documents (+ metadata) for a structure or directory of them.
    python cli.py generate --input cifs/ --out docs.parquet \
        --summary-out summary.json
    # Eyeball a few documents + their contact tables in the terminal.
    python cli.py view --input tests/data/1QYS.cif
    # Build / save / push the contacts-v1 tokenizer.
    python cli.py tokenizer --save-local ./tok/
    python cli.py tokenizer --push open-athena/contacts-v1-tokenizer

``generate`` writes one row per input structure — the ``document`` token
string plus the metadata columns from :meth:`GenerationResult.metadata_row`
(seq length, contact counts, truncation flag, …) — via the shared
:func:`marinfold.write_docs`. With ``--summary-out`` it also dumps a rich
JSON summary (full sequence + every emitted contact's degree) for local
inspection. ``view`` is the same generation but printed, no files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from marinfold import build_tokenizer, write_docs

from . import generate
from .parse import DEFAULT_CIF_COLUMN, DEFAULT_ID_COLUMN
from .vocab import CONTEXT_LENGTH, NAME, all_domain_tokens, position_token


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _assembly_arg(text: str) -> int | str | None:
    """Parse ``--assembly`` into pyconfind's ``int | str | None`` surface."""
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered == "none":
        return None
    try:
        return int(stripped)
    except ValueError:
        if not stripped:
            raise argparse.ArgumentTypeError(
                "assembly must be an integer, a named assembly, or 'none'"
            ) from None
        return stripped


def _config_from_args(args: argparse.Namespace) -> generate.GenerationConfig:
    return generate.GenerationConfig(
        native_only=args.native_only,
        contact_distance=args.contact_distance,
        dcut=args.dcut,
        clash_distance=args.clash_distance,
        assembly=args.assembly,
        min_contact_degree=args.min_contact_degree,
    )


def _write_summary(
    out: Path,
    results: list[generate.GenerationResult],
    *,
    context_length: int,
    config: generate.GenerationConfig,
) -> None:
    """Dump the rich per-protein summary JSON for local inspection."""
    summary: dict[str, Any] = {
        "structure": NAME,
        "context_length": context_length,
        "config": vars(config),
        "num_documents": len(results),
        "num_truncated": sum(1 for r in results if r.truncated),
        "documents": [r.summary_dict() for r in results],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)


# --------------------------------------------------------------------------
# Subcommand handlers
# --------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    results = list(generate.generate_documents(
        input_path=args.input,
        num_docs=args.num_docs,
        context_length=args.context_length,
        config=config,
        rotamer_library=args.rotamer_library,
        cif_column=args.cif_column,
        id_column=args.id_column,
    ))
    write_docs(
        args.out,
        (r.metadata_row() for r in results),
        structure_name=NAME,
    )
    print(f"[{NAME}] wrote {len(results)} documents to {args.out}", file=sys.stderr)
    if args.summary_out is not None:
        _write_summary(
            args.summary_out, results,
            context_length=args.context_length, config=config,
        )
        print(f"[{NAME}] wrote summary to {args.summary_out}", file=sys.stderr)


def cmd_view(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    shown = 0
    for result in generate.generate_documents(
        input_path=args.input,
        num_docs=args.num_docs,
        context_length=args.context_length,
        config=config,
        rotamer_library=args.rotamer_library,
        cif_column=args.cif_column,
        id_column=args.id_column,
    ):
        shown += 1
        print(f"\n=== {result.entry_id} ===")
        print(
            f"  residues={result.seq_len}  "
            f"global_plddt={result.global_plddt:.1f}  "
            f"start_index={result.start_index}  "
            f"n_term={position_token(result.n_term_index)}  "
            f"c_term={position_token(result.c_term_index)}"
        )
        print(
            f"  contacts: {result.contacts_emitted} included / "
            f"{result.contacts_excluded} excluded / "
            f"{result.contacts_passing_min_degree} pass min-degree / "
            f"{result.contacts_pre_filter} found  "
            f"truncated={result.truncated}  tokens={result.num_tokens}"
        )
        if result.contacts_pre_filter:
            print(
                f"  degree: highest={result.highest_contact_degree:.4f}  "
                f"lowest_nonzero={result.lowest_nonzero_contact_degree:.4f}  "
                f"lowest_included={result.lowest_included_contact_degree:.4f}"
            )
        ncap = args.max_contacts
        # Contacts appear in random order in the document; sort for display.
        by_degree = sorted(result.contacts, key=lambda c: -c.degree)
        print(f"  strongest contacts (up to {ncap}, sorted for display):")
        for c in by_degree[:ncap]:
            print(
                f"    {position_token(c.pos_i)}/{position_token(c.pos_j)}  "
                f"{c.resname_i}{c.resnum_i}–{c.resname_j}{c.resnum_j}  "
                f"degree={c.degree:.4f}"
            )
        if result.contacts_emitted > ncap:
            print(f"    … and {result.contacts_emitted - ncap} more")
        print(f"  document:\n{result.document}")
    if shown == 0:
        print(f"[{NAME}] no documents generated for {args.input}", file=sys.stderr)


def cmd_tokenizer(args: argparse.Namespace) -> None:
    tokenizer = build_tokenizer(all_domain_tokens())
    print(f"[{NAME}] built tokenizer with {len(tokenizer)} tokens", file=sys.stderr)
    did_anything = False
    if args.save_local is not None:
        out_dir = Path(args.save_local)
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(out_dir))
        print(f"[{NAME}] saved tokenizer to {out_dir}", file=sys.stderr)
        did_anything = True
    if args.push is not None:
        tokenizer.push_to_hub(args.push, private=args.private)
        print(
            f"[{NAME}] pushed tokenizer to https://huggingface.co/{args.push}",
            file=sys.stderr,
        )
        did_anything = True
    if not did_anything:
        sample = " ".join(all_domain_tokens()[:8])
        encoded = tokenizer.encode(sample, add_special_tokens=False)
        print(f"sample: {sample!r}")
        print(f"  ids:  {encoded}")
        print(f"  back: {tokenizer.decode(encoded)!r}")
        print("Use --save-local DIR or --push REPO to persist the tokenizer.")


# --------------------------------------------------------------------------
# Argparse wiring
# --------------------------------------------------------------------------


def _add_generation_common(p: argparse.ArgumentParser) -> None:
    """Args shared by ``generate`` and ``view``."""
    cfg = generate.GenerationConfig()
    p.add_argument("--input", type=Path, required=True,
                   help="PDB / mmCIF (.gz) file or directory of them, OR a "
                        ".parquet shard / directory of shards in the afdb-24M "
                        "layout (structures read from --cif-column).")
    p.add_argument("--cif-column", default=DEFAULT_CIF_COLUMN,
                   help="For parquet input: column holding the mmCIF text "
                        f"(default {DEFAULT_CIF_COLUMN!r}).")
    p.add_argument("--id-column", default=DEFAULT_ID_COLUMN,
                   help="For parquet input: column holding the entry id / "
                        f"generation seed (default {DEFAULT_ID_COLUMN!r}).")
    p.add_argument("--num-docs", type=int, default=None,
                   help="Cap on documents produced (default: one per input).")
    p.add_argument("--context-length", type=int, default=CONTEXT_LENGTH,
                   help="Token budget per document; contacts truncate to fit.")
    p.add_argument("--rotamer-library", type=Path, default=None,
                   help="pyconfind rotamer-library directory (EBL.out + "
                        "BEBL.out). Default: auto-download + cache.")
    p.add_argument("--assembly", type=_assembly_arg, default=cfg.assembly,
                   help="Biological assembly passed to pyconfind. Default "
                        "'none' uses the asymmetric unit as-is; pass an "
                        "integer or assembly name to expand explicitly.")
    # pyconfind geometry knobs (SPEC / confind defaults).
    p.add_argument("--native-only", action=argparse.BooleanOptionalAction,
                   default=cfg.native_only,
                   help="Only place rotamers of the native amino acid "
                        "(SPEC default). --no-native-only substitutes all.")
    p.add_argument("--contact-distance", type=float, default=cfg.contact_distance,
                   help="Side-chain contact cutoff in Å.")
    p.add_argument("--dcut", type=float, default=cfg.dcut,
                   help="CA-CA pair cutoff in Å for the contact-degree search.")
    p.add_argument("--clash-distance", type=float, default=cfg.clash_distance,
                   help="Backbone-clash cutoff in Å used while pruning rotamers.")
    p.add_argument("--min-contact-degree", type=float, default=cfg.min_contact_degree,
                   help="Drop contacts with degree below this before "
                        "selection; they are never included even if there is "
                        f"room (default {cfg.min_contact_degree}).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description=f"{NAME} driver — generate / view / tokenizer.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- generate ----------------------------------------------------------
    p_gen = sub.add_parser("generate", help="Generate training documents.")
    _add_generation_common(p_gen)
    p_gen.add_argument("--out", type=Path, required=True,
                       help="Output path for documents (.parquet or .jsonl).")
    p_gen.add_argument("--summary-out", type=Path, default=None,
                       help="Optional rich per-protein summary (.json): each "
                            "protein's sequence, every emitted contact with "
                            "its degree, and the truncation flag.")
    p_gen.set_defaults(func=cmd_generate)

    # ---- view --------------------------------------------------------------
    p_view = sub.add_parser("view", help="Print documents + contact tables to stdout.")
    _add_generation_common(p_view)
    p_view.add_argument("--max-contacts", type=int, default=20,
                        help="Max contacts to list per document (default 20).")
    p_view.set_defaults(func=cmd_view)

    # ---- tokenizer ---------------------------------------------------------
    p_tok = sub.add_parser("tokenizer",
                           help="Build / save / push the contacts-v1 tokenizer.")
    p_tok.add_argument("--save-local", type=Path, default=None,
                       help="Save via tokenizer.save_pretrained() to this dir.")
    p_tok.add_argument("--push", type=str, default=None,
                       help="Push to this HF Hub repo "
                            "(e.g. open-athena/contacts-v1-tokenizer).")
    p_tok.add_argument("--private", action="store_true",
                       help="If --push, create the repo private.")
    p_tok.set_defaults(func=cmd_tokenizer)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
