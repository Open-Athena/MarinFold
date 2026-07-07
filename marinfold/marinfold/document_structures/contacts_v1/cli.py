# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-v1 driver — generate / view / infer / evaluate / tokenizer.

Usage::

    # Write documents (+ metadata) for a structure or directory of them.
    python cli.py generate --input cifs/ --out docs.parquet \
        --summary-out summary.json
    # Eyeball a few documents + their contact tables in the terminal.
    python cli.py view --input tests/data/1QYS.cif
    # Predict a contact map (P(contact)) for a sequence — no ground truth.
    python cli.py infer --model M --input-sequence SIINFEKL... --out preds.json
    # Score the model's contacts against pyconfind ground truth.
    python cli.py evaluate --model M --input tests/data/1QYS.cif --out metrics.json
    # Build / save / push the contacts-v1 tokenizer.
    python cli.py tokenizer --save-local ./tok/
    python cli.py tokenizer --push open-athena/contacts-v1-tokenizer

``generate`` writes one row per input structure — the ``document`` token
string plus the metadata columns from :meth:`GenerationResult.metadata_row`
(seq length, contact counts, truncation flag, …) — via the shared
:func:`marinfold.write_docs`. With ``--summary-out`` it also dumps a rich
JSON summary (full sequence + every emitted contact's degree) for local
inspection. ``view`` is the same generation but printed, no files.

``infer`` / ``evaluate`` are the same surface the top-level ``marinfold``
CLI dispatches to, exposed here with the extra contacts-v1 knobs
(``--min-seq-separation``, ``--ensemble-k``, …); see :mod:`.inference`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from marinfold import build_tokenizer, write_docs, write_eval, write_predictions

from . import generate, inference, plots
from .parse import DEFAULT_CIF_COLUMN, DEFAULT_ID_COLUMN
from .vocab import CONTEXT_LENGTH, NAME, all_domain_tokens, position_token


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _pdf_path(s: str) -> Path:
    """Argparse type validator: only accept ``*.pdf`` paths for ``--out-plots``."""
    p = Path(s)
    if p.suffix.lower() != ".pdf":
        raise argparse.ArgumentTypeError(f"--out-plots must end in .pdf; got {s!r}")
    return p


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
        min_seq_separation=args.min_seq_separation,
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


def _format_optional_degree(value: float | None) -> str:
    """Render an optional degree for terminal display."""
    return "n/a" if value is None else f"{value:.4f}"


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
            f"{result.contacts_pre_filter} survive seq-sep>="
            f"{result.min_seq_separation}  "
            f"truncated={result.truncated}  tokens={result.num_tokens}"
        )
        if result.contacts_pre_filter:
            print(
                f"  degree: highest={_format_optional_degree(result.highest_contact_degree)}  "
                f"lowest_nonzero={_format_optional_degree(result.lowest_nonzero_contact_degree)}  "
                f"lowest_included={_format_optional_degree(result.lowest_included_contact_degree)}"
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


def _inference_config(args: argparse.Namespace) -> inference.InferenceConfig:
    return inference.InferenceConfig(
        model=args.model,
        input_path=getattr(args, "input", None),
        backend=args.backend,
        batch_size=args.batch_size,
        dtype=args.dtype,
        method=args.method,
        min_seq_separation=args.min_seq_separation,
        ensemble_k=args.ensemble_k,
        top_k_logprobs=args.top_k_logprobs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        keep_matrix=getattr(args, "keep_matrix", False),
        n_rollouts=args.n_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )


def cmd_infer(args: argparse.Namespace) -> None:
    cfg = _inference_config(args)
    if args.input_sequence is not None:
        structures = [inference.structure_from_sequence(args.input_sequence)]
        records = list(inference.predict(cfg, structures=structures))
    else:
        records = list(inference.predict(cfg))
    write_predictions(args.out, records, structure_name=NAME)
    print(f"[{NAME}] wrote {args.out}", file=sys.stderr)
    if args.out_plots is not None:
        plots.plot_infer_pdf(args.out_plots, records)
        print(f"[{NAME}] wrote {args.out_plots}", file=sys.stderr)


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = _inference_config(args)
    result = inference.evaluate(cfg)
    write_eval(args.out, result, structure_name=NAME)
    print(f"[{NAME}] wrote {args.out}", file=sys.stderr)
    for k, v in result.metrics.items():
        print(f"  {k} = {v}")
    if args.out_plots is not None:
        plots.plot_evaluate_pdf(args.out_plots, result)
        print(f"[{NAME}] wrote {args.out_plots}", file=sys.stderr)


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
    p.add_argument("--min-seq-separation", type=int, default=cfg.min_seq_separation,
                   help="Minimum primary-sequence separation |i-j| for a pair "
                        "to count as a contact; closer pairs are never "
                        f"contacts (default {cfg.min_seq_separation}).")
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

    # ---- shared inference args (infer + evaluate) --------------------------
    def _add_inference_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True,
                       help="Local directory holding the model + tokenizer, or "
                            "a nickname listed in MODELS.yaml "
                            "(e.g. 'contacts-v1-exp75-1.5B').")
        p.add_argument("--backend", choices=("vllm", "transformers", "mlx"),
                       default="vllm",
                       help="Inference runtime. 'vllm' (Linux+GPU, default), "
                            "'transformers' (anywhere torch installs), or 'mlx' "
                            "(Apple Silicon native).")
        p.add_argument("--method", choices=("pairwise", "rollout"),
                       default="pairwise",
                       help="Contact readout. 'pairwise' (default, fast) scores "
                            "P(contact) per pair; 'rollout' (exp82's best, "
                            "~150x slower) votes over sampled completions with a "
                            "pairwise tie-break, and needs --backend vllm or "
                            "transformers (not mlx).")
        p.add_argument("--min-seq-separation", type=int, default=6,
                       help="Smallest |i-j| that can be a contact (default 6, "
                            "matching the contacts-v1 data).")
        p.add_argument("--ensemble-k", type=int, default=1,
                       help="(--method pairwise) resample the sequence definition "
                            "this many times and average P(contact) (test-time "
                            "augmentation; default 1).")
        p.add_argument("--n-rollouts", type=int, default=100,
                       help="(--method rollout) sampled completions to vote over "
                            "(default 100).")
        p.add_argument("--temperature", type=float, default=1.0,
                       help="(--method rollout) sampling temperature (default 1.0).")
        p.add_argument("--top-p", type=float, default=0.95,
                       help="(--method rollout) nucleus top-p (default 0.95).")
        p.add_argument("--top-k", type=int, default=50,
                       help="(--method rollout) sampling top-k; 0 disables "
                            "(default 50).")
        p.add_argument("--top-k-logprobs", type=int, default=256,
                       help="vLLM-only; ignored by other backends.")
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--dtype", default="bfloat16",
                       help="Model dtype. Honored by vllm + transformers; MLX "
                            "loads whatever's on disk.")
        p.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                       help="vLLM-only; ignored by other backends.")
        p.add_argument("--out", type=Path, required=True, help="Output path.")
        p.add_argument("--out-plots", type=_pdf_path, default=None,
                       help="Optional multi-page PDF; one P(contact) heatmap "
                            "page per structure (infer), or GT-vs-model "
                            "side-by-side (evaluate).")

    # ---- infer -------------------------------------------------------------
    p_inf = sub.add_parser("infer", help="Predict a contact map (no ground truth).")
    _add_inference_common(p_inf)
    src = p_inf.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-sequence", default=None,
                     help="One-letter amino-acid sequence (e.g. SIINFEKLLLSKP).")
    src.add_argument("--input", type=Path, default=None,
                     help="Structure file (PDB / mmCIF / .gz) or directory; the "
                          "sequence is read from each (no pyconfind needed).")
    p_inf.add_argument("--keep-matrix", action="store_true",
                       help="Include the dense [L,L] P(contact) matrix per "
                            "structure in the output records.")
    p_inf.set_defaults(func=cmd_infer)

    # ---- evaluate ----------------------------------------------------------
    p_eval = sub.add_parser(
        "evaluate", help="Predict + score against pyconfind ground-truth contacts.")
    _add_inference_common(p_eval)
    p_eval.add_argument("--input", type=Path, required=True,
                        help="Structure file (PDB / mmCIF / .gz) or directory. "
                             "The input IS the ground truth — pyconfind contacts "
                             "on it are scored against the model (needs the "
                             "contacts-v1 extra).")
    p_eval.set_defaults(func=cmd_evaluate)

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
