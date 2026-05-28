# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v2 driver — generate / tokenizer.

Usage::

    python cli.py generate  --input cifs/ --num-docs 100 --out docs.parquet
    python cli.py tokenizer --save-local ./tok/
    python cli.py tokenizer --push open-athena/contacts-and-distances-v2-tokenizer

Scope is intentionally narrower than v1's cli: this experiment only
implements training-data *generation*. Inference and evaluation will
land in follow-up experiments (one to run generation at scale, then
one to train on the result, then one to eval the trained model).
"""

import argparse
import os
import sys
import typing
from pathlib import Path

from marinfold import build_tokenizer
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

import generate
import parse
from vocab import CONTEXT_LENGTH, NAME, all_domain_tokens


# Input extensions that mean "parquet rows with an mmCIF column", as opposed to
# directories/globs of individual .cif/.pdb structure files.
_PARQUET_INPUT_SUFFIXES = (".parquet", ".pq")


def _is_parquet_input(spec: str) -> bool:
    return os.path.splitext(spec)[1].lower() in _PARQUET_INPUT_SUFFIXES


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = generate.GenerationConfig(
        contact_cutoff_angstrom=args.contact_cutoff_angstrom,
        contact_f_range=tuple(args.contact_f_range),
        contact_rank_mean=args.contact_rank_mean,
        distance_rank_mean=args.distance_rank_mean,
        rank_std=args.rank_std,
        residue_plddt_min=args.residue_plddt_min,
        think_initial_prob=args.think_initial_prob,
        think_initial_geom_p=args.think_initial_geom_p,
        think_additional_count_range=tuple(args.think_additional_count_range),
        think_run_length_geom_p=args.think_run_length_geom_p,
    )

    # Input modalities, dispatched on the --input extension (and on
    # --cif-uri-column for the parquet case):
    #   * parquet + --cif-uri-column — each row holds a URI (e.g. gs://...) to
    #     the structure file; read just the URI + entry_id columns from the
    #     manifest, then fetch each structure in-region. Avoids egressing the
    #     bulky cif_content column cross-cloud.
    #   * parquet (default) — each row holds an inline mmCIF in --cif-column
    #     plus an entry_id; parse the text in memory.
    #   * structure files — each path is a .cif/.pdb that gemmi reads directly.
    # entry_id matters in every mode: _generate_one seeds its RNG from it, and
    # passing the manifest's canonical id keeps URI-mode docs byte-identical
    # to inline-cif-mode docs on the same dataset.
    if _is_parquet_input(args.input):
        cif_col = args.cif_uri_column or args.cif_column
        rows = Dataset.from_files(args.input).load_parquet(
            columns=[cif_col, "entry_id"]
        )
        if args.num_docs is not None:
            rows = rows.reshard(1).take_per_shard(args.num_docs)
        if args.cif_uri_column:
            structures = rows.map(
                lambda r: parse.try_parse_structure(r.get(cif_col), entry_id=r.get("entry_id"))
            )
        else:
            structures = rows.map(
                lambda r: parse.try_parse_cif_content(r.get(cif_col), r.get("entry_id"))
            )
    else:
        glob = parse.input_glob(args.input)
        if args.num_docs is not None:
            source = Dataset.from_iterable(parse.list_structure_files(glob, limit=args.num_docs))
        else:
            source = Dataset.from_files(glob)
        structures = source.map(parse.try_parse_structure)

    ds = (
        structures
        # try_parse_* returns None for unparseable inputs; _generate_one returns
        # None for structures with < 2 residues or that don't fit the budget.
        .filter(lambda s: s is not None)
        .map(lambda s: generate._generate_one(s, context_length=args.context_length, cfg=cfg))
        .filter(lambda d: d is not None)
    )

    # Each input is its own shard, so a {shard} --out writes one file per input;
    # otherwise collapse everything into a single file.
    if "{shard" not in args.out:
        ds = ds.reshard(1)

    suffix = os.path.splitext(args.out)[1]
    match suffix:
        case ".parquet":
            ds = ds.map(lambda d: {"document": d, "structure": NAME}).write_parquet(args.out)
        case ".jsonl" | ".json":
            ds = ds.write_jsonl(args.out)
        case _:
            typing.assert_never(suffix)

    ctx = ZephyrContext(
        max_workers=args.max_workers,
        resources=ResourceConfig(
            cpu=args.worker_cpu,
            ram=args.worker_memory,
            disk=args.worker_disk,
        ),
    )
    ctx.execute(ds)
    print(f"[{NAME}] wrote {args.out}", file=sys.stderr)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description=f"{NAME} driver — generate / tokenizer.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- generate ----------------------------------------------------------
    p_gen = sub.add_parser("generate", help="Generate training documents.")
    p_gen.add_argument("--input", type=str, required=True,
                       help="Structures to read. A .cif/.pdb (.gz) file / "
                            "directory / glob, OR a .parquet file/glob whose "
                            "rows carry mmCIF text in --cif-column (e.g. "
                            "hf://datasets/timodonnell/afdb-1.6M/**/*.parquet). "
                            "Local path or cloud URL (gs://, s3://, hf://, ...).")
    p_gen.add_argument("--cif-column", type=str, default="cif_content",
                       help="For parquet --input: column holding the mmCIF text "
                            "(default: cif_content). Rows also need an 'entry_id' "
                            "column (seeds per-structure generation).")
    p_gen.add_argument("--cif-uri-column", type=str, default=None,
                       help="For parquet --input: column holding a URI per row "
                            "(e.g. 'gcs_uri' in afdb-1.6M -> gs://...). When set, "
                            "only the URI + entry_id columns are read from the "
                            "manifest and workers fetch each structure from the "
                            "URI directly — much faster than streaming the bulky "
                            "--cif-column cross-cloud. Overrides --cif-column.")
    p_gen.add_argument("--num-docs", type=int, default=None,
                       help="Process only the first N input files (one doc per "
                            "structure, so ~N docs). With a {shard} --out this "
                            "writes up to N single-doc files; otherwise one file "
                            "with up to N docs. Pair with a bounded --input — the "
                            "glob is enumerated in full before truncating.")
    p_gen.add_argument("--context-length", type=int, default=CONTEXT_LENGTH,
                       help="Token budget per document.")
    p_gen.add_argument("--out", type=str, required=True,
                       help="Output path (.parquet or .jsonl), local or cloud "
                            "URL. Include a {shard} placeholder (e.g. "
                            "corpus-{shard:05d}-of-{total:05d}.parquet) to write "
                            "one file per input; omit it for a single file.")
    # Zephyr worker resources. Each shard's rows are parsed + generated
    # single-threaded, so one CPU per worker is the efficient default — raise
    # --max-workers (not --worker-cpu) to scale out. See README.
    p_gen.add_argument("--worker-cpu", type=float, default=1,
                       help="CPUs per Zephyr worker task.")
    p_gen.add_argument("--worker-memory", type=str, default="4g",
                       help="RAM per Zephyr worker task (e.g. 4g, 8g).")
    p_gen.add_argument("--worker-disk", type=str, default="32g",
                       help="Ephemeral disk per Zephyr worker task (spill scratch).")
    p_gen.add_argument("--max-workers", type=int, default=None,
                       help="Upper bound on concurrent Zephyr workers. Actual "
                            "count is min(max_workers, num_shards). Default: "
                            "Zephyr's own (128 on a cluster) or ZEPHYR_MAX_WORKERS.")
    cfg_defaults = generate.GenerationConfig()
    # ---- v1-inherited knobs ----
    p_gen.add_argument("--contact-cutoff-angstrom", type=float,
                       default=cfg_defaults.contact_cutoff_angstrom)
    p_gen.add_argument("--residue-plddt-min", type=float,
                       default=cfg_defaults.residue_plddt_min)
    p_gen.add_argument("--contact-f-range", type=float, nargs=2,
                       metavar=("LOW", "HIGH"),
                       default=list(cfg_defaults.contact_f_range))
    p_gen.add_argument("--contact-rank-mean", type=float,
                       default=cfg_defaults.contact_rank_mean)
    p_gen.add_argument("--distance-rank-mean", type=float,
                       default=cfg_defaults.distance_rank_mean)
    p_gen.add_argument("--rank-std", type=float, default=cfg_defaults.rank_std)
    # ---- v2 think-token knobs ----
    p_gen.add_argument("--think-initial-prob", type=float,
                       default=cfg_defaults.think_initial_prob,
                       help="P(any think tokens right after <begin_statements>). "
                            "Default 0.75 per issue #34.")
    p_gen.add_argument("--think-initial-geom-p", type=float,
                       default=cfg_defaults.think_initial_geom_p,
                       help="Geometric p for the initial run length (support >= 1). "
                            "Default 0.13.")
    p_gen.add_argument("--think-additional-count-range", type=float, nargs=2,
                       metavar=("LOW", "HIGH"),
                       default=list(cfg_defaults.think_additional_count_range),
                       help="Uniform range for k2; n_additional_runs = max(int(k2), 0). "
                            "Default (-4, 4).")
    p_gen.add_argument("--think-run-length-geom-p", type=float,
                       default=cfg_defaults.think_run_length_geom_p,
                       help="Geometric p for the length of each additional run. "
                            "Default 0.25.")
    p_gen.set_defaults(func=cmd_generate)

    # ---- tokenizer ---------------------------------------------------------
    p_tok = sub.add_parser("tokenizer",
                           help="Build / save / push the v2 tokenizer.")
    p_tok.add_argument("--save-local", type=Path, default=None,
                       help="Save via tokenizer.save_pretrained() to this dir.")
    p_tok.add_argument("--push", type=str, default=None,
                       help="Push to this HF Hub repo "
                            "(e.g. open-athena/<name>-tokenizer).")
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
