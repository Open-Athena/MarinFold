# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``marinfold`` CLI: high-level entry point for inference + eval.

Two subcommands::

    marinfold infer    --backend mlx --input-sequence SIINFEKL... --out preds.json
    marinfold infer    --backend mlx --input /path/to/file-or-dir --out preds.json
    marinfold evaluate --backend mlx --input-dir /path/to/pdbs/ \
        --out preds.json --metrics-out metrics.json

Dispatch is driven by repo-root ``MODELS.yaml``:

1. ``--model <nickname>`` (or omitted → the entry with ``default: true``)
   resolves to a :class:`marinfold.ModelEntry`.
2. ``--document-structure <name>`` (or omitted → the first entry in
   the model's ``document_structures`` list) picks the impl package.
3. The impl subpackage is imported (e.g.
   ``marinfold.document_structures.contacts_and_distances_v1``) and
   the appropriate function (``predict`` / ``evaluate``) is called.

For impl-specific flags (seed-N sweeps, distance cap, batch size, …)
use the per-impl ``cli.py`` instead. The top-level CLI keeps its
surface narrow on purpose.
"""

import argparse
import dataclasses
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from marinfold import write_eval, write_predictions
from marinfold.registry import ModelEntry, resolve_model_entry


_BACKEND_CHOICES = ("vllm", "transformers", "mlx")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _pdf_path(s: str) -> Path:
    """Argparse type validator: only accept ``*.pdf`` paths for ``--out-plots``."""
    p = Path(s)
    if p.suffix.lower() != ".pdf":
        raise argparse.ArgumentTypeError(
            f"--out-plots must end in .pdf; got {s!r}"
        )
    return p


def _impl_module_name(structure_name: str) -> str:
    """Map a doc-structure NAME (kebab-case) to its subpackage module path."""
    return f"marinfold.document_structures.{structure_name.replace('-', '_')}"


def _load_impl(structure_name: str) -> ModuleType:
    """Import the impl subpackage; emit a helpful error on miss.

    Two failure modes for graduated impls:

    - The impl subpackage doesn't exist (typo / not yet graduated).
    - The impl exists but its optional-deps extra
      (e.g. ``marinfold[contacts-and-distances-v1]``) isn't installed,
      so a transitive import (gemmi etc.) fails.
    """
    module_name = _impl_module_name(structure_name)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Distinguish "impl package missing" from "impl's deps missing".
        impl_missing = exc.name == module_name or (
            exc.name is not None and module_name.startswith(exc.name + ".")
        )
        if impl_missing:
            raise SystemExit(
                f"Document structure {structure_name!r} is not available. "
                f"Expected subpackage {module_name!r}. Graduate the impl "
                f"into marinfold/marinfold/document_structures/ first."
            ) from exc
        raise SystemExit(
            f"Document structure {structure_name!r} is installed but a "
            f"required dep is missing ({exc.name!r}). Install the matching "
            f"extra: pip install 'marinfold[{structure_name}]'."
        ) from exc


def _pick_document_structure(
    entry: ModelEntry,
    requested: str | None,
) -> str:
    """Pick the doc-structure name to dispatch to.

    Without ``--document-structure``, defaults to the first entry in
    the model's ``document_structures`` list.
    """
    if requested is None:
        if not entry.document_structures:
            raise SystemExit(
                f"Model {entry.nickname!r} declares no supported "
                f"document_structures in MODELS.yaml. Pass --document-structure "
                f"or fix the MODELS.yaml entry."
            )
        return entry.document_structures[0]
    if requested not in entry.document_structures:
        supported = ", ".join(entry.document_structures) or "(none)"
        raise SystemExit(
            f"Model {entry.nickname!r} does not support document structure "
            f"{requested!r}. Supported: {supported}."
        )
    return requested


def _resolve_model_and_structure(
    model_spec: str | None,
    requested_structure: str | None,
) -> tuple[str | None, str]:
    """Resolve the model spec and document structure for CLI dispatch.

    Nickname/default flows still go through ``MODELS.yaml`` so we can
    pick the model's declared document structure. Local checkpoint
    directories bypass the registry, which means the caller must pass
    ``--document-structure`` explicitly.
    """
    if model_spec is not None and Path(model_spec).is_dir():
        if requested_structure is None:
            raise SystemExit(
                "Local model directories require --document-structure because "
                "there is no MODELS.yaml entry to infer it from."
            )
        return model_spec, requested_structure

    entry = resolve_model_entry(model_spec)
    ds_name = _pick_document_structure(entry, requested_structure)
    return entry.nickname, ds_name


def _make_inference_config(
    impl: ModuleType,
    model_spec: str | None,
    args: argparse.Namespace,
    *,
    input_path: Path | None = None,
) -> Any:
    """Build the impl's InferenceConfig from the parsed CLI args."""
    return impl.InferenceConfig(
        model=model_spec,
        input_path=input_path,
        backend=args.backend,
        dtype=args.dtype,
    )


def _structures_from_sequence(impl: ModuleType, seq: str) -> list:
    """Synthesize a single-structure list from a one-letter AA sequence.

    The impl must expose ``structure_from_sequence(aa: str)`` at its
    top level. Doc structures that can't be exercised on sequence-only
    input (e.g. ones that need atom coordinates even for ``infer``)
    don't expose this — they raise here.
    """
    fn = getattr(impl, "structure_from_sequence", None)
    if fn is None:
        raise SystemExit(
            f"Document structure {impl.__name__} does not support "
            f"--input-sequence (no structure_from_sequence helper). "
            f"Pass --input <file-or-dir> instead."
        )
    return [fn(seq)]


# --------------------------------------------------------------------------
# Subcommand handlers
# --------------------------------------------------------------------------


def cmd_infer(args: argparse.Namespace) -> None:
    model_spec, ds_name = _resolve_model_and_structure(
        args.model,
        args.document_structure,
    )
    impl = _load_impl(ds_name)

    if args.input_sequence is not None:
        structures = _structures_from_sequence(impl, args.input_sequence)
        cfg = _make_inference_config(impl, model_spec, args)
        records = list(impl.predict(cfg, structures=structures))
    else:
        cfg = _make_inference_config(impl, model_spec, args, input_path=args.input)
        records = list(impl.predict(cfg))

    write_predictions(args.out, records, structure_name=ds_name)
    print(f"[marinfold] wrote predictions to {args.out}", file=sys.stderr)

    if args.out_plots is not None:
        _emit_infer_plots(impl, ds_name, args.out_plots, records)


def cmd_evaluate(args: argparse.Namespace) -> None:
    model_spec, ds_name = _resolve_model_and_structure(
        args.model,
        args.document_structure,
    )
    impl = _load_impl(ds_name)

    cfg = _make_inference_config(impl, model_spec, args, input_path=args.input_dir)
    # `evaluate` runs predict + GT comparison in one pass; we re-run
    # predict for the predictions output so the per-example records
    # in the metrics file stay aligned with what we write to --out.
    # Cheaper alternative would be teaching evaluate to also emit raw
    # prediction records, but that complicates the EvalResult shape.
    result = impl.evaluate(cfg)
    write_eval(args.metrics_out, result, structure_name=ds_name)
    print(f"[marinfold] wrote metrics to {args.metrics_out}", file=sys.stderr)

    if args.out is not None:
        # Re-run predict on the same inputs to emit a predictions
        # file. Backends cache the model load and prompt KV is rebuilt
        # per call — fine for the per-pair workloads this CLI targets.
        records = list(impl.predict(dataclasses.replace(cfg)))
        write_predictions(args.out, records, structure_name=ds_name)
        print(f"[marinfold] wrote predictions to {args.out}", file=sys.stderr)

    if args.out_plots is not None:
        _emit_evaluate_plots(impl, ds_name, args.out_plots, result)


def _emit_infer_plots(
    impl: ModuleType,
    ds_name: str,
    out_path: Path,
    records: list[Any],
) -> None:
    fn = getattr(impl, "plot_infer_pdf", None)
    if fn is None:
        raise SystemExit(
            f"Document structure {ds_name!r} does not expose plot_infer_pdf "
            f"— --out-plots is not supported for this impl."
        )
    fn(out_path, records)
    print(f"[marinfold] wrote plots to {out_path}", file=sys.stderr)


def _emit_evaluate_plots(
    impl: ModuleType,
    ds_name: str,
    out_path: Path,
    result: Any,
) -> None:
    fn = getattr(impl, "plot_evaluate_pdf", None)
    if fn is None:
        raise SystemExit(
            f"Document structure {ds_name!r} does not expose plot_evaluate_pdf "
            f"— --out-plots is not supported for this impl."
        )
    fn(out_path, result)
    print(f"[marinfold] wrote plots to {out_path}", file=sys.stderr)


# --------------------------------------------------------------------------
# Argparse wiring
# --------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model", default=None,
        help="MODELS.yaml nickname (e.g. '1B') or a local directory "
             "with a model + tokenizer. Defaults to the MODELS.yaml "
             "entry marked default: true.",
    )
    p.add_argument(
        "--backend", choices=_BACKEND_CHOICES, default="vllm",
        help="Inference runtime. 'vllm' (Linux+GPU, default), "
             "'transformers' (anywhere torch installs), or 'mlx' "
             "(Apple Silicon native).",
    )
    p.add_argument(
        "--document-structure", default=None,
        help="Override which document structure to use. Defaults to "
             "the first entry in the model's MODELS.yaml "
             "document_structures list.",
    )
    p.add_argument(
        "--dtype", default="bfloat16",
        help="Model dtype. Honored by vllm + transformers; MLX loads "
             "whatever's on disk. On MPS prefer 'bfloat16' or "
             "'float32'.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marinfold",
        description="MarinFold CLI: run a trained protein-document LLM "
                    "for inference or evaluation.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- infer ------------------------------------------------------------
    p_inf = sub.add_parser(
        "infer",
        help="Predict residue-pair distances. No ground truth used.",
    )
    _add_common(p_inf)
    src = p_inf.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input-sequence", default=None,
        help="One-letter amino-acid sequence (e.g. SIINFEKLLLSKP).",
    )
    src.add_argument(
        "--input", type=Path, default=None,
        help="Path to a structure file (.pdb / .cif / .gz) or a "
             "directory of them. The sequence is taken from each file.",
    )
    p_inf.add_argument(
        "--out", type=Path, required=True,
        help="Predictions output (.json/.jsonl/.parquet).",
    )
    p_inf.add_argument(
        "--out-plots", type=_pdf_path, default=None,
        help="Optional multi-page PDF; one predicted-distance heatmap "
             "page per (structure, n_seeded). Requires the impl to "
             "expose plot_infer_pdf; for contacts-and-distances-v1 "
             "this is pulled in by the [contacts-and-distances-v1] extra.",
    )
    p_inf.set_defaults(func=cmd_infer)

    # ---- evaluate ---------------------------------------------------------
    p_eval = sub.add_parser(
        "evaluate",
        help="Predict + score against ground truth. Requires structure files.",
    )
    _add_common(p_eval)
    p_eval.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory of structure files (.pdb / .cif / .gz). The "
             "atom coordinates are the ground truth.",
    )
    p_eval.add_argument(
        "--metrics-out", type=Path, required=True,
        help="Aggregated metrics output (.json/.parquet).",
    )
    p_eval.add_argument(
        "--out", type=Path, default=None,
        help="Optional predictions output (.json/.jsonl/.parquet). "
             "Omit to skip the second predict pass.",
    )
    p_eval.add_argument(
        "--out-plots", type=_pdf_path, default=None,
        help="Optional multi-page PDF; one (GT, predicted) heatmap "
             "page per (structure, n_seeded). Requires the impl to "
             "expose plot_evaluate_pdf.",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
