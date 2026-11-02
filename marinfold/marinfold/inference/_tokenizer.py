# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared tokenizer loading for the inference backends.

Kept torch-free so the MLX / vLLM backends can reuse it without
pulling in the ``[transformers]`` extra's torch dependency.
"""

import json
import tempfile
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from marinfold.inference._config import needs_rope_repair, read_config, repair_rope

# Special-token / length keys we carry over from a checkpoint's
# ``tokenizer_config.json`` when falling back to a raw
# ``PreTrainedTokenizerFast`` (see :func:`load_tokenizer`). The
# marinfold training export also writes bookkeeping keys — ``backend``,
# ``is_local``, ``local_files_only``, ``tokenizer_class`` — which are
# either meaningless to transformers or (in the case of a custom
# ``tokenizer_class``) the very thing that breaks ``AutoTokenizer``; we
# deliberately drop those.
_FALLBACK_TOKENIZER_KEYS = (
    "bos_token",
    "eos_token",
    "unk_token",
    "pad_token",
    "cls_token",
    "sep_token",
    "mask_token",
    "model_max_length",
    "clean_up_tokenization_spaces",
)


def load_tokenizer(model_path: Path):
    """Load a checkpoint tokenizer, tolerating marinfold-custom exports.

    ``AutoTokenizer.from_pretrained`` is tried first. Training-path
    checkpoints (e.g. ``contacts-v1-exp120-1.5B``) declare
    ``"tokenizer_class": "TokenizersBackend"`` in ``tokenizer_config.json``
    — a levanter export class name ``AutoTokenizer`` can't resolve, so the
    call raises ``ValueError: Tokenizer class TokenizersBackend does not
    exist``. Those checkpoints do ship a valid WordLevel ``tokenizer.json``
    though, so on any load failure we fall back to constructing a
    ``PreTrainedTokenizerFast`` straight from ``tokenizer.json``, carrying
    the special tokens over from the config. If there is no
    ``tokenizer.json`` to fall back to, the original error is re-raised.
    """
    try:
        return AutoTokenizer.from_pretrained(str(model_path))
    except Exception:
        tokenizer_file = Path(model_path) / "tokenizer.json"
        if not tokenizer_file.exists():
            raise
        config: dict = {}
        config_path = Path(model_path) / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path) as fh:
                config = json.load(fh)
        kwargs = {k: config[k] for k in _FALLBACK_TOKENIZER_KEYS if k in config}
        return PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file), **kwargs
        )


def tokenizer_source_path(model_path: Path) -> str:
    """Return a filesystem path a path-based loader can read the tokenizer from.

    Unlike :func:`load_tokenizer`, this is for consumers that load the
    tokenizer *themselves* from a path — notably vLLM, whose engine
    constructs its own tokenizer internally via ``AutoTokenizer`` and so
    can't be handed a Python tokenizer object.

    If the checkpoint's own directory loads cleanly, it is returned
    unchanged. If ``AutoTokenizer`` can't resolve it (a marinfold-custom
    ``tokenizer_class`` such as ``"TokenizersBackend"``), a repaired
    ``PreTrainedTokenizerFast`` copy — whose ``tokenizer_config.json``
    declares a class ``AutoTokenizer`` understands — is written to a fresh
    temp dir and that path is returned instead. Raises if the tokenizer
    can't be loaded at all (no ``tokenizer.json`` to fall back to).
    """
    try:
        AutoTokenizer.from_pretrained(str(model_path))
        return str(model_path)
    except Exception:
        # load_tokenizer re-raises the original error when there is no
        # tokenizer.json to fall back to, so we never write a bogus repair.
        repaired = load_tokenizer(model_path)
        out_dir = tempfile.mkdtemp(prefix="marinfold-tokenizer-")
        repaired.save_pretrained(out_dir)
        return out_dir


def model_source_path(model_path: Path) -> str:
    """Return a checkpoint directory a path-based loader can read as-is.

    Some consumers — ``mlx_lm.load``, and vLLM's engine — load the model and
    its tokenizer from a directory themselves, so they cannot be handed a
    repaired Python object. This builds them one.

    Two things may need repairing, both from the same transformers-5 exporter
    and both silent rather than fatal:

    * the tokenizer, whose ``tokenizer_class`` (``"TokenizersBackend"``)
      ``AutoTokenizer`` cannot resolve — see :func:`tokenizer_source_path`;
    * ``config.json``, whose ``rope_parameters`` block our pinned
      transformers 4.x ignores, falling back to the architecture's default
      rope — see :mod:`marinfold.inference._config`.

    If neither needs repair, ``model_path`` is returned unchanged. Otherwise a
    temp directory is populated with symlinks to every checkpoint artifact —
    keeping the large weights in place — and the repaired files are written
    over the corresponding links.
    """
    tokenizer_path = Path(tokenizer_source_path(model_path))
    raw_config = read_config(model_path)
    repair_config = needs_rope_repair(raw_config)

    if tokenizer_path == model_path and not repair_config:
        return str(model_path)

    source_path = tokenizer_path
    if source_path == model_path:
        # Only the config is broken, so there is no repaired tokenizer dir to
        # build on; start a fresh overlay.
        source_path = Path(tempfile.mkdtemp(prefix="marinfold-checkpoint-"))

    for model_entry in model_path.iterdir():
        destination = source_path / model_entry.name
        if destination.exists() or destination.is_symlink():
            continue
        destination.symlink_to(
            model_entry.resolve(), target_is_directory=model_entry.is_dir()
        )

    if repair_config:
        # Replace the symlink rather than writing through it — that would
        # rewrite the user's checkpoint in place.
        destination = source_path / "config.json"
        destination.unlink(missing_ok=True)
        destination.write_text(json.dumps(repair_rope(raw_config), indent=2))
    return str(source_path)
