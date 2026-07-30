# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared model-config loading for the inference backends.

Companion to :mod:`._tokenizer`, and kept torch-free for the same reason:
the MLX / vLLM backends reuse it without pulling in the ``[transformers]``
extra's torch dependency.

The problem it solves. A levanter checkpoint exported by **transformers 5.x**
writes its rope settings as a single ``rope_parameters`` block::

    "rope_parameters": {"rope_type": "llama3", "rope_theta": 500000,
                        "factor": 8.0, "low_freq_factor": 1.0,
                        "high_freq_factor": 4.0,
                        "original_max_position_embeddings": 8192}

marinfold pins ``transformers>=4.40,<5``, which reads ``rope_theta`` and
``rope_scaling`` instead. It does not error on the 5.x shape — it *ignores*
it and falls back to the architecture default, so a checkpoint that was
trained with Llama3 rope at theta 500000 loads with **theta 10000 and no
scaling**. A 50x error in the rope base, silently, at load time.

This is the config-side twin of the ``tokenizer_class: TokenizersBackend``
breakage in :mod:`._tokenizer`, from the same exporter, and both were first
seen on the #117 checkpoints (MarinFold #89, #169). Every future
transformers-5 export carries it.

Measured on the #117 checkpoint over three real contacts-v1 documents
(experiments/exp82 benchmark set), repaired vs as-published: mean NLL **2.414
vs 3.179 nats/token**. For scale, the whole #75 -> #117 model generation was
worth 0.053 nats. The damage grows with sequence length — +0.25 nats at 361
tokens, +1.34 at 683 — which is the signature of a wrong rope base.

One expected noise source: transformers 4.x warns that
``original_max_position_embeddings`` (8192) is not *less than*
``max_position_embeddings`` (8192). That equality is what levanter's
``Llama3RotaryEmbeddingsConfig()`` default produces and what the model was
trained with, so it is reproduced deliberately. The warning does not stop the
llama3 scaling being applied. Do not "fix" it by changing either value — that
would make inference diverge from training.
"""

import json
from pathlib import Path

# Keys inside a transformers-5 ``rope_parameters`` block that are the *base*
# frequency rather than part of the scaling spec.
_ROPE_THETA_KEYS = ("rope_theta", "theta", "base")
# ``rope_type`` values that mean "no scaling" — for these, transformers 4.x
# wants ``rope_scaling`` left as None rather than a dict.
_UNSCALED_ROPE_TYPES = (None, "default")


def needs_rope_repair(config: dict) -> bool:
    """True if ``config`` carries transformers-5 rope that 4.x would ignore."""
    if not isinstance(config.get("rope_parameters"), dict):
        return False
    # If the 4.x keys are already present the exporter (or a previous repair)
    # wrote both shapes; nothing to do.
    return "rope_theta" not in config and "rope_scaling" not in config


def repair_rope(config: dict) -> dict:
    """Return ``config`` with ``rope_parameters`` restated in 4.x terms.

    The input is not mutated. ``rope_parameters`` is kept as well as
    translated: it is inert under 4.x, and leaving it in place means a
    repaired directory still round-trips under transformers 5.x.
    """
    if not needs_rope_repair(config):
        return config

    out = dict(config)
    params = dict(out["rope_parameters"])

    theta = next((params.pop(k) for k in _ROPE_THETA_KEYS if k in params), None)
    if theta is not None:
        out["rope_theta"] = theta

    rope_type = params.get("rope_type", params.get("type"))
    if rope_type in _UNSCALED_ROPE_TYPES:
        out["rope_scaling"] = None
    else:
        # transformers 4.x reads `rope_type`; older configs used `type`. Write
        # both so either validator is satisfied.
        params.setdefault("rope_type", rope_type)
        params.setdefault("type", rope_type)
        out["rope_scaling"] = params
    return out


def read_config(model_path: Path) -> dict:
    """Read ``config.json`` from a checkpoint directory. Empty dict if absent."""
    path = Path(model_path) / "config.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def load_config(model_path: Path):
    """Load a checkpoint's config, repairing transformers-5 rope if present.

    Returns a ``PretrainedConfig`` suitable for passing to
    ``from_pretrained(..., config=...)``. Import of transformers is deferred
    so the module stays importable in torch-free environments.
    """
    from transformers import AutoConfig

    raw = read_config(model_path)
    if not needs_rope_repair(raw):
        return AutoConfig.from_pretrained(str(model_path))
    return AutoConfig.for_model(**repair_rope(raw))
