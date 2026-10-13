"""Pinned redesign subset and the three requested learning-rate trials."""

from experiments.exp277_models_single_mpnn_pilot.config import (
    CORPORA as DEFAULT_CORPORA,
)
from experiments.exp277_models_single_mpnn_pilot.config import (
    TOKENIZER,
    VALIDATION_CACHE,
)

PREFIX = "s3://marin-us-east-02a/MarinFold/exp364_models_mpnn_only_lr_sweep"
VERSION = "2026.10.13.1"
CORPORA = tuple(corpus for corpus in DEFAULT_CORPORA if corpus.name.startswith("mpnn-"))
EPOCH_PACKED_EXAMPLES = 5_010_642 + 18_910_547
EPOCHS = 2
TRAIN_STEPS = (EPOCHS * EPOCH_PACKED_EXAMPLES + 127) // 128
LEARNING_RATES = {"lr5e-4": 5e-4, "lr1e-3": 1e-3, "lr2e-3": 2e-3}


def run_id_for(trial: str, *, smoke: bool = False, attempt: int = 1) -> str:
    """Keep production identity stable across placements and retry attempts."""
    if trial not in LEARNING_RATES:
        raise ValueError(f"Unknown trial: {trial}")
    run_id = f"contacts-v1-exp364-mpnn-only-2ep-{trial}-1.5B"
    return f"{run_id}-smoke-a{attempt:02d}" if smoke else run_id
