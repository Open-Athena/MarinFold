"""Only the model's hidden and feed-forward widths change from exp277."""

from dataclasses import replace

from experiments.exp232_sweep_cv1_decontam.training_contract import (
    MODEL_CONFIG as BASELINE_MODEL_CONFIG,
)

PREFIX = "s3://marin-us-east-02a/MarinFold/exp363_models_double_channel_width"
VERSION = "2026.10.13.1"
RUN_ID = "contacts-v1-exp363-m2-p06-full-epoch-5.86B"
TOKENIZER = (
    "eczech/contacts-v1-tokenizer-5d68a24a899f"
    "@80f4e411b957641e8c350b07bbe1e2c832697518"
)
MODEL_CONFIG = replace(
    BASELINE_MODEL_CONFIG,
    hidden_dim=4096,
    intermediate_dim=16384,
)
EXPECTED_PARAMETERS = 5_861_984_256
