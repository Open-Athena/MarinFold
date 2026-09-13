"""Audited complete corpora for the single-epoch native/MPNN pilot."""

from dataclasses import dataclass

PREFIX = "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot"
VERSION = "2026.09.09.1"
RUN_ID = "contacts-v1-exp277-m2-p06-full-epoch-1.5B"
TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"
EPOCH_PACKED_EXAMPLES = 34_092_146
EPOCH_TRAIN_STEPS = (EPOCH_PACKED_EXAMPLES + 127) // 128


@dataclass(frozen=True)
class Corpus:
    """One immutable source and its completed token cache."""

    name: str
    source: str
    cache: str
    documents: int
    shards: int
    tokens: int | None = None


CORPORA = (
    Corpus(
        "native-afdb",
        "",
        "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/tokenized/contacts_v1/afdb/2026.08.14",
        3_963_003,
        2067,
        4_432_940_838,
    ),
    Corpus(
        "native-esm",
        "",
        "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/tokenized/contacts_v1/esm/2026.08.14",
        65_553_178,
        3338,
        70_042_923_165,
    ),
    Corpus(
        "mpnn-afdb",
        "s3://marin-us-east-02a/MarinFold/exp266/documents/*.parquet",
        f"{PREFIX}/tokenized/mpnn-afdb/{VERSION}",
        31_702_680,
        199,
        35_352_543_972,
    ),
    Corpus(
        "mpnn-esm",
        "s3://marin-us-east-02a/MarinFold/exp266/esm_documents/*.parquet",
        f"{PREFIX}/tokenized/mpnn-esm/{VERSION}",
        130_872_044,
        3338,
        138_755_354_859,
    ),
)
VALIDATION_CACHE = "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1-val/2026.07.25"
