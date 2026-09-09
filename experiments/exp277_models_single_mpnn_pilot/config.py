"""Fixed scientific configuration for the single MPNN mixture pilot."""

from dataclasses import dataclass, replace

PREFIX = "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot"
VERSION = "2026.09.09.1"
RUN_ID = "contacts-v1-exp277-m2-p06-native-mpnn-1.5B"
TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"
AFDB_FRACTION = 4_432_940_838 / 74_475_864_003


@dataclass(frozen=True)
class Corpus:
    """One immutable source and its completed token cache."""

    name: str
    source: str
    cache: str
    documents: int
    shards: int
    weight: float
    tokens: int | None = None


CORPORA = (
    Corpus(
        "native-afdb",
        "",
        "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/tokenized/contacts_v1/afdb/2026.08.14",
        3_963_003,
        2067,
        AFDB_FRACTION / 2,
        4_432_940_838,
    ),
    Corpus(
        "native-esm",
        "",
        "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/tokenized/contacts_v1/esm/2026.08.14",
        65_553_178,
        3338,
        (1 - AFDB_FRACTION) / 2,
        70_042_923_165,
    ),
    Corpus(
        "mpnn-afdb",
        "s3://marin-us-east-02a/MarinFold/exp266/documents/*.parquet",
        f"{PREFIX}/tokenized/mpnn-afdb/{VERSION}",
        31_702_680,
        199,
        AFDB_FRACTION / 2,
    ),
    Corpus(
        "mpnn-esm",
        "s3://marin-us-east-02a/MarinFold/exp266/esm_documents/*.parquet",
        f"{PREFIX}/tokenized/mpnn-esm/{VERSION}",
        130_872_044,
        3338,
        (1 - AFDB_FRACTION) / 2,
    ),
)
VALIDATION_CACHE = "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1-val/2026.07.25"


def training_corpora(*, smoke: bool) -> tuple[Corpus, ...]:
    """Use audited small redesign caches for the isolated GPU startup test."""
    if not smoke:
        return CORPORA
    sizes = {"mpnn-afdb": (160000, 21584525), "mpnn-esm": (39180, 40616645)}
    return tuple(
        replace(
            corpus,
            cache=f"{PREFIX}/smoke-tokenized/{corpus.name}",
            documents=sizes[corpus.name][0],
            tokens=sizes[corpus.name][1],
        )
        if corpus.name in sizes
        else corpus
        for corpus in CORPORA
    )
