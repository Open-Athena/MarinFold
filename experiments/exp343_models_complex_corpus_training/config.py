# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Corpora for exp277's full training set plus the #294 predicted-complex corpus.

The four exp277 caches are adopted by path and pinned count -- nothing is
re-tokenized. Only the complex corpus is new, and it arrives from the public
`open-athena/MarinFold` HF bucket rather than from CoreWeave storage, so it is
staged in-region first (`stage.py`) and tokenized from there (`prepare.py`).

One published shard is held out as a complex validation set. The monomer eval
sets contain no complexes, so without it the experiment cannot measure the thing
it is testing.
"""

from dataclasses import dataclass

from experiments.exp277_models_single_mpnn_pilot.config import (
    CORPORA as EXP277_CORPORA,
)
from experiments.exp277_models_single_mpnn_pilot.config import (
    VALIDATION_CACHE as MONOMER_VALIDATION_CACHE,
)

PREFIX = "s3://marin-us-east-02a/MarinFold/exp343_models_complex_corpus_training"
VERSION = "2026.09.28.1"
RUN_ID = "contacts-v1-exp343-m2-p06-complex-1.5B"
TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"

#: The published #294 corpus. Its co-located tokenizer carries three tokens the
#: 1.5B model has no embeddings for (`<contacts-v1.sequence_only>`,
#: `<retract>`, `<contacts-v1.backtracking>`, ids 2845-2847, appended by other
#: document structures). Ids 0-2844 are identical to TOKENIZER's, and no complex
#: document uses a higher id -- `prepare.py` enforces that per record rather
#: than trusting it, so the model's vocabulary stays at 2845.
COMPLEX_BUCKET = "open-athena/MarinFold"
COMPLEX_BUCKET_PREFIX = "data/document_structures/contacts_v1_complex"
COMPLEX_SHARDS = 171
COMPLEX_DOCUMENTS = 3_410_738
#: `num_tokens` summed over the published corpus, which counts each document
#: without the `<eos>` the tokenizer appends. The cache therefore reports
#: COMPLEX_TOKENS + one token per document, less the `<eos>` trimmed from every
#: document already at the 8192 cap.
COMPLEX_PUBLISHED_TOKENS = 11_151_472_024

#: Held out for validation, and excluded from training. Shard assignment in #294
#: randomised document order, so this is a random sample of the corpus rather
#: than a slice of one arm.
VALIDATION_SHARD = 170

STAGED_TRAIN = f"{PREFIX}/documents/train"
STAGED_VALIDATION = f"{PREFIX}/documents/validation"
STAGE_MANIFEST = f"{PREFIX}/documents/stage-manifest.json"


@dataclass(frozen=True)
class Corpus:
    """One immutable source and its token cache, under one levanter split.

    `source` is an fsspec glob, empty for a cache adopted from another
    experiment. `split` is the subdirectory levanter reads the cache from, and
    must be `validation` for anything passed to `train_lm(validation=...)`.
    """

    name: str
    source: str
    cache: str
    documents: int
    shards: int
    tokens: int | None = None
    split: str = "train"


#: Pinned by `stage.py`, which reads every staged shard's parquet footer. They
#: must sum to COMPLEX_DOCUMENTS; `prepare.py` re-derives and rechecks both.
COMPLEX_TRAIN_DOCUMENTS = 3_400_000
COMPLEX_VALIDATION_DOCUMENTS = 10_738

COMPLEX_TRAIN = Corpus(
    "complex",
    f"{STAGED_TRAIN}/*.parquet",
    f"{PREFIX}/tokenized/complex/{VERSION}",
    COMPLEX_TRAIN_DOCUMENTS,
    COMPLEX_SHARDS - 1,
)
COMPLEX_VALIDATION = Corpus(
    "complex-validation",
    f"{STAGED_VALIDATION}/*.parquet",
    f"{PREFIX}/tokenized/complex-validation/{VERSION}",
    COMPLEX_VALIDATION_DOCUMENTS,
    1,
    split="validation",
)

#: exp277's four caches verbatim, then the new corpus. Order fixes the
#: concatenation the one-epoch shuffle permutes, so it is part of the run's
#: identity; exp277's own order is preserved and the new corpus appended.
CORPORA = tuple(
    Corpus(
        corpus.name,
        corpus.source,
        corpus.cache,
        corpus.documents,
        corpus.shards,
        corpus.tokens,
    )
    for corpus in EXP277_CORPORA
) + (COMPLEX_TRAIN,)

VALIDATION_CACHE = MONOMER_VALIDATION_CACHE

#: Pinned by `audit_epoch.py` before launch, using the trainer's own packer.
EPOCH_PACKED_EXAMPLES = 0
GLOBAL_BATCH_SIZE = 128
EPOCH_TRAIN_STEPS = (EPOCH_PACKED_EXAMPLES + GLOBAL_BATCH_SIZE - 1) // GLOBAL_BATCH_SIZE
