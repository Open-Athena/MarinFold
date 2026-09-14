# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for exp157 CoreWeave training smokes."""

import os

from fray.types import ResourceConfig

CONTACTS_V1_S3_PREFIX = "s3://marin-us-east-02a/MarinFold/exp157_fixed_position_embeddings"
os.environ.setdefault("MARIN_PREFIX", CONTACTS_V1_S3_PREFIX)

CONTACTS_V1_TOKENIZER_REPO = "timodonnell/contacts-v1-tokenizer"
CONTACTS_V1_TOKENIZER_REVISION = "5d68a24a899f"
CONTACTS_V1_TOKENIZER = CONTACTS_V1_TOKENIZER_REPO

CONTACTS_V1_S3_CORPUS_BASE = "s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1"
# Reuse the already-built exp108 contacts-v1 token cache: same tokenizer, same
# corpus paths, same text column. This keeps the exp157 smoke as a model-training
# smoke rather than a Zephyr cache-build/dependency smoke.
CONTACTS_V1_TOKEN_CACHE_BASE = "s3://marin-us-east-02a/MarinFold/exp108_qwen_3b_contacts_v1/tokenized"
CONTACTS_V1_DATA_SEED = 0

# contacts-v1 prepends five native tokens before carrying the contacts-and-
# distances-v1 block. In that carried block, <p0> starts after 136 domain tokens;
# the tokenizer prepends <pad>/<eos>. Therefore <p0> id = 2 + 5 + 136 = 143.
CONTACTS_V1_P0_TOKEN_ID = 143
CONTACTS_V1_NUM_POSITION_TOKENS = 2000

PROTEIN_RESOURCES_H100 = ResourceConfig.with_gpu(
    "H100",
    count=8,
    cpu=32,
    ram="256g",
    disk="256g",
    replicas=1,
)
