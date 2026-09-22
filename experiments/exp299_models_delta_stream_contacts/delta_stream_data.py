# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter data-format adapters for variable-length pretokenized documents."""

from dataclasses import dataclass

from levanter.data.text.formats import PrebuiltCacheProcessor, TextLmDatasetFormat
from levanter.tokenizers import MarinTokenizer


@dataclass(frozen=True)
class PackableTokenIdsFormat(TextLmDatasetFormat):
    """Load ``input_ids`` caches while selecting Levanter's document packer.

    Levanter's ``PrebuiltLmDatasetFormat`` denotes already fixed-length model
    examples and therefore bypasses packing. These caches instead contain one
    variable-length pretokenized protein document per row. Subclassing the text
    format selects ``PackedTokenDataset`` without retokenizing an existing
    cache; the preprocessor is retained for completeness if cache construction
    is requested accidentally.
    """

    def build_preprocessor(
        self,
        tokenizer: MarinTokenizer,
        *,
        enforce_eos: bool = True,
        enforce_bos: bool = True,
    ) -> PrebuiltCacheProcessor:
        del tokenizer, enforce_eos, enforce_bos
        return PrebuiltCacheProcessor("input_ids", None)
