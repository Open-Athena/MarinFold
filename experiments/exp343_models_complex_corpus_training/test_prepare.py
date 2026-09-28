"""The record contract is what keeps the corpus inside the model's vocabulary.

The published corpus ships a 2848-token tokenizer; the model has 2845
embeddings. Tokenizing with the model's tokenizer is only safe because no
complex document uses one of the three appended ids, and these tests pin the
check that enforces it rather than the assumption.
"""

import numpy as np
import pytest

from experiments.exp343_models_complex_corpus_training.prepare import (
    UNK_ID,
    VOCAB_SIZE,
    _validate_tokenized_record,
)

#: `<contacts-v1> <begin_sequence> <p1> <ALA> <begin_statements> <contact> <p1>
#: <p2> <end> <eos>`, with plausible ids for the non-special tokens.
VALID = [2, 8, 1200, 30, 9, 11, 1200, 1201, 10, 1]


def record(ids: list[int]) -> dict:
    return {"input_ids": np.asarray(ids, dtype=np.int32)}


def test_a_well_formed_complex_record_passes() -> None:
    assert _validate_tokenized_record(record(VALID))["input_ids"][0] == 2


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param([2, 8, 9, 10], id="too-short"),
        pytest.param([8, 1200, 30, 9, 11, 10, 1], id="missing-structure-token"),
        pytest.param([2, 1200, 30, 9, 11, 10, 1], id="missing-begin-sequence"),
        pytest.param([2, 8, 1200, 30, 11, 1200, 10, 1], id="missing-begin-statements"),
        pytest.param([2, 8, 1200, 30, 9, 11, 1200, 1201, 1], id="missing-end"),
        pytest.param([2, 8, 1200, 30, 9, 11, 1200, 1201, 10], id="missing-eos"),
        pytest.param([2, 8, 1200, 0, 9, 11, 1200, 1201, 10, 1], id="padding-token"),
    ],
)
def test_malformed_records_are_rejected(ids: list[int]) -> None:
    with pytest.raises(ValueError):
        _validate_tokenized_record(record(ids))


def test_an_appended_token_the_model_has_no_embedding_for_is_rejected() -> None:
    # 2845 is `<contacts-v1.sequence_only>` in the corpus's co-located
    # tokenizer, one past the model's last embedding row. A document reaching it
    # must fail the build, not index out of the table at training time.
    for out_of_contract in (UNK_ID, VOCAB_SIZE, VOCAB_SIZE + 2):
        ids = list(VALID)
        ids[2] = out_of_contract
        with pytest.raises(ValueError, match="out-of-range"):
            _validate_tokenized_record(record(ids))


def test_the_last_in_contract_id_is_accepted() -> None:
    ids = list(VALID)
    ids[2] = UNK_ID - 1
    assert _validate_tokenized_record(record(ids))["input_ids"][2] == UNK_ID - 1
