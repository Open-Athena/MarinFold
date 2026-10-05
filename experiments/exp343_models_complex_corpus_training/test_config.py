"""Pin the corpus inventory: exp277's four caches, unchanged, plus complexes."""

from experiments.exp277_models_single_mpnn_pilot.config import (
    CORPORA as EXP277_CORPORA,
)
from experiments.exp343_models_complex_corpus_training.config import (
    COMPLEX_DOCUMENTS,
    COMPLEX_TRAIN,
    COMPLEX_VALIDATION,
    CORPORA,
    EPOCH_PACKED_EXAMPLES,
    EPOCH_TRAIN_STEPS,
    GLOBAL_BATCH_SIZE,
    VALIDATION_CACHE,
)


def test_training_corpus_is_exp277s_plus_one() -> None:
    assert len(CORPORA) == len(EXP277_CORPORA) + 1
    for adopted, source in zip(CORPORA, EXP277_CORPORA, strict=False):
        assert (adopted.name, adopted.cache, adopted.documents, adopted.tokens) == (
            source.name,
            source.cache,
            source.documents,
            source.tokens,
        )
        assert adopted.split == "train"
    assert CORPORA[-1] == COMPLEX_TRAIN


def test_held_out_shard_is_excluded_from_training_and_accounted_for() -> None:
    assert COMPLEX_TRAIN.documents + COMPLEX_VALIDATION.documents == COMPLEX_DOCUMENTS
    assert COMPLEX_TRAIN.shards == 170
    assert COMPLEX_VALIDATION.shards == 1
    assert COMPLEX_TRAIN.source != COMPLEX_VALIDATION.source
    # Levanter reads a validation set from the `validation` subdirectory; a
    # training split name here would make the cache invisible as a validation
    # set and silently train on the held-out shard instead.
    assert COMPLEX_VALIDATION.split == "validation"
    assert COMPLEX_VALIDATION not in CORPORA


def test_cache_paths_are_distinct() -> None:
    caches = [f"{corpus.cache}/{corpus.split}" for corpus in CORPORA]
    caches.append(f"{COMPLEX_VALIDATION.cache}/{COMPLEX_VALIDATION.split}")
    caches.append(f"{VALIDATION_CACHE}/validation")
    assert len(set(caches)) == len(caches)


def test_step_count_follows_the_pinned_packed_example_count() -> None:
    assert GLOBAL_BATCH_SIZE == 128
    expected = -(-EPOCH_PACKED_EXAMPLES // GLOBAL_BATCH_SIZE)
    assert EPOCH_TRAIN_STEPS == expected


def test_a_limited_run_writes_under_its_own_label() -> None:
    """A smoke must not share the production `parts/` prefix.

    The worker's resume path reads back any part already present and checks its
    row count, so a 32-row smoke part left in the production prefix makes the
    next full run abort deterministically. `--name-suffix` only renames the Iris
    job, so the isolation has to come from the output label.
    """
    from experiments.exp343_models_complex_corpus_training.dispatch_complex_eval_cw import (
        output_label,
    )

    assert output_label("exp343-step280154", None) == "exp343-step280154"
    assert output_label("exp343-step280154", 32) == "exp343-step280154-smoke32"
    assert output_label("exp343-step280154", 32) != output_label("exp343-step280154", None)
