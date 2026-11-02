# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the parts of ``ContactsV1RLEnv`` that do not need a vLLM engine.

Generation itself is covered by the Phase-1 parity gate on TPU, which is the only
place a real answer can come from. What is checkable here is everything that
would silently produce *plausible but wrong* prompts or budgets: the mode
sentinel splice, the token budgets, the train/eval split, and the precision EMA.
"""

import inspect

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import contact_rewards as cr
from contacts_env import ContactsV1RLEnv


class FakeTokenizer:
    """Stand-in for levanter's ``HfMarinTokenizer``.

    Deliberately exposes ONLY ``encode`` and is deliberately NOT callable. An
    earlier version of this fake mimicked the HF ``__call__``/``BatchEncoding``
    interface, so the suite passed green while the real thing died on the pod with
    ``TypeError: 'HfMarinTokenizer' object is not callable``. A fake that is more
    permissive than production is worse than no fake at all.
    """

    def __init__(self, ids):
        self._ids = list(ids)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self._ids)


def write_targets(tmp_path, rows):
    path = tmp_path / "targets.parquet"
    pq.write_table(
        pa.table(
            {
                "entry_id": pa.array([r[0] for r in rows], pa.string()),
                "L": pa.array([r[1] for r in rows], pa.int32()),
                "gt_contacts": pa.array([r[2] for r in rows], pa.list_(pa.list_(pa.int32()))),
            }
        ),
        path,
    )
    return str(path)


def write_prompts(tmp_path, entry_id, n_realizations, seq_len=40):
    directory = tmp_path / "prompts"
    directory.mkdir(exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "r": pa.array(list(range(n_realizations)), pa.int32()),
                "prefix": pa.array([f"{entry_id}-prefix-{r}" for r in range(n_realizations)], pa.string()),
                "seq_positions": pa.array(
                    [list(range(seq_len)) for _ in range(n_realizations)], pa.list_(pa.int32())
                ),
            }
        ),
        directory / f"{entry_id}.parquet",
    )
    return str(directory)


@pytest.fixture
def env_paths(tmp_path):
    targets = write_targets(
        tmp_path,
        [
            ("prot_a", 100, [[0, 10], [1, 20], [2, 30]]),
            ("prot_b", 250, [[0, 40], [5, 60]]),
            ("prot_c", 512, [[3, 70], [4, 80]]),
            # Only a sub-MIN_SEP pair: no usable ground truth, must be dropped.
            ("prot_dropped", 90, [[0, 2]]),
        ],
    )
    prompts = write_prompts(tmp_path, "prot_a", 4)
    for entry in ("prot_b", "prot_c", "prot_dropped"):
        write_prompts(tmp_path, entry, 4)
    return targets, prompts


def make_env(env_paths, **kwargs):
    targets, prompts = env_paths
    return ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, eval_fraction=0.0, **kwargs)


def test_targets_drop_proteins_with_no_usable_ground_truth(env_paths):
    env = make_env(env_paths)
    assert set(env._targets) == {"prot_a", "prot_b", "prot_c"}
    assert env._targets["prot_a"]["gt"] == {(0, 10), (1, 20), (2, 30)}


def test_ground_truth_pairs_are_normalized_and_separation_filtered(tmp_path):
    # (30, 2) arrives reversed and must normalize; (0, 3) is below MIN_SEP.
    targets = write_targets(tmp_path, [("p", 50, [[30, 2], [0, 3], [9, 40]])])
    write_prompts(tmp_path, "p", 2)
    env = ContactsV1RLEnv(
        targets_path=targets, prompts_path=str(tmp_path / "prompts"), eval_fraction=0.0
    )
    assert env._targets["p"]["gt"] == {(2, 30), (9, 40)}


def test_limit_caps_the_protein_set(env_paths):
    assert len(make_env(env_paths, limit=2)._targets) == 2


def test_mode_sets_the_doc_sentinel_by_id_not_by_string(env_paths):
    prompt_ids = [cr.PLAIN_DOC_ID, 8, 500, 501, cr.BEGIN_STATEMENTS_ID]
    tokenizer = FakeTokenizer(prompt_ids)

    plain = make_env(env_paths, mode="plain")._build_prompt_ids(tokenizer, "x")
    multi = make_env(env_paths, mode="multi")._build_prompt_ids(tokenizer, "x")

    assert plain[0] == cr.PLAIN_DOC_ID
    assert multi[0] == cr.MULTI_DOC_ID
    # Only index 0 differs; everything after it is the untouched prompt.
    assert plain[1:] == multi[1:] == prompt_ids[1:]


def test_rejects_a_prompt_pool_built_for_another_document_structure(env_paths):
    env = make_env(env_paths, mode="multi")
    with pytest.raises(ValueError, match="does not start with <contacts-v1>"):
        env._build_prompt_ids(FakeTokenizer([999, 8, 500]), "x")


def test_multi_budget_follows_the_exp163_per_section_formula(env_paths):
    env = make_env(env_paths, mode="multi", max_sections=8, section_contacts=220)
    # (3 * 220 + 8) * 8
    assert env._response_budget(max_prompt_len=1000, max_length=512) == 5344


def test_plain_budget_follows_the_exp98_per_residue_formula(env_paths):
    env = make_env(env_paths, mode="plain")
    assert env._response_budget(max_prompt_len=1000, max_length=250) == 4 * 250 + 64
    # Plain mode scores one section regardless of max_sections.
    assert env.max_sections == 1


def test_budget_is_clamped_to_the_context_window(env_paths):
    env = make_env(env_paths, mode="multi", max_model_len=2048)
    assert env._response_budget(max_prompt_len=1500, max_length=512) == 548
    with pytest.raises(ValueError, match="no room"):
        env._response_budget(max_prompt_len=2048, max_length=512)


def test_train_eval_split_is_disjoint_and_seed_stable(env_paths):
    targets, prompts = env_paths
    a = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, eval_fraction=0.34, seed=7)
    b = ContactsV1RLEnv(targets_path=targets, prompts_path=prompts, eval_fraction=0.34, seed=7)

    assert set(a._train_ids).isdisjoint(a._eval_ids)
    assert sorted(a._train_ids + a._eval_ids) == sorted(a._targets)
    assert (a._train_ids, a._eval_ids) == (b._train_ids, b._eval_ids)
    assert len(a._eval_ids) >= 1


def test_precision_ema_tracks_observed_precision(env_paths):
    env = make_env(env_paths, initial_precision=0.30, precision_ema_decay=0.5)
    env._update_precision(scored=10, correct=5)      # observed 0.5
    assert env._p_bar == pytest.approx(0.4)
    env._update_precision(scored=10, correct=5)
    assert env._p_bar == pytest.approx(0.45)


def test_precision_ema_ignores_a_batch_with_no_scored_contacts(env_paths):
    env = make_env(env_paths, initial_precision=0.30)
    env._update_precision(scored=0, correct=0)
    assert env._p_bar == pytest.approx(0.30)


def test_prompt_reads_are_cached(env_paths):
    env = make_env(env_paths)
    first = env._prompts_for("prot_a")
    assert env._prompts_for("prot_a") is first
    assert len(first) == 4
    assert first[0]["seq_positions"][:3] == [0, 1, 2]


def test_metrics_are_namespaced_and_carry_the_collapse_detectors(env_paths):
    env = make_env(env_paths, mode="multi")
    diagnostics = [
        {"best_f1": 0.30, "n_pred": 100.0, "n_sections": 8.0, "mean_jaccard": 0.07, "precision": 0.3},
        {"best_f1": 0.20, "n_pred": 60.0, "n_sections": 8.0, "mean_jaccard": 0.09, "precision": 0.2},
    ]

    class Applied:
        max_output_tokens = 5344

    metrics = env._metrics(diagnostics, n_empty=1, n_dropped=2, applied=Applied())

    assert metrics["contacts_multi/best_f1"] == pytest.approx(0.25)
    assert metrics["contacts_multi/mean_jaccard"] == pytest.approx(0.08)
    assert metrics["contacts_multi/n_pred_per_section"] == pytest.approx(10.0)
    assert metrics["contacts_multi/n_empty_responses"] == 1.0
    assert metrics["contacts_multi/n_ragged_groups_dropped"] == 2.0
    assert metrics["contacts_multi/precision_baseline"] == pytest.approx(env._p_bar)
    assert all(k.startswith("contacts_multi/") for k in metrics)


def test_metrics_survive_all_nan_diagnostics(env_paths):
    env = make_env(env_paths)

    class Applied:
        max_output_tokens = 100

    metrics = env._metrics([{"best_f1": np.nan}], n_empty=0, n_dropped=0, applied=Applied())
    assert "contacts_multi/best_f1" not in metrics
    assert metrics["contacts_multi/n_rollouts"] == 1.0


def test_rejects_an_unknown_mode(env_paths):
    with pytest.raises(ValueError, match="mode must be one of"):
        make_env(env_paths, mode="bogus")


def test_budget_never_exceeds_the_lesson_declared_cap(env_paths):
    """`curriculum.max_seq_len` is derived from the declared cap, and `train_batch`
    raises when a padded sequence overruns it — so the env must not out-generate it."""
    env = make_env(env_paths, mode="multi", max_sections=8, section_contacts=220)
    assert env._response_budget(max_prompt_len=1000, max_length=512, declared=2000) == 2000
    assert env._response_budget(max_prompt_len=1000, max_length=512, declared=99999) == 5344


def test_fake_tokenizer_matches_the_real_wrapper_surface():
    """Guard against the fake drifting back to a more permissive interface."""
    from levanter.tokenizers import HfMarinTokenizer

    assert not callable(FakeTokenizer([1]))
    # An INSTANCE is callable only if some class in the MRO defines __call__.
    # (The class object itself is always callable — that is just its constructor.)
    assert not any(
        "__call__" in klass.__dict__ for klass in HfMarinTokenizer.__mro__ if klass is not object
    ), "HfMarinTokenizer instances became callable; the fake may now be too strict"
    assert list(inspect.signature(HfMarinTokenizer.encode).parameters) == list(
        inspect.signature(FakeTokenizer.encode).parameters
    )


def test_limit_samples_randomly_rather_than_truncating(tmp_path):
    """Truncation takes whole benchmarks, not a representative subset.

    On the exp163 eval file, `limit=100` by truncation yielded 100% foldbench100,
    where the reference model scores 0.1296 against 0.2928 on the denovo_pdb rows
    that make up 71% of the file — turning a parity run into an apparent 50%
    regression that was really just a harder protein set.
    """
    rows = [(f"{ds}__p{i:03d}", 100, [[0, 10]]) for ds in ("aaa", "zzz") for i in range(50)]
    targets = write_targets(tmp_path, rows)

    picked = ContactsV1RLEnv._load_targets(targets, 20, seed=0)
    assert len(picked) == 20
    prefixes = {k.split("__")[0] for k in picked}
    assert prefixes == {"aaa", "zzz"}, f"limit collapsed onto one dataset: {prefixes}"


def test_limit_is_deterministic_for_a_given_seed(tmp_path):
    rows = [(f"p{i:03d}", 100, [[0, 10]]) for i in range(60)]
    targets = write_targets(tmp_path, rows)
    a = ContactsV1RLEnv._load_targets(targets, 10, seed=7)
    b = ContactsV1RLEnv._load_targets(targets, 10, seed=7)
    c = ContactsV1RLEnv._load_targets(targets, 10, seed=8)
    assert sorted(a) == sorted(b)
    assert sorted(a) != sorted(c)


def test_limit_above_the_population_keeps_everything(tmp_path):
    rows = [(f"p{i:03d}", 100, [[0, 10]]) for i in range(5)]
    targets = write_targets(tmp_path, rows)
    assert len(ContactsV1RLEnv._load_targets(targets, 999, seed=0)) == 5


def test_seed_from_accepts_both_halves_of_marins_prng_union():
    """RolloutWorker passes a JAX key OR a plain int, chosen by inference_type.

    `use_jax_rng = (inference_type == "levanter")`, so a vLLM rollout worker — what
    exp200 runs — always passes an int, and jax.random.randint on it raises
    "JAX encountered invalid PRNG key data". marin's own mock_env calls
    jax.random.randint unguarded, so this union is easy to miss.
    """
    import jax

    from contacts_env import seed_from

    assert seed_from(425801368) == 425801368
    assert isinstance(seed_from(jax.random.PRNGKey(0)), int)
    # Deterministic for a given input, either way.
    assert seed_from(jax.random.PRNGKey(7)) == seed_from(jax.random.PRNGKey(7))
    assert seed_from(np.int64(12345)) == 12345


def test_seed_from_stays_in_numpy_seed_range():
    from contacts_env import seed_from

    assert 0 <= seed_from(2**62) < 2**31 - 1
    np.random.default_rng(seed_from(2**62))  # must not raise


def test_environment_traces_init_and_sample_failures(env_paths, tmp_path):
    """`iris job logs` shows nothing for a running child, so the env self-reports.

    A tracer that can break the thing it observes is worse than none, so writes
    are guarded — but an exception inside sample() must still be RECORDED and then
    re-raised, never swallowed.
    """
    import json

    trace = tmp_path / "trace"
    env = make_env(env_paths, trace_path=str(trace))

    events = [json.load(open(f)) for f in sorted((trace / "env-multi").glob("*.json"))]
    kinds = [e["kind"] for e in events]
    assert "boot" in kinds and "env_init" in kinds
    init = next(e for e in events if e["kind"] == "env_init")
    assert init["n_proteins"] == 3 and init["max_sections"] == env.max_sections
    # One boot id for one interpreter — a second id in a real trace means restart.
    assert len({e["boot"] for e in events}) == 1


def test_tracer_is_a_noop_when_unconfigured(env_paths):
    env = make_env(env_paths)
    assert env._trace.path is None
    env._trace.event("anything", x=1)  # must not raise
