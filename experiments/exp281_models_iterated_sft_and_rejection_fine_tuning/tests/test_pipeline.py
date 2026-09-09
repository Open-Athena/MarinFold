"""Behavioral checks for interruption, selection, and weighted optimization."""

import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from marinfold.document_structures.contacts_v1_multi import (
    BEGIN,
    END,
    FINAL,
    MULTI,
    parse_history,
    truncate_history,
)
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from common import read_json, rows, write_json, write_rows
from compare import paired_difference
from corpus import LossProfile, build_example, select_candidate
from evaluate import score_pool
from generate import scored_candidate, validate_bootstrap_draft
from prepare import prepare_model, prepare_targets, target_from_document
from round_plan import plan
from train import ParquetStream, collate, weighted_loss_sum

HEADER = [MULTI, "<begin_sequence>", "<p0>", "<ALA>"]
HISTORY = [BEGIN, "<contact>", "<p0>", "<p6>", BEGIN, "<contact>", "<p1>", "<p7>"]
REFERENCE = ["<contact>", "<p0>", "<p7>"]


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    words = list(dict.fromkeys(["<pad>", "<unk>", "<contacts-v1>", *HEADER, *HISTORY, *REFERENCE, FINAL, END]))
    raw = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>")


def test_arbitrary_budgets_only_retain_complete_statements() -> None:
    expected_lengths = [0, 0, 0, 0, 4, 4, 4, 4, 8]
    for budget, length in enumerate(expected_lengths):
        result = truncate_history(HISTORY, budget)
        assert result == HISTORY[:length]
        parsed = parse_history([*result, FINAL, *REFERENCE, END])
        assert parsed.finished
    with pytest.raises(ValueError):
        truncate_history([BEGIN, "<contact>", "<p0>", "oops"], 4)


def test_explicit_final_boundary_does_not_merge_hypotheses() -> None:
    parsed = parse_history([*HISTORY, FINAL, *REFERENCE, END])
    assert parsed.hypotheses == (((0, 6),), ((1, 7),))
    assert parsed.final == ((0, 7),)
    assert truncate_history([*HISTORY, FINAL, *REFERENCE, END], 100) == HISTORY
    for invalid in ([*HISTORY, END], [BEGIN, FINAL, END], [FINAL, *REFERENCE, FINAL, END]):
        with pytest.raises(ValueError):
            parse_history(invalid)


def test_marker_supervision_and_first_answer_are_shifted_correctly(tokenizer) -> None:
    kwargs = dict(header=HEADER, history=HISTORY, reference=REFERENCE,
                  profile=LossProfile(), tokenizer=tokenizer, context=64)
    complete = build_example(**kwargs, forced=False)
    forced = build_example(**kwargs, forced=True)
    boundary = len(HEADER) + len(HISTORY)
    assert forced["loss_weights"][boundary] == 0
    assert complete["loss_weights"][boundary] == 1
    assert forced["loss_weights"][boundary + 1] == 1
    assert forced["loss_weights"][boundary - 1] == 0.1
    batch = collate([forced], tokenizer.pad_token_id, 64)
    logits = torch.zeros((1, len(forced["input_ids"]), len(tokenizer)), requires_grad=True)
    weighted_loss_sum(logits, batch["input_ids"], batch["weights"]).backward()
    assert logits.grad[0, boundary - 1].abs().sum() == 0
    assert logits.grad[0, boundary].abs().sum() > 0
    with pytest.raises(ValueError, match="do not truncate"):
        build_example(**{**kwargs, "context": 5}, forced=True)


def test_global_weighted_gradient_is_invariant_to_microbatch_partition() -> None:
    torch.manual_seed(281)
    network = torch.nn.Linear(5, 7).double()
    inputs = torch.randn(3, 6, 5, dtype=torch.float64)
    ids = torch.randint(0, 7, (3, 6))
    weights = torch.tensor([[0, .1, .1, .1, 1, 1], [0, 0, 0, 1, 1, 0], [0, .1, 1, 1, 0, 0]])
    together = copy.deepcopy(network)
    split = copy.deepcopy(network)
    (weighted_loss_sum(together(inputs), ids, weights) / weights[:, 1:].sum()).backward()
    for i in range(3):
        (weighted_loss_sum(split(inputs[i:i + 1]), ids[i:i + 1], weights[i:i + 1]) / weights[:, 1:].sum()).backward()
    for left, right in zip(together.parameters(), split.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-6, atol=1e-7)


def test_rejection_selects_by_final_answer_not_draft_quality() -> None:
    target = {"target_id": "protein", "header": HEADER, "reference": REFERENCE,
              "positions": list(range(8)), "n_residues": 8}
    bad = scored_candidate(target, [BEGIN, *REFERENCE, FINAL, "<contact>", "<p0>", "<p6>", END],
                           forced=False, budget=20, candidate_id=0, generator="checkpoint")
    good = scored_candidate(target, [*HISTORY, FINAL, *REFERENCE, END],
                            forced=False, budget=20, candidate_id=1, generator="checkpoint")
    assert select_candidate([bad, good], "best", 281)["candidate_id"] == 1
    assert {select_candidate([bad, good], "random", seed)["candidate_id"] for seed in range(20)} == {0, 1}
    with pytest.raises(ValueError, match="share target"):
        select_candidate([bad, {**good, "budget": 21}], "best", 281)
    invalid = scored_candidate(target, [FINAL, "<contact>", "<p0>", END],
                               forced=False, budget=20, candidate_id=2, generator="checkpoint")
    assert not invalid["valid"] and invalid["error"]
    with pytest.raises(ValueError, match="all candidate"):
        select_candidate([invalid], "best", 281)


def test_stream_rank_disjointness_and_resume_across_epochs(tmp_path: Path) -> None:
    paths = []
    for shard in range(2):
        path = str(tmp_path / f"part-{shard}.parquet")
        write_rows(path, [{"input_ids": [shard * 40 + i, 1], "loss_weights": [0.0, 1.0]} for i in range(40)])
        paths.append(path)
    streams = [ParquetStream(paths, rank, 2, 281) for rank in range(2)]
    values = [{row["input_ids"][0] for row in s.iterate(repeat=False)} for s in streams]
    assert not values[0] & values[1]
    assert values[0] | values[1] == set(range(80))
    reference = list(itertools.islice(streams[0].iterate(), 200))
    assert list(itertools.islice(streams[0].iterate(skip=77), 30)) == reference[77:107]


def test_prepare_retains_wrapped_position_mapping() -> None:
    header = ["<contacts-v1>", "<begin_sequence>"]
    for p in (1998, 1999, 0, 1, 2, 3, 4, 5):
        header.extend([f"<p{p}>", "<ALA>"])
    header.extend(["<n-term>", "<p1998>", "<c-term>", "<p5>"])
    document = " ".join([*header, BEGIN, "<contact>", "<p1998>", "<p5>", END])
    target = target_from_document(document, "wrapped")
    assert target["positions"] == [1998, 1999, 0, 1, 2, 3, 4, 5]
    assert target["n_residues"] == 8


def test_corpus_cli_keeps_reference_labels_after_selecting_imperfect_answer(tmp_path, tokenizer) -> None:
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    target = {"target_id": "protein", "header": HEADER,
              "reference": [*REFERENCE, "<contact>", "<p0>", "<p6>"],
              "positions": list(range(8)), "n_residues": 8}
    candidate = scored_candidate(target, [*HISTORY, FINAL, *REFERENCE, END],
                                 forced=True, budget=8, candidate_id=0, generator="checkpoint")
    candidate["bootstrap"] = False
    path = str(tmp_path / "candidates.parquet")
    write_rows(path, [candidate])
    write_json(str(tmp_path / "candidates.json"), {"config": {"model": "checkpoint"}})
    subprocess.run(["uv", "run", "--no-project", sys.executable,
                    str(Path(__file__).resolve().parents[1] / "build_corpus.py"),
                    "--candidates", path, "--tokenizer", str(tmp_path / "tokenizer"),
                    "--output", str(tmp_path / "corpus"), "--selection", "best"], check=True)
    manifest = read_json(str(tmp_path / "corpus/manifest.json"))
    example = next(rows(manifest["shards"][0]))
    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    assert tokens[tokens.index(FINAL) + 1:-1] == target["reference"]
    assert example["selected_f1"] == pytest.approx(2 / 3)
    assert example["loss_weights"][tokens.index(FINAL)] == 0


def test_evaluation_includes_failed_candidates() -> None:
    target = {"target_id": "protein", "header": HEADER, "reference": REFERENCE,
              "positions": list(range(8)), "n_residues": 8}
    good = scored_candidate(target, [*HISTORY, FINAL, *REFERENCE, END],
                            forced=False, budget=8, candidate_id=0, generator="checkpoint")
    bad = scored_candidate(target, [FINAL, "<contact>"],
                           forced=False, budget=8, candidate_id=1, generator="checkpoint")
    good["bootstrap"] = bad["bootstrap"] = False
    result = score_pool([good, bad])
    assert result["final_f1"] == 0.5
    assert result["valid_fraction"] == 0.5
    assert result["final_r_precision_all"] == 1.0


def test_round_plan_orders_stages_and_never_selects_validation_winners() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/format.json"
    config = json.loads(path.read_text())
    config.update(phase="rejection", generator=config["initial_model"], candidates=4)
    stages = plan(config)
    assert [s["group"] for s in stages] == sorted(s["group"] for s in stages)
    corpora = [s for s in stages if s["stage"] == "corpus"]
    assert corpora[0]["arguments"][corpora[0]["arguments"].index("--selection") + 1] == "best"
    assert corpora[1]["arguments"][corpora[1]["arguments"].index("--selection") + 1] == "random"
    assert stages[-1]["stage"] == "train" and stages[-1]["gpus"] == 8


def test_model_initialization_appends_tokens_without_id_or_weight_drift(tmp_path) -> None:
    vocabulary = {"<pad>": 0, "<unk>": 1, "<contacts-v1>": 2, BEGIN: 3, END: 4, "<contact>": 5}
    raw = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    original_tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>")
    original = Qwen3ForCausalLM(Qwen3Config(vocab_size=6, hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                         head_dim=16, max_position_embeddings=128))
    source = tmp_path / "source"
    original.save_pretrained(source)
    original_tokenizer.save_pretrained(source)
    prepare_model(str(source), tmp_path / "prepared")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "prepared")
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "prepared", dtype=torch.bfloat16)
    assert len(tokenizer) == 8
    assert all(tokenizer.convert_tokens_to_ids(token) == index for token, index in vocabulary.items())
    assert tokenizer.convert_tokens_to_ids(MULTI) == 6
    assert tokenizer.convert_tokens_to_ids(FINAL) == 7
    torch.testing.assert_close(model.get_input_embeddings().weight[:6],
                               original.get_input_embeddings().weight.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(model.get_output_embeddings().weight[:6],
                               original.get_output_embeddings().weight.to(torch.bfloat16), rtol=0, atol=0)


def test_bootstrap_comparison_weights_proteins_not_budget_rows() -> None:
    baseline = {("a", "True", "0"): 0.1, ("a", "True", "100"): 0.1, ("b", "True", "0"): 0.2}
    candidate = {("a", "True", "0"): 0.3, ("a", "True", "100"): 0.5, ("b", "True", "0"): 0.1}
    result = paired_difference(baseline, candidate)
    assert result["proteins"] == 2
    assert result["delta"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="identical"):
        paired_difference(baseline, {("a", "True", "0"): 0.3})


def test_small_validation_set_can_leave_ddp_ranks_empty(tmp_path: Path) -> None:
    path = str(tmp_path / "validation.parquet")
    write_rows(path, [{"input_ids": [1, 2], "loss_weights": [0.0, 1.0]}])
    empty = ParquetStream([path], 7, 8, 281, allow_empty=True)
    assert list(empty.iterate(repeat=False)) == []
    with pytest.raises(ValueError, match="empty"):
        list(empty.iterate())
    with pytest.raises(ValueError, match="nonempty"):
        ParquetStream([path], 7, 8, 281)


def test_bounded_target_pool_balances_sources_and_preserves_labels(tmp_path: Path) -> None:
    header = ["<contacts-v1>", "<begin_sequence>"]
    for position in range(8):
        header.extend([f"<p{position}>", "<ALA>"])
    header.extend(["<n-term>", "<p0>", "<c-term>", "<p7>"])
    document = " ".join([*header, BEGIN, *REFERENCE, END])
    sources = []
    for name in ("afdb", "esm"):
        path = str(tmp_path / f"{name}.parquet")
        write_rows(path, [{"entry_id": str(i), "document": document} for i in range(10)])
        sources.append({"name": name, "uri": path, "id_column": "entry_id", "document_column": "document"})
    output = str(tmp_path / "targets")
    prepare_targets({"sources": sources, "source_revision": "pinned", "decontamination_reference": "test"},
                    output, 8, 4, per_source_limit=2)
    manifest = read_json(output + "/manifest.json")
    assert manifest["source_counts"] == {"afdb": 2, "esm": 2}
    retained = [row for paths in manifest["shards"].values() for path in paths for row in rows(path)]
    assert len(retained) == 4 and all(row["reference"] == REFERENCE for row in retained)
    assert sum(bool(paths) for paths in manifest["shards"].values()) == 1  # identical sequences stay together


def test_bootstrap_validation_uses_format_and_positions_without_reference() -> None:
    validate_bootstrap_draft(["<contact>", "<p0>", "<p7>", END], list(range(8)))
    for tokens in ([END], ["<contact>", "<p0>", END], ["<contact>", "<p0>", "<p100>", END],
                   ["<contact>", "<p0>", "<p1>", END], [BEGIN, "<contact>", "<p0>", "<p7>", END]):
        with pytest.raises(ValueError):
            validate_bootstrap_draft(tokens, list(range(8)))
