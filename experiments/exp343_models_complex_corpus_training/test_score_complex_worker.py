"""Resume bookkeeping for the complex-loss worker.

The worker runs standalone in a vendor GPU image and imports torch, which this
environment has no reason to carry -- the training stack here is jax. A stub is
installed so the pure bookkeeping (scoring order, part partitioning, resume,
coverage) can be covered without a GPU. Only `torch` is stubbed; every function
under test is exercised as written.
"""

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

WORKER = Path(__file__).resolve().parent / "score_complex_worker.py"


def load_worker():
    """Import the standalone worker with torch stubbed out."""
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        # `transformers` probes for torch with `importlib.util.find_spec`, which
        # raises on a module whose `__spec__` is None, so the stub needs one.
        torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
        torch.long = "long"
        torch.bfloat16 = "bfloat16"
        sys.modules["torch"] = torch
    spec = importlib.util.spec_from_file_location("exp343_score_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load_worker()


def rows(lengths: list[int]) -> list[dict]:
    return [
        {"document_id": f"d{index}", "num_tokens": length}
        for index, length in enumerate(lengths)
    ]


def test_scoring_order_is_longest_first_and_stable() -> None:
    order = worker.part_order(rows([100, 900, 300, 900, 50]))
    assert order == [1, 3, 2, 0, 4]
    # Stability is what makes a written part safe to skip on restart: the same
    # shard must partition into the same parts every time.
    assert worker.part_order(rows([100, 900, 300, 900, 50])) == order


def test_every_document_appears_in_exactly_one_part() -> None:
    documents = rows([(i * 37) % 911 for i in range(2000)])
    order = worker.part_order(documents)
    size = worker.DOCUMENTS_PER_PART
    parts = [order[start : start + size] for start in range(0, len(order), size)]
    assert sum(len(part) for part in parts) == len(documents)
    assert sorted(index for part in parts for index in part) == list(range(2000))
    assert len(parts) == -(-len(documents) // size)


def test_score_resumes_from_written_parts_and_recomputes_the_rest(tmp_path, monkeypatch) -> None:
    documents = rows([500 - index for index in range(40)])
    monkeypatch.setattr(worker, "DOCUMENTS_PER_PART", 10)
    monkeypatch.setattr(worker, "BATCH_DOCUMENTS", 4)
    computed: list[int] = []

    def fake_score_batch(model, tokenizer, source, chunk, sections):
        computed.extend(chunk)
        return [
            {"document_index": index, "document_id": source[index]["document_id"],
             "scored_positions": 1, "nll_total": float(index)}
            for index in chunk
        ]

    monkeypatch.setattr(worker, "score_batch", fake_score_batch)
    parts = tmp_path / "parts"
    parts.mkdir()
    # Pre-write part 0 as if a previous attempt had finished it.
    order = worker.part_order(documents)
    done = order[:10]
    pq.write_table(
        pa.Table.from_pylist(
            [{"document_index": i, "document_id": documents[i]["document_id"],
              "scored_positions": 1, "nll_total": float(i)} for i in done]
        ),
        parts / "part-00000.parquet",
    )
    scored = worker.score(None, None, documents, None, parts_prefix=str(parts))
    assert len(scored) == 40
    assert [row["document_index"] for row in scored] == list(range(40))
    # The finished part was read back, not recomputed.
    assert set(computed).isdisjoint(done)
    assert sorted(computed) == sorted(order[10:])
    assert len(list(parts.glob("part-*.parquet"))) == 4


def test_a_short_part_is_rejected_rather_than_trusted(tmp_path, monkeypatch) -> None:
    documents = rows([500 - index for index in range(20)])
    monkeypatch.setattr(worker, "DOCUMENTS_PER_PART", 10)
    parts = tmp_path / "parts"
    parts.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{"document_index": 0, "nll_total": 1.0}]),
        parts / "part-00000.parquet",
    )
    with pytest.raises(ValueError, match="expected 10"):
        worker.score(None, None, documents, None, parts_prefix=str(parts))
