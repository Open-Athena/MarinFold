"""Exercise real parquet coverage, bounded batches, lineage filtering and replay."""

import copy
import hashlib

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from common import convert_document, split_key
from corpus_catalog import canonical_bytes
from corpus_stream import BATCH_ROWS, CorpusStream, encode_document
from test_documents import TEXT


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]")


def make_catalog(tmp_path) -> tuple[str, dict]:
    catalog = {}
    for source, size in [("native-afdb", 279), ("mpnn-afdb", 137), ("complex", 3)]:
        shards = []
        for shard in range(4):
            rows = [
                {
                    "document": TEXT,
                    "entry_id": f"{source}-{shard}-{i}",
                    "seq_cluster_id": "lineage",
                }
                for i in range(size)
            ]
            path = tmp_path / f"{source}-{shard}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=71)
            fs, remote = fsspec.core.url_to_fs(str(path))
            info = fs.info(remote)
            shards.append(
                {
                    "uri": str(path),
                    "size": info["size"],
                    "etag": info.get("ETag"),
                    "rows": size,
                    "columns": list(rows[0]),
                }
            )
        catalog[source] = {"rows": size * 4, "shards": shards}
    encoded = canonical_bytes(catalog)
    (tmp_path / "catalog.json").write_bytes(encoded)
    return str(tmp_path), {"catalog_sha256": hashlib.sha256(encoded).hexdigest()}


def test_every_source_row_once_and_no_rank_overlap(tmp_path) -> None:
    prefix, manifest = make_catalog(tmp_path)
    ranks = [
        CorpusStream(prefix, manifest, tiny_tokenizer(), rank, 2) for rank in range(2)
    ]
    seen = []
    for stream in ranks:
        n = sum(stream.totals.values())
        rows = [stream.next_raw()[1]["entry_id"] for _ in range(n)]
        assert len(set(rows)) == n
        assert max(len(batch[2]) for batch in stream.batches.values()) <= BATCH_ROWS
        seen.append(set(rows))
        stream.close()
    assert not seen[0] & seen[1]
    assert len(seen[0] | seen[1]) == 4 * (279 + 137 + 3)


@pytest.mark.parametrize("advance", [1, 130, 834])
def test_exact_replay_across_batches_files_and_epoch(tmp_path, advance) -> None:
    prefix, manifest = make_catalog(tmp_path)
    stream = CorpusStream(prefix, manifest, tiny_tokenizer(), 0, 2)
    for _ in range(advance):
        stream.next_raw()
    state = copy.deepcopy(stream.state)
    expected = [stream.next_raw() for _ in range(100)]
    resumed = CorpusStream(prefix, manifest, tiny_tokenizer(), 0, 2)
    resumed.state = state
    assert [resumed.next_raw() for _ in range(100)] == expected
    assert resumed.state == stream.state
    stream.close()
    resumed.close()


def test_reject_changed_catalog_rank_and_source(tmp_path) -> None:
    prefix, manifest = make_catalog(tmp_path)
    stream = CorpusStream(prefix, manifest, tiny_tokenizer(), 0, 2)
    state = copy.deepcopy(stream.state)
    state["world"] = 4
    with pytest.raises(ValueError, match="rank layout"):
        stream.state = state
    for path in tmp_path.glob("*.parquet"):
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        stream.next_raw()
    (tmp_path / "catalog.json").write_text("{}")
    with pytest.raises(ValueError, match="catalog differs"):
        CorpusStream(prefix, manifest, tiny_tokenizer(), 0, 2)


def test_native_and_redesigns_share_pilot_holdout() -> None:
    cluster = next(
        str(i) for i in range(10000) if int(split_key("", str(i))[:8], 16) % 100 == 0
    )
    native = {"document": TEXT, "entry_id": "native", "seq_cluster_id": cluster}
    redesign = {
        **native,
        "document": TEXT.replace("<ALA>", "<ARG>"),
        "entry_id": "design",
    }
    for source, row in [("native-afdb", native), ("mpnn-afdb", redesign)]:
        assert encode_document(row, source, tiny_tokenizer()) == (None, "validation")


def test_complex_wraparound_and_interchain_adjacent_contact() -> None:
    text = (
        "<contacts-v1> <begin_sequence> <p1999> <ALA> <p0> <CYS> "
        "<p10> <GLY> <p11> <HIS> <n-term> <p1999> <c-term> <p0> "
        "<n-term> <p10> <c-term> <p11> <begin_statements> "
        "<contact> <p11> <p1999> <end>"
    )
    doc = convert_document(text)
    assert doc.sequence == "GHAC"
    assert doc.contacts == ((2, 3),)
    assert "Chain 1 (residues 1-2):\nGH" in doc.prompted_prefix
    assert "Chain 2 (residues 3-4):\nAC" in doc.prompted_prefix
    assert doc.raw_prefix + doc.raw_completion == text
    with pytest.raises(ValueError, match="too-close"):
        convert_document(
            text.replace("<contact> <p11> <p1999>", "<contact> <p11> <p10>")
        )
    with pytest.raises(ValueError, match="overlapping"):
        convert_document(text.replace("<c-term> <p0>", "<c-term> <p10>"))


def test_malformed_complex_duplicate_termini() -> None:
    with pytest.raises(ValueError, match="termini"):
        convert_document(
            TEXT.replace("<n-term> <p1998>", "<n-term> <p1998> <n-term> <p1998>")
        )


def test_exclude_both_renderings_when_only_raw_exceeds_context() -> None:
    lineage = next(
        str(i) for i in range(100) if int(split_key("", str(i))[:8], 16) % 100 != 0
    )
    text = (
        "<contacts-v1> <begin_sequence> "
        + " ".join(f"<p{i}> <ALA>" for i in range(100))
        + " <n-term> <p0> <c-term> <p99> <begin_statements> "
        + " ".join(
            f"<contact> <p{i}> <p{j}>" for i in range(100) for j in range(i + 6, 100)
        )
        + " <end>"
    )
    tokenizer = tiny_tokenizer()
    doc = convert_document(text)
    assert len(tokenizer(text)["input_ids"]) > 16384
    assert (
        len(tokenizer(doc.prompted_prefix + doc.prompted_completion)["input_ids"])
        < 16384
    )
    row = {"document": text, "entry_id": "long-raw", "seq_cluster_id": lineage}
    assert encode_document(row, "native-afdb", tokenizer) == (None, "too_long")
