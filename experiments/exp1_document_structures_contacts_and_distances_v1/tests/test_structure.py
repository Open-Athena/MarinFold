# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contacts-and-distances-v1 document structure.

The impl is a set of plain modules in the experiment directory:
``vocab.py`` (token list), ``parse.py`` (gemmi-backed parsing),
``generate.py`` (training-document generation), ``inference.py``
(predict + evaluate), and ``cli.py`` (argparse driver). Tests import
these as plain modules — the experiment dir is on ``sys.path``.

The network-marked tests download the pinned published tokenizer
(``timodonnell/protein-docs-tokenizer@83f597d88e9b``) and assert
byte-equivalence. Skip them with ``pytest -m 'not network'`` if
you're offline.

Run::

    uv sync --extra test
    uv run pytest tests/ -v
"""

import argparse
import math
import sys
import textwrap
from pathlib import Path

import pytest


# Make the experiment dir importable so we can reach vocab / parse /
# generate / inference / cli directly.
_EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXP_DIR))

from vocab import (  # noqa: E402
    AMINO_ACIDS, ATOM_NAMES, CONTACT_TYPES, CONTROL_TOKENS,
    CONTEXT_LENGTH, DISTANCE_BINS, DISTANCE_MARKER, MAX_POSITION,
    NAME, PLDDT_BINS, UNK_TOKEN, all_domain_tokens,
)
from parse import ParsedStructure, Residue, parse_structure  # noqa: E402
import generate  # noqa: E402
import inference  # noqa: E402
import cli  # noqa: E402

from marinfold_document_structures import (  # noqa: E402
    EvalResult,
    build_tokenizer,
    write_eval,
    write_predictions,
)


# ---------------------------------------------------------------------------
# Vocab shape
# ---------------------------------------------------------------------------


def test_tokens_count_matches_canonical_2838():
    """Domain vocab is 2838 tokens; +2 specials = 2840 total."""
    tokens = all_domain_tokens()
    n_expected = (
        len(CONTROL_TOKENS)
        + len(CONTACT_TYPES)
        + len(DISTANCE_MARKER)
        + len(DISTANCE_BINS)
        + len(PLDDT_BINS)
        + len(AMINO_ACIDS)
        + len(ATOM_NAMES)
        + (MAX_POSITION + 1)
        + len(UNK_TOKEN)
    )
    assert n_expected == 2838
    assert len(tokens) == 2838


def test_tokens_unique_and_returned_as_copy():
    tokens = all_domain_tokens()
    assert len(tokens) == len(set(tokens))
    a = all_domain_tokens()
    a.append("<broken>")
    assert "<broken>" not in all_domain_tokens()


def test_token_order_invariants():
    tokens = all_domain_tokens()
    assert tokens[0] == "<contacts-and-distances-v1>"
    assert tokens[1] == "<begin_sequence>"
    assert tokens[4] == "<long-range-contact>"
    assert tokens[7] == "<distance>"
    assert tokens[8] == "<d0.5>"
    assert tokens[8 + 63] == "<d32.0>"
    assert tokens[-1] == "<UNK>"


# ---------------------------------------------------------------------------
# build_tokenizer
# ---------------------------------------------------------------------------


def test_build_tokenizer_from_vocab():
    tok = build_tokenizer(all_domain_tokens())
    assert len(tok) == 2840
    assert tok.convert_tokens_to_ids("<pad>") == 0
    assert tok.convert_tokens_to_ids("<eos>") == 1


def test_build_tokenizer_roundtrip_sample():
    tok = build_tokenizer(all_domain_tokens())
    sample = (
        "<contacts-and-distances-v1> <begin_sequence> "
        "<MET> <LYS> <PHE> <CYS> <ASP> <TYR> <GLY> <LEU> "
        "<begin_statements> "
        "<long-range-contact> <p1> <p50> "
        "<distance> <p10> <p45> <CA> <CB> <d4.5> "
        "<plddt_80_85> <end>"
    )
    ids = tok.encode(sample, add_special_tokens=False)
    assert len(ids) == len(sample.split())
    unk_id = tok.convert_tokens_to_ids("<UNK>")
    assert unk_id not in ids


REVISION_PIN = "83f597d88e9b"


@pytest.mark.network
def test_published_tokenizer_vocab_matches():
    pytest.importorskip("huggingface_hub")
    from transformers import AutoTokenizer

    published = AutoTokenizer.from_pretrained(
        "timodonnell/protein-docs-tokenizer",
        revision=REVISION_PIN,
    )
    local = build_tokenizer(all_domain_tokens())
    assert published.get_vocab().keys() == local.get_vocab().keys()
    mismatches = [
        (t, published.get_vocab()[t], local.get_vocab()[t])
        for t in published.get_vocab()
        if published.get_vocab()[t] != local.get_vocab()[t]
    ]
    assert not mismatches, f"first 5: {mismatches[:5]}"


@pytest.mark.network
def test_published_tokenizer_encodes_identically():
    pytest.importorskip("huggingface_hub")
    from transformers import AutoTokenizer

    published = AutoTokenizer.from_pretrained(
        "timodonnell/protein-docs-tokenizer",
        revision=REVISION_PIN,
    )
    local = build_tokenizer(all_domain_tokens())
    sample = (
        "<contacts-and-distances-v1> <begin_sequence> "
        "<MET> <LYS> <PHE> <CYS> <ASP> <TYR> <GLY> <LEU> "
        "<begin_statements> "
        "<long-range-contact> <p1> <p50> "
        "<distance> <p10> <p45> <CA> <CB> <d4.5> "
        "<medium-range-contact> <p3> <p20> "
        "<short-range-contact> <p5> <p12> "
        "<distance> <p2> <p80> <NZ> <O> <d15.0> "
        "<plddt_80_85> <end>"
    )
    assert (
        published.encode(sample, add_special_tokens=False)
        == local.encode(sample, add_special_tokens=False)
    )


# ---------------------------------------------------------------------------
# Generator helpers (pure stdlib)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d, expected", [
    (0.0, "d0.5"), (0.25, "d0.5"), (0.5, "d0.5"), (0.51, "d1.0"),
    (4.0, "d4.0"), (4.5, "d4.5"), (31.99, "d32.0"), (32.0, "d32.0"),
    (45.7, "d32.0"),
])
def test_distance_token_bins(d, expected):
    assert generate._distance_token(d) == expected


@pytest.mark.parametrize("p, expected", [
    (50.0, "plddt_lt70"), (69.99, "plddt_lt70"),
    (70.0, "plddt_70_75"), (74.99, "plddt_70_75"),
    (75.0, "plddt_75_80"), (84.0, "plddt_80_85"),
    (95.0, "plddt_95_100"), (100.0, "plddt_95_100"),
])
def test_plddt_bin_token(p, expected):
    edges = (70.0, 75.0, 80.0, 85.0, 90.0, 95.0)
    assert generate._plddt_bin_token(p, edges) == expected


# ---------------------------------------------------------------------------
# Parsing + generation (gemmi-dependent)
# ---------------------------------------------------------------------------


_HAS_GEMMI = True
try:
    import gemmi  # noqa: F401
except ImportError:
    _HAS_GEMMI = False


_PDB_FIXTURE = textwrap.dedent("""\
    HEADER    TEST PROTEIN                            01-JAN-26   TEST
    ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00 80.00           N
    ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 80.00           C
    ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00 80.00           C
    ATOM      4  O   MET A   1      27.886  26.463   4.263  1.00 80.00           O
    ATOM      5  CB  MET A   1      25.112  24.880   3.649  1.00 80.00           C
    ATOM      6  CG  MET A   1      25.353  24.860   5.134  1.00 80.00           C
    ATOM      7  SD  MET A   1      23.930  23.959   5.904  1.00 80.00           S
    ATOM      8  CE  MET A   1      24.447  23.971   7.620  1.00 80.00           C
    ATOM      9  N   ALA A   2      26.335  27.770   3.258  1.00 90.00           N
    ATOM     10  CA  ALA A   2      26.881  29.013   3.793  1.00 90.00           C
    ATOM     11  C   ALA A   2      27.183  28.992   5.282  1.00 90.00           C
    ATOM     12  O   ALA A   2      28.250  28.583   5.713  1.00 90.00           O
    ATOM     13  CB  ALA A   2      25.857  30.131   3.494  1.00 90.00           C
    ATOM     14  N   GLY A   3      26.255  29.485   6.077  1.00 88.00           N
    ATOM     15  CA  GLY A   3      26.444  29.581   7.515  1.00 88.00           C
    ATOM     16  C   GLY A   3      26.999  28.293   8.075  1.00 88.00           C
    ATOM     17  O   GLY A   3      28.038  28.300   8.745  1.00 88.00           O
    ATOM     18  N   LYS A   4      26.301  27.187   7.838  1.00 75.00           N
    ATOM     19  CA  LYS A   4      26.756  25.876   8.302  1.00 75.00           C
    ATOM     20  C   LYS A   4      28.218  25.629   7.961  1.00 75.00           C
    ATOM     21  O   LYS A   4      28.799  24.679   8.484  1.00 75.00           O
    ATOM     22  CB  LYS A   4      25.901  24.752   7.701  1.00 75.00           C
    ATOM     23  CG  LYS A   4      24.401  24.880   7.918  1.00 75.00           C
    ATOM     24  CD  LYS A   4      23.610  23.681   7.395  1.00 75.00           C
    ATOM     25  CE  LYS A   4      22.118  23.787   7.687  1.00 75.00           C
    ATOM     26  NZ  LYS A   4      21.412  22.591   7.137  1.00 75.00           N
    ATOM     27  N   PHE A   5      28.787  26.470   7.103  1.00 65.00           N
    ATOM     28  CA  PHE A   5      30.181  26.282   6.681  1.00 65.00           C
    ATOM     29  C   PHE A   5      30.350  26.395   5.169  1.00 65.00           C
    ATOM     30  O   PHE A   5      31.444  26.246   4.622  1.00 65.00           O
    ATOM     31  CB  PHE A   5      31.111  27.286   7.395  1.00 65.00           C
    ATOM     32  CG  PHE A   5      32.589  27.156   7.114  1.00 65.00           C
    END
""")


@pytest.fixture
def tiny_pdb_path(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.pdb"
    p.write_text(_PDB_FIXTURE)
    return p


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_parse_tiny_pdb(tiny_pdb_path):
    parsed = parse_structure(tiny_pdb_path)
    assert isinstance(parsed, ParsedStructure)
    assert parsed.sequence == ["MET", "ALA", "GLY", "LYS", "PHE"]
    plddts = {r.name: r.plddt for r in parsed.residues}
    assert math.isclose(plddts["MET"], 80.0)
    assert math.isclose(plddts["PHE"], 65.0)
    vocab = set(ATOM_NAMES)
    for r in parsed.residues:
        for name, *_ in r.atoms:
            assert name in vocab


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generate_documents(tiny_pdb_path):
    docs = list(generate.generate_documents(
        input_path=tiny_pdb_path,
        num_docs=None,
        context_length=CONTEXT_LENGTH,
        config=generate.GenerationConfig(),
    ))
    assert len(docs) == 1
    parts = docs[0].split()
    assert parts[0] == "<contacts-and-distances-v1>"
    assert parts[1] == "<begin_sequence>"
    assert parts[-1] == "<end>"


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generate_documents_num_docs_cap(tmp_path: Path):
    sub = tmp_path / "many"
    sub.mkdir()
    for i in range(5):
        (sub / f"copy{i}.pdb").write_text(_PDB_FIXTURE)
    docs = list(generate.generate_documents(
        input_path=sub,
        num_docs=3,
        context_length=CONTEXT_LENGTH,
        config=generate.GenerationConfig(),
    ))
    assert len(docs) == 3


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generate_documents_is_deterministic(tiny_pdb_path):
    kwargs = dict(
        input_path=tiny_pdb_path,
        num_docs=None,
        context_length=CONTEXT_LENGTH,
        config=generate.GenerationConfig(),
    )
    a = list(generate.generate_documents(**kwargs))
    b = list(generate.generate_documents(**kwargs))
    assert a == b


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generated_doc_is_in_vocab(tiny_pdb_path):
    doc = next(iter(generate.generate_documents(
        input_path=tiny_pdb_path,
        num_docs=None,
        context_length=CONTEXT_LENGTH,
        config=generate.GenerationConfig(),
    )))
    tok = build_tokenizer(all_domain_tokens())
    unk_id = tok.convert_tokens_to_ids("<UNK>")
    ids = tok.encode(doc, add_special_tokens=False)
    assert unk_id not in ids
    assert len(ids) == len(doc.split())


# ---------------------------------------------------------------------------
# Inference helpers (vllm-free)
# ---------------------------------------------------------------------------


def test_distance_bin_midpoints():
    assert len(inference._DISTANCE_BIN_MIDPOINTS) == 64
    assert math.isclose(inference._DISTANCE_BIN_MIDPOINTS[0], 0.25)
    assert math.isclose(inference._DISTANCE_BIN_MIDPOINTS[-1], 31.75)


def test_resolve_distance_token_ids():
    tok = build_tokenizer(all_domain_tokens())
    ids = inference._resolve_distance_token_ids(tok)
    assert len(ids) == 64
    assert tok.decode([ids[0]]).strip() == "<d0.5>"
    assert tok.decode([ids[-1]]).strip() == "<d32.0>"


def test_resolve_distance_token_ids_rejects_unk_collapse():
    """A tokenizer missing the <d_X.X> vocab must error loudly."""

    class _DummyTokenizer:
        unk_token_id = 42

        def encode(self, tok, add_special_tokens=False):
            return [self.unk_token_id]

    with pytest.raises(ValueError, match="UNK"):
        inference._resolve_distance_token_ids(_DummyTokenizer())


def test_resolve_distance_token_ids_rejects_partial_collapse():
    """Partial collapse (some bins valid, some UNK) still errors."""

    class _PartialTokenizer:
        unk_token_id = 42
        _next_id = 100

        def encode(self, tok, add_special_tokens=False):
            if tok in ("<d0.5>", "<d1.0>"):
                return [100]
            self._next_id += 1
            return [self._next_id]

    with pytest.raises(ValueError, match="collapsed|unique IDs"):
        inference._resolve_distance_token_ids(_PartialTokenizer())


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_gt_long_range_contacts_filters_by_sep(tiny_pdb_path):
    parsed = parse_structure(tiny_pdb_path)
    assert inference._gt_long_range_contacts(parsed) == []


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_gt_query_distance_matrix_ca(tiny_pdb_path):
    parsed = parse_structure(tiny_pdb_path)
    gt = inference._gt_query_distance_matrix(parsed, "CA")
    assert gt.shape == (5, 5)
    for k in range(5):
        assert gt[k, k] == 0.0
    for i in range(5):
        for j in range(5):
            assert math.isfinite(gt[i, j])
            assert math.isclose(gt[i, j], gt[j, i], rel_tol=1e-5)


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_gt_query_distance_matrix_missing_atom_nans(tiny_pdb_path):
    parsed = parse_structure(tiny_pdb_path)
    gt = inference._gt_query_distance_matrix(parsed, "CB")
    gly_idx = next(i for i, r in enumerate(parsed.residues) if r.name == "GLY")
    for j in range(5):
        if j == gly_idx:
            continue
        assert math.isnan(gt[gly_idx, j])


def test_build_base_prompt_tokens_no_seeds():
    structure = ParsedStructure(
        entry_id="t",
        residues=tuple(
            Residue(index=k + 1, name=n, plddt=99.0, atoms=())
            for k, n in enumerate(["MET", "LYS", "PHE"])
        ),
        source_path=Path("/dev/null"),
    )
    assert inference._build_base_prompt_tokens(structure, seeded_contacts=[]) == [
        "<contacts-and-distances-v1>",
        "<begin_sequence>",
        "<MET>", "<LYS>", "<PHE>",
        "<begin_statements>",
    ]


def test_build_base_prompt_tokens_with_seeds():
    structure = ParsedStructure(
        entry_id="t",
        residues=tuple(
            Residue(index=k + 1, name="ALA", plddt=99.0, atoms=())
            for k in range(5)
        ),
        source_path=Path("/dev/null"),
    )
    toks = inference._build_base_prompt_tokens(
        structure, seeded_contacts=[(1, 30), (2, 50)],
    )
    assert toks[-7:] == [
        "<begin_statements>",
        "<long-range-contact>", "<p1>", "<p30>",
        "<long-range-contact>", "<p2>", "<p50>",
    ]


def test_pair_tail_tokens():
    assert inference._pair_tail_tokens(7, 42, "CA", "CB") == [
        "<distance>", "<p7>", "<p42>", "<CA>", "<CB>",
    ]


# ---------------------------------------------------------------------------
# predict / evaluate (vllm-free paths only)
# ---------------------------------------------------------------------------


def _empty_cfg(tmp_path: Path) -> inference.InferenceConfig:
    empty = tmp_path / "empty"
    empty.mkdir()
    return inference.InferenceConfig(model="/nonexistent", input_path=empty)


def test_predict_empty_input_short_circuits(tmp_path: Path):
    """Empty input yields nothing — vllm is never imported."""
    records = list(inference.predict(_empty_cfg(tmp_path)))
    assert records == []


def test_evaluate_empty_input_warns(tmp_path: Path):
    result = inference.evaluate(_empty_cfg(tmp_path))
    assert result.metrics == {}
    assert result.per_example == []
    assert result.extras.get("warning") == "no input structures"


# ---------------------------------------------------------------------------
# CLI surface (argparse smoke)
# ---------------------------------------------------------------------------


def test_cli_generate_parses():
    parser = cli.build_parser()
    args = parser.parse_args([
        "generate", "--input", "/tmp/x", "--num-docs", "10",
        "--out", "/tmp/docs.parquet",
    ])
    assert args.cmd == "generate"
    assert args.input == Path("/tmp/x")
    assert args.num_docs == 10
    assert args.func is cli.cmd_generate


def test_cli_infer_includes_keep_bin_probs():
    parser = cli.build_parser()
    args = parser.parse_args([
        "infer", "--model", "M", "--input", "/tmp/x",
        "--out", "/tmp/preds.parquet", "--keep-bin-probs",
    ])
    assert args.keep_bin_probs is True


def test_cli_evaluate_omits_keep_bin_probs():
    """--keep-bin-probs is infer-only; evaluate parser rejects it."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "evaluate", "--model", "M", "--input", "/tmp/x",
            "--out", "/tmp/metrics.json", "--keep-bin-probs",
        ])


def test_cli_distance_cap_only_in_evaluate():
    """--distance-cap-angstrom is evaluate-only."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "infer", "--model", "M", "--input", "/tmp/x",
            "--out", "/tmp/preds.parquet", "--distance-cap-angstrom", "16.0",
        ])
    args = parser.parse_args([
        "evaluate", "--model", "M", "--input", "/tmp/x",
        "--out", "/tmp/metrics.json", "--distance-cap-angstrom", "16.0",
    ])
    assert args.distance_cap_angstrom == 16.0


def test_cli_seed_n_values_parses_list():
    parser = cli.build_parser()
    args = parser.parse_args([
        "evaluate", "--model", "M", "--input", "/tmp/x",
        "--out", "/tmp/metrics.json", "--seed-n-values", "0,5,20",
    ])
    assert args.seed_n_values == (0, 5, 20)


def test_cli_seed_n_values_rejects_bad():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._seed_n_values("")
    with pytest.raises(argparse.ArgumentTypeError):
        cli._seed_n_values("-1")


def test_cli_top_level_help_does_not_error(capsys):
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "generate" in captured.out
    assert "infer" in captured.out
    assert "evaluate" in captured.out
    assert "tokenizer" in captured.out


# ---------------------------------------------------------------------------
# Writers (lifted from cli into the library)
# ---------------------------------------------------------------------------


def test_write_predictions_parquet_includes_structure(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    out = tmp_path / "preds.parquet"
    write_predictions(
        out,
        [{"entry_id": "X", "expected_distances": [1.0, 2.0]}],
        structure_name="contacts-and-distances-v1",
    )
    tbl = pq.read_table(str(out))
    cols = tbl.column_names
    assert "structure" in cols, cols
    assert tbl.column("structure").to_pylist() == ["contacts-and-distances-v1"]


def test_write_eval_parquet_preserves_extras(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    out = tmp_path / "eval.parquet"
    result = EvalResult(
        metrics={"mae_at_n0_angstrom": 3.6},
        per_example=[],
        extras={
            "model": "open-athena/foo@step-31337",
            "query_atom": "CA",
            "seed_n_values": [0, 5, 20],
            "per_structure_mae": {"1QYS": 3.6},
        },
    )
    write_eval(out, result, structure_name="contacts-and-distances-v1")
    tbl = pq.read_table(str(out))
    row = {k: tbl.column(k).to_pylist()[0] for k in tbl.column_names}
    assert row["model"] == "open-athena/foo@step-31337"
    assert row["query_atom"] == "CA"
    assert row["seed_n_values"] == [0, 5, 20]
    assert "extras_json" in row
    assert "per_structure_mae" in row["extras_json"]


def test_write_docs_jsonl_empty_does_not_create_file(tmp_path: Path):
    """Empty generator must raise before any file is opened."""
    from marinfold_document_structures import write_docs

    out = tmp_path / "docs.jsonl"
    with pytest.raises(SystemExit):
        write_docs(out, iter([]), structure_name="x")
    assert not out.exists()


def test_write_predictions_structure_name_wins_over_record_key(tmp_path: Path):
    """A record's own 'structure' key must not clobber the caller's name."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    out = tmp_path / "preds.parquet"
    write_predictions(
        out,
        [{"entry_id": "X", "structure": "format-other", "value": 1}],
        structure_name="contacts-and-distances-v1",
    )
    tbl = pq.read_table(str(out))
    assert tbl.column("structure").to_pylist() == ["contacts-and-distances-v1"]


def test_write_eval_structure_name_wins_over_extras(tmp_path: Path):
    """An extras 'structure' key must not clobber the caller's name."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    out = tmp_path / "eval.parquet"
    result = EvalResult(
        metrics={"mae": 1.0},
        extras={"structure": "format-other", "model": "M"},
    )
    write_eval(out, result, structure_name="contacts-and-distances-v1")
    tbl = pq.read_table(str(out))
    assert tbl.column("structure").to_pylist() == ["contacts-and-distances-v1"]
