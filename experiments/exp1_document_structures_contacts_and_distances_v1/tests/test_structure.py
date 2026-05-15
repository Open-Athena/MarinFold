# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contacts-and-distances-v1 Generator + Inference.

The Generator + Inference split lives in ``generate.py`` and
``inference.py``; shared vocab + parsing in ``_vocab.py`` /
``_parse.py``. Tests load each module via the standard
``marinfold_document_structures.load_*`` helpers (the same path the
CLI uses), so a regression in the Protocol surface fails here.

The network-marked tests download the pinned published tokenizer
(``timodonnell/protein-docs-tokenizer@83f597d88e9b``) and assert
byte-equivalence. Skip them with ``pytest -m 'not network'`` if
you're offline.

Run:

    uv sync --extra test
    uv run pytest tests/ -v
"""

import argparse
import math
import sys
import textwrap
from pathlib import Path

import pytest

# Make the experiment dir importable so we can reach _vocab / _parse /
# generate / inference directly (the CLI does this via spec_from_file_location,
# but tests use sys.path for ergonomic in-process imports).
_EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXP_DIR))

from _vocab import (  # noqa: E402
    AMINO_ACIDS, ATOM_NAMES, CONTACT_TYPES, CONTROL_TOKENS,
    CONTEXT_LENGTH, DISTANCE_BINS, DISTANCE_MARKER, MAX_POSITION,
    NAME, PLDDT_BINS, UNK_TOKEN, all_domain_tokens,
)
from _parse import ParsedStructure, Residue, parse_structure  # noqa: E402
from generate import GenerationConfig, V1Generator, get_generator, _distance_token, _plddt_bin_token  # noqa: E402
from inference import (  # noqa: E402
    V1Inference, _build_base_prompt_tokens, _gt_long_range_contacts,
    _gt_query_distance_matrix, _pair_tail_tokens, _parse_seed_n_values,
    _resolve_distance_token_ids, _DISTANCE_BIN_MIDPOINTS,
    get_inference,
)

from marinfold_document_structures import (  # noqa: E402
    Generator,
    Inference,
    build_tokenizer,
    load_generator,
    load_inference,
)


# ---------------------------------------------------------------------------
# Protocol conformance via the CLI loader path
# ---------------------------------------------------------------------------


def test_load_generator_satisfies_protocol():
    gen = load_generator(_EXP_DIR)
    assert isinstance(gen, Generator)
    assert gen.name == NAME
    assert gen.context_length == CONTEXT_LENGTH


def test_load_inference_satisfies_protocol():
    inf = load_inference(_EXP_DIR)
    assert isinstance(inf, Inference)
    assert inf.name == NAME
    assert inf.context_length == CONTEXT_LENGTH


def test_generator_and_inference_agree_on_vocab():
    """Both files must build the same WordLevel tokenizer."""
    gen = get_generator()
    inf = get_inference()
    assert gen.tokens() == inf.tokens()


# ---------------------------------------------------------------------------
# Vocab shape
# ---------------------------------------------------------------------------


def test_tokens_count_matches_canonical_2838():
    """Domain vocab is 2838 tokens; +2 specials = 2840 total."""
    tokens = get_generator().tokens()
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
    g = V1Generator()
    a = g.tokens()
    a.append("<broken>")
    assert "<broken>" not in g.tokens()


def test_token_order_invariants():
    tokens = get_generator().tokens()
    assert tokens[0] == "<contacts-and-distances-v1>"
    assert tokens[1] == "<begin_sequence>"
    assert tokens[4] == "<long-range-contact>"
    assert tokens[7] == "<distance>"
    assert tokens[8] == "<d0.5>"
    assert tokens[8 + 63] == "<d32.0>"
    assert tokens[-1] == "<UNK>"


# ---------------------------------------------------------------------------
# build_tokenizer (works on Generator AND Inference)
# ---------------------------------------------------------------------------


def test_build_tokenizer_on_generator():
    tok = build_tokenizer(get_generator())
    assert len(tok) == 2840
    assert tok.convert_tokens_to_ids("<pad>") == 0
    assert tok.convert_tokens_to_ids("<eos>") == 1


def test_build_tokenizer_on_inference():
    tok = build_tokenizer(get_inference())
    assert len(tok) == 2840


def test_build_tokenizer_roundtrip_sample():
    tok = build_tokenizer(get_generator())
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
    local = build_tokenizer(get_generator())
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
    local = build_tokenizer(get_generator())
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
    assert _distance_token(d) == expected


@pytest.mark.parametrize("p, expected", [
    (50.0, "plddt_lt70"), (69.99, "plddt_lt70"),
    (70.0, "plddt_70_75"), (74.99, "plddt_70_75"),
    (75.0, "plddt_75_80"), (84.0, "plddt_80_85"),
    (95.0, "plddt_95_100"), (100.0, "plddt_95_100"),
])
def test_plddt_bin_token(p, expected):
    edges = (70.0, 75.0, 80.0, 85.0, 90.0, 95.0)
    assert _plddt_bin_token(p, edges) == expected


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
def test_generator_run_via_args(tiny_pdb_path):
    """Build args via the impl's own add_args and exercise run()."""
    g = get_generator()
    p = argparse.ArgumentParser()
    p.add_argument("impl_dir", type=Path)
    p.add_argument("--out", type=Path, default=Path("/tmp/x.parquet"))
    g.add_args(p)
    args = p.parse_args([str(_EXP_DIR), "--input", str(tiny_pdb_path)])
    docs = list(g.run(args))
    assert len(docs) == 1
    parts = docs[0].split()
    assert parts[0] == "<contacts-and-distances-v1>"
    assert parts[1] == "<begin_sequence>"
    assert parts[-1] == "<end>"


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generator_run_num_docs_cap(tmp_path: Path):
    sub = tmp_path / "many"
    sub.mkdir()
    for i in range(5):
        (sub / f"copy{i}.pdb").write_text(_PDB_FIXTURE)
    g = get_generator()
    p = argparse.ArgumentParser()
    p.add_argument("impl_dir", type=Path)
    p.add_argument("--out", type=Path, default=Path("/tmp/x.parquet"))
    g.add_args(p)
    args = p.parse_args([str(_EXP_DIR), "--input", str(sub), "--num-docs", "3"])
    docs = list(g.run(args))
    assert len(docs) == 3


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generator_run_is_deterministic(tiny_pdb_path):
    g = get_generator()
    p = argparse.ArgumentParser()
    p.add_argument("impl_dir", type=Path)
    p.add_argument("--out", type=Path, default=Path("/tmp/x.parquet"))
    g.add_args(p)
    args = p.parse_args([str(_EXP_DIR), "--input", str(tiny_pdb_path)])
    a = list(g.run(args))
    b = list(g.run(args))
    assert a == b


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generated_doc_is_in_vocab(tiny_pdb_path):
    g = get_generator()
    p = argparse.ArgumentParser()
    p.add_argument("impl_dir", type=Path)
    p.add_argument("--out", type=Path, default=Path("/tmp/x.parquet"))
    g.add_args(p)
    args = p.parse_args([str(_EXP_DIR), "--input", str(tiny_pdb_path)])
    doc = next(iter(g.run(args)))
    tok = build_tokenizer(g)
    unk_id = tok.convert_tokens_to_ids("<UNK>")
    ids = tok.encode(doc, add_special_tokens=False)
    assert unk_id not in ids
    assert len(ids) == len(doc.split())


# ---------------------------------------------------------------------------
# Inference helpers (vllm-free)
# ---------------------------------------------------------------------------


def test_distance_bin_midpoints():
    assert len(_DISTANCE_BIN_MIDPOINTS) == 64
    assert math.isclose(_DISTANCE_BIN_MIDPOINTS[0], 0.25)
    assert math.isclose(_DISTANCE_BIN_MIDPOINTS[-1], 31.75)


def test_resolve_distance_token_ids():
    tok = build_tokenizer(get_inference())
    ids = _resolve_distance_token_ids(tok)
    assert len(ids) == 64
    assert tok.decode([ids[0]]).strip() == "<d0.5>"
    assert tok.decode([ids[-1]]).strip() == "<d32.0>"


def test_parse_seed_n_values():
    assert _parse_seed_n_values("0") == (0,)
    assert _parse_seed_n_values("0,5,20,50") == (0, 5, 20, 50)
    assert _parse_seed_n_values("0, 5 , 20") == (0, 5, 20)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_seed_n_values("")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_seed_n_values("-1")


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_gt_long_range_contacts_filters_by_sep(tiny_pdb_path):
    parsed = parse_structure(tiny_pdb_path)
    assert _gt_long_range_contacts(parsed) == []


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_gt_query_distance_matrix_ca(tiny_pdb_path):
    parsed = parse_structure(tiny_pdb_path)
    gt = _gt_query_distance_matrix(parsed, "CA")
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
    gt = _gt_query_distance_matrix(parsed, "CB")
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
    assert _build_base_prompt_tokens(structure, seeded_contacts=[]) == [
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
    toks = _build_base_prompt_tokens(structure, seeded_contacts=[(1, 30), (2, 50)])
    assert toks[-7:] == [
        "<begin_statements>",
        "<long-range-contact>", "<p1>", "<p30>",
        "<long-range-contact>", "<p2>", "<p50>",
    ]


def test_pair_tail_tokens():
    assert _pair_tail_tokens(7, 42, "CA", "CB") == [
        "<distance>", "<p7>", "<p42>", "<CA>", "<CB>",
    ]


# ---------------------------------------------------------------------------
# Inference add_args registers different flags per subcommand
# ---------------------------------------------------------------------------


def _inference_args(subcommand: str) -> argparse.ArgumentParser:
    inf = get_inference()
    p = argparse.ArgumentParser()
    p.add_argument("impl_dir", type=Path)
    p.add_argument("--out", type=Path, default=Path("/tmp/x"))
    inf.add_args(p, subcommand=subcommand)
    return p


def test_inference_add_args_infer_includes_keep_bin_probs():
    p = _inference_args("infer")
    # --keep-bin-probs is the infer-only flag.
    args = p.parse_args([
        str(_EXP_DIR), "--model", "M", "--input", "/tmp/x", "--keep-bin-probs",
    ])
    assert args.keep_bin_probs is True


def test_inference_add_args_evaluate_omits_keep_bin_probs():
    p = _inference_args("evaluate")
    with pytest.raises(SystemExit):
        # --keep-bin-probs only exists in infer mode
        p.parse_args([
            str(_EXP_DIR), "--model", "M", "--input", "/tmp/x", "--keep-bin-probs",
        ])


def test_inference_add_args_seed_n_values_parses_list():
    p = _inference_args("evaluate")
    args = p.parse_args([
        str(_EXP_DIR), "--model", "M", "--input", "/tmp/x",
        "--seed-n-values", "0,5,20",
    ])
    assert args.seed_n_values == (0, 5, 20)


def test_inference_predict_empty_input_short_circuits(tmp_path: Path):
    """Empty input directory yields nothing — vllm is never imported."""
    empty = tmp_path / "empty"
    empty.mkdir()
    inf = get_inference()
    p = _inference_args("infer")
    args = p.parse_args([str(_EXP_DIR), "--model", "/nonexistent", "--input", str(empty)])
    records = list(inf.predict(args))
    assert records == []


def test_inference_evaluate_empty_input_warns(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    inf = get_inference()
    p = _inference_args("evaluate")
    args = p.parse_args([str(_EXP_DIR), "--model", "/nonexistent", "--input", str(empty)])
    result = inf.evaluate(args)
    assert result.metrics == {}
    assert result.per_example == []
    assert result.extras.get("warning") == "no input structures"


# ---------------------------------------------------------------------------
# Ultrareview regression tests
# ---------------------------------------------------------------------------


def test_inference_add_args_distance_cap_only_in_evaluate():
    """bug_012: --distance-cap-angstrom is consumed only by evaluate."""
    infer_p = _inference_args("infer")
    with pytest.raises(SystemExit):
        infer_p.parse_args([
            str(_EXP_DIR), "--model", "M", "--input", "/tmp/x",
            "--distance-cap-angstrom", "16.0",
        ])
    eval_p = _inference_args("evaluate")
    args = eval_p.parse_args([
        str(_EXP_DIR), "--model", "M", "--input", "/tmp/x",
        "--distance-cap-angstrom", "16.0",
    ])
    assert args.distance_cap_angstrom == 16.0


def test_resolve_distance_token_ids_rejects_unk_collapse():
    """bug_008: a tokenizer missing the <d_X.X> vocab must error loudly."""

    class _DummyTokenizer:
        unk_token_id = 42

        def encode(self, tok, add_special_tokens=False):
            return [self.unk_token_id]

    with pytest.raises(ValueError, match="UNK"):
        _resolve_distance_token_ids(_DummyTokenizer())


def test_resolve_distance_token_ids_rejects_partial_collapse():
    """bug_008: partial collapse (some bins valid, some UNK) still errors."""

    class _PartialTokenizer:
        unk_token_id = 42
        _next_id = 100

        def encode(self, tok, add_special_tokens=False):
            # Two adjacent bins both collide on id=100.
            if tok in ("<d0.5>", "<d1.0>"):
                return [100]
            self._next_id += 1
            return [self._next_id]

    with pytest.raises(ValueError, match="collapsed|unique IDs"):
        _resolve_distance_token_ids(_PartialTokenizer())


def test_load_isolates_per_impl_vocab(tmp_path: Path):
    """bug_003: two impls in the same process must load independent vocab."""

    def _make_impl(dir_: Path, *, name: str, ctx: int) -> None:
        dir_.mkdir(parents=True, exist_ok=True)
        (dir_ / "_vocab.py").write_text(
            textwrap.dedent(
                f"""
                NAME = {name!r}
                CONTEXT_LENGTH = {ctx}
                def all_domain_tokens():
                    return [{name!r}, "<begin>", "<end>"]
                """
            )
        )
        (dir_ / "generate.py").write_text(
            textwrap.dedent(
                """
                import argparse
                from _vocab import NAME, CONTEXT_LENGTH, all_domain_tokens

                class _Gen:
                    name = NAME
                    context_length = CONTEXT_LENGTH
                    def tokens(self):
                        return all_domain_tokens()
                    def add_args(self, p):
                        p.add_argument('--input', required=True)
                    def run(self, args):
                        return iter([])

                def get_generator():
                    return _Gen()
                """
            )
        )

    impl_a = tmp_path / "impl_a"
    impl_b = tmp_path / "impl_b"
    _make_impl(impl_a, name="format-a", ctx=1024)
    _make_impl(impl_b, name="format-b", ctx=2048)

    gen_a = load_generator(impl_a)
    gen_b = load_generator(impl_b)
    # Pre-fix: gen_b inherits gen_a's vocab/name because Python caches
    # bare `_vocab` in sys.modules across the two loads.
    assert gen_a.name == "format-a"
    assert gen_b.name == "format-b"
    assert gen_a.tokens() == ["format-a", "<begin>", "<end>"]
    assert gen_b.tokens() == ["format-b", "<begin>", "<end>"]
    assert gen_a.context_length == 1024
    assert gen_b.context_length == 2048
    # Reload the first impl after the second — must still see its own vocab.
    gen_a_again = load_generator(impl_a)
    assert gen_a_again.name == "format-a"
    assert gen_a_again.tokens() == ["format-a", "<begin>", "<end>"]


def test_load_does_not_leak_sys_path_or_sys_modules(tmp_path: Path):
    """bug_003 hygiene: loader restores sys.path and sys.modules after load."""
    impl = tmp_path / "impl_x"
    impl.mkdir()
    (impl / "_vocab.py").write_text("NAME = 'x'\nCONTEXT_LENGTH = 16\n"
                                    "def all_domain_tokens():\n    return ['x']\n")
    (impl / "generate.py").write_text(textwrap.dedent("""
        from _vocab import NAME, CONTEXT_LENGTH, all_domain_tokens

        class _Gen:
            name = NAME
            context_length = CONTEXT_LENGTH
            def tokens(self): return all_domain_tokens()
            def add_args(self, p): pass
            def run(self, args): return iter([])

        def get_generator(): return _Gen()
    """))

    path_before = list(sys.path)
    vocab_before = sys.modules.get("_vocab")

    gen = load_generator(impl)
    assert gen.name == "x"

    assert sys.path == path_before
    assert sys.modules.get("_vocab") is vocab_before


# bug_001: parquet writers must include structure_name / extras


def test_write_predictions_parquet_includes_structure(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from marinfold_document_structures.cli import _write_predictions

    out = tmp_path / "preds.parquet"
    _write_predictions(
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
    from marinfold_document_structures import EvalResult
    from marinfold_document_structures.cli import _write_eval

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
    _write_eval(out, result, structure_name="contacts-and-distances-v1")
    tbl = pq.read_table(str(out))
    row = {k: tbl.column(k).to_pylist()[0] for k in tbl.column_names}
    assert row["model"] == "open-athena/foo@step-31337"
    assert row["query_atom"] == "CA"
    assert row["seed_n_values"] == [0, 5, 20]
    # Nested dict gets JSON-stringified
    assert "extras_json" in row
    assert "per_structure_mae" in row["extras_json"]


# bug_005: <cmd> --help works without impl_dir


def test_cli_help_without_impl_dir_does_not_error(capsys):
    from marinfold_document_structures.cli import _cmd_generate

    rc = _cmd_generate(["--help"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "impl_dir" in captured.out
