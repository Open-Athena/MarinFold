# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contacts-and-distances-v1 DocumentStructure.

The network-marked tests download the pinned published tokenizer
(``timodonnell/protein-docs-tokenizer@83f597d88e9b``) and assert
byte-equivalence with the locally-constructed one. Skip them with
``pytest -m 'not network'`` if you're offline.

Run:

    uv sync --extra test
    uv run pytest tests/ -v
"""

import math
import sys
import textwrap
from pathlib import Path

import pytest

# The experiment dir is not a package; make ``structure`` importable.
_EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXP_DIR))

from structure import (  # noqa: E402
    AMINO_ACIDS,
    ATOM_NAMES,
    CONTACT_TYPES,
    CONTROL_TOKENS,
    DISTANCE_BINS,
    DISTANCE_MARKER,
    MAX_POSITION,
    PLDDT_BINS,
    UNK_TOKEN,
    ContactsAndDistancesV1,
    EvaluationConfig,
    ParsedStructure,
    _build_eval_prompt_tokens,
    _DISTANCE_BIN_MIDPOINTS,
    _distance_token,
    _gt_long_range_contacts,
    _gt_query_distance_matrix,
    _pair_tail_tokens,
    _plddt_bin_token,
    _resolve_distance_token_ids,
    get_structure,
    parse_structure,
)

from marinfold_document_structures import (  # noqa: E402
    DocumentStructure,
    build_tokenizer,
)


# ---------------------------------------------------------------------------
# Protocol conformance + metadata
# ---------------------------------------------------------------------------


def test_get_structure_satisfies_protocol():
    s = get_structure()
    assert isinstance(s, DocumentStructure), (
        f"get_structure() returned a {type(s).__name__} that does not "
        "satisfy the DocumentStructure protocol."
    )


def test_structure_metadata():
    s = get_structure()
    assert s.name == "contacts-and-distances-v1"
    assert s.context_length == 8192


# ---------------------------------------------------------------------------
# tokens() shape
# ---------------------------------------------------------------------------


def test_tokens_count_matches_canonical_2838():
    """Domain vocab is 2838 tokens; +2 specials = 2840 total."""
    tokens = get_structure().tokens()
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
    assert n_expected == 2838, f"category counts don't sum to 2838: {n_expected}"
    assert len(tokens) == 2838


def test_tokens_unique():
    tokens = get_structure().tokens()
    assert len(tokens) == len(set(tokens))


def test_tokens_returns_a_copy():
    s = ContactsAndDistancesV1()
    a = s.tokens()
    a.append("<broken>")
    assert "<broken>" not in s.tokens()


def test_token_order_invariants():
    """Spot-check positions baked into every v1 checkpoint."""
    tokens = get_structure().tokens()
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


def test_build_tokenizer_size():
    tok = build_tokenizer(get_structure())
    assert len(tok) == 2840
    assert tok.convert_tokens_to_ids("<pad>") == 0
    assert tok.convert_tokens_to_ids("<eos>") == 1
    assert tok.convert_tokens_to_ids("<contacts-and-distances-v1>") == 2


def test_build_tokenizer_roundtrip_sample():
    tok = build_tokenizer(get_structure())
    sample = (
        "<contacts-and-distances-v1> <begin_sequence> "
        "<MET> <LYS> <PHE> <CYS> <ASP> <TYR> <GLY> <LEU> "
        "<begin_statements> "
        "<long-range-contact> <p1> <p50> "
        "<distance> <p10> <p45> <CA> <CB> <d4.5> "
        "<plddt_80_85> <end>"
    )
    ids = tok.encode(sample, add_special_tokens=False)
    decoded = tok.decode(ids)
    assert len(ids) == len(sample.split())
    unk_id = tok.convert_tokens_to_ids("<UNK>")
    assert unk_id not in ids
    assert decoded.strip().split() == sample.split()


REVISION_PIN = "83f597d88e9b"


@pytest.mark.network
def test_published_tokenizer_vocab_matches():
    pytest.importorskip("huggingface_hub")
    from transformers import AutoTokenizer

    published = AutoTokenizer.from_pretrained(
        "timodonnell/protein-docs-tokenizer",
        revision=REVISION_PIN,
    )
    local = build_tokenizer(get_structure())
    pub_vocab = published.get_vocab()
    loc_vocab = local.get_vocab()
    assert pub_vocab.keys() == loc_vocab.keys()
    mismatches = [
        (t, pub_vocab[t], loc_vocab[t])
        for t in pub_vocab
        if pub_vocab[t] != loc_vocab[t]
    ]
    assert not mismatches, f"first 5 mismatches: {mismatches[:5]}"


@pytest.mark.network
def test_published_tokenizer_encodes_identically():
    pytest.importorskip("huggingface_hub")
    from transformers import AutoTokenizer

    published = AutoTokenizer.from_pretrained(
        "timodonnell/protein-docs-tokenizer",
        revision=REVISION_PIN,
    )
    local = build_tokenizer(get_structure())
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
    assert published.encode(sample, add_special_tokens=False) == local.encode(
        sample, add_special_tokens=False
    )


# ---------------------------------------------------------------------------
# Bin / token helpers (pure-stdlib)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d, expected", [
    (0.0, "d0.5"),         # clamped to lowest bin
    (0.25, "d0.5"),
    (0.5, "d0.5"),         # exactly on edge -> ceil(0.5/0.5)=1 -> d0.5
    (0.51, "d1.0"),
    (4.0, "d4.0"),
    (4.5, "d4.5"),
    (31.99, "d32.0"),
    (32.0, "d32.0"),
    (45.7, "d32.0"),       # clamped to top bin
])
def test_distance_token_bins(d, expected):
    assert _distance_token(d) == expected


@pytest.mark.parametrize("p, expected", [
    (50.0, "plddt_lt70"),
    (69.99, "plddt_lt70"),
    (70.0, "plddt_70_75"),
    (74.99, "plddt_70_75"),
    (75.0, "plddt_75_80"),
    (84.0, "plddt_80_85"),
    (95.0, "plddt_95_100"),
    (100.0, "plddt_95_100"),
])
def test_plddt_bin_token(p, expected):
    edges = (70.0, 75.0, 80.0, 85.0, 90.0, 95.0)
    assert _plddt_bin_token(p, edges) == expected


# ---------------------------------------------------------------------------
# Structure parsing + doc generation (use gemmi; skip if unavailable)
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
    assert len(parsed.residues) == 5
    # B-factors of MET (80) and PHE (65) — stored verbatim in
    # residue.plddt (mean over heavy atoms).
    plddts = {r.name: r.plddt for r in parsed.residues}
    assert math.isclose(plddts["MET"], 80.0)
    assert math.isclose(plddts["PHE"], 65.0)
    # All atoms must be in our v1 atom vocab.
    vocab = set(ATOM_NAMES)
    for r in parsed.residues:
        for name, *_ in r.atoms:
            assert name in vocab, f"out-of-vocab atom: {name}"


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_parse_skips_hydrogens(tiny_pdb_path):
    parsed = parse_structure(tiny_pdb_path)
    for r in parsed.residues:
        for name, *_ in r.atoms:
            assert not name.startswith("H")


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_iter_inputs_directory_walk(tmp_path: Path):
    """``iter_inputs`` walks subdirs and only picks up structure-shaped files."""
    sub = tmp_path / "many"
    sub.mkdir()
    (sub / "a.pdb").write_text(_PDB_FIXTURE)
    (sub / "b.pdb").write_text(_PDB_FIXTURE)
    (sub / "ignore.txt").write_text("not a structure")
    structures = list(get_structure().iter_inputs(sub))
    assert len(structures) == 2
    assert all(s.sequence == ["MET", "ALA", "GLY", "LYS", "PHE"] for s in structures)


# ---------------------------------------------------------------------------
# generate_documents
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generate_documents_yields_one_per_input(tiny_pdb_path):
    s = get_structure()
    docs = list(s.generate_documents(s.iter_inputs(tiny_pdb_path)))
    assert len(docs) == 1


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generated_doc_well_formed(tiny_pdb_path):
    s = get_structure()
    doc = next(iter(s.generate_documents(s.iter_inputs(tiny_pdb_path))))

    parts = doc.split()
    assert parts[0] == "<contacts-and-distances-v1>"
    assert parts[1] == "<begin_sequence>"
    assert "<begin_statements>" in parts
    assert parts[-1] == "<end>"

    seq_start = parts.index("<begin_sequence>") + 1
    stmts_start = parts.index("<begin_statements>")
    assert parts[seq_start:stmts_start] == ["<MET>", "<ALA>", "<GLY>", "<LYS>", "<PHE>"]

    # Exactly one pLDDT bin token in the statements/post-statements
    # region (the algorithm places it either mid-statements or at the
    # very end).
    plddt_tokens = [t for t in parts[stmts_start:] if t.startswith("<plddt_")]
    assert len(plddt_tokens) == 1


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generated_doc_is_in_vocab(tiny_pdb_path):
    """Every emitted token must be in the v1 vocabulary (no UNK)."""
    s = get_structure()
    doc = next(iter(s.generate_documents(s.iter_inputs(tiny_pdb_path))))
    tok = build_tokenizer(s)
    unk_id = tok.convert_tokens_to_ids("<UNK>")
    ids = tok.encode(doc, add_special_tokens=False)
    assert unk_id not in ids, (
        f"document contains out-of-vocab tokens; doc head: {doc[:200]!r}"
    )
    # WordLevel + whitespace-split = 1:1.
    assert len(ids) == len(doc.split())


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generation_is_deterministic(tiny_pdb_path):
    """Same input → same output (seed is sha1(entry_id))."""
    s = get_structure()
    doc1 = next(iter(s.generate_documents(s.iter_inputs(tiny_pdb_path))))
    doc2 = next(iter(s.generate_documents(s.iter_inputs(tiny_pdb_path))))
    assert doc1 == doc2


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generate_num_docs_cap(tmp_path: Path):
    """``num_docs`` caps the output even when there are more inputs."""
    sub = tmp_path / "many"
    sub.mkdir()
    for i in range(5):
        (sub / f"copy{i}.pdb").write_text(_PDB_FIXTURE)
    s = get_structure()
    docs = list(s.generate_documents(s.iter_inputs(sub), num_docs=3))
    assert len(docs) == 3


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generate_skips_when_sequence_exceeds_budget(tiny_pdb_path):
    """A ``context_length`` smaller than ``5 + sequence_length`` skips the doc."""
    s = get_structure()
    docs = list(s.generate_documents(s.iter_inputs(tiny_pdb_path), context_length=4))
    assert docs == []


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_iter_ground_truth_shares_parser(tiny_pdb_path):
    """``iter_ground_truth`` yields the same parses as ``iter_inputs``."""
    s = get_structure()
    a = list(s.iter_inputs(tiny_pdb_path))
    b = list(s.iter_ground_truth(tiny_pdb_path))
    assert len(a) == len(b) == 1
    assert a[0].sequence == b[0].sequence
    assert a[0].entry_id == b[0].entry_id


# ---------------------------------------------------------------------------
# evaluate() — vllm-free pieces (prompt builders, GT helpers, midpoints)
# ---------------------------------------------------------------------------


def test_distance_bin_midpoints_canonical():
    # 64 bins at 0.5Å resolution. Bin 1 = (0, 0.5] → midpoint 0.25.
    # Bin 64 = (31.5, 32.0] → midpoint 31.75.
    assert len(_DISTANCE_BIN_MIDPOINTS) == 64
    assert math.isclose(_DISTANCE_BIN_MIDPOINTS[0], 0.25)
    assert math.isclose(_DISTANCE_BIN_MIDPOINTS[1], 0.75)
    assert math.isclose(_DISTANCE_BIN_MIDPOINTS[-1], 31.75)


def test_resolve_distance_token_ids_uses_canonical_tokenizer():
    tok = build_tokenizer(get_structure())
    ids = _resolve_distance_token_ids(tok)
    assert len(ids) == 64
    # bin 0 == <d0.5> at id (CONTROL + CONTACT_TYPES + DISTANCE_MARKER + 0)
    # We don't need to know the exact ID — just that decoding round-trips.
    assert tok.decode([ids[0]]).strip() == "<d0.5>"
    assert tok.decode([ids[-1]]).strip() == "<d32.0>"


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_gt_long_range_contacts_filters_by_sep_and_distance(tiny_pdb_path):
    """The tiny 5-residue fixture has no long-range pairs (sep < 24 for all)."""
    parsed = parse_structure(tiny_pdb_path)
    assert _gt_long_range_contacts(parsed) == []


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_gt_query_distance_matrix_ca_ca(tiny_pdb_path):
    parsed = parse_structure(tiny_pdb_path)
    gt = _gt_query_distance_matrix(parsed, "CA")
    # 5x5, diagonal is zero, symmetric, all finite (all 5 residues have CA).
    assert gt.shape == (5, 5)
    for k in range(5):
        assert gt[k, k] == 0.0
    for i in range(5):
        for j in range(5):
            assert math.isfinite(gt[i, j])
            assert math.isclose(gt[i, j], gt[j, i], rel_tol=1e-5)


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_gt_query_distance_matrix_missing_atom_nans(tiny_pdb_path):
    """A residue without the requested atom yields NaN rows/cols.

    The fixture's GLY (residue 3) has no CB — so a CB-CB matrix has NaN
    for any pair touching row/col 2.
    """
    parsed = parse_structure(tiny_pdb_path)
    gt = _gt_query_distance_matrix(parsed, "CB")
    gly_idx = next(i for i, r in enumerate(parsed.residues) if r.name == "GLY")
    for j in range(5):
        if j == gly_idx:
            continue
        assert math.isnan(gt[gly_idx, j]), f"expected NaN at GLY row, got {gt[gly_idx, j]}"


def test_build_eval_prompt_tokens_zero_shot():
    """Zero-shot prompt: structure tag + sequence + begin_statements only."""
    s = ContactsAndDistancesV1()
    structure = ParsedStructure(
        entry_id="test",
        residues=tuple([
            # 3 dummy residues; atoms don't matter for this prompt-shape test
            __import__("structure").Residue(index=k + 1, name=name, plddt=99.0, atoms=())
            for k, name in enumerate(["MET", "LYS", "PHE"])
        ]),
        source_path=Path("/dev/null"),
    )
    toks = _build_eval_prompt_tokens(structure, seeded_contacts=[], structure_name=s.name)
    assert toks == [
        "<contacts-and-distances-v1>",
        "<begin_sequence>",
        "<MET>", "<LYS>", "<PHE>",
        "<begin_statements>",
    ]


def test_build_eval_prompt_tokens_with_seeded_contacts():
    """Seeded contacts appear as `<long-range-contact> <p_i> <p_j>` triples."""
    s = ContactsAndDistancesV1()
    structure = ParsedStructure(
        entry_id="test",
        residues=tuple([
            __import__("structure").Residue(index=k + 1, name="ALA", plddt=99.0, atoms=())
            for k in range(5)
        ]),
        source_path=Path("/dev/null"),
    )
    toks = _build_eval_prompt_tokens(
        structure,
        seeded_contacts=[(1, 30), (2, 50)],
        structure_name=s.name,
    )
    # tail of toks: <begin_statements>, then 3 tokens per contact
    assert toks[-7:] == [
        "<begin_statements>",
        "<long-range-contact>", "<p1>", "<p30>",
        "<long-range-contact>", "<p2>", "<p50>",
    ]


def test_pair_tail_tokens_shape():
    """The 5-token tail elicits a distance-bin next-token prediction."""
    assert _pair_tail_tokens(7, 42, "CA", "CB") == [
        "<distance>", "<p7>", "<p42>", "<CA>", "<CB>",
    ]


def test_evaluation_config_defaults():
    cfg = EvaluationConfig()
    assert cfg.query_atom == "CA"
    assert cfg.seed_n_values == (0,)
    assert cfg.top_k_logprobs >= 64
    assert cfg.distance_cap_angstrom == 32.0


def test_evaluate_picks_up_evaluation_config():
    """The class-level config knob is what evaluate() consults."""
    custom = EvaluationConfig(query_atom="CB", seed_n_values=(0, 5, 20))
    s = ContactsAndDistancesV1(evaluation_config=custom)
    assert s.evaluation_config is custom


# ---------------------------------------------------------------------------
# Full vllm-backed evaluate() — GPU-only smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_evaluate_raises_without_vllm_or_records():
    """Empty input is safely handled — no vllm spin-up, no metrics."""
    s = get_structure()
    result = s.evaluate(model_path="/nonexistent", ground_truth_records=iter([]))
    assert result.metrics == {}
    assert result.per_example == []
    assert result.extras.get("warning") == "no input structures"
