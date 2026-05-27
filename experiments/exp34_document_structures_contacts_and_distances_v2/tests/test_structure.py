# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contacts-and-distances-v2 document structure.

v2 is a serialization-only delta on v1 (same parsing, same statement
sampling, same context-length contract — only with ``<think>``
tokens spliced in between statements). The tests here therefore
focus on the v2-specific surface: the vocab delta, the think-token
sampling distributions, the placement contract, and the budget
accounting that keeps documents within 8192 tokens.
"""

import math
import random
import statistics
import textwrap
from pathlib import Path

import pytest

import generate
import vocab
from parse import parse_structure
from vocab import (
    NAME, THINK_TOKEN, V2_NEW_TOKENS, CONTEXT_LENGTH,
    all_domain_tokens, v1_all_domain_tokens,
)

from marinfold import build_tokenizer


# ---------------------------------------------------------------------------
# Vocab shape: v2 = v1 + 2 tokens, in order
# ---------------------------------------------------------------------------


def test_name_is_v2():
    assert NAME == "contacts-and-distances-v2"


def test_v2_tokens_are_v1_plus_two_appended():
    v1 = v1_all_domain_tokens()
    v2 = all_domain_tokens()
    assert len(v2) == len(v1) + 2
    assert v2[: len(v1)] == v1, "v1 IDs must be stable in v2"
    assert v2[-2:] == V2_NEW_TOKENS


def test_v2_new_tokens_are_what_we_expect():
    assert V2_NEW_TOKENS == [f"<{NAME}>", THINK_TOKEN]
    assert THINK_TOKEN == "<think>"


def test_v2_vocab_total_count_is_2840_with_specials():
    """2838 v1 tokens + 2 v2 tokens + (<pad>, <eos>) specials = 2842."""
    tok = build_tokenizer(all_domain_tokens())
    assert len(tok) == 2842
    assert tok.convert_tokens_to_ids("<pad>") == 0
    assert tok.convert_tokens_to_ids("<eos>") == 1
    # v2 tokens are at the end; specifically the last two domain ids.
    assert tok.convert_tokens_to_ids(f"<{NAME}>") == 2 + len(v1_all_domain_tokens())
    assert tok.convert_tokens_to_ids(THINK_TOKEN) == 2 + len(v1_all_domain_tokens()) + 1


def test_v1_ids_unchanged_under_v2_tokenizer():
    """A v1-pretrained model's embedding table can be reused 1:1.

    Every v1 token must keep the same id when we build the v2
    tokenizer — that's what makes warm-starting v1 → v2 a 2-row
    embedding extension instead of a from-scratch retrain.
    """
    v1_tok = build_tokenizer(v1_all_domain_tokens())
    v2_tok = build_tokenizer(all_domain_tokens())
    for t in v1_all_domain_tokens():
        assert v1_tok.convert_tokens_to_ids(t) == v2_tok.convert_tokens_to_ids(t), t


def test_v2_tokenizer_encodes_think_token():
    tok = build_tokenizer(all_domain_tokens())
    ids = tok.encode(THINK_TOKEN, add_special_tokens=False)
    assert len(ids) == 1
    assert ids[0] != tok.convert_tokens_to_ids("<UNK>")


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
# Geometric helper
# ---------------------------------------------------------------------------


def test_geometric_support_starts_at_1():
    rng = random.Random(0)
    for _ in range(200):
        k = generate._geometric(rng, 0.3)
        assert k >= 1


def test_geometric_p_one_returns_one():
    """p=1: every trial succeeds, so k=1 deterministically."""
    rng = random.Random(0)
    for _ in range(10):
        assert generate._geometric(rng, 1.0) == 1


def test_geometric_mean_matches_1_over_p():
    """E[X] = 1/p for support-≥-1 geometric. Be generous on tolerance."""
    rng = random.Random(42)
    samples = [generate._geometric(rng, 0.13) for _ in range(20_000)]
    expected = 1.0 / 0.13  # ≈ 7.69
    assert abs(statistics.mean(samples) - expected) < 0.3


def test_geometric_rejects_invalid_p():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        generate._geometric(rng, 0.0)
    with pytest.raises(ValueError):
        generate._geometric(rng, -0.1)
    with pytest.raises(ValueError):
        generate._geometric(rng, 1.5)


# ---------------------------------------------------------------------------
# Think-overhead sampling (statistical sanity)
# ---------------------------------------------------------------------------


def test_sample_think_overhead_p075_gate():
    """About 75% of draws should fire the initial run."""
    cfg = generate.GenerationConfig()
    rng = random.Random(1)
    n_fired = 0
    n = 5000
    for _ in range(n):
        k1, _ = generate._sample_think_overhead(rng, cfg)
        if k1 > 0:
            n_fired += 1
    frac = n_fired / n
    assert abs(frac - cfg.think_initial_prob) < 0.02, frac


def test_sample_think_overhead_k1_geom_p013():
    """When the gate fires, k1 ~ Geom(0.13); mean ~ 7.69."""
    cfg = generate.GenerationConfig()
    rng = random.Random(2)
    fired_k1: list[int] = []
    for _ in range(20_000):
        k1, _ = generate._sample_think_overhead(rng, cfg)
        if k1 > 0:
            fired_k1.append(k1)
    assert fired_k1
    assert abs(statistics.mean(fired_k1) - 1.0 / cfg.think_initial_geom_p) < 0.3


def test_sample_think_overhead_additional_run_count():
    """E[max(int(U(-4, 4)), 0)] ≈ 1.5 (∫₀⁴ floor(x)·(1/8) dx = (0+1+2+3)/8 = 0.75 ; mass on [-4,0) → 0).

    Actually for U(-4, 4) with width 8, ``int`` truncates toward 0,
    so values in [k, k+1) map to k (for k≥0) and to k+1 (for k<0).
    On [-4, 0) every draw maps to a non-positive int → max(.,0)=0.
    On [0, 4) draws of [0,1) → 0, [1,2) → 1, [2,3) → 2, [3,4) → 3.
    Density 1/8 each, so E = (0+1+2+3)/8 = 0.75.
    """
    cfg = generate.GenerationConfig()
    rng = random.Random(3)
    counts = []
    for _ in range(20_000):
        _, additional = generate._sample_think_overhead(rng, cfg)
        counts.append(len(additional))
    assert abs(statistics.mean(counts) - 0.75) < 0.05, statistics.mean(counts)
    assert all(0 <= c <= 4 for c in counts)


def test_sample_think_overhead_run_length_geom_p025():
    """Additional run lengths ~ Geom(0.25); mean ~ 4."""
    cfg = generate.GenerationConfig()
    rng = random.Random(4)
    lengths: list[int] = []
    for _ in range(20_000):
        _, additional = generate._sample_think_overhead(rng, cfg)
        lengths.extend(additional)
    assert lengths
    assert abs(statistics.mean(lengths) - 1.0 / cfg.think_run_length_geom_p) < 0.2


# ---------------------------------------------------------------------------
# Parsing + generation fixtures
# ---------------------------------------------------------------------------


_HAS_GEMMI = True
try:
    import gemmi  # noqa: F401
except ImportError:
    _HAS_GEMMI = False


# A larger fixture than v1's: enough residues to support a few
# long-range contacts and many distance statements, so the
# generated v2 docs reliably contain both pre-statement and inter-
# statement think runs to validate placement.
def _make_long_pdb(n_residues: int = 50) -> str:
    """Build a synthetic ALA chain ``n_residues`` long.

    Coordinates are placed on a stretched 3D helix (4 Å pitch in z
    plus a 12 Å diameter circle) so contacts at sequence-sep ≥ 6
    actually exist after the 8 Å CB-CB filter.
    """
    lines = ["HEADER    TEST PROTEIN                            01-JAN-26   TEST"]
    serial = 1
    for i in range(1, n_residues + 1):
        z = 1.5 * i
        # ALA backbone + CB.
        atoms = [
            ("N",  6.0, 0.0, z + 0.0),
            ("CA", 6.0, 1.5, z + 0.5),
            ("C",  6.0, 3.0, z + 0.0),
            ("O",  6.0, 3.0, z - 1.2),
            ("CB", 4.5, 1.5, z + 0.5),
        ]
        for name, x, y, zc in atoms:
            element = name[0]
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} ALA A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{zc:8.3f}  1.00 90.00          {element:>2s}"
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


@pytest.fixture
def long_pdb_path(tmp_path: Path) -> Path:
    p = tmp_path / "long.pdb"
    p.write_text(_make_long_pdb(50))
    return p


# ---------------------------------------------------------------------------
# Document-level structural invariants
# ---------------------------------------------------------------------------


_STATEMENT_STARTERS = frozenset({
    "<distance>",
    "<long-range-contact>",
    "<medium-range-contact>",
    "<short-range-contact>",
})


def _gen_one(structure, *, cfg=None, context_length=CONTEXT_LENGTH):
    cfg = cfg or generate.GenerationConfig()
    return generate._generate_one(structure, context_length=context_length, cfg=cfg)


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_doc_starts_with_v2_marker_and_ends_with_end(long_pdb_path):
    parsed = parse_structure(long_pdb_path)
    doc = _gen_one(parsed)
    assert doc is not None
    parts = doc.split()
    assert parts[0] == f"<{NAME}>"
    assert parts[1] == "<begin_sequence>"
    assert parts[-1] == "<end>"


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_doc_fits_in_context_window(long_pdb_path):
    parsed = parse_structure(long_pdb_path)
    doc = _gen_one(parsed)
    assert doc is not None
    assert len(doc.split()) <= CONTEXT_LENGTH


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_doc_is_deterministic(long_pdb_path):
    parsed = parse_structure(long_pdb_path)
    a = _gen_one(parsed)
    b = _gen_one(parsed)
    assert a == b


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_doc_tokenizes_without_unk(long_pdb_path):
    """Every token in a v2 doc must be in the v2 vocab."""
    parsed = parse_structure(long_pdb_path)
    doc = _gen_one(parsed)
    tok = build_tokenizer(all_domain_tokens())
    ids = tok.encode(doc, add_special_tokens=False)
    unk_id = tok.convert_tokens_to_ids("<UNK>")
    assert unk_id not in ids
    assert len(ids) == len(doc.split())


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_think_never_breaks_within_a_statement(long_pdb_path):
    """A run of <think> tokens always lands immediately before a
    statement-starting token (or before <plddt_*> / <end> at slot 0
    if the doc has no statements). It never appears mid-statement.

    Concretely: every maximal run of <think> tokens in
    ``<begin_statements>..<end>`` must be followed by an opener
    (a contact-mode marker, ``<distance>``, ``<plddt_*>``, or
    ``<end>``) — never by a position token, atom token, distance
    bin, or amino-acid token (those would mean we'd split a
    statement).
    """
    parsed = parse_structure(long_pdb_path)
    doc = _gen_one(parsed)
    parts = doc.split()
    start = parts.index("<begin_statements>") + 1
    end = parts.index("<end>")
    i = start
    while i < end:
        if parts[i] != THINK_TOKEN:
            i += 1
            continue
        # Walk past the maximal run of <think> tokens.
        j = i
        while j < end and parts[j] == THINK_TOKEN:
            j += 1
        # The token *immediately* after the run must be a statement
        # opener — anything else means we sliced a statement.
        if j < end:
            nxt = parts[j]
            ok = (
                nxt in _STATEMENT_STARTERS
                or nxt.startswith("<plddt_")
            )
            assert ok, (
                f"<think> run at position {i}..{j} is followed by "
                f"{nxt!r}, which is not a statement opener — that "
                f"means the run is splitting a statement."
            )
        i = j


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_initial_think_run_lands_right_after_begin_statements_when_present(long_pdb_path):
    """When the (a)-gate fires (P=0.75), the initial run sits at
    slot 0 — i.e. the first non-pLDDT token after
    ``<begin_statements>`` should be ``<think>``.

    Pick a config that *always* fires (think_initial_prob=1) so the
    test doesn't flake.
    """
    parsed = parse_structure(long_pdb_path)
    # think_initial_prob=1.0 makes the (a)-gate deterministic.
    cfg = generate.GenerationConfig(think_initial_prob=1.0)
    doc = _gen_one(parsed, cfg=cfg)
    parts = doc.split()
    bs = parts.index("<begin_statements>")
    assert parts[bs + 1] == THINK_TOKEN, parts[bs + 1 : bs + 5]


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_no_think_when_all_gates_disabled(long_pdb_path):
    """think_initial_prob=0 + additional_count_range=(-4,0) → no think tokens."""
    parsed = parse_structure(long_pdb_path)
    cfg = generate.GenerationConfig(
        think_initial_prob=0.0,
        think_additional_count_range=(-4.0, 0.0),
    )
    doc = _gen_one(parsed, cfg=cfg)
    assert THINK_TOKEN not in doc.split()


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_total_think_count_matches_sampled_overhead(long_pdb_path):
    """Sanity: the # of <think> tokens in the doc equals the
    pre-sampled (k1 + sum(additional_run_lengths)) for the same
    deterministic seed.
    """
    parsed = parse_structure(long_pdb_path)
    cfg = generate.GenerationConfig()
    # Reproduce the seed/RNG dance from _generate_one so we can
    # inspect what _sample_think_overhead returned without rerunning
    # the whole generator.
    import hashlib as _hashlib
    seed = int(_hashlib.sha1(parsed.entry_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    k1, additional = generate._sample_think_overhead(rng, cfg)
    expected_total = k1 + sum(additional)
    doc = _gen_one(parsed, cfg=cfg)
    actual = doc.split().count(THINK_TOKEN)
    assert actual == expected_total, (actual, expected_total, k1, additional)


# ---------------------------------------------------------------------------
# generate_documents / num_docs cap
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generate_documents_iter(long_pdb_path):
    docs = list(generate.generate_documents(
        input_path=long_pdb_path,
        num_docs=None,
        context_length=CONTEXT_LENGTH,
        config=generate.GenerationConfig(),
    ))
    assert len(docs) == 1
    assert docs[0].split()[0] == f"<{NAME}>"


@pytest.mark.skipif(not _HAS_GEMMI, reason="gemmi not installed")
def test_generate_documents_num_docs_cap(tmp_path: Path):
    sub = tmp_path / "many"
    sub.mkdir()
    pdb_text = _make_long_pdb(50)
    for i in range(5):
        (sub / f"copy{i}.pdb").write_text(pdb_text)
    docs = list(generate.generate_documents(
        input_path=sub,
        num_docs=3,
        context_length=CONTEXT_LENGTH,
        config=generate.GenerationConfig(),
    ))
    assert len(docs) == 3


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_generate_parses():
    import cli
    parser = cli.build_parser()
    args = parser.parse_args([
        "generate", "--input", "/tmp/x", "--num-docs", "10",
        "--out", "/tmp/docs.parquet",
    ])
    assert args.cmd == "generate"
    assert args.input == "/tmp/x"
    assert args.num_docs == 10
    assert args.func is cli.cmd_generate
    # v2-specific defaults wired through.
    assert args.think_initial_prob == 0.75
    assert args.think_initial_geom_p == 0.13
    assert tuple(args.think_additional_count_range) == (-4.0, 4.0)
    assert args.think_run_length_geom_p == 0.25


def test_cli_top_level_help_does_not_error(capsys):
    import cli
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "generate" in captured.out
    assert "tokenizer" in captured.out
    # v2's CLI is generate-only — no infer / evaluate yet.
    assert "infer" not in captured.out
    assert "evaluate" not in captured.out


def test_cli_does_not_expose_infer_or_evaluate():
    """v2 is generation-only this experiment; later issues will add
    inference + eval surfaces."""
    import cli
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["infer", "--model", "M", "--input", "/tmp/x", "--out", "/tmp/y"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--model", "M", "--input", "/tmp/x", "--out", "/tmp/y"])
