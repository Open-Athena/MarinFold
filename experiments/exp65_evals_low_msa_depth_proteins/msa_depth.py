"""Compute MSA depth (Neff) for ColabFold ``.a3m`` alignments.

Reusable reference implementation for issue #65 (continuation of #41:
"identify low-MSA eval datapoints"). Given the per-protein ``non_pairing.a3m``
files that exp12 already computed for the FoldBench-100 (and that
exp12 persists to the Modal ``protenix-foldbench-msa`` Volume and the
``open-athena/MarinFold`` HF bucket under
``data/protenix-foldbench-monomers/msa/<stem>/msa/0/0/non_pairing.a3m``),
this emits one CSV row per protein with several depth measures.

Why Neff and not the raw line count: a ColabFold MSA can contain
hundreds of near-identical orthologs that carry almost no independent
evolutionary signal. Neff down-weights that redundancy by clustering
at an identity threshold (AlphaFold2 uses 80%), so it measures
*independent* sequences -- which is what coevolution-based methods
actually consume. We report the raw count too, since "MSA depth" is
used loosely in the literature for both.

Definitions (all configurable):

- ``n_seqs``       raw number of aligned sequences incl. the query.
- ``neff``         sum_i 1 / |{ j : seqid(i, j) >= theta }|, theta=0.80
                   by default (AF2's 80% cluster-reweighting scheme).
                   Pairwise identity is measured over columns that are
                   non-gap in *either* sequence (AF2's convention).
- ``neff_per_L``   neff / L  (per-residue depth; AF2-comparable).
- ``neff_per_sqrtL`` neff / sqrt(L) (trRosetta's ``Nf``).

L is the number of match-state columns = the query length. a3m
lowercase letters are insertions relative to the query and are
dropped before any identity is computed, so every row is length L.

This is intentionally dependency-light (numpy only) and self-contained
so it can run anywhere the a3m files can be staged. For a citable,
externally-maintained cross-check, NEFFy
(https://github.com/Maryam-Haghani/NEFFy, Bioinformatics 2025)
computes the same clustering Neff from an a3m; ``--cross-check-neffy``
documents the equivalent invocation in the output but is not required.

Usage::

    # one a3m -> printed summary
    python msa_depth.py one path/to/non_pairing.a3m

    # a tree of <stem>/msa/0/0/non_pairing.a3m (exp12 layout) -> CSV
    python msa_depth.py dir MSA_ROOT --layout exp12 --out data/msa_depth.csv

    # a flat dir of <stem>.a3m -> CSV
    python msa_depth.py dir SOME_DIR --layout flat --out data/msa_depth.csv

    # self-test (no external data needed)
    python msa_depth.py selftest
"""

import argparse
import csv
import math
import random
import string
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# a3m insertion columns are lowercase letters / "." -- everything we
# strip to recover the query-aligned match-state columns.
_INSERTION_CHARS = set(string.ascii_lowercase) | {"."}
_GAP = ord("-")

# 20 aa + X/B/Z/U/O ambiguity codes + gap; anything else -> "X".
_AA = "ACDEFGHIKLMNPQRSTVWYXBZUO"
_CODE = {c: i for i, c in enumerate(_AA)}
_GAP_CODE = len(_AA)  # gap gets its own code, never counts as a match


def _match_columns(seq: str) -> str:
    """Drop a3m insertion state (lowercase + '.'), keep match columns."""
    return "".join(c for c in seq if c not in _INSERTION_CHARS)


def parse_a3m(text: str) -> list[str]:
    """Parse a3m text into a list of match-state sequences.

    The first record is the query and defines the match-state column
    count L. Every returned sequence has length L (insertions removed,
    uppercase). Raises ValueError if a record's match length disagrees
    with the query -- a sign of a malformed a3m, which we do not want to
    silently mis-score.
    """
    seqs: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
            continue
        cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    if not seqs:
        raise ValueError("empty a3m (no sequences)")

    match = [_match_columns(s).upper() for s in seqs]
    length = len(match[0])
    for i, m in enumerate(match):
        if len(m) != length:
            raise ValueError(
                f"record {i} has {len(m)} match columns, query has {length}"
            )
    return match


def _encode(seqs: list[str]) -> np.ndarray:
    """(N, L) uint8 matrix; gaps -> _GAP_CODE, unknown aa -> X."""
    n, length = len(seqs), len(seqs[0])
    arr = np.empty((n, length), dtype=np.uint8)
    x_code = _CODE["X"]
    for i, s in enumerate(seqs):
        arr[i] = [
            _GAP_CODE if ord(c) == _GAP else _CODE.get(c, x_code) for c in s
        ]
    return arr


def neff(
    seqs: list[str], theta: float = 0.80, chunk: int = 512, max_seqs: int = 0
) -> tuple[float, int]:
    """Compute clustering Neff at identity threshold ``theta``.

    Returns ``(neff, n_seqs)`` where ``n_seqs`` is the TRUE sequence count.
    Pairwise identity for sequences a, b is ``(# match columns where a == b,
    both non-gap) / (# columns non-gap in a OR b)`` -- AlphaFold2's "measured
    on the region that is non-gap in either sequence" convention. Sequence
    i's cluster size is the count of j (including i) with identity >= theta;
    its weight is the reciprocal, and Neff is the sum of weights.

    Vectorised in row-chunks to bound memory at O(chunk * N). Because the
    computation is O(N^2 * L), ``max_seqs`` caps it: for deep alignments
    (N > max_seqs) the query plus a deterministic random sample of
    ``max_seqs - 1`` other rows are used, so the returned Neff is over the
    sample (still unambiguously "deep") while ``n_seqs`` reports the true
    depth. ``max_seqs == 0`` disables capping (exact, but can be very large).
    """
    n_true = len(seqs)
    if max_seqs and n_true > max_seqs:
        rng = random.Random(0)  # deterministic subsample, query kept first
        seqs = [seqs[0]] + rng.sample(seqs[1:], max_seqs - 1)
    arr = _encode(seqs)
    n, _ = arr.shape
    is_res = arr != _GAP_CODE  # (N, L) non-gap mask
    cluster_sizes = np.zeros(n, dtype=np.int64)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = arr[start:stop]  # (b, L)
        block_res = is_res[start:stop]  # (b, L)
        # both non-gap and equal -> identical residue match
        # (b, 1, L) vs (1, N, L) broadcast, summed over L.
        both_res = block_res[:, None, :] & is_res[None, :, :]  # (b, N, L)
        equal = block[:, None, :] == arr[None, :, :]  # (b, N, L)
        n_match = (both_res & equal).sum(axis=2)  # (b, N)
        # union of non-gap columns = |a| + |b| - |a & b|
        a_len = block_res.sum(axis=1)[:, None]  # (b, 1)
        b_len = is_res.sum(axis=1)[None, :]  # (1, N)
        n_union = a_len + b_len - both_res.sum(axis=2)  # (b, N)
        with np.errstate(divide="ignore", invalid="ignore"):
            ident = np.where(n_union > 0, n_match / n_union, 0.0)
        cluster_sizes[start:stop] = (ident >= theta).sum(axis=1)

    cluster_sizes = np.maximum(cluster_sizes, 1)
    return float((1.0 / cluster_sizes).sum()), n_true


@dataclass
class DepthRow:
    stem: str
    length: int
    n_seqs: int
    neff: float
    neff_per_L: float
    neff_per_sqrtL: float
    theta: float


def depth_for_a3m(stem: str, text: str, theta: float = 0.80, max_seqs: int = 0) -> DepthRow:
    seqs = parse_a3m(text)
    length = len(seqs[0])
    nf, n = neff(seqs, theta=theta, max_seqs=max_seqs)
    return DepthRow(
        stem=stem,
        length=length,
        n_seqs=n,
        neff=round(nf, 4),
        neff_per_L=round(nf / length, 6) if length else 0.0,
        neff_per_sqrtL=round(nf / math.sqrt(length), 4) if length else 0.0,
        theta=theta,
    )


def _iter_a3m_paths(root: Path, layout: str):
    """Yield (stem, path) for a3m files under ``root``.

    ``exp12``: ``<stem>/msa/0/0/non_pairing.a3m`` (the Protenix colabfold
    layout exp12 writes). ``flat``: ``<stem>.a3m`` directly under root.
    """
    if layout == "exp12":
        for p in sorted(root.glob("*/msa/0/0/non_pairing.a3m")):
            yield p.parents[3].name, p
    elif layout == "flat":
        for p in sorted(root.glob("*.a3m")):
            yield p.stem, p
    else:
        raise ValueError(f"unknown layout {layout!r}")


def run_dir(root: Path, layout: str, out: Path, theta: float, max_seqs: int = 0) -> list[DepthRow]:
    rows: list[DepthRow] = []
    for stem, path in _iter_a3m_paths(root, layout):
        rows.append(depth_for_a3m(stem, path.read_text(), theta=theta, max_seqs=max_seqs))
    if not rows:
        raise SystemExit(
            f"no a3m files found under {root} with layout={layout!r}"
        )
    rows.sort(key=lambda r: r.neff)  # shallowest first -- the interesting tail
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return rows


_SELFTEST_A3M = """\
>query
ACDEFGHIKL
>identical_copy
ACDEFGHIKL
>one_mismatch
ACDEFGHIKM
>half_different
ACDEFPQRST
>with_insertions
ACDEFggGHIKL
>gappy
ACDE--HIKL
"""


def _selftest() -> None:
    seqs = parse_a3m(_SELFTEST_A3M)
    assert all(len(s) == 10 for s in seqs), [len(s) for s in seqs]
    # insertions were stripped -> the row stays length 10.
    assert seqs[4] == "ACDEFGHIKL", seqs[4]

    # At theta=0.80, identities to the query (query-length=10, all
    # non-gap, so identity = matches/10):
    #   identical_copy 1.0, with_insertions 1.0 (insertions stripped),
    #   one_mismatch 0.9, gappy 0.8 (8 matched, 2 query-only gaps in the
    #   union denominator), half_different 0.5.
    # Per-sequence cluster sizes (count of neighbours with id>=0.80,
    # self included):
    #   query=5 {q,copy,mis,ins,gappy}      -> 1/5
    #   copy=5  (same set)                  -> 1/5
    #   one_mismatch=4 {q,copy,mis,ins}     -> 1/4  (gappy is 0.7 to it)
    #   half_different=1                    -> 1
    #   with_insertions=5 (== query)        -> 1/5
    #   gappy=4 {q,copy,ins,gappy}          -> 1/4
    # Neff = 3*(1/5) + 2*(1/4) + 1 = 0.6 + 0.5 + 1 = 2.1.
    nf, n = neff(seqs, theta=0.80)
    assert n == 6, n
    assert abs(nf - 2.1) < 1e-9, nf

    # Raising theta to 1.0 keeps only exact matches: {query, copy,
    # with_insertions} form a cluster of 3; one_mismatch (0.9), gappy
    # (0.8) and half_different (0.5) become singletons.
    # Neff = 3*(1/3) + 1 + 1 + 1 = 4.0.
    nf_strict, _ = neff(seqs, theta=1.0)
    assert abs(nf_strict - 4.0) < 1e-9, nf_strict

    # Chunking must not change the answer.
    nf_chunked, _ = neff(seqs, theta=0.80, chunk=2)
    assert abs(nf_chunked - nf) < 1e-9, (nf_chunked, nf)

    row = depth_for_a3m("selftest", _SELFTEST_A3M)
    assert row.length == 10 and row.n_seqs == 6
    assert abs(row.neff - 2.1) < 1e-4
    print("selftest OK: Neff(0.80)=2.1  Neff(1.0)=4.0  (6 seqs, L=10)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_one = sub.add_parser("one", help="score a single a3m")
    p_one.add_argument("a3m", type=Path)
    p_one.add_argument("--theta", type=float, default=0.80)
    p_one.add_argument("--max-seqs", type=int, default=0,
                       help="Cap sequences used for Neff (0 = no cap; deep MSAs are slow uncapped).")

    p_dir = sub.add_parser("dir", help="score a directory of a3m -> CSV")
    p_dir.add_argument("root", type=Path)
    p_dir.add_argument("--layout", choices=("exp12", "flat"), default="exp12")
    p_dir.add_argument("--out", type=Path, default=Path("data/msa_depth.csv"))
    p_dir.add_argument("--theta", type=float, default=0.80)
    p_dir.add_argument("--max-seqs", type=int, default=0,
                       help="Cap sequences used for Neff (0 = no cap; deep MSAs are slow uncapped).")

    sub.add_parser("selftest", help="run the built-in self-test")

    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        _selftest()
    elif args.cmd == "one":
        row = depth_for_a3m(args.a3m.stem, args.a3m.read_text(), theta=args.theta,
                            max_seqs=args.max_seqs)
        for k, v in asdict(row).items():
            print(f"{k:16s} {v}")
    elif args.cmd == "dir":
        rows = run_dir(args.root, args.layout, args.out, args.theta, max_seqs=args.max_seqs)
        n_low = sum(1 for r in rows if r.neff < 10)
        print(f"wrote {len(rows)} rows to {args.out}")
        print(f"shallowest: {rows[0].stem} Neff={rows[0].neff}")
        print(f"Neff < 10: {n_low} / {len(rows)} proteins")


if __name__ == "__main__":
    sys.exit(main())
