# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Corpus-quality numbers for the ESM-Atlas arm, from the documents alone.

`analyze_documents.py` needs the staged backbones to recover the native
contact count. This arm never staged anything -- it reads source cif straight
from HF and discards it -- but it does not need to: every row carries both
sides of the comparison already.

    designed density = contacts_emitted      / seq_len
    native   density = native_contacts_emitted / parent_seq_len

`native_contacts_emitted` is the *published parent document's* count, so this
compares against `contacts_v1_esm_atlas_decontam` itself rather than against a
recomputation, and any disagreement would be a real difference rather than a
backend artifact.

Stratified by length, because on the AFDB arm the effect was not
length-uniform (1.040 below L=100, 0.989 above L=800) and the corpus-wide
ratio alone would have hidden that.

Shards are indexed across the whole range rather than taken as a prefix. This
experiment has been misled four times by samples that looked representative;
the ESM corpus is not length-sorted, but the habit is cheap to keep.

    uv run python analyze_esm_documents.py \\
        --documents-glob 's3://.../esm_documents/*.parquet' --shards 24
"""

from __future__ import annotations

import argparse
import collections
import sys

import fsspec
import pyarrow.parquet as pq

LENGTH_BINS = ((0, 100), (100, 200), (200, 400), (400, 800), (800, 10 ** 6))

COLUMNS = ["seq_len", "contacts_emitted", "parent_seq_len",
           "native_contacts_emitted", "mpnn_temperature", "mpnn_score",
           "identity_to_native", "num_tokens"]


def _log(msg: str) -> None:
    print(f"[exp266-analyze] {msg}", file=sys.stderr, flush=True)


def _bin(length: int) -> str:
    for lo, hi in LENGTH_BINS:
        if lo <= length < hi:
            return f"{lo}-{hi if hi < 10 ** 6 else '+'}"
    return "?"


class Acc:
    """Sums, kept as sums so the ratio is over the whole population.

    Deliberately not a mean of per-document ratios: short documents dominate
    that average and it is not the quantity the density question asks about.
    """

    def __init__(self) -> None:
        self.docs = 0
        self.res = self.con = 0
        self.nres = self.ncon = 0

    def add(self, sl, ce, psl, nce) -> None:
        self.docs += 1
        self.res += sl
        self.con += ce
        self.nres += psl
        self.ncon += nce

    def row(self) -> tuple[float, float, float]:
        d = self.con / self.res if self.res else 0.0
        n = self.ncon / self.nres if self.nres else 0.0
        return d, n, (d / n if n else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents-glob", required=True)
    ap.add_argument("--shards", type=int, default=24,
                    help="Shards to read, indexed across the whole range.")
    args = ap.parse_args()

    fs, _ = fsspec.core.url_to_fs(args.documents_glob)
    files = sorted(fs.glob(args.documents_glob))
    n = max(1, min(args.shards, len(files)))
    picked = [files[round(i * (len(files) - 1) / max(n - 1, 1))] for i in range(n)]
    _log(f"{len(files)} shards, reading {len(picked)}")

    overall = Acc()
    by_len: dict[str, Acc] = collections.defaultdict(Acc)
    by_temp: dict[float, list] = collections.defaultdict(
        lambda: [0, 0.0, 0.0, Acc()])   # docs, sum identity, sum score, density

    for k, path in enumerate(picked, 1):
        with fs.open(path, "rb") as handle:
            t = pq.read_table(handle, columns=COLUMNS)
        cols = {c: t.column(c).to_pylist() for c in COLUMNS}
        for i in range(t.num_rows):
            sl, ce = cols["seq_len"][i], cols["contacts_emitted"][i]
            psl, nce = cols["parent_seq_len"][i], cols["native_contacts_emitted"][i]
            if not sl or not psl:
                continue
            overall.add(sl, ce, psl, nce)
            by_len[_bin(sl)].add(sl, ce, psl, nce)
            slot = by_temp[cols["mpnn_temperature"][i]]
            slot[0] += 1
            slot[1] += cols["identity_to_native"][i]
            slot[2] += cols["mpnn_score"][i]
            slot[3].add(sl, ce, psl, nce)
        _log(f"  {k}/{len(picked)} shards, {overall.docs:,} documents")

    d, nat, ratio = overall.row()
    print(f"\ndocuments analysed: {overall.docs:,}")
    print(f"contacts/residue  designed {d:.4f}  native {nat:.4f}  "
          f"ratio {ratio:.4f}")

    print("\nby designed length:")
    print(f"  {'bin':>10}  {'docs':>12}  {'designed':>9}  {'native':>9}  {'ratio':>6}")
    for lo, hi in LENGTH_BINS:
        key = f"{lo}-{hi if hi < 10 ** 6 else '+'}"
        acc = by_len.get(key)
        if not acc or not acc.docs:
            continue
        dd, nn, rr = acc.row()
        print(f"  {key:>10}  {acc.docs:>12,}  {dd:>9.4f}  {nn:>9.4f}  {rr:>6.4f}")

    print("\nby temperature:")
    print(f"  {'T':>5}  {'docs':>12}  {'identity':>8}  {'score':>7}  {'ratio':>6}")
    for temp in sorted(by_temp):
        docs, ident, score, acc = by_temp[temp]
        _dd, _nn, rr = acc.row()
        print(f"  {temp:>5.2f}  {docs:>12,}  {ident/docs:>8.4f}  "
              f"{score/docs:>7.4f}  {rr:>6.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
