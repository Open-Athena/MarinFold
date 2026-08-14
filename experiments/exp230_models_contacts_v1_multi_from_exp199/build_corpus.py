# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 — build exp230's multi-draft + plain-rehearsal corpus.

The document format is #163's, unchanged, because #163 is what established it:

    <contacts-v1.multi> <begin_sequence> ...sequence, shuffled...
    <begin_statements> ...draft 1...          (NOT closed by <end>)
    <begin_statements> ...draft 2...
    <begin_statements> ...ground truth... <end>

``<begin_statements>`` may repeat and means *"discard the previous candidate,
here is a new one"*; only the final section is closed by ``<end>``, so ``<end>``
keeps its meaning as the document terminator and no inference path changes.
``<contacts-v1.multi>`` is vocab id **7 renamed in place** — vocab size stays
2,845, every other id is untouched, there is no embedding resize and no id
drift, and exp200's RL stack already assumes id 7 means this.

Three things are exp230's, not #163's:

* **Drafts are on-policy.**  They are exp199's own rollouts (``gen_rollouts_worker``),
  not E8's.  Measured on the smoke sample, an exp199 draft is **0.41** precise
  against ground truth where #163's E8 drafts were ~0.12 and arm F's were 0.23.
* **The plain rehearsal half is generated from the same proteins**, by calling
  the ordinary contacts-v1 generator on the same ground truth.  #163 mixed in
  documents from a different corpus; drawing both halves from one pool makes the
  token-0 marker the *only* systematic difference between them, which is the
  property that has to hold for the marker to become a clean switch.
* **``--docs-per-protein`` defaults to 3** (#163 used 2).  Each document redraws
  K, which drafts are shown, how far each is subsampled, the N-terminal offset
  and the statement order — the same nuisance-symmetry augmentation #166 got
  +0.026 R-precision from.

The 50:50 plain mix is **not optional**.  #163's v2 sweep trained four weight
profiles on 100k multi-draft documents with *no* rehearsal and every arm lost
~44 % of the base task; the published arm F differs in exactly two ways —
``w_draft = 1.0`` and this mix — and is a statistical tie with base.

    uv run python build_corpus.py --targets /data/exp230_multi/targets.parquet \\
        --rollouts /data/exp230_multi/rollouts --out /data/exp230_multi/corpus
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from marinfold.document_structures.contacts_v1.generate import (
    GenerationConfig, build_document,
)
from marinfold.document_structures.contacts_v1.parse import (
    RawContact, residues_from_sequence,
)
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_STRUCTURE_TOKEN as BEGIN, NUM_POSITION_INDICES as NUM_POS, position_token,
)

CTX = 8192
MIN_SEP = 6
PLAIN_DOC_TOKEN = "<contacts-v1>"
#: Vocab id 7, renamed in place by ``make_multi_tokenizer.py``.  The published
#: tokenizer spells id 7 ``<contacts-and-distances-v1>``; a document written with
#: the literal string below only tokenizes correctly under the renamed
#: tokenizer, which is why it ships co-located with the weights.
MULTI_DOC_TOKEN = "<contacts-v1.multi>"

SCHEMA = pa.schema([
    ("doc_id", pa.string()), ("target_id", pa.string()), ("arm", pa.string()),
    ("kind", pa.string()), ("document", pa.string()),
    ("L", pa.int32()), ("K", pa.int32()), ("n_gt", pa.int32()),
    ("n_tokens", pa.int32()), ("draft_f1_mean", pa.float32()),
])


def seq_section(doc_id: str, seq: str):
    """The header of a fresh realization: tokens, ring->seq map, length.

    Reuses the library's own builder (with an empty contact list) rather than
    re-implementing the wrap-around numbering, then peels off everything up to
    ``<begin_statements>``.  #163's approach, kept so a multi-draft header is
    byte-for-byte a contacts-v1 header.
    """
    built = build_document(doc_id, residues_from_sequence(seq), [], config=GenerationConfig())
    if built is None:
        return None
    L, nterm, doc = built.seq_len, built.n_term_index, built.document
    seq_pos = [(nterm + k) % NUM_POS for k in range(L)]
    head = doc[: doc.index(BEGIN)].rstrip().split()
    return head, seq_pos, L


def emit(pairs, seq_pos, rng) -> list[str]:
    """A contact list in random order with random i/j orientation.

    Both shuffles are the format's own nuisance symmetries: contacts-v1
    statements are unordered and ``<contact> <pi> <pj>`` is symmetric, so
    re-randomising them per document is augmentation, not noise.
    """
    order = list(pairs)
    rng.shuffle(order)
    toks: list[str] = []
    for i, j in order:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        toks += ["<contact>", position_token(seq_pos[a]), position_token(seq_pos[b])]
    return toks


def canon(flat) -> list[tuple[int, int]]:
    """Flat ``[i0,j0,i1,j1,...]`` -> sorted unique (lo, hi) at or above MIN_SEP."""
    a = np.asarray(flat, dtype=np.int64).reshape(-1, 2)
    if a.size == 0:
        return []
    lo = np.minimum(a[:, 0], a[:, 1])
    hi = np.maximum(a[:, 0], a[:, 1])
    keep = (hi - lo) >= MIN_SEP
    return sorted(set(zip(lo[keep].tolist(), hi[keep].tolist())))


def draft_size(m: int, alpha: float, rng) -> int:
    """Draw a draft length from a truncated power law on {1, ..., m}.

    ``P(n) proportional to n**-alpha``, so the density decays smoothly from a
    single contact upward and the support reaches the **whole rollout** -- there
    is no fixed cap. The previous ``Uniform{1, min(m, 250)}`` could never emit a
    draft longer than 250, and 32 % of real rollouts are longer than that, so a
    large *complete* candidate -- exactly what every prior section looks like at
    generation time -- was impossible to train on.

    Sampled by inverting the continuous CDF, which is closed-form and O(1):
    for ``F(x) = (x**(1-a) - 1) / ((m+1)**(1-a) - 1)`` on ``[1, m+1)``,
    ``x = (1 + u * ((m+1)**(1-a) - 1))**(1/(1-a))``. Discretising by floor costs
    nothing at these magnitudes and avoids an O(m) table per draw.

    At ``alpha = 0.5`` over the measured rollout-size distribution this gives
    mean 81 contacts (the old scheme's realised mean was 80), median 46, and
    5.8 % of drafts at or above 90 % of their rollout.
    """
    if m <= 1:
        return max(1, m)
    b = 1.0 - alpha
    hi = (m + 1) ** b
    x = (1.0 + rng.random() * (hi - 1.0)) ** (1.0 / b)
    return int(min(m, max(1, int(x))))


def build_multi(doc_id, seq, gt_pairs, rollouts, f1s, *, alpha, rng, budget=CTX):
    """One multi-draft document.  Budget: ground truth is prioritised over drafts.

    A draft costs ``1 + 3n`` tokens (its ``<begin_statements>`` plus the triples)
    because drafts carry no ``<end>``; the final section costs ``1 + 3n + 1``.
    """
    section = seq_section(doc_id, seq)
    if section is None:
        return None
    head, seq_pos, L = section
    fixed = len(head) + 2                                   # final BEGIN + <end>
    gt = [(i, j) for (i, j) in gt_pairs if 0 <= i < L and 0 <= j < L]
    gt_cap = max(0, (budget - fixed) // 3)
    if len(gt) > gt_cap:
        gt = [gt[t] for t in rng.choice(len(gt), gt_cap, replace=False)]
    gt_toks = emit(gt, seq_pos, rng)
    remaining = budget - fixed - len(gt_toks)

    # PACK the context: use as many rollout slots as fit, rather than drawing a
    # count. Every slot is treated identically -- same size law, same weight, no
    # positional privilege -- and the number of sections is decided by the
    # budget, not by a Uniform{0..K} draw.
    drafts: list[str] = []
    kept_f1: list[float] = []
    if len(rollouts):
        pick = list(rng.permutation(len(rollouts)))         # unordered: #163 Phase 0
        for ri in pick:
            if remaining < 4:                               # BEGIN + one triple
                break
            pairs = [(i, j) for (i, j) in canon(rollouts[ri]) if i < L and j < L]
            if not pairs:
                continue
            n = draft_size(len(pairs), alpha, rng)
            n = min(n, (remaining - 1) // 3)
            if n <= 0:
                continue
            sub = [pairs[t] for t in rng.choice(len(pairs), n, replace=False)]
            sec = [BEGIN] + emit(sub, seq_pos, rng)
            drafts += sec
            remaining -= len(sec)
            kept_f1.append(float(f1s[ri]))

    if head[0] != PLAIN_DOC_TOKEN:
        raise AssertionError(f"expected {PLAIN_DOC_TOKEN} at position 0, got {head[0]}")
    head = [MULTI_DOC_TOKEN] + head[1:]
    toks = head + drafts + [BEGIN] + gt_toks + ["<end>"]
    return " ".join(toks), dict(
        L=L, K=len(kept_f1), n_gt=len(gt), n_tokens=len(toks),
        draft_f1_mean=float(np.mean(kept_f1)) if kept_f1 else float("nan"),
    )


def build_plain(doc_id, seq, gt_pairs):
    """One ordinary contacts-v1 document — the rehearsal half.

    Built by the library generator from the same ground truth, so it is exactly
    a training document of the corpus exp199 was pretrained on.
    """
    contacts = [RawContact(seq_i=int(i), seq_j=int(j), degree=1.0) for i, j in gt_pairs]
    built = build_document(doc_id, residues_from_sequence(seq), contacts,
                           config=GenerationConfig())
    if built is None:
        return None
    doc = built.document
    return doc, dict(L=built.seq_len, K=0, n_gt=len(contacts),
                     n_tokens=len(doc.split()), draft_f1_mean=float("nan"))


def load_rollouts(path, log) -> dict[str, tuple[list, list]]:
    """``target_id -> (flat pred arrays, per-rollout f1)``, local dir or ``gs://``.

    Predictions are kept as numpy int16 arrays rather than Python lists: at
    2.5 M rollouts of ~300 values each, the list-of-int representation costs
    several GB of pure object overhead for data that is never indexed
    element-wise.
    """
    uri = str(path)
    if "://" in uri:
        import fsspec

        fs = fsspec.core.url_to_fs(uri)[0]
        scheme = uri.split("://", 1)[0]
        files = [p if "://" in p else f"{scheme}://{p}"
                 for p in sorted(fs.glob(f"{uri.rstrip('/')}/*.parquet"))]
        opener = lambda f: fs.open(f, "rb")  # noqa: E731
    else:
        local = Path(uri)
        files = sorted(local.glob("*.parquet")) if local.is_dir() else [local]
        opener = lambda f: open(f, "rb")  # noqa: E731
    if not files:
        raise SystemExit(f"no rollout parquets under {uri}")

    by_target: dict[str, tuple[list, list]] = {}
    n = 0
    for f in files:
        with opener(f) as fh:
            t = pq.read_table(fh, columns=["target_id", "pred", "f1"])
        tids = t.column("target_id").to_pylist()
        f1s_all = t.column("f1").to_pylist()
        preds_col = t.column("pred")
        for k, (tid, f1) in enumerate(zip(tids, f1s_all)):
            preds, f1s = by_target.setdefault(tid, ([], []))
            preds.append(np.asarray(preds_col[k].as_py(), dtype=np.int16))
            f1s.append(f1)
            n += 1
    log(f"[rollouts] {n:,} rollouts over {len(by_target):,} proteins "
        f"from {len(files)} file(s)")
    return by_target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--rollouts", required=True,
                    help="local dir or gs:// prefix of the rollout parquets")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--docs-per-protein", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="draft-size power law exponent; P(n) ~ n**-alpha on "
                         "{1..|rollout|}. 0 = the old uniform draw")
    ap.add_argument("--mix-plain", type=float, default=0.5)
    ap.add_argument("--rows-per-file", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=230)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N proteins")
    a = ap.parse_args()

    def log(*msg):
        print(" ".join(str(m) for m in msg), flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    targets = pq.read_table(a.targets).to_pylist()
    if a.limit:
        targets = targets[: a.limit]
    rollouts = load_rollouts(a.rollouts, log)

    rng = np.random.default_rng(a.seed)
    rows: list[dict] = []
    n_file = 0
    stats = {"multi": 0, "plain": 0, "skipped_no_rollouts": 0, "skipped_build": 0}
    t0 = time.time()

    def flush():
        nonlocal rows, n_file
        if not rows:
            return
        out = a.out / f"corpus-{n_file:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out)
        log(f"[corpus] wrote {len(rows):,} docs -> {out}")
        rows = []
        n_file += 1

    for rec in targets:
        tid = rec["target_id"]
        preds, f1s = rollouts.get(tid, ([], []))
        if not preds:
            stats["skipped_no_rollouts"] += 1
            continue
        gt = [(int(i), int(j)) for i, j in rec["gt_contacts"]]
        for d in range(a.docs_per_protein):
            built = build_multi(f"{tid}:m{d}", rec["sequence"], gt, preds, f1s,
                                alpha=a.alpha, rng=rng)
            if built is None:
                stats["skipped_build"] += 1
                continue
            doc, meta = built
            rows.append({"doc_id": f"{tid}:m{d}", "target_id": tid, "arm": rec["arm"],
                         "kind": "multi", "document": doc, **meta})
            stats["multi"] += 1
        # The plain half is drawn 1:1 against the multi half, from the SAME
        # protein, so mode is the only thing the marker has to explain.
        n_plain = int(round(a.docs_per_protein * a.mix_plain / (1.0 - a.mix_plain)))
        for d in range(n_plain):
            built = build_plain(f"{tid}:p{d}", rec["sequence"], gt)
            if built is None:
                stats["skipped_build"] += 1
                continue
            doc, meta = built
            rows.append({"doc_id": f"{tid}:p{d}", "target_id": tid, "arm": rec["arm"],
                         "kind": "plain", "document": doc, **meta})
            stats["plain"] += 1
        if len(rows) >= a.rows_per_file:
            flush()
    flush()

    log(f"[corpus] {stats['multi']:,} multi + {stats['plain']:,} plain "
        f"= {stats['multi'] + stats['plain']:,} documents in {(time.time() - t0) / 60:.1f} min")
    log(f"[corpus] skipped: {stats['skipped_no_rollouts']:,} without rollouts, "
        f"{stats['skipped_build']:,} unbuildable")
    (a.out / "corpus.provenance.json").write_text(json.dumps({
        "seed": a.seed, "docs_per_protein": a.docs_per_protein, "mix_plain": a.mix_plain,
        "alpha": a.alpha, "stats": stats,
        "targets": str(a.targets), "rollouts": str(a.rollouts),
        "multi_doc_token": MULTI_DOC_TOKEN,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
