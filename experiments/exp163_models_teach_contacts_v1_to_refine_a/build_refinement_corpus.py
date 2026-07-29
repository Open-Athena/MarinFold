# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""exp163 Phase 1 — build the rollout-refinement corpus.

A refinement document = a normal contacts-v1 document with K candidate
rollouts prepended as separate blocks:

    <contacts-v1> <begin_sequence> ...seq...
      <CAND> <contact> pi pj ...          (candidate 1, subsampled/shuffled/flipped)
      ...  (K ~ Uniform{0..Kmax} separate blocks, unordered)
    <begin_statements> <contact> ...GT... <end>

Loss (Phase 2) is on the <begin_statements> section only; candidates are context.
Design rationale + probe evidence: see README.md (Steps 0-2).

Zero vocab change: the candidate-block marker CAND repurposes the spare token
"<contacts-and-distances-v1>" (the *other* format's doc-type sentinel, never
emitted inside a contacts-v1 document, so it is collision-proof; E8 never saw it
in training, so its embedding is effectively fresh). Configurable via --marker.

Inputs are LOCAL parquet (pre-stage bucket data in a separate hf>=1.5 process;
this builder imports marinfold, whose transformers pins huggingface_hub<1.0).

    # validate format on the 50 pre-staged proteins:
    uv run python build_refinement_corpus.py --rollouts <local>.parquet \
        --targets <exp98>/targets.parquet --n-docs 400 --validate
    # write a corpus shard:
    uv run python build_refinement_corpus.py ... --out corpus.jsonl --docs-per-protein 8
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np, pandas as pd

from marinfold.document_structures.contacts_v1.generate import build_document, GenerationConfig
from marinfold.document_structures.contacts_v1.parse import residues_from_sequence
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_STRUCTURE_TOKEN as BEGIN, NUM_POSITION_INDICES as NUM_POS,
    all_domain_tokens, position_token,
)

CTX = 8192
MIN_SEP = 6
DEFAULT_MARKER = "<contacts-and-distances-v1>"  # v1 <CAND>; repurposed spare
# v2 doc-type sentinel: id 7 renamed in place (see make_multi_tokenizer.py), so a
# multi-draft document announces its mode in position 0 instead of masquerading as
# a plain contacts-v1 document. Same id, same embedding row -- only the spelling.
MULTI_DOC_TOKEN = "<contacts-v1.multi>"
PLAIN_DOC_TOKEN = "<contacts-v1>"
VOCAB = set(all_domain_tokens()) | {MULTI_DOC_TOKEN}

def canon(flat) -> list[tuple[int, int]]:
    a = np.asarray(flat).reshape(-1, 2)
    if a.size == 0:
        return []
    lo = np.minimum(a[:, 0], a[:, 1]); hi = np.maximum(a[:, 0], a[:, 1])
    k = (hi - lo) >= MIN_SEP
    return sorted(set(zip(lo[k].tolist(), hi[k].tolist())))

def seq_section(entry_id, seq):
    """Reuse contacts-v1's own builder for the sequence section + wrap-around
    position map, then peel off everything up to <begin_statements>."""
    res = build_document(entry_id, residues_from_sequence(seq), [], config=GenerationConfig())
    if res is None:
        return None
    L, nterm, doc = res.seq_len, res.n_term_index, res.document
    seq_pos = [(nterm + k) % NUM_POS for k in range(L)]
    head = doc[: doc.rindex(BEGIN)].rstrip().split()
    return head, seq_pos, L

def emit(pairs, seq_pos, rng, marker=None):
    order = list(pairs); rng.shuffle(order)
    toks = [marker] if marker else []
    for (i, j) in order:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        toks += ["<contact>", position_token(seq_pos[a]), position_token(seq_pos[b])]
    return toks

def emit_draft(pairs, seq_pos, rng):
    """A v2 draft section: ``<begin_statements> …contacts…`` — deliberately NOT
    closed by ``<end>``.

    A draft is superseded by the next ``<begin_statements>``; only the FINAL
    section is closed by ``<end>``. That leaves both tokens with exactly their
    existing meanings — ``<end>`` still terminates the document, so it stays the
    generation stop token and no inference path changes — while
    ``<begin_statements>`` means "discard the previous structure, here is a new
    candidate". Termination becomes a learned choice: after a contact triple,
    emit another ``<contact>``, or ``<begin_statements>`` to start over, or
    ``<end>`` to finish.
    """
    return [BEGIN] + emit(pairs, seq_pos, rng)


def build_doc_multidraft(entry_id, seq, gt_pairs, rollouts, scores, Kmax, N_cap, rng,
                         budget=CTX, order="random"):
    """v2 document: K draft sections (worst -> best) then the TRUE section.

        <contacts-v1> <begin_sequence> …seq…
        <begin_statements> …draft 1…            (no <end>)
        <begin_statements> …draft 2…
        <begin_statements> …TRUE contacts… <end>

    Drafts are shown in **random order** by default — Phase 0's conclusion, and
    what v1 did, so the format change stays the only difference from the Phase-3
    baseline.

    ``order="ascending-f1"`` (needs ``scores``) instead sorts them worst-first
    into a refinement ramp. That is an ABLATION, not the default, and it carries
    two specific hazards: position alone then encodes quality, so the model can
    learn "later = better" without reading the drafts at all; and every training
    context ends on the best draft so far, so at generation time the model has
    never seen what follows a GOOD draft.

    Budget accounting matches v1: a draft costs ``1 + 3n`` (its
    ``<begin_statements>`` plus the contact triples), since drafts carry no
    ``<end>``.
    """
    r = seq_section(entry_id, seq)
    if r is None:
        return None
    head, seq_pos, L = r
    fixed = len(head) + 2  # final section's BEGIN + <end>
    gt = [(i, j) for (i, j) in gt_pairs if 0 <= i < L and 0 <= j < L]
    gt_cap = max(0, (budget - fixed) // 3)                 # GT is prioritised
    if len(gt) > gt_cap:
        gt = [gt[t] for t in rng.choice(len(gt), gt_cap, replace=False)]
    gt_toks = emit(gt, seq_pos, rng)
    remaining = budget - fixed - len(gt_toks)

    K = int(rng.integers(0, Kmax + 1))                     # Uniform{0..Kmax}
    drafts, kept = [], []
    if K > 0 and rollouts:
        pick = list(rng.choice(len(rollouts), min(K, len(rollouts)), replace=False))
        if order == "ascending-f1":
            if scores is None:
                raise ValueError("--draft-order ascending-f1 needs per-rollout scores")
            pick.sort(key=lambda t: scores[t])             # worst draft first (ablation)
        else:
            rng.shuffle(pick)                              # default: unordered (Phase 0)
        for ri in pick:
            pairs = [(i, j) for (i, j) in canon(rollouts[ri]) if i < L and j < L]
            if not pairs or remaining < 4:                 # BEGIN + one triple
                continue
            cap = min(len(pairs), N_cap)
            n = int(rng.integers(1, cap + 1))              # subsample Uniform[1,cap]
            n = min(n, (remaining - 1) // 3)               # fit budget (1 = BEGIN)
            if n <= 0:
                continue
            sub = [pairs[t] for t in rng.choice(len(pairs), n, replace=False)]
            sec = emit_draft(sub, seq_pos, rng)
            drafts += sec
            remaining -= len(sec)
            kept.append(float(scores[ri]) if scores is not None else float("nan"))
    # Announce refine mode in position 0. WITHOUT this, a multi-draft document is
    # prefix-identical to a plain contacts-v1 one, so the SAME prefix is supervised
    # toward true contacts in only the K=0 draws (~6% of documents) and toward a
    # ~13%-precision draft in the rest -- a direct train-time conflict on the base
    # task's own prefix, and the leading suspect for the K0 collapse.
    assert head[0] == PLAIN_DOC_TOKEN, f"expected {PLAIN_DOC_TOKEN} at position 0, got {head[0]}"
    head = [MULTI_DOC_TOKEN] + head[1:]
    toks = head + drafts + [BEGIN] + gt_toks + ["<end>"]
    meta = dict(entry_id=entry_id, L=L, K=len(kept), n_gt=len(gt), n_tokens=len(toks),
                draft_f1_first=kept[0] if kept else float("nan"),
                draft_f1_last=kept[-1] if kept else float("nan"))
    return " ".join(toks), meta


def build_doc(entry_id, seq, gt_pairs, rollouts, Kmax, N_cap, marker, rng, budget=CTX):
    r = seq_section(entry_id, seq)
    if r is None:
        return None
    head, seq_pos, L = r
    fixed = len(head) + 2  # + BEGIN + <end>
    gt = [(i, j) for (i, j) in gt_pairs if 0 <= i < L and 0 <= j < L]
    gt_cap = max(0, (budget - fixed) // 3)                 # GT is prioritised
    if len(gt) > gt_cap:
        gt = [gt[t] for t in rng.choice(len(gt), gt_cap, replace=False)]
    gt_toks = emit(gt, seq_pos, rng)
    remaining = budget - fixed - len(gt_toks)
    K = int(rng.integers(0, Kmax + 1))                     # Uniform{0..Kmax}
    blocks, kept = [], 0
    if K > 0 and rollouts:
        pick = rng.choice(len(rollouts), min(K, len(rollouts)), replace=False)
        for ri in pick:
            pairs = [(i, j) for (i, j) in canon(rollouts[ri]) if i < L and j < L]
            if not pairs or remaining < 4:
                continue
            cap = min(len(pairs), N_cap)
            n = int(rng.integers(1, cap + 1))              # subsample Uniform[1,cap]
            n = min(n, (remaining - 1) // 3)               # fit budget
            if n <= 0:
                continue
            sub = [pairs[t] for t in rng.choice(len(pairs), n, replace=False)]
            blk = emit(sub, seq_pos, rng, marker=marker)
            blocks += blk; remaining -= len(blk); kept += 1
    toks = head + blocks + [BEGIN] + gt_toks + ["<end>"]
    meta = dict(entry_id=entry_id, L=L, K=kept, n_gt=len(gt), n_tokens=len(toks))
    return " ".join(toks), meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="local parquet: entry_id,r,pred")
    ap.add_argument("--targets", required=True, help="exp98 targets.parquet (local)")
    ap.add_argument("--out", default=None, help="jsonl output; omit for --validate only")
    ap.add_argument("--marker", default=DEFAULT_MARKER)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--n-cap", type=int, default=120, help="max contacts shown per candidate")
    ap.add_argument("--docs-per-protein", type=int, default=8)
    ap.add_argument("--n-docs", type=int, default=None, help="cap total docs (validate)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["candidate", "multi-draft"], default="candidate",
                    help="candidate = v1 (<CAND> blocks, loss on the answer only); "
                         "multi-draft = v2 (every section is <begin_statements>..<end>, "
                         "drafts ordered worst->best, weights set at tokenize time)")
    ap.add_argument("--mix-plain", type=float, default=0.0, metavar="FRAC",
                    help="multi-draft only: fraction of the FINAL corpus that is plain "
                         "contacts-v1 documents, read from --plain-corpus. These carry no "
                         "drafts and open with <contacts-v1>, so with the mode token they "
                         "no longer collide with refine mode -- they train the base task "
                         "at full weight, which is the mechanistic anti-forgetting lever "
                         "(the header-weight lever was falsified in the 100k sweep).")
    ap.add_argument("--plain-corpus", default=None,
                    help="parquet glob of real contacts-v1 documents (a `document` column) "
                         "-- exactly what E8 trained on")
    ap.add_argument("--draft-order", choices=["random", "ascending-f1"], default="random",
                    help="multi-draft only. RANDOM is the default and matches Phase 0's "
                         "conclusion (train on unordered candidate sets). 'ascending-f1' "
                         "builds a quality ramp -- an ABLATION, not the default: it lets "
                         "the model learn 'later = better' positionally without reading "
                         "the drafts, and it never shows what follows a GOOD draft.")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.format == "candidate":
        assert a.marker in VOCAB, f"marker {a.marker} not in vocab (would need a vocab change)"

    tgt = pd.read_parquet(a.targets).set_index("entry_id")
    _cols = ["entry_id", "r", "pred"]
    if a.format == "multi-draft" and a.draft_order == "ascending-f1":
        _cols.append("all_f1")          # orders the refinement ramp; not used otherwise
    roll = pd.read_parquet(a.rollouts, columns=_cols)
    preds_by = {e: list(g["pred"].to_numpy()) for e, g in roll.groupby("entry_id")}
    scores_by = ({e: list(g["all_f1"].to_numpy()) for e, g in roll.groupby("entry_id")}
                 if "all_f1" in _cols else {})
    eids = [e for e in preds_by if e in tgt.index]
    rng = np.random.default_rng(a.seed)

    out = open(a.out, "w") if a.out else None
    metas, oov, over = [], 0, 0
    made = 0
    for eid in eids:
        gt = canon(np.concatenate([np.asarray(p).ravel() for p in tgt.loc[eid, "gt_contacts"]]))
        if len(gt) < 5:
            continue
        for _ in range(a.docs_per_protein):
            if a.n_docs and made >= a.n_docs:
                break
            if a.format == "multi-draft":
                r = build_doc_multidraft(eid, tgt.loc[eid, "sequence"], gt, preds_by[eid],
                                         scores_by.get(eid), a.kmax, a.n_cap, rng,
                                         order=a.draft_order)
            else:
                r = build_doc(eid, tgt.loc[eid, "sequence"], gt, preds_by[eid],
                              a.kmax, a.n_cap, a.marker, rng)
            if r is None:
                continue
            doc, meta = r
            made += 1; metas.append(meta)
            if a.validate:
                bad = [t for t in doc.split() if t not in VOCAB]
                if bad:
                    oov += 1
                    if oov <= 3:
                        print(f"  OOV in {eid}: {sorted(set(bad))[:6]}", flush=True)
                if meta["n_tokens"] > CTX:
                    over += 1
            if out:
                out.write(json.dumps({"entry_id": eid, "text": doc,
                                      "doc_type": "multi" if a.format == "multi-draft"
                                                  else "candidate"}) + "\n")
        if a.n_docs and made >= a.n_docs:
            break

    # ---- mix in real plain contacts-v1 documents -------------------------------
    # These open with <contacts-v1> and contain no drafts. With the mode token they
    # no longer share a prefix with refine mode, so they can train the base task at
    # full weight without the two objectives fighting over the same prefix. This is
    # the mechanistic anti-forgetting lever; the header-weight lever was falsified
    # in the 100k sweep (four arms within 0.002 on K0).
    n_plain = 0
    if a.format == "multi-draft" and a.mix_plain > 0:
        if not a.plain_corpus:
            raise SystemExit("--mix-plain needs --plain-corpus")
        if not 0 < a.mix_plain < 1:
            raise SystemExit(f"--mix-plain must be in (0, 1), got {a.mix_plain}")
        # frac of the FINAL corpus -> n_plain / (made + n_plain) = frac
        want = int(round(made * a.mix_plain / (1.0 - a.mix_plain)))
        print(f"\nmixing in plain contacts-v1 docs: target {want} "
              f"({a.mix_plain:.0%} of the final corpus) from {a.plain_corpus}", flush=True)
        import glob as _g
        import pyarrow.parquet as _pq
        shards = sorted(_g.glob(a.plain_corpus))
        if not shards:
            raise SystemExit(f"no shards matched {a.plain_corpus}")
        for sh in shards:
            if n_plain >= want:
                break
            docs = _pq.read_table(sh, columns=["document"]).column("document").to_pylist()
            for d in docs:
                if n_plain >= want:
                    break
                toks = d.split()
                if len(toks) > CTX or toks[0] != PLAIN_DOC_TOKEN:
                    continue
                n_plain += 1
                if out:
                    out.write(json.dumps({"entry_id": f"plain-{n_plain}", "text": d,
                                          "doc_type": "plain"}) + "\n")
            print(f"  {sh.split('/')[-1]}: plain={n_plain}/{want}", flush=True)
        print(f"  mixed {n_plain} plain docs -> final corpus {made + n_plain} "
              f"({n_plain/(made+n_plain):.1%} plain)", flush=True)

    if out:
        out.close()
    m = pd.DataFrame(metas)
    print(f"\nbuilt {len(m)} docs over {m.entry_id.nunique()} proteins", flush=True)
    print(f"  tokens: min={m.n_tokens.min()} med={int(m.n_tokens.median())} "
          f"max={m.n_tokens.max()} (ctx={CTX}); over-budget={over}", flush=True)
    print(f"  K (candidate blocks): {m.K.value_counts().sort_index().to_dict()}", flush=True)
    print(f"  n_gt: med={int(m.n_gt.median())} max={m.n_gt.max()}", flush=True)
    if a.validate:
        print(f"  VALIDATION: out-of-vocab docs={oov}, over-budget docs={over}  "
              f"-> {'PASS' if oov==0 and over==0 else 'FAIL'}", flush=True)
        # show one sample doc (head)
        r = build_doc(eids[0], tgt.loc[eids[0], "sequence"],
                      canon(np.concatenate([np.asarray(p).ravel() for p in tgt.loc[eids[0], "gt_contacts"]])),
                      preds_by[eids[0]], a.kmax, a.n_cap, a.marker, np.random.default_rng(1))
        if r:
            toks = r[0].split()
            ci = [i for i, t in enumerate(toks)
                  if t == (BEGIN if a.format == "multi-draft" else a.marker)][:-1]
            bi = toks.index(BEGIN)
            print(f"\n  sample doc: {len(toks)} toks; {len(ci)} candidate blocks; "
                  f"BEGIN at {bi}; first 24 toks:\n    {' '.join(toks[:24])}", flush=True)
            if ci:
                print(f"    first candidate block head: {' '.join(toks[ci[0]:ci[0]+10])}", flush=True)
            print(f"    GT section head: {' '.join(toks[bi:bi+10])}", flush=True)

if __name__ == "__main__":
    sys.exit(main())
