# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""exp163 Phase 3 eval worker — refiner@K vs refiner@K0 vs consensus, on GPU.

Scores one shard of the exp89 eval set under three candidate contexts, for one
model, and writes per-protein R-precision/AUC to S3:

* **K0**        — sequence prefix only, no candidates. The one-shot readout.
* **raw K**     — K candidate rollout blocks, the trained format.
* **consensus** — ONE block holding contacts with vote >= frac·M over M sampled
  rollouts: a deployable high-precision partial (the Step-2 lever).

This is the cluster port of ``mvp_eval.py``. Two things forced a rewrite rather
than a lift-and-shift:

* **Self-contained.** ``mvp_eval.py`` imports ``Scorer`` from exp82 and
  ``score_matrix`` / metric helpers from exp89 via ``sys.path`` surgery, and
  ``prefix_and_positions`` needs ``marinfold``. A pod has no repo checkout, so
  the ~60 lines that actually matter are reproduced here and the prefixes are
  read from the **pre-built prompt files** (``gen_prompts_exp163.py``) instead of
  being rebuilt — same deterministic ``build_document`` output, no marinfold.
* **No sklearn.** exp89's ``metric_rows`` pulls in ``roc_auc_score``; AUC here is
  the equivalent Mann-Whitney rank statistic in pure numpy, so the pod needs only
  torch + transformers + numpy + pyarrow.

Metric definitions match exp89 exactly: bands ``all``/``short``/``medium``/``long``
by sequence separation, R-precision = precision at the top-R scored pairs where R
is the number of true contacts in that band.

    python eval_refiner_worker.py --model s3://…/hf/step-51 \\
        --targets s3://…/eval554/targets.parquet \\
        --prompts s3://…/eval554/prompts \\
        --rollouts s3://…/eval554/runs/rollout_metrics \\
        --out s3://…/eval554/scores/refiner --shard 0/4 --k 16
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MIN_SEP = 6
BEGIN = "<begin_statements>"
MARKER = "<contacts-and-distances-v1>"   # the <CAND> block marker
RANGES = {"all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None)}
S3_KW = {"s3": {"addressing_style": "virtual"}}


def _fs(url: str):
    if not url.startswith("s3://"):
        return fsspec.filesystem("file")
    return fsspec.filesystem("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
                             config_kwargs=S3_KW)


def _strip(url: str) -> str:
    return url.split("://", 1)[1] if "://" in url else url


# --------------------------------------------------------------------------
# metrics (exp89 compute_metrics, numpy-only)
# --------------------------------------------------------------------------
def true_matrix(L: int, pairs) -> np.ndarray:
    m = np.zeros((L, L), bool)
    for i, j in pairs:
        if 0 <= i < j < L:
            m[i, j] = True
    return m


def resolved_pairs(L: int):
    a, b = np.triu_indices(L, k=1)
    return a, b, (b - a)


def _auc(g: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC — identical to sklearn.roc_auc_score, no sklearn."""
    n_pos = int(g.sum())
    n_neg = int(g.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    # average ranks within ties, so ties contribute 0.5 as sklearn does
    sv = s[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[g == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def band_metrics(score: np.ndarray, tmat: np.ndarray, pi, pj, psep) -> dict:
    """{band: (R-precision, AUC)} over the upper-triangular candidate universe."""
    cs, cg = score[pi, pj], tmat[pi, pj].astype(int)
    out = {}
    for rng, (lo, hi) in RANGES.items():
        inr = psep >= lo
        if hi is not None:
            inr = inr & (psep <= hi)
        s, g = cs[inr], cg[inr]
        nc, nt = int(s.size), int(g.sum())
        if nc == 0 or nt == 0:
            out[rng] = (float("nan"), float("nan"))
            continue
        order = np.argsort(-s, kind="mergesort")
        top = min(nt, nc)
        out[rng] = (float(g[order][:top].sum()) / top, _auc(g, s))
    return out


# --------------------------------------------------------------------------
# scorer (exp82 Scorer + exp89 score_matrix)
# --------------------------------------------------------------------------
class Scorer:
    def __init__(self, model_path, device="cuda", batch=16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16).to(device).eval()
        self.device, self.batch = device, batch
        self.contact_id = self.tok.convert_tokens_to_ids("<contact>")

    def ptoken(self, pos):
        return self.tok.convert_tokens_to_ids(f"<p{pos}>")

    def score_matrix(self, prefix: str, seq_positions: list[int]) -> np.ndarray:
        """Symmetrized geo-mean log-score [L, L] in input-sequence coords."""
        import torch.nn.functional as F
        torch = self.torch
        prefix_ids = self.tok(prefix, add_special_tokens=False).input_ids
        pos_ids = [self.ptoken(p) for p in seq_positions]
        base = list(prefix_ids) + [self.contact_id]
        with torch.no_grad():
            X = torch.tensor([base], device=self.device)
            lp_all = F.log_softmax(self.model(X).logits[0, -1].float(), -1)
            lp1 = lp_all[pos_ids].cpu().numpy()
            n = len(seq_positions)
            lp2 = np.empty((n, n), np.float32)
            seqs = [base + [pid] for pid in pos_ids]
            for s in range(0, len(seqs), self.batch):
                chunk = seqs[s:s + self.batch]
                X = torch.tensor(chunk, device=self.device)
                lp = F.log_softmax(self.model(X).logits[:, -1].float(), -1)[:, pos_ids]
                lp2[s:s + len(chunk)] = lp.cpu().numpy()
        fwd = lp1[:, None] + lp2
        return (0.5 * (fwd + fwd.T)).astype(np.float32)


# --------------------------------------------------------------------------
# candidate-context construction (mirrors build_refinement_corpus.emit)
# --------------------------------------------------------------------------
def canon(flat) -> list[tuple[int, int]]:
    a = np.asarray(flat).reshape(-1, 2)
    if a.size == 0:
        return []
    lo = np.minimum(a[:, 0], a[:, 1]); hi = np.maximum(a[:, 0], a[:, 1])
    k = (hi - lo) >= MIN_SEP
    return sorted(set(zip(lo[k].tolist(), hi[k].tolist())))


def emit_block(pairs, seq_pos, rng) -> list[str]:
    order = list(pairs); rng.shuffle(order)
    toks = [MARKER]
    for (i, j) in order:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        toks += ["<contact>", f"<p{seq_pos[a]}>", f"<p{seq_pos[b]}>"]
    return toks


def prefix_with(prefix: str, blocks) -> str:
    head = prefix[: prefix.rindex(BEGIN)].rstrip()
    toks = [head]
    for blk in blocks:
        toks += blk
    toks.append(BEGIN)
    return " ".join(toks)


def consensus(pool, L, frac, M, ncap, rng):
    idx = rng.choice(len(pool), min(M, len(pool)), replace=False)
    votes = Counter()
    for t in idx:
        for (i, j) in canon(pool[t]):
            if i < L and j < L:
                votes[(i, j)] += 1
    thr = max(2, int(frac * len(idx)))
    keep = [p for p, c in votes.items() if c >= thr]
    keep.sort(key=lambda p: -votes[p])
    return keep[:ncap]


def fetch_model(url: str, dest: str) -> str:
    """transformers cannot read s3://, so stage the checkpoint dir locally."""
    if not url.startswith("s3://"):
        return url
    fs = _fs(url)
    os.makedirs(dest, exist_ok=True)
    for f in fs.ls(_strip(url), detail=True):
        name = f["name"].split("/")[-1]
        out = os.path.join(dest, name)
        if os.path.exists(out) and os.path.getsize(out) == f["size"]:
            continue
        print(f"  fetching {name} ({f['size']/1e6:.0f} MB)", flush=True)
        fs.get(f["name"], out)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--rollouts", required=True, help="dir of rollout_metrics parquet parts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-cap", type=int, default=120)
    ap.add_argument("--cons-frac", type=float, default=0.3)
    ap.add_argument("--cons-pool", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None, help="skip proteins longer than this")
    a = ap.parse_args()

    si, sm = (int(x) for x in a.shard.split("/"))

    with fsspec.open(a.targets, "rb") as fh:
        targets = {t["entry_id"]: t for t in pq.read_table(fh).to_pylist()}

    rfs = _fs(a.rollouts)
    preds: dict[str, list] = {}
    for f in sorted(rfs.glob(f"{_strip(a.rollouts)}/*.parquet")):
        with rfs.open(f, "rb") as fh:
            t = pq.read_table(fh, columns=["entry_id", "pred"])
        for e, p in zip(t.column("entry_id").to_pylist(), t.column("pred").to_pylist()):
            preds.setdefault(e, []).append(p)

    stems = sorted(e for e in targets if e in preds)
    if a.max_len:
        stems = [e for e in stems if int(targets[e]["L"]) <= a.max_len]
    mine = [e for n, e in enumerate(stems) if n % sm == si]
    if a.limit:
        mine = mine[: a.limit]
    print(f"[exp163-eval] shard {si}/{sm}: {len(mine)}/{len(stems)} proteins, k={a.k}", flush=True)

    local = fetch_model(a.model, "/tmp/exp163_model")
    scorer = Scorer(local)
    rng = np.random.default_rng(0)
    pfs = _fs(a.prompts)

    rows, t0 = [], time.time()
    for n, eid in enumerate(mine):
        t = targets[eid]
        L = int(t["L"])
        gt = [(int(i), int(j)) for i, j in
              canon(np.concatenate([np.asarray(p).ravel() for p in t["gt_contacts"]]))
              if i < L and j < L]
        if len(gt) < 5:
            continue
        with pfs.open(f"{_strip(a.prompts)}/{eid}.parquet", "rb") as fh:
            prow = pq.read_table(fh).to_pylist()[0]
        prefix, seq_pos = prow["prefix"], [int(x) for x in prow["seq_positions"]]
        if len(seq_pos) != L:
            print(f"  !! {eid}: prompt L={len(seq_pos)} != target L={L}, skipping", flush=True)
            continue

        tmat = true_matrix(L, gt)
        pi, pj, psep = resolved_pairs(L)
        gts = set(gt)

        out = dict(entry_id=eid, L=L, n_gt=len(gt))
        # K0
        m = band_metrics(scorer.score_matrix(prefix, seq_pos), tmat, pi, pj, psep)
        for b in ("all", "long"):
            out[f"R0_{b}"], out[f"A0_{b}"] = m[b]
        # raw K blocks
        pool = preds[eid]
        blocks = []
        for ri in rng.choice(len(pool), min(a.k, len(pool)), replace=False):
            p = [(i, j) for (i, j) in canon(pool[ri]) if i < L and j < L]
            if p:
                blocks.append(emit_block(p[: a.n_cap], seq_pos, rng))
        m = band_metrics(scorer.score_matrix(prefix_with(prefix, blocks), seq_pos),
                         tmat, pi, pj, psep)
        for b in ("all", "long"):
            out[f"Rraw_{b}"], out[f"Araw_{b}"] = m[b]
        out["n_blocks"] = len(blocks)
        # consensus block
        cons = consensus(pool, L, a.cons_frac, a.cons_pool, a.n_cap, rng)
        out["cons_n"] = len(cons)
        out["cons_prec"] = (len(set(cons) & gts) / len(cons)) if cons else float("nan")
        m = band_metrics(
            scorer.score_matrix(prefix_with(prefix, [emit_block(cons, seq_pos, rng)]), seq_pos),
            tmat, pi, pj, psep)
        for b in ("all", "long"):
            out[f"Rcons_{b}"], out[f"Acons_{b}"] = m[b]
        rows.append(out)

        if (n + 1) % 10 == 0 or n == 0:
            el = time.time() - t0
            print(f"  [{n+1}/{len(mine)}] {eid} L={L} "
                  f"K0={out['R0_all']:.3f} rawK={out['Rraw_all']:.3f} "
                  f"cons={out['Rcons_all']:.3f}  ({el:.0f}s)", flush=True)

    ofs = _fs(a.out)
    dest = f"{_strip(a.out).rstrip('/')}/shard-{si}-of-{sm}.parquet"
    with ofs.open(dest, "wb") as fh:
        pq.write_table(pa.Table.from_pylist(rows), fh)
    r0 = np.nanmean([r["R0_all"] for r in rows]) if rows else float("nan")
    rr = np.nanmean([r["Rraw_all"] for r in rows]) if rows else float("nan")
    rc = np.nanmean([r["Rcons_all"] for r in rows]) if rows else float("nan")
    print(f"SHARD_DONE {si}/{sm}: {len(rows)} proteins in {time.time()-t0:.0f}s | "
          f"all-band K0={r0:.4f} rawK{a.k}={rr:.4f} cons={rc:.4f} -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
