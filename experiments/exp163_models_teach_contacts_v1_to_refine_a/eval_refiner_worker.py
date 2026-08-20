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
# Special-token keys carried over when falling back to a raw
# PreTrainedTokenizerFast. Mirrors marinfold.inference._tokenizer, inlined
# because the pod has no marinfold.
_FALLBACK_TOKENIZER_KEYS = (
    "bos_token", "eos_token", "unk_token", "pad_token", "cls_token",
    "sep_token", "mask_token", "model_max_length", "clean_up_tokenization_spaces",
)


def load_tokenizer(model_path: str):
    """Load a checkpoint tokenizer, tolerating levanter-exported ones.

    Levanter training exports declare ``"tokenizer_class": "TokenizersBackend"``
    in ``tokenizer_config.json`` — a class ``AutoTokenizer`` cannot resolve, so
    it raises ``ValueError: Tokenizer class TokenizersBackend does not exist``.
    They do ship a valid WordLevel ``tokenizer.json``, so fall back to building a
    ``PreTrainedTokenizerFast`` straight from it, carrying the special tokens
    over. Same repair as ``marinfold.inference._tokenizer.load_tokenizer``.
    """
    import json
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(str(model_path))
    except Exception:
        tok_file = os.path.join(model_path, "tokenizer.json")
        if not os.path.exists(tok_file):
            raise
        cfg = {}
        cfg_path = os.path.join(model_path, "tokenizer_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                cfg = json.load(fh)
        kwargs = {k: cfg[k] for k in _FALLBACK_TOKENIZER_KEYS if k in cfg}
        print("  tokenizer_class unresolvable -> rebuilt from tokenizer.json", flush=True)
        return PreTrainedTokenizerFast(tokenizer_file=tok_file, **kwargs)


def _load_model(model_path: str, torch):
    """``AutoModelForCausalLM`` with the right dtype kwarg for this transformers.

    transformers 5.x renamed ``torch_dtype`` to ``dtype``. On 4.x an unknown
    ``dtype=`` kwarg is swallowed into the model config, which then blows up
    serializing it (``TypeError: Object of type dtype is not JSON serializable``)
    — so pick by version rather than guessing.
    """
    import transformers
    from transformers import AutoModelForCausalLM
    major = int(transformers.__version__.split(".")[0])
    key = "dtype" if major >= 5 else "torch_dtype"
    print(f"  transformers {transformers.__version__} -> loading with {key}=bfloat16", flush=True)
    return AutoModelForCausalLM.from_pretrained(model_path, **{key: torch.bfloat16})


class Scorer:
    def __init__(self, model_path, device="cuda", batch=16):
        import torch
        self.torch = torch
        self.tok = load_tokenizer(model_path)
        self.model = _load_model(model_path, torch).to(device).eval()
        self.device, self.batch = device, batch
        self.contact_id = self.tok.convert_tokens_to_ids("<contact>")

    def ptoken(self, pos):
        return self.tok.convert_tokens_to_ids(f"<p{pos}>")

    def score_matrix(self, prefix: str, seq_positions: list[int],
                     mode_id: int | None = None) -> np.ndarray:
        """Symmetrized geo-mean log-score [L, L] in input-sequence coords.

        ``mode_id`` overwrites the document-type sentinel at position 0 AFTER
        encoding. Doing it on ids rather than text is deliberate: the v3 mode token
        is vocab id 7 RENAMED in place, so whether a given checkpoint's co-located
        tokenizer spells id 7 ``<contacts-and-distances-v1>`` (published) or
        ``<contacts-v1.multi>`` (renamed) is irrelevant — the id is the same and
        putting the string in the prefix would tokenize to UNK under the wrong one.
        """
        import torch.nn.functional as F
        torch = self.torch
        prefix_ids = self.tok(prefix, add_special_tokens=False).input_ids
        if mode_id is not None:
            prefix_ids = [mode_id] + list(prefix_ids[1:])
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


def emit_block(pairs, seq_pos, rng, fmt: str = "candidate") -> list[str]:
    """One block of candidate contacts, in whichever format the model was trained on.

    v1 ``candidate``: ``<CAND> <contact> pi pj …`` (a repurposed spare marker).
    v2 ``multi-draft``: ``<begin_statements> <contact> pi pj …`` with **no**
    ``<end>`` — a draft is superseded by the next ``<begin_statements>``, and only
    the final section is closed. Feeding a multi-draft model v1-format context
    (or vice versa) is out of distribution and meaningless, so this must match
    the corpus the checkpoint was trained on.
    """
    order = list(pairs); rng.shuffle(order)
    toks = [BEGIN] if fmt == "multi-draft" else [MARKER]
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
    ap.add_argument("--ks", default="2,4,8,16",
                    help="comma list of candidate-block counts to score (the test-time-"
                         "scaling sweep). See the note below on staying in-distribution.")
    ap.add_argument("--k", type=int, default=None, help="deprecated alias for --ks <k>")
    ap.add_argument("--n-cap", type=int, default=120)
    ap.add_argument("--cons-frac", type=float, default=0.3)
    ap.add_argument("--cons-pool", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None, help="skip proteins longer than this")
    ap.add_argument("--mode-id", type=int, default=None,
                    help="v3: doc-type sentinel id for refine mode (7). When set, the "
                         "candidate-conditioned arms use it and a SEPARATE base-mode K0 "
                         "is also recorded, so base-task retention and refine-mode "
                         "one-shot become distinct measurements. The old eval probed K0 "
                         "with the colliding plain prefix, which may have understated it.")
    ap.add_argument("--k0-only", action="store_true",
                    help="score ONLY the no-draft prefixes (R0, and R0multi if "
                         "--mode-id is set) and skip the K sweep + consensus. Those "
                         "are ~4x the work and are the only draft-SAMPLED (hence "
                         "run-to-run noisy) metrics; R0 is deterministic. Use for a "
                         "base-task-retention profile across many checkpoints.")
    ap.add_argument("--format", choices=["candidate", "multi-draft"], default="candidate",
                    help="candidate context format — MUST match what the checkpoint "
                         "was trained on (v1 <CAND> blocks vs v2 statements sections)")
    a = ap.parse_args()

    si, sm = (int(x) for x in a.shard.split("/"))
    ks = sorted({int(x) for x in a.ks.split(",") if x.strip()}) if a.k is None else [a.k]

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
    print(f"[exp163-eval] shard {si}/{sm}: {len(mine)}/{len(stems)} proteins, "
          f"ks={ks} format={a.format}", flush=True)

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
        # K0 in BASE mode (plain <contacts-v1> prefix) -- base-task retention,
        # directly comparable to the base model's 0.3355.
        m = band_metrics(scorer.score_matrix(prefix, seq_pos), tmat, pi, pj, psep)
        for b in ("all", "long"):
            out[f"R0_{b}"], out[f"A0_{b}"] = m[b]
        # K0 in REFINE mode (mode sentinel, still no drafts) -- what the model does
        # when told it is refining but given nothing to refine.
        if a.mode_id is not None:
            m = band_metrics(scorer.score_matrix(prefix, seq_pos, a.mode_id),
                             tmat, pi, pj, psep)
            for b in ("all", "long"):
                out[f"R0multi_{b}"], out[f"A0multi_{b}"] = m[b]
        if a.k0_only:
            rows.append(out)
            if (n + 1) % 25 == 0 or n == 0:
                print(f"  [{n+1}/{len(targets)}] {eid} R0={out['R0_all']:.3f}", flush=True)
            continue
        # raw candidate blocks, swept over K (test-time scaling).
        #
        # STAY IN DISTRIBUTION. The training corpus caps the candidate section by a
        # token budget: mean 471 contacts / 1,422 tokens per document, MAX 1,282 /
        # 3,862. K x n_cap can blow straight past that (16 x 120 = 1,920 contacts =
        # ~5,776 tokens, ~4x the training mean and well beyond anything seen), and a
        # context that far out of distribution scores BELOW random. So sweep K and
        # record the realized size, rather than assuming a single K is safe.
        pool = preds[eid]
        order = rng.permutation(len(pool))
        blocks: list[list[str]] = []
        built = 0
        for K in ks:
            while built < min(K, len(pool)):
                p = [(i, j) for (i, j) in canon(pool[order[built]]) if i < L and j < L]
                built += 1
                if p:
                    blocks.append(emit_block(p[: a.n_cap], seq_pos, rng, a.format))
            n_con = sum((len(b) - 1) // 3 for b in blocks)
            m = band_metrics(scorer.score_matrix(prefix_with(prefix, blocks), seq_pos, a.mode_id),
                             tmat, pi, pj, psep)
            for b in ("all", "long"):
                out[f"Rraw{K}_{b}"], out[f"Araw{K}_{b}"] = m[b]
            out[f"ncon{K}"] = n_con
        out["n_blocks"] = len(blocks)
        # consensus block
        cons = consensus(pool, L, a.cons_frac, a.cons_pool, a.n_cap, rng)
        out["cons_n"] = len(cons)
        out["cons_prec"] = (len(set(cons) & gts) / len(cons)) if cons else float("nan")
        m = band_metrics(
            scorer.score_matrix(prefix_with(prefix, [emit_block(cons, seq_pos, rng, a.format)]),
                                seq_pos, a.mode_id),
            tmat, pi, pj, psep)
        for b in ("all", "long"):
            out[f"Rcons_{b}"], out[f"Acons_{b}"] = m[b]
        rows.append(out)

        if (n + 1) % 10 == 0 or n == 0:
            el = time.time() - t0
            sweep = " ".join(f"K{K}={out[f'Rraw{K}_all']:.3f}" for K in ks)
            k0m = f" K0multi={out['R0multi_all']:.3f}" if a.mode_id is not None else ""
            print(f"  [{n+1}/{len(mine)}] {eid} L={L} K0base={out['R0_all']:.3f}{k0m} {sweep} "
                  f"cons={out['Rcons_all']:.3f} ncon@{ks[-1]}={out[f'ncon{ks[-1]}']}  ({el:.0f}s)",
                  flush=True)

    ofs = _fs(a.out)
    dest = f"{_strip(a.out).rstrip('/')}/shard-{si}-of-{sm}.parquet"
    with ofs.open(dest, "wb") as fh:
        pq.write_table(pa.Table.from_pylist(rows), fh)
    def mean(key):
        return np.nanmean([r[key] for r in rows]) if rows else float("nan")
    sweep = " ".join(f"K{K}={mean(f'Rraw{K}_all'):.4f}(n{mean(f'ncon{K}'):.0f})" for K in ks)
    k0m = f" K0multi={mean('R0multi_all'):.4f}" if a.mode_id is not None else ""
    print(f"SHARD_DONE {si}/{sm}: {len(rows)} proteins in {time.time()-t0:.0f}s | "
          f"all-band K0base={mean('R0_all'):.4f}{k0m} {sweep} cons={mean('Rcons_all'):.4f} -> {dest}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
