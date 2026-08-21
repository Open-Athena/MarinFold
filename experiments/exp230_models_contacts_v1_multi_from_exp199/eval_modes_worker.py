# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Gate B and the multi-mode report: free generation in BOTH modes, one checkpoint.

Prompts the same protein twice -- once with ``<contacts-v1>`` at position 0 and
once with ``<contacts-v1.multi>`` -- and reports what the model actually emits.
With a working mode marker this is a **generation-time** experiment on a single
checkpoint, which is strictly more informative than a readout-time ablation
(#175's argument, and its result).

**Gate B is the point of the plain arm.**  #163's arm F emitted **~2.94
sections** under the plain sentinel: the multi-draft habit leaked, so the marker
was not a clean on/off switch.  This measures the same number.

The multi arm reports what #200/#208 need from the hand-off: how many candidates
per generation, how disjoint they are, and how much better the best one is than
the last -- because best-of-N is paid for by *spread*, and the preregistered
risk (H3) is that on-policy drafts from a much stronger sampler make the
candidates too similar.

**Cap the sections.**  Only ~56 % of #163's multi generations emitted ``<end>``;
the rest ran to the context limit.  ``--max-sections`` bounds both the token
budget and the parsed output.

Output: one parquet per shard.

    rollouts/<mode>/shard-<i>-part-<k>.parquet
      target_id, dataset, stem, r, L, mode, n_sections, n_sections_raw,
      finished, n_gen_tokens, mean_jaccard, first_f1, best_f1, last_f1,
      best_section, section_sizes, section_f1s
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BEGIN, END = "<begin_statements>", "<end>"
NUM_POS, MIN_SEP = 2000, 6
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")

PLAIN_TOKEN = "<contacts-v1>"
MULTI_TOKEN = "<contacts-v1.multi>"

#: Sized from #163's measured ~220-contact section: 3 tokens a contact plus the
#: section marker, so ~668 tokens each.
TOKENS_PER_SECTION = 700

SCHEMA = pa.schema([
    ("target_id", pa.string()), ("dataset", pa.string()), ("stem", pa.string()),
    ("r", pa.int16()), ("L", pa.int32()), ("mode", pa.string()),
    ("n_sections", pa.int32()), ("n_sections_raw", pa.int32()),
    ("finished", pa.bool_()), ("n_gen_tokens", pa.int32()),
    ("mean_jaccard", pa.float32()),
    ("first_f1", pa.float32()), ("best_f1", pa.float32()), ("last_f1", pa.float32()),
    ("best_section", pa.int32()),
    ("section_sizes", pa.list_(pa.int32())), ("section_f1s", pa.list_(pa.float32())),
])


def fs_for(uri: str):
    import fsspec

    return fsspec.core.url_to_fs(uri)[0]


def read_parquet(uri: str, columns=None):
    fs = fs_for(uri)
    with fs.open(uri, "rb") as fh:
        return pq.read_table(fh, columns=columns)


def stage_model(src: str, dst: Path) -> str:
    if not src.startswith(("s3://", "gs://")):
        return src
    if dst.exists() and (dst / "config.json").exists():
        return str(dst)
    dst.mkdir(parents=True, exist_ok=True)
    fs = fs_for(src)
    t0 = time.time()
    for name in fs.ls(src.rstrip("/"), detail=False):
        leaf = name.rsplit("/", 1)[-1]
        if leaf:
            fs.get(name, str(dst / leaf))
    got = sum(p.stat().st_size for p in dst.iterdir() if p.is_file())
    print(f"[eval] staged {got / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)
    return str(dst)


def split_sections(text: str) -> list[str]:
    """Split a completion on ``<begin_statements>``.

    The prompt ends ON a ``<begin_statements>``, so the completion's first
    section is un-prefixed and every later one starts at a marker.  A trailing
    EMPTY section after a final marker is dropped: #200 found #163's regex kept
    it, which changed the section COUNT on 3/2216 truncated rollouts (never a
    score -- an empty section scores zero and never wins a max).
    """
    parts = text.split(BEGIN)
    return [p for p in parts if p.strip()]


def contacts_of(section: str, nterm: int, L: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for a, b in CONTACT_RE.findall(section):
        ia, ib = (int(a) - nterm) % NUM_POS, (int(b) - nterm) % NUM_POS
        if ia >= L or ib >= L or ia == ib or abs(ia - ib) < MIN_SEP:
            continue
        out.add((min(ia, ib), max(ia, ib)))
    return out


def f1_of(pred: set, gt: set) -> float:
    if not pred or not gt:
        return 0.0
    tp = len(pred & gt)
    p, r = tp / len(pred), tp / len(gt)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["plain", "multi"], required=True)
    ap.add_argument("--shard", required=True, help="i/N")
    ap.add_argument("--n-rollouts", type=int, default=4)
    ap.add_argument("--max-sections", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    shard_i, num_shards = (int(x) for x in a.shard.split("/"))

    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig, build_document, residues_from_sequence,
    )
    from vllm import LLM, SamplingParams

    targets = read_parquet(a.targets).to_pylist()
    targets.sort(key=lambda r: (r["L"], r["target_id"]))
    mine = [r for k, r in enumerate(targets) if k % num_shards == shard_i]
    if a.limit:
        mine = mine[: a.limit]
    print(f"[eval] mode={a.mode} shard {shard_i}/{num_shards}: {len(mine)} proteins", flush=True)

    model_dir = stage_model(a.model, Path("/tmp/exp230_eval_model"))
    llm = LLM(model=model_dir, max_model_len=a.max_model_len,
              tensor_parallel_size=a.tensor_parallel_size,
              max_num_seqs=a.max_num_seqs, trust_remote_code=False)
    tok = llm.get_tokenizer()
    end_id = tok.convert_tokens_to_ids(END)
    mode_token = PLAIN_TOKEN if a.mode == "plain" else MULTI_TOKEN
    mode_id = tok.convert_tokens_to_ids(mode_token)
    unk = getattr(tok, "unk_token_id", None)
    if mode_id is None or mode_id < 0 or (unk is not None and mode_id == unk):
        raise SystemExit(
            f"tokenizer does not know {mode_token!r} -- the RENAMED tokenizer "
            "(id 7 = <contacts-v1.multi>) must ship with these weights")
    print(f"[eval] {mode_token} id={mode_id} <end> id={end_id} vocab={len(tok)}", flush=True)

    fs = fs_for(a.out)
    out_dir = f"{a.out.rstrip('/')}/{a.mode}"
    rows: list[dict] = []
    part = 0

    def flush():
        nonlocal rows, part
        if not rows:
            return
        uri = f"{out_dir}/shard-{shard_i:04d}-part-{part:04d}.parquet"
        fs.makedirs(out_dir, exist_ok=True)
        with fs.open(uri, "wb") as fh:
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), fh)
        print(f"[eval] wrote {len(rows)} rows -> {uri}", flush=True)
        rows = []
        part += 1

    t0 = time.time()
    for n_done, rec in enumerate(mine, 1):
        L = int(rec["L"])
        gt = {(int(i), int(j)) for i, j in rec["gt_contacts"]}
        residues = residues_from_sequence(rec["input_seq"])
        prompts, nterms = [], []
        for k in range(a.n_rollouts):
            built = build_document(f"{rec['target_id']}:e{k}", residues, [],
                                   config=GenerationConfig())
            if built is None:
                continue
            doc = built.document
            head = doc[: doc.index(BEGIN) + len(BEGIN)]
            # Swap token 0 for the mode marker. Done on the STRING here because
            # both markers are single tokens with identical surrounding text; the
            # id check above is what guarantees the tokenizer can express it.
            assert head.startswith(PLAIN_TOKEN)
            prompts.append(mode_token + head[len(PLAIN_TOKEN):])
            nterms.append(built.n_term_index)
        if not prompts:
            continue
        plen = len(tok(prompts[0], add_special_tokens=False).input_ids)
        budget = (a.max_sections * TOKENS_PER_SECTION if a.mode == "multi"
                  else 6 * L + 128)
        params = SamplingParams(
            temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
            max_tokens=min(a.max_model_len - plen, budget), n=1,
            stop_token_ids=[end_id], skip_special_tokens=False,
        )
        outs = llm.generate(prompts, params, use_tqdm=False)
        for r, (out, nterm) in enumerate(zip(outs, nterms)):
            comp = out.outputs[0]
            raw = split_sections(comp.text)
            n_raw = len(raw)
            secs = [contacts_of(s, nterm, L) for s in raw[: a.max_sections]]
            f1s = [f1_of(s, gt) for s in secs]
            js = [jaccard(x, y) for x, y in combinations(secs, 2)] if len(secs) > 1 else []
            best = int(np.argmax(f1s)) if f1s else -1
            rows.append({
                "target_id": rec["target_id"], "dataset": rec["dataset"],
                "stem": rec["stem"], "r": r, "L": L, "mode": a.mode,
                "n_sections": len(secs), "n_sections_raw": n_raw,
                "finished": comp.finish_reason == "stop",
                "n_gen_tokens": len(comp.token_ids),
                "mean_jaccard": float(np.mean(js)) if js else float("nan"),
                "first_f1": f1s[0] if f1s else 0.0,
                "best_f1": max(f1s) if f1s else 0.0,
                "last_f1": f1s[-1] if f1s else 0.0,
                "best_section": best,
                "section_sizes": [len(s) for s in secs],
                "section_f1s": [float(v) for v in f1s],
            })
        if n_done % 50 == 0:
            flush()
            print(f"[eval] {n_done}/{len(mine)} proteins, "
                  f"{n_done / max(time.time() - t0, 1) * 3600:.0f}/h", flush=True)
    flush()
    print(f"[eval] mode={a.mode} shard {shard_i} done in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
