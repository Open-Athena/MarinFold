# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step B (canonical / iris-TPU variant) — MarinFold pairwise contact scores
with **vLLM**, runnable on the iris TPU pool.

Identical scoring definition to ``score_eval_set.py`` (exp82 *pairwise*: the
symmetrized geo-mean log-prob of the ``<contact> <pi> <pj>`` statement) and the
**same** ``scores/<dataset>__<stem>.npz`` output layout, so the metric step is
backend-agnostic — the local transformers run and this vLLM run are
interchangeable and should agree to float precision.

vLLM gives us per-token logprobs via *generation* (the path eric's
``experiments.protein.eval_protein_contacts`` already runs on TPU). For each
protein we need only ``L+1`` forwards:

* one over ``<prefix> <contact>`` → ``lp1[i] = log P(<pi> | prefix,<contact>)``
* one over ``<prefix> <contact> <pi>`` for each i → ``lp2[i,j]``

We read the full next-token distribution (``logprobs=vocab``; set
``max_logprobs`` at init) and gather it at the position tokens. The shared
``<prefix> <contact>`` is reused by vLLM prefix caching.

Launch on iris (from a marin checkout, mirroring eric's recipe)::

    HF_TOKEN=... uv run iris --config=lib/iris/examples/marin.yaml job run \
        --region us-east5 --tpu=v5p-8 --memory=64GB --disk=64GB --cpu=16 \
        --extra=vllm --extra=tpu -- \
        python -m score_eval_set_vllm \
            --model gs://marin-us-east5/checkpoints/prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1-bc3084/hf/step-35679 \
            --out-dir gs://marin-us-east5/eval/exp89-contacts-v1/scores

It also runs locally with a CUDA vLLM build (``--model /path/to/hf``).
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd

from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)

NUM_POS = 2000
BEGIN = "<begin_statements>"
EXP78 = Path("/home/bizon/git/MarinFold-exp78/experiments/exp78_evals_esmfold_contacts")
MANIFESTS = (EXP78 / "data/eval_manifest_foldbench.csv", EXP78 / "data/eval_manifest_exp65.csv")


def load_eval_proteins() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for m in MANIFESTS:
        df = pd.read_csv(m)
        for _, r in df.iterrows():
            rows.append((r["dataset"], r["stem"], r["input_seq"]))
    return rows


def prefix_and_positions(stem: str, input_seq: str):
    residues = residues_from_sequence(input_seq)
    result = build_document(stem, residues, [], config=GenerationConfig())
    if result is None:
        return None
    L = result.seq_len
    seq_positions = [(result.n_term_index + k) % NUM_POS for k in range(L)]
    doc = result.document
    return doc[: doc.index(BEGIN) + len(BEGIN)], seq_positions, L


def _save_npz(path: str, score: np.ndarray) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, score=score.astype(np.float16))
    with fsspec.open(path, "wb") as fh:
        fh.write(buf.getvalue())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF dir (local or gs://)")
    ap.add_argument("--out-dir", required=True, help="scores dir (local or gs://)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    # stage a gs:// model dir locally (vLLM wants a local path), mirroring eric.
    model_path = args.model
    if str(args.model).startswith("gs://"):
        local = Path("/tmp/exp89_model")
        local.mkdir(parents=True, exist_ok=True)
        fs = fsspec.filesystem("gs")
        for f in fs.ls(args.model):
            fs.get(f, str(local / Path(f).name))
        model_path = str(local)

    tok_probe = LLM  # noqa: F841 (keep import obvious)
    llm = LLM(model=model_path, max_model_len=args.max_model_len,
              tensor_parallel_size=args.tensor_parallel_size,
              enforce_eager=True, max_logprobs=2845)
    tok = llm.get_tokenizer()
    contact_id = tok.convert_tokens_to_ids("<contact>")

    def ptoken(p: int) -> int:
        return tok.convert_tokens_to_ids(f"<p{p}>")

    one = SamplingParams(max_tokens=1, temperature=0.0, logprobs=2845)

    def next_logprobs(prompt_ids_list: list[list[int]]) -> list[dict[int, float]]:
        """Full next-token logprob dict for each prompt (batched generate)."""
        from vllm import TokensPrompt
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids_list], one)
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]  # {token_id: Logprob}
            res.append({tid: v.logprob for tid, v in lp.items()})
        return res

    proteins = load_eval_proteins()
    if args.limit:
        proteins = proteins[: args.limit]
    print(f"vLLM scoring {len(proteins)} proteins -> {args.out_dir}", flush=True)

    n_ok = 0
    for k, (dataset, stem, seq) in enumerate(proteins):
        built = prefix_and_positions(stem, seq)
        if built is None:
            continue
        prefix, seq_positions, L = built
        pos_ids = [ptoken(p) for p in seq_positions]
        base = list(tok(prefix, add_special_tokens=False).input_ids) + [contact_id]
        # lp1: one forward over prefix+<contact>
        d1 = next_logprobs([base])[0]
        neg = float(np.log(1e-12))
        lp1 = np.array([d1.get(pid, neg) for pid in pos_ids], dtype=np.float64)
        # lp2[i, :]: forward over prefix+<contact>+<pi> for each i (batched)
        dists = next_logprobs([base + [pid] for pid in pos_ids])
        lp2 = np.array([[d.get(pj, neg) for pj in pos_ids] for d in dists], dtype=np.float64)
        fwd = lp1[:, None] + lp2
        sym = 0.5 * (fwd + fwd.T)
        _save_npz(f"{args.out_dir.rstrip('/')}/{dataset}__{stem}.npz", sym.astype(np.float16))
        n_ok += 1
        if (k + 1) % 25 == 0:
            print(f"  ...{k + 1}/{len(proteins)} (last {stem} L={L})", flush=True)
    print(f"[vllm-score] {n_ok} scored -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
