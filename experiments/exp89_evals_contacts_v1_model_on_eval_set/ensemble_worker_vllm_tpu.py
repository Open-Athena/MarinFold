"""vLLM/TPU worker: K-resample-ensembled pairwise P(contact) for a shard of
the exp89 eval proteins. Reads pre-generated prompts from GCS, writes one
``<dataset>__<stem>.npz`` (key ``score`` = ensembled [L,L] matrix) per protein
to GCS — the same layout the local transformers scorer used, so exp89's
compute_metrics.py consumes it unchanged.

  python _cv1_ensemble_worker.py --model gs://.../hf/step-35679 \
      --prompts gs://marin-us-east5/eval/exp89/ensemble_prompts.parquet \
      --out gs://marin-us-east5/eval/exp89/scores_ens --shard 0/4 --method gen
"""
import argparse, hashlib, io, os, tempfile, time
from collections import defaultdict
import fsspec
import numpy as np
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

NEG = float(np.log(1e-12))


def stage(model_path):
    if not model_path.startswith("gs://"):
        return model_path
    fs, root = url_to_fs(model_path.rstrip("/"))
    local = os.path.join(tempfile.gettempdir(), "cv1model", hashlib.sha256(model_path.encode()).hexdigest()[:12])
    os.makedirs(local, exist_ok=True)
    for e in fs.find(root, detail=True, maxdepth=1).values():
        if e.get("type") != "file":
            continue
        name = os.path.basename(e["name"]); lp = os.path.join(local, name)
        if not (os.path.exists(lp) and os.path.getsize(lp) == e.get("size")):
            fs.get(e["name"], lp)
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--method", choices=["gen", "prompt"], default="gen")
    ap.add_argument("--max-logprobs", type=int, default=2845)
    a = ap.parse_args()
    si, sm = (int(x) for x in a.shard.split("/"))
    from vllm import LLM, SamplingParams, TokensPrompt

    with fsspec.open(a.prompts, "rb") as fh:
        rows = pq.read_table(fh).to_pylist()
    by_stem = defaultdict(list)
    for r in rows:
        by_stem[(r["dataset"], r["stem"])].append(r)
    stems = sorted(by_stem)
    mine = stems[si::sm]
    print(f"shard {si}/{sm}: {len(mine)}/{len(stems)} proteins, method={a.method}", flush=True)

    llm = LLM(model=stage(a.model), max_model_len=8192, tensor_parallel_size=1,
              enforce_eager=True, max_logprobs=a.max_logprobs, dtype="bfloat16")
    tok = llm.get_tokenizer()
    contact_id = tok.convert_tokens_to_ids("<contact>")
    pid_cache = {}

    def pid(pos):
        if pos not in pid_cache:
            pid_cache[pos] = tok.convert_tokens_to_ids(f"<p{pos}>")
        return pid_cache[pos]

    def gen_dists(prompt_id_lists, want_ids_per):
        """For each prompt, return {token_id: logprob} for the next token."""
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=a.max_logprobs)
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_id_lists], sp, use_tqdm=False)
        return [{tid: v.logprob for tid, v in o.outputs[0].logprobs[0].items()} for o in outs]

    def prompt_lp(prompt_id_lists):
        """Logprob of each prompt's final token (prompt_logprobs path)."""
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_id_lists], sp, use_tqdm=False)
        res = []
        for p, o in zip(prompt_id_lists, outs):
            last = o.prompt_logprobs[-1]
            res.append(last[p[-1]].logprob if (last and p[-1] in last) else NEG)
        return res

    def score_one(prefix, positions):
        base = list(tok(prefix, add_special_tokens=False).input_ids) + [contact_id]
        pos_ids = [pid(p) for p in positions]
        L = len(positions)
        if a.method == "gen":
            d1 = gen_dists([base], None)[0]
            lp1 = np.array([d1.get(x, NEG) for x in pos_ids])
            d2 = gen_dists([base + [x] for x in pos_ids], None)
            lp2 = np.array([[d.get(y, NEG) for y in pos_ids] for d in d2])
        else:
            lp1 = np.array(prompt_lp([base + [x] for x in pos_ids]))
            flat = prompt_lp([base + [pos_ids[i], pos_ids[j]] for i in range(L) for j in range(L)])
            lp2 = np.array(flat).reshape(L, L)
        fwd = lp1[:, None] + lp2
        return np.exp(fwd) + np.exp(fwd.T)

    ofs, _ = url_to_fs(a.out)
    t0 = time.time()
    for n, (dataset, stem) in enumerate(mine):
        sub = sorted(by_stem[(dataset, stem)], key=lambda r: r["k"])
        mats = [score_one(r["prefix"], list(r["seq_positions"])) for r in sub]
        ens = np.mean(mats, axis=0).astype(np.float16)
        # also keep the k=0 single realization, so ensemble-vs-single is a
        # clean same-backend (bf16/TPU) comparison.
        single = mats[0].astype(np.float16)
        buf = io.BytesIO(); np.savez_compressed(buf, score=ens, score_single=single)
        with fsspec.open(f"{a.out.rstrip('/')}/{dataset}__{stem}.npz", "wb") as fh:
            fh.write(buf.getvalue())
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(mine)}  {stem} L={ens.shape[0]}  {(time.time()-t0):.0f}s", flush=True)
    print(f"SHARD_DONE {si}/{sm} ({len(mine)} proteins, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
