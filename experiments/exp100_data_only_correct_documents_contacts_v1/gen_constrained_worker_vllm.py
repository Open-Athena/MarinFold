"""vLLM only-correct rollout worker (exp100) — structured-output-backend path.

The portable scale-out worker: masks via vLLM's structured-output grammar bitmask
(our OnlyCorrectBackend, see only_correct_backend.py) instead of a custom
LogitsProcessor (which the TPU stack ignores). Runs on **GPU vLLM** as-is
(develop/verify here) and on **iris TPU** once OnlyCorrectBackend is registered in
every engine process via a `vllm.general_plugins` entry point. Unmodified NLL is
recovered by a separate `prompt_logprobs` pass (supported on the pinned
tpu_inference rev; computed from raw pre-mask logits).

Same outputs as gen_constrained_worker_hf_gpu.py (nll/, documents/,
all_documents/, timings/), so aggregate_results.py / publish_to_hf.py are shared.

  # GPU (in-process engine so the monkeypatch reaches EngineCore):
  VLLM_ENABLE_V1_MULTIPROCESSING=0 python gen_constrained_worker_vllm.py \
      --model <hf_bf16 dir> --targets data/targets.parquet --prompts <dir> \
      --out <run dir> --shard 0/1 --n-rollouts 10

  # iris TPU: install this package with a vllm.general_plugins entry point that
  # calls only_correct_backend.register(), then drop --enforce-eager etc. per the
  # exp89/exp98 iris recipe and add --tensor-parallel-size 4.
"""
import argparse
import csv
import io
import json
import math
import os
import time

# Force an in-process EngineCore so register()'s monkeypatch of grammar_init (which
# runs in the EngineCore) takes effect without a pip-installed vllm.general_plugins
# entry point. Must be set before vllm is imported. On the JAX/TPU stack the engine
# is single-process (SPMD) anyway; on GPU this makes the driver == EngineCore.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from only_correct_backend import make_grammar_spec, register
from rollout_metrics import gt_by_band, parse_pred, score_rollout


def stage(model_path):
    """Download a gs:// model dir to local tmp (TPU worker needs local files)."""
    if not model_path.startswith("gs://"):
        return model_path
    import hashlib
    import tempfile
    from fsspec.core import url_to_fs
    fs, root = url_to_fs(model_path.rstrip("/"))
    local = os.path.join(tempfile.gettempdir(), "cv1model",
                         hashlib.sha256(model_path.encode()).hexdigest()[:12])
    os.makedirs(local, exist_ok=True)
    for e in fs.find(root, detail=True, maxdepth=1).values():
        if e.get("type") != "file":
            continue
        lp = os.path.join(local, os.path.basename(e["name"]))
        if not (os.path.exists(lp) and os.path.getsize(lp) == e.get("size")):
            fs.get(e["name"], lp)
    return local


def write_parquet(table, dest):
    with fsspec.open(dest, "wb") as fh:
        pq.write_table(table, fh)


def seq_nll(prompt_logprobs, lo, hi):
    """Sum of -logprob over prompt positions [lo, hi); prompt_logprobs=0 gives the
    realized token's logprob at each position (dict with a single entry)."""
    total, n = 0.0, 0
    for t in range(lo, hi):
        d = prompt_logprobs[t]
        if not d:
            continue
        total += -next(iter(d.values())).logprob
        n += 1
    return total, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n-rollouts", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=96,
                    help="keep < 128 so vLLM uses the serial grammar-bitmask fill "
                         "path; the parallel (threaded) path has a rare fill/accept "
                         "desync that yields occasional non-correct rollouts")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    # Device label recorded in each timings row; pass the real accelerator
    # (e.g. v5p-8, v6e-4). Defaults to "unknown" so it is never silently wrong —
    # the r0_full run's v5p-8 rows predate this and are mislabeled "A5000".
    ap.add_argument("--tpu-type", default="unknown")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save-all-documents", action="store_true",
                    help="also write all N rollouts verbatim per target (off by "
                         "default — at ~941k targets that's millions of extra files)")
    a = ap.parse_args()
    si, sm = (int(x) for x in a.shard.split("/"))
    out = a.out.rstrip("/")
    ofs, _ = fsspec.core.url_to_fs(out)

    register()  # monkeypatch the only-correct structured-output backend
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.sampling_params import StructuredOutputsParams

    with fsspec.open(a.targets, "rb") as fh:
        targets = {t["entry_id"]: t for t in pq.read_table(fh).to_pylist()}
    stems = sorted(targets)
    mine = stems[si::sm]
    if a.limit:
        mine = mine[: a.limit]

    if not a.overwrite:
        # Resume by listing the output dirs ONCE (a per-entry exists() check is
        # ~200k serial GCS calls at shard sizes ~117k -> hours of startup).
        def _done_entries():
            got = None
            for sub in ("nll", "documents"):
                try:
                    names = {os.path.basename(p).rsplit(".", 1)[0]
                             for p in ofs.ls(f"{out}/{sub}", detail=False)}
                except FileNotFoundError:
                    names = set()
                got = names if got is None else (got & names)
            return got or set()
        done = _done_entries()
        mine = [e for e in mine if e not in done]
    print(f"shard {si}/{sm}: {len(mine)} targets to do, n_rollouts={a.n_rollouts}", flush=True)
    if not mine:
        print(f"SHARD_DONE {si}/{sm}: nothing to do", flush=True)
        return

    t_load = time.time()
    llm = LLM(model=stage(a.model), max_model_len=a.max_model_len,
              tensor_parallel_size=a.tensor_parallel_size, enforce_eager=True,
              dtype="bfloat16", max_num_seqs=a.max_num_seqs,
              structured_outputs_config={"backend": "auto"})
    tok = llm.get_tokenizer()
    contact_id = tok.convert_tokens_to_ids("<contact>")
    end_id = tok.convert_tokens_to_ids("<end>")
    print(f"model loaded in {time.time()-t_load:.0f}s (contact_id={contact_id}, end_id={end_id})",
          flush=True)

    def pos_token_id(pos):
        return tok.convert_tokens_to_ids(f"<p{pos}>")

    timings = []
    t_all = time.time()
    for n, entry in enumerate(mine):
        t = targets[entry]
        L = int(t["L"])
        gt_seq = {(int(i), int(j)) for i, j in t["gt_contacts"]}
        gtb = gt_by_band(gt_seq)
        with fsspec.open(f"{a.prompts}/{entry}.parquet", "rb") as fh:
            prows = pq.read_table(fh).to_pylist()[: a.n_rollouts]

        # --- constrained generation via the structured-output bitmask ---
        # A rollout is viable only if the context leaves room for the full
        # structure section (3*n_gt+1 tokens). Reserve ONE extra token of headroom:
        # the scoring pass below re-feeds prompt+generated as a prompt with
        # max_tokens=1, so prompt+generated must be <= max_model_len-1 or vLLM
        # raises "prompt (N) + output (>=1) > max_model_len". If the resampled prefix
        # leaves no room, the rollout could at best produce a truncated, non-correct
        # document (and would crash the scoring pass) -> skip it.
        budget = a.max_model_len - 1  # leave room for the scoring pass's forced token
        need = 3 * len(gt_seq) + 1
        gprompts, gsp, meta = [], [], []
        for p in prows:
            prompt_ids = tok(p["prefix"], add_special_tokens=False).input_ids
            room = budget - len(prompt_ids)
            if room < need:
                continue
            seq_positions = list(p["seq_positions"])
            pos_to_seq = {int(pos): i for i, pos in enumerate(seq_positions)}
            gt_pos_ids = [(pos_token_id(seq_positions[i]), pos_token_id(seq_positions[j]))
                          for i, j in gt_seq]
            spec = make_grammar_spec(gt_pos_ids, contact_id, end_id)
            gprompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
            # exact generated length is 3*n_gt+1 (+small slack); cap to the room
            # left (against the reserved budget) so prompt+generated <= budget and
            # the scoring pass never exceeds max_model_len.
            max_new = min(3 * len(gt_seq) + 4, room)
            gsp.append(SamplingParams(
                n=1, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                max_tokens=max_new, stop_token_ids=[end_id],
                structured_outputs=StructuredOutputsParams(grammar=spec)))
            meta.append(dict(r=int(p["r"]), prefix=p["prefix"], prompt_ids=prompt_ids,
                             prompt_len=len(prompt_ids), pos_to_seq=pos_to_seq))

        # No rollout fits in context -> this protein cannot be regenerated within
        # max_model_len. Write a skip marker (both nll + documents, so resume marks
        # it done and never retries it) and move on, instead of crashing the shard.
        if not meta:
            write_parquet(pa.Table.from_pylist([dict(
                r=-1, n_gen_tokens=0, n_contacts=0, finished=False, n_pred=0,
                all_prec=math.nan, all_rec=math.nan, all_f1=math.nan,
                struct_nll=math.nan, struct_ntok=0, struct_nll_per_tok=math.nan)]),
                f"{out}/nll/{entry}.parquet")
            with fsspec.open(f"{out}/documents/{entry}.json", "w") as fh:
                json.dump(dict(entry_id=entry, L=L, n_gt=len(gt_seq), n_rollouts=0,
                               n_correct=0, skipped=True,
                               reason="prompt_exceeds_context"), fh)
            print(f"  [{n+1}/{len(mine)}] {entry} L={L} n_gt={len(gt_seq)}  "
                  f"SKIP prompt_exceeds_context (need {need}, ctx {a.max_model_len})",
                  flush=True)
            continue

        t0 = time.time()
        gouts = llm.generate(gprompts, gsp, use_tqdm=False)
        gen_s = time.time() - t0
        gen_ids = [list(o.outputs[0].token_ids) for o in gouts]
        gen_text = [tok.decode(g, skip_special_tokens=False) for g in gen_ids]

        # --- unmodified NLL via prompt_logprobs over prefix+generated ---
        spr = [TokensPrompt(prompt_token_ids=meta[i]["prompt_ids"] + gen_ids[i])
               for i in range(len(meta))]
        ssp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        t1 = time.time()
        souts = llm.generate(spr, ssp, use_tqdm=False)
        score_s = time.time() - t1

        rows, docs, total_gen = [], [], 0
        for i, o in enumerate(souts):
            plen = meta[i]["prompt_len"]
            g = gen_ids[i]
            total_gen += len(g)
            struct_nll, nstruct = seq_nll(o.prompt_logprobs, plen, plen + len(g))
            text = gen_text[i]
            pred = parse_pred(text, meta[i]["pos_to_seq"])
            sc = score_rollout(pred, gtb)
            finished = gouts[i].outputs[0].finish_reason in ("stop", "length")
            n_contacts = sum(1 for x in g if x == contact_id)
            rows.append(dict(
                r=meta[i]["r"], n_gen_tokens=len(g), n_contacts=n_contacts,
                finished=(g and g[-1] == end_id), n_pred=len(pred),
                all_prec=sc["all_prec"], all_rec=sc["all_rec"], all_f1=sc["all_f1"],
                struct_nll=struct_nll, struct_ntok=nstruct,
                struct_nll_per_tok=(struct_nll / nstruct) if nstruct else math.nan))
            docs.append(dict(r=meta[i]["r"], document=meta[i]["prefix"] + " " + text,
                             struct_nll=struct_nll, n_contacts=n_contacts,
                             finished=(g and g[-1] == end_id),
                             all_prec=sc["all_prec"], all_rec=sc["all_rec"],
                             pred_contacts=sorted([list(p) for p in pred])))

        # select the most-likely rollout ONLY among 100%-correct ones, so a rare
        # grammar-rejected (truncated) rollout can never win on its shorter NLL.
        correct = [i for i in range(len(rows))
                   if rows[i]["all_prec"] == 1.0 and rows[i]["all_rec"] == 1.0]
        pool = correct if correct else list(range(len(rows)))
        best = min(pool, key=lambda i: rows[i]["struct_nll"])
        n_correct = len(correct)
        write_parquet(pa.Table.from_pylist(rows), f"{out}/nll/{entry}.parquet")
        with fsspec.open(f"{out}/documents/{entry}.json", "w") as fh:
            json.dump(dict(entry_id=entry, L=L, n_gt=len(gt_seq), n_rollouts=len(rows),
                           n_correct=n_correct, selected_by="struct_nll", selected=docs[best],
                           sampling=dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)), fh)
        if a.save_all_documents:
            with fsspec.open(f"{out}/all_documents/{entry}.json", "w") as fh:
                json.dump(dict(entry_id=entry, documents=docs), fh)

        tps = total_gen / gen_s if gen_s > 0 else 0.0
        trow = dict(entry_id=entry, L=L, n_gt=len(gt_seq), n_rollouts=len(rows),
                    n_correct=n_correct, tensor_parallel=a.tensor_parallel_size,
                    total_gen_tokens=total_gen, gen_seconds=round(gen_s, 3),
                    score_seconds=round(score_s, 3), tokens_per_s=round(tps, 1),
                    mean_gen_tokens=round(total_gen / len(rows), 1),
                    best_struct_nll=round(rows[best]["struct_nll"], 3), tpu_type=a.tpu_type)
        timings.append(trow)
        tbuf = io.StringIO()
        tw = csv.DictWriter(tbuf, fieldnames=list(trow.keys()))
        tw.writeheader(); tw.writerow(trow)
        with fsspec.open(f"{out}/timings/{entry}.csv", "w") as fh:
            fh.write(tbuf.getvalue())
        print(f"  [{n+1}/{len(mine)}] {entry} L={L} n_gt={len(gt_seq)}  gen {gen_s:.1f}s "
              f"score {score_s:.1f}s  {tps:.0f} tok/s  correct={n_correct}/{len(rows)}  "
              f"best_struct_nll={rows[best]['struct_nll']:.1f}", flush=True)
        if n_correct != len(rows):
            print(f"  WARNING {entry}: {len(rows)-n_correct} not 100%-correct", flush=True)

    wall = time.time() - t_all
    print(f"\nSHARD_DONE {si}/{sm}: {len(timings)} targets, wall {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
