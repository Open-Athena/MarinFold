# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""One shard of the rollout+resample contact eval, on one CoreWeave H100.

Same measurement as ``score_rollout_vllm.py`` (which is the single-GPU local
version) — n resampled contacts-v1 rollouts per eval protein, voted into an
``[L, L]`` per-pair occurrence matrix — but sharded, S3-backed and resumable so a
fleet of single-H100 batch-band pods can cover the 554-protein eval set in
minutes instead of the ~80 min one A5000 takes per checkpoint.

Differences from the local version, all I/O:

* **targets come from a parquet on S3** (``dataset``/``stem``/``L``/``input_seq``)
  rather than the workstation's GT-universe + manifest files, so the pod needs no
  repo checkout. Shards interleave the length-sorted order (``idx % n``) so every
  shard gets the same mix of short and long proteins — the L=761 tail otherwise
  decides the whole job's wall clock.
* **the model is copied S3 → local disk first.** vLLM wants a directory.
* **output is sparse triplets**, one parquet part per chunk:
  ``(dataset, stem, L, i, j, votes)`` for the upper triangle only. Dense
  ``[L, L]`` npz per protein would be ~3k tiny S3 objects; the caller rebuilds
  the exact same matrices from these.
* **resumable**: on restart it reads the shard's existing parts and skips those
  stems, so a batch-band preemption resumes instead of redoing the shard.

Run (the dispatcher's bootstrap does this)::

    python score_rollout_worker.py --model s3://…/model --targets s3://…/targets.parquet \
        --out s3://…/scores --label exp117 --shard 3/16 --n-rollouts 100 --top-k -1
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BEGIN, THINK, NUM_POS, MIN_SEP = "<begin_statements>", "<think>", 2000, 6
CONTACT_RE = re.compile(r"<contact>\s+<p(\d+)>\s+<p(\d+)>")

SCHEMA = pa.schema([
    ("dataset", pa.string()), ("stem", pa.string()), ("L", pa.int32()),
    ("i", pa.int16()), ("j", pa.int16()), ("votes", pa.int16()),
])


"""All S3 access goes through **fsspec**, never pyarrow's own S3FileSystem and
never the `aws` CLI: iris injects CoreWeave's endpoint + credentials as an
``FSSPEC_S3`` blob (plus the virtual-addressing ``config_kwargs`` the bootstrap
exports), which only fsspec/s3fs reads. pyarrow's native S3 would go to AWS, and
the CLI isn't on PATH in the vLLM image."""


def stage_model(src: str, dst: Path) -> Path:
    """Copy a remote model directory to local disk (vLLM needs a real directory).

    Any fsspec URL works — ``s3://`` on CoreWeave, ``gs://`` on the marin TPU
    pool. A plain path is returned untouched.
    """
    import fsspec
    if "://" not in src:
        return Path(src)
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fs, root = fsspec.core.url_to_fs(src)
    files = [f for f in fs.ls(root, detail=True) if f["type"] == "file"]
    assert files, f"no files under {src}"
    for f in files:
        fs.get_file(f["name"], str(dst / os.path.basename(f["name"])))
    size = sum(f["size"] for f in files)
    print(f"[worker] staged model {src} -> {dst} ({len(files)} files, "
          f"{size / 2**30:.2f} GiB, {time.time() - t0:.0f}s)", flush=True)
    return dst


def read_parquet(uri: str, **kw):
    import fsspec
    with fsspec.open(uri, "rb") as fh:
        return pq.read_table(fh, **kw)


def write_parquet(tbl, uri: str) -> None:
    import fsspec
    with fsspec.open(uri, "wb") as fh:
        pq.write_table(tbl, fh, compression="zstd")


def load_targets(path: str):
    recs = read_parquet(path).to_pylist()
    recs.sort(key=lambda r: r["L"])          # short -> long, then interleave
    return recs


def normalize_tokenizer_metadata(model_dir: Path) -> None:
    """Make Marin's HF export tokenizer loadable by vanilla Transformers/vLLM.

    The export writes ``tokenizer_class=TokenizersBackend`` because that is the
    in-repo wrapper used during training. The vLLM image has only stock
    Transformers, which can load the same ``tokenizer.json`` as a
    ``PreTrainedTokenizerFast`` once the metadata names the stock class.
    """
    import json

    tokenizer_config = model_dir / "tokenizer_config.json"
    if not tokenizer_config.exists():
        return
    with tokenizer_config.open() as handle:
        cfg = json.load(handle)
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        with tokenizer_config.open("w") as handle:
            json.dump(cfg, handle, indent=2, sort_keys=True)
        print("[worker] rewrote tokenizer_class TokenizersBackend -> PreTrainedTokenizerFast", flush=True)
    special_tokens_map = model_dir / "special_tokens_map.json"
    if not special_tokens_map.exists():
        special = {key: cfg[key] for key in ("unk_token", "eos_token", "pad_token") if key in cfg}
        with special_tokens_map.open("w") as handle:
            json.dump(special, handle, indent=2, sort_keys=True)
        print("[worker] wrote special_tokens_map.json", flush=True)


def done_stems(out_dir: str, shard_i: int, num_shards: int) -> tuple[set[str], int]:
    """Stems this shard has already covered, and how many part files hold them.

    The part count is what the caller must resume *writing* from. Restarting the
    part counter at 0 on a resume silently overwrites the shard's own earlier
    parts — the stems in them are skipped as "done" and then their file is
    clobbered, so they vanish from the output entirely. exp169 lost 2 proteins
    that way (a smoke run's part-0000 replaced by the full run's part-0000)
    before this returned the count.
    """
    import fsspec
    fs, _ = fsspec.core.url_to_fs(out_dir)
    pat = f"{out_dir.rstrip('/')}/shard-{shard_i:03d}-of-{num_shards:03d}-part-*.parquet"
    try:
        parts = fs.glob(pat)
    except FileNotFoundError:
        return set(), 0
    seen: set[str] = set()
    for p in parts:
        try:
            # `glob` returns bare keys, so the protocol has to go back on before
            # fsspec can reopen them. `unstrip_protocol` is what keeps this worker
            # backend-agnostic — exp169 runs the same file against gs:// on TPU.
            t = read_parquet(fs.unstrip_protocol(p), columns=["dataset", "stem"])
            seen |= {f"{d}__{s}" for d, s in zip(t.column("dataset").to_pylist(),
                                                 t.column("stem").to_pylist())}
        except Exception as e:                       # a half-written part from a kill
            print(f"[worker] ignoring unreadable part {p}: {e}", flush=True)
    return seen, len(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True, help="s3 prefix; parts land under <out>/<label>/")
    ap.add_argument("--label", required=True)
    ap.add_argument("--shard", required=True, help="i/n")
    ap.add_argument("--n-rollouts", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--contact-mult", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-per-request-seed", dest="per_request_seed", action="store_false",
                    help="Required on TPU: the JAX backend rejects SamplingParams.seed "
                         "outright (`ValueError: JAX does not support per-request seed.`). "
                         "The engine-level --seed still applies, and the 100 rollouts per "
                         "protein are independent draws from the same distribution either "
                         "way, so the estimator is unchanged — only bitwise replay of one "
                         "specific run is lost.")
    ap.add_argument("--prompt-think-after-sequence", action="store_true",
                    help="Append one literal <think> token after the sequence prefix. "
                         "Deprecated alias for --prompt-think-count 1.")
    ap.add_argument("--prompt-think-count", type=int, default=0,
                    help="Append this many literal <think> tokens after the sequence prefix "
                         "before generating contacts.")
    ap.add_argument("--gpu-frac", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    shard_i, num_shards = (int(x) for x in a.shard.split("/"))
    out_dir = f"{a.out.rstrip('/')}/{a.label}"
    prompt_think_count = max(a.prompt_think_count, 1 if a.prompt_think_after_sequence else 0)
    if prompt_think_count < 0:
        raise ValueError(f"--prompt-think-count must be nonnegative, got {prompt_think_count}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from marinfold.document_structures.contacts_v1 import (
        GenerationConfig, build_document, residues_from_sequence,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    recs = load_targets(a.targets)
    mine = [r for k, r in enumerate(recs) if k % num_shards == shard_i]
    skip, n_existing_parts = done_stems(out_dir, shard_i, num_shards)
    todo = [r for r in mine if f"{r['dataset']}__{r['stem']}" not in skip]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[worker] shard {shard_i}/{num_shards}: {len(mine)} assigned, {len(skip)} already done, "
          f"{len(todo)} to do | n_rollouts={a.n_rollouts} top_k={a.top_k} top_p={a.top_p} "
          f"T={a.temperature} per_request_seed={a.per_request_seed} "
          f"prompt_think_count={prompt_think_count} label={a.label}", flush=True)
    if not todo:
        print("[worker] nothing to do")
        return 0

    model_dir = stage_model(a.model, Path("/tmp/marinfold_model"))
    normalize_tokenizer_metadata(model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    end_id = tok.convert_tokens_to_ids("<end>")
    assert end_id is not None and end_id >= 0, "no <end> token in the tokenizer"
    llm = LLM(model=str(model_dir), dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=a.gpu_frac, enable_prefix_caching=False,
              generation_config="vllm", max_num_seqs=a.max_num_seqs, seed=a.seed)

    # Continue the part numbering past whatever a previous attempt wrote.
    t0, n_unfinished, n_total, part = time.time(), 0, 0, n_existing_parts
    for s in range(0, len(todo), a.chunk):
        group = todo[s:s + a.chunk]
        prompts, per, sps = [], [], []
        for r in group:
            residues = residues_from_sequence(r["input_seq"])
            first, maps = len(prompts), []
            for k in range(a.n_rollouts):
                doc = build_document(f"{r['stem']}:r{k}", residues, [], config=GenerationConfig())
                prompt = doc.document[: doc.document.index(BEGIN) + len(BEGIN)]
                if prompt_think_count:
                    prompt = f"{prompt} {' '.join([THINK] * prompt_think_count)}"
                prompts.append(prompt)
                maps.append({(doc.n_term_index + t) % NUM_POS: t for t in range(doc.seq_len)})
            plen = len(tok(prompts[first], add_special_tokens=False).input_ids)
            max_new = min(8192 - plen, a.contact_mult * r["L"] + 128)
            per.append((r, first, maps))
            sps += [SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                                   max_tokens=max_new, stop_token_ids=[end_id],
                                   skip_special_tokens=False,
                                   **({"seed": a.seed * 1_000_003 + first + k}
                                      if a.per_request_seed else {}))
                    for k in range(a.n_rollouts)]
        ts = time.time()
        outs = llm.generate(prompts, sps, use_tqdm=False)
        dt = time.time() - ts

        rows = {c: [] for c in ("dataset", "stem", "L", "i", "j", "votes")}
        for r, first, maps in per:
            chunk_outs = outs[first:first + a.n_rollouts]
            n_unfinished += sum(1 for o in chunk_outs if o.outputs[0].finish_reason != "stop")
            n_total += a.n_rollouts
            L = r["L"]
            M = np.zeros((L, L), np.int32)
            for o, seqidx in zip(chunk_outs, maps):
                seen = set()
                for x, y in CONTACT_RE.findall(o.outputs[0].text):
                    ia, ib = seqidx.get(int(x)), seqidx.get(int(y))
                    if ia is None or ib is None or ia == ib:
                        continue
                    key = (min(ia, ib), max(ia, ib))
                    if abs(ia - ib) >= MIN_SEP and key not in seen:
                        seen.add(key)
                        M[key] += 1
            ii, jj = np.nonzero(np.triu(M, k=1))
            rows["dataset"] += [r["dataset"]] * len(ii)
            rows["stem"] += [r["stem"]] * len(ii)
            rows["L"] += [L] * len(ii)
            rows["i"] += ii.astype(np.int16).tolist()
            rows["j"] += jj.astype(np.int16).tolist()
            rows["votes"] += M[ii, jj].astype(np.int16).tolist()

        dest = (f"{out_dir}/shard-{shard_i:03d}-of-{num_shards:03d}-part-{part:04d}.parquet")
        write_parquet(pa.table(rows, schema=SCHEMA), dest)
        part += 1
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"[worker] [{s + len(group)}/{len(todo)}] L={group[0]['L']}-{group[-1]['L']} "
              f"{dt:6.1f}s {ntok / dt:7.0f} tok/s unfinished={n_unfinished}/{n_total} "
              f"-> {dest} (elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

    print(f"[worker] DONE shard {shard_i}/{num_shards}: {len(todo)} proteins in "
          f"{(time.time() - t0) / 60:.1f} min | prompt_think_count={prompt_think_count} "
          f"| unfinished {n_unfinished}/{n_total} "
          f"({100 * n_unfinished / max(n_total, 1):.2f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
