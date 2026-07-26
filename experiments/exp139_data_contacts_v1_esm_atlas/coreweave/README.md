# CoreWeave operational scripts (exp139)

The scripts that actually produced the published corpus. They run as CoreWeave
jobs on `cw-us-east-02a` in a **minimal env** (this dir's `pyproject.toml`:
`huggingface_hub>=1.5` + `s3fs` + `pyarrow`) — deliberately NOT the exp139 env,
because `marinfold -> transformers` pins `huggingface_hub<1.0`, which cannot
read/write `hf://buckets/...` paths.

Object-storage creds are auto-injected into task pods (`AWS_*` + `AWS_ENDPOINT_URL`);
only `HF_TOKEN` needs passing.

| Script | Role |
| --- | --- |
| `publish_to_bucket.py` | s3 `analyzed/` → the two published views on the HF bucket (one streaming pass, resumable). **Use ~4 workers** — HF rate-limits the bucket `xet-write-token` endpoint; 16 workers → sustained 429s. |
| `verify_and_dedupe.py` | Map every output part back to its input part, delete redundant/unmappable copies, assert full coverage. Run this **before publishing** — a killed job's zombie workers keep flushing outputs. |
| `copy_missing_parts.py` | Copy uncovered input parts to `source_missing/` so a resume can glob exactly them (zephyr's `{shard}` naming can't select a scattered set). |
| `corpus_stats.py` | Exact document / token / contact counts across all parts; also copies the tokenizer next to the corpus. |

Submit (from this dir, bundle must be <25 MB):

```bash
KUBECONFIG=~/.kube/coreweave-iris-gpu \
/home/bizon/git/marin-freshiris/.venv/bin/iris --cluster=cw-us-east-02a job run \
  --no-wait --enable-extra-resources --cpu 8 --memory 24GB --disk 64GB \
  -e HF_TOKEN "$HF_TOKEN" -e PUBLISH_WORKERS 4 \
  -- python publish_to_bucket.py
```

Note `--disk`: CoreWeave pods default to **5 Gi** ephemeral storage, which anything
staging files locally will blow past.
