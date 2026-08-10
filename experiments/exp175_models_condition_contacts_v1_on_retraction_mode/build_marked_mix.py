# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the 50:50 training mix from the *fixed* corpus, cloud-side (#175).

#160's mix was built from the corpus whose closing flush appended ground truth
in sorted order (#159's flush bug), so every accuracy number it produced is
void. This rebuilds it from the regenerated `flush="shuffled"` corpus and marks
the backtracking half in one pass:

1. read the 4,096 generation parts from **CoreWeave S3**,
2. consolidate into training-sized shards,
3. draw an equal number of clean ESM-Atlas documents from **disjoint proteins**,
4. stamp the backtracking half's token 0 with ``<contacts-v1.backtracking>``,
5. write to **GCS us-east5**, where the training job reads.

Runs on a **marin CPU pod**: it is the one place that can reach both stores —
CoreWeave S3 with the injected credentials, and GCS natively. The workstation
would have to pull ~3 GB down and push ~3 GB back up at ~2.9 MB/s.

It **streams**. The pool is ``e2-highmem-2`` (2 vCPU, 16 GB), so holding 1M
documents to shuffle them is not an option — asking for 64 GB simply comes back
``unschedulable``. Instead: one pass for entry_ids alone (to filter proteins),
then both halves read in tandem 1:1 into a bounded shuffle buffer that drains a
shard at a time. Training shuffles again anyway, so a local shuffle is enough.

Credentials go in as job env vars and the launcher never prints argv, since
``CalledProcessError`` would render them into a traceback.

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    uv run --no-project python build_marked_mix.py
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

MARIN = Path(os.environ.get("MARIN_CHECKOUT", "/home/bizon/git/marin-freshiris"))
IRIS = os.environ.get("IRIS_BIN", str(MARIN / ".venv/bin/iris"))

BT_S3 = ("s3://marin-us-east-02a/protein-structure/MarinFold/"
         "exp159_backtracking_shuffled/documents")
CLEAN_GCS = ("gs://marin-us-east5/protein-structure/MarinFold/"
             "exp160_backtracking_training/corpus/train")
OUT_GCS = ("gs://marin-us-east5/protein-structure/MarinFold/"
           "exp175_backtracking_mode/corpus_v2/train")

POD_SCRIPT = r'''
import os, sys, random
import gcsfs, s3fs, pandas as pd

bt_s3, clean_gcs, out_gcs, marked_token, plain_token = sys.argv[1:6]
DOCS_PER_SHARD = 64_000
BUFFER = 192_000          # bounded shuffle buffer: ~3 shards, ~250 MB

s3 = s3fs.S3FileSystem(endpoint_url="https://cwobject.com",
                       key=os.environ["CW_KEY_ID"], secret=os.environ["CW_KEY_SECRET"],
                       config_kwargs={"s3": {"addressing_style": "virtual"}})
gcs = gcsfs.GCSFileSystem()
rng = random.Random(0)

parts = sorted(p for p in s3.find(bt_s3[len("s3://"):]) if p.endswith(".parquet"))
clean_shards = sorted(gcs.glob(f"{clean_gcs}/*.parquet"))
print(f"[mix] {len(parts)} backtracking parts, {len(clean_shards)} clean shards", flush=True)

# Pass 1: entry_ids only, so the disjointness filter needs no document text.
bt_ids = set()
for i, p in enumerate(parts):
    with s3.open(p, "rb") as fh:
        bt_ids |= set(pd.read_parquet(fh, columns=["entry_id"]).entry_id)
    if (i + 1) % 1000 == 0:
        print(f"[mix]   ids {i+1}/{len(parts)} ({len(bt_ids):,})", flush=True)
print(f"[mix] {len(bt_ids):,} backtracking proteins", flush=True)

def clean_stream():
    """Yield clean documents from #160's mix, skipping backtracking proteins."""
    for key in clean_shards:
        with gcs.open(key, "rb") as fh:
            df = pd.read_parquet(fh)
        df = df[(df.kind == "clean") & (~df.entry_id.isin(bt_ids))]
        for row in df[["entry_id", "document", "num_tokens"]].itertuples(index=False):
            yield {"entry_id": row.entry_id, "document": row.document,
                   "num_tokens": row.num_tokens, "kind": "clean"}

clean_it = clean_stream()
buf, shard_i = [], 0
totals = {"documents": 0, "backtracking": 0, "clean": 0, "tokens": 0, "silent": 0}

def flush(force=False):
    """Write whole shards out of the shuffled buffer."""
    global buf, shard_i
    while len(buf) >= (DOCS_PER_SHARD if not force else 1):
        rng.shuffle(buf)
        chunk, buf = buf[:DOCS_PER_SHARD], buf[DOCS_PER_SHARD:]
        out = f"{out_gcs}/shard-{shard_i:05d}.parquet"
        with gcs.open(out, "wb") as fh:
            pd.DataFrame(chunk).to_parquet(fh, index=False, compression="zstd")
        print(f"[mix]   wrote {out}: {len(chunk):,}", flush=True)
        shard_i += 1
        if force and not buf:
            return

# Pass 2: stream both halves in tandem, 1:1, into a bounded shuffle buffer.
for i, p in enumerate(parts):
    with s3.open(p, "rb") as fh:
        df = pd.read_parquet(fh, columns=["entry_id", "document", "num_tokens"])
    # Every document came from the backtracking engine, including the ~20% in
    # which the trigger never fired. They ALL get the marker: it means "you may
    # retract", not "a retraction follows".
    for row in df.itertuples(index=False):
        assert row.document.startswith(plain_token + " "), "unexpected doc type"
        doc = marked_token + " " + row.document[len(plain_token) + 1:]
        buf.append({"entry_id": row.entry_id, "document": doc,
                    "num_tokens": row.num_tokens, "kind": "backtracking"})
        totals["backtracking"] += 1
        totals["silent"] += ("<retract>" not in doc)
        try:
            buf.append(next(clean_it))
            totals["clean"] += 1
        except StopIteration:
            pass
    if len(buf) >= BUFFER:
        flush()
    if (i + 1) % 500 == 0:
        print(f"[mix]   parts {i+1}/{len(parts)}  buffered {len(buf):,}", flush=True)

flush(force=True)
totals["documents"] = totals["backtracking"] + totals["clean"]
print(f"[mix] DONE {totals}", flush=True)
print(f"[mix] backtracking share "
      f"{totals['backtracking']/max(totals['documents'],1):.3f}", flush=True)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt-s3", default=BT_S3)
    ap.add_argument("--clean-gcs", default=CLEAN_GCS)
    ap.add_argument("--out-gcs", default=OUT_GCS)
    ap.add_argument("--job-name", default="exp175-build-mix-v2")
    ap.add_argument("--zone", default="us-east5-a")
    a = ap.parse_args()

    for var in ("CW_KEY_ID", "CW_KEY_SECRET"):
        if not os.environ.get(var):
            raise SystemExit(f"{var} not set — source ~/.config/marin/cw-rno2a.env")

    b64 = base64.b64encode(POD_SCRIPT.encode()).decode()
    bootstrap = f"""
set -euo pipefail
echo "[mix] host=$(hostname)"
echo {b64} | base64 -d > /tmp/build_mix.py
exec uv run --no-project --with gcsfs --with s3fs --with pandas --with pyarrow \\
    python /tmp/build_mix.py '{a.bt_s3}' '{a.clean_gcs}' '{a.out_gcs}' \\
    '<contacts-v1.backtracking>' '<contacts-v1>'
""".strip()

    command = [
        IRIS, "--cluster=marin", "job", "run",
        "--job-name", a.job_name, "--no-wait", "--enable-extra-resources",
        "--priority", "interactive", "--zone", a.zone,
        "--cpu", "2", "--memory", "12GB", "--disk", "32GB",
        "-e", "CW_KEY_ID", os.environ["CW_KEY_ID"],
        "-e", "CW_KEY_SECRET", os.environ["CW_KEY_SECRET"],
        "--", "bash", "-lc", bootstrap,
    ]
    print(f"[mix] {a.bt_s3}\n   +  {a.clean_gcs}\n   -> {a.out_gcs}")
    result = subprocess.run(command, cwd=MARIN, check=False)
    if result.returncode != 0:
        print(f"[mix] iris job run failed (exit {result.returncode}) — argv withheld "
              "because it carries CoreWeave credentials", file=sys.stderr)
        return result.returncode
    print(f"[mix] submitted /bizon/{a.job_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
