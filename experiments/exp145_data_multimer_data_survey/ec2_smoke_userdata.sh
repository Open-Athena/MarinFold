#!/bin/bash
# exp145 phase 2 smoke test. Proves the EC2 *instance role* can write to S3, which
# is the one credential path the workstation test could not exercise.
# Everything lands in s3://marinfold-exp91-usw2/exp145/smoke-output/ ; the box
# terminates itself when done (shutdown behaviour is set to terminate).
set -x
exec > /var/log/exp145-smoke.log 2>&1

B=marinfold-exp91-usw2
OUT=exp145/smoke-output
cd /root

apt-get update -qq
apt-get install -y -qq python3-pip
# pyarrow only. Do NOT install boto3: it wedges these boxes ~60s in (exp91).
pip3 install --quiet pyarrow

# Pull the staged scripts using the instance role, via pyarrow rather than the
# aws CLI, which is not on this AMI.
python3 - <<'PY'
import pyarrow.fs as pafs
fs = pafs.S3FileSystem(region=pafs.resolve_s3_region("marinfold-exp91-usw2"))
# compression=None is required: open_input_stream defaults to "detect" and would
# transparently gunzip a .gz object, so the bytes written back out would be the
# decompressed tar under a .tar.gz name and `tar xzf` would fail.
data = fs.open_input_stream(
    "marinfold-exp91-usw2/exp145/staging/scripts.tar.gz", compression=None).read()
open("/root/scripts.tar.gz", "wb").write(data)
print("staged", len(data), "bytes")
PY
tar xzf scripts.tar.gz
# The fetcher appends per-file timings under --data-dir, which does not exist on
# a fresh box.
mkdir -p data/curation

python3 curate_fetch_structures.py --split val \
    --ids smoke_ids.csv \
    --out s3://$B/exp145/smoke-structures/ \
    --shard-size 25 --workers 8
RC=$?

# Read one member back, and check BOTH chains are present (label_asym_id, field 6).
python3 - <<'PY'
import io, tarfile, gzip, pyarrow.fs as pafs
fs = pafs.S3FileSystem(region=pafs.resolve_s3_region("marinfold-exp91-usw2"))
raw = fs.open_input_stream(
    "marinfold-exp91-usw2/exp145/smoke-structures/shard_000_00000.tar").read()
t = tarfile.open(fileobj=io.BytesIO(raw))
print("members:", len(t.getnames()))
name = t.getnames()[0]
text = gzip.decompress(t.extractfile(name).read()).decode()
print(name, text.splitlines()[0])
print("chains:", sorted({l.split()[6] for l in text.splitlines() if l.startswith("ATOM")}))
PY

# Publish the log and a terminal marker so the result is readable without SSH.
python3 - <<PY
import pyarrow.fs as pafs
fs = pafs.S3FileSystem(region=pafs.resolve_s3_region("$B"))
with fs.open_output_stream("$B/$OUT/smoke.log") as h:
    h.write(open("/var/log/exp145-smoke.log","rb").read())
with fs.open_output_stream("$B/$OUT/_DONE_rc$RC") as h:
    h.write(b"")
PY
shutdown -h now
