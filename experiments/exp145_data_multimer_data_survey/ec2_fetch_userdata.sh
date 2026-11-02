#!/bin/bash
# exp145 phase 2, full run: 881,010 val-eligible AFDB dimers into tar shards on S3.
#
# Runs unattended for ~21 h under the EC2 instance role, with no SSH and no
# credentials in the environment (short-lived STS would expire ~80 times over the
# course of this run). Progress is published to S3 every 5 minutes so the run is
# observable without a login, and a _DONE_rc<N> marker carries the exit code.
#
# Interrupt-safe: relaunching this same user-data lists the destination, skips
# every tar already written, and continues. At most one shard is redone.
set -x
exec > /var/log/exp145-fetch.log 2>&1

B=marinfold-exp91-usw2
OUT=exp145/fetch-output
DEST=s3://$B/MarinFold/exp145-afdb-dimers/val/
cd /root

apt-get update -qq
apt-get install -y -qq python3-pip
# pyarrow only. Do NOT install boto3: it wedges these boxes ~60s in (exp91).
pip3 install --quiet pyarrow

# Pull the staged scripts with the instance role. compression=None is required:
# open_input_stream defaults to "detect" and would silently gunzip the .gz object.
python3 - <<'PY'
import pyarrow.fs as pafs
fs = pafs.S3FileSystem(region=pafs.resolve_s3_region("marinfold-exp91-usw2"))
data = fs.open_input_stream(
    "marinfold-exp91-usw2/exp145/staging/scripts.tar.gz", compression=None).read()
open("/root/scripts.tar.gz", "wb").write(data)
print("staged", len(data), "bytes")
PY
tar xzf scripts.tar.gz
mkdir -p data/curation

publish() {  # copy a local file to the output prefix; used for progress and results
  python3 - "$1" "$2" <<'PY'
import sys, pyarrow.fs as pafs
local, key = sys.argv[1], sys.argv[2]
fs = pafs.S3FileSystem(region=pafs.resolve_s3_region("marinfold-exp91-usw2"))
with fs.open_output_stream(key, compression=None) as h:
    h.write(open(local, "rb").read())
PY
}
export -f publish

# Heartbeat: the run is 21 h, so a log that only appears at the end is useless.
( while true; do sleep 300; publish /var/log/exp145-fetch.log "$B/$OUT/fetch.log"; done ) &
HEARTBEAT=$!

python3 curate_fetch_structures.py --split val \
    --ids val_ids.csv.gz --out "$DEST" --workers 8
RC=$?

kill $HEARTBEAT 2>/dev/null
publish /var/log/exp145-fetch.log "$B/$OUT/fetch.log"
# AGENTS.md requires per-input timings to be kept; they live only on this box.
publish data/curation/fetch_timings.csv "$B/$OUT/fetch_timings.csv"
touch /root/_marker && publish /root/_marker "$B/$OUT/_DONE_rc$RC"
shutdown -h now
