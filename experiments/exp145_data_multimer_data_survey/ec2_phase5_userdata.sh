#!/bin/bash
# exp145 phase 5: interface residues for the 31,956 drawn candidates.
#
# One pass over the 441 val shards. The candidates are 3.6% of the corpus but they
# are spread across every shard, so the read is the whole cost: ~111 GB in-region,
# about 37 minutes on 8 workers, against ~40 minutes of single-core compute that
# the same 8 workers absorb in five. No data volume, because the output is one
# 32,000-row CSV rather than 45 GB of chains.
#
# Headless: no SSH, no credentials in the environment, log published to S3 every
# 5 minutes, and the box terminates itself. Artifacts are published before the
# strict completeness check runs, so a run that fails on a missing candidate still
# leaves its measurements behind to look at.
set -x
set -o pipefail
exec > /var/log/exp145-phase5.log 2>&1

B=marinfold-exp91-usw2
OUT=exp145/phase5-output
SRC=s3://$B/MarinFold/exp145-afdb-dimers/val/
cd /root

publish() {
  python3 - "$1" "$2" <<'PY' || echo "PUBLISH FAILED: $1"
import sys, pyarrow.fs as pafs
local, key = sys.argv[1], sys.argv[2]
fs = pafs.S3FileSystem(region=pafs.resolve_s3_region("marinfold-exp91-usw2"))
with fs.open_output_stream(key, compression=None) as h:
    h.write(open(local, "rb").read())
PY
}

die() {
  local stage="$1"
  echo "===== FAILED AT STAGE: $stage ====="
  df -h; free -g
  ls -la /root /root/data/curation 2>&1 | head -50
  tail -100 /var/log/exp145-phase5.log
  publish /var/log/exp145-phase5.log "$B/$OUT/phase5.log"
  touch /root/_marker
  publish /root/_marker "$B/$OUT/_FAILED_$stage"
  shutdown -h now
  exit 1
}

mkdir -p data/curation || die setup

apt-get update -qq || die apt_update
apt-get install -y -qq python3-pip || die apt_install
# pyarrow and numpy only. Do NOT install boto3: it wedges these boxes ~60s in (exp91).
pip3 install --quiet pyarrow numpy || die pip_deps

# compression=None is required: open_input_stream defaults to "detect" and would
# silently gunzip a .gz object.
python3 - <<'PY' || die stage_scripts
import pyarrow.fs as pafs
fs = pafs.S3FileSystem(region=pafs.resolve_s3_region("marinfold-exp91-usw2"))
data = fs.open_input_stream(
    "marinfold-exp91-usw2/exp145/staging/phase5_scripts.tar.gz", compression=None).read()
open("/root/scripts.tar.gz", "wb").write(data)
print("staged", len(data), "bytes")
PY
tar xzf scripts.tar.gz || die untar
for f in curate_phase5_interfaces.py curate_interface_residues.py \
         curate_extract_chains.py data/curation/phase5_candidates.csv.gz; do
  [ -f "$f" ] || die "missing_staged_$(basename "$f")"
done

( while true; do sleep 300; publish /var/log/exp145-phase5.log "$B/$OUT/phase5.log"; done ) &
HEARTBEAT=$!

# --max-absent 0: every candidate comes from a complex the fetch retrieved, so a
# candidate missing from the shards means the shards and the assignment disagree,
# which is a finding rather than a rounding error. The run still publishes first.
python3 -u curate_phase5_interfaces.py --source "$SRC" \
    --candidates data/curation/phase5_candidates.csv.gz \
    --workers "$(nproc)" --contact-angstroms 5.0 --min-interface 7 \
    --max-absent 0 --out data/curation/phase5_interfaces.csv
RC=$?

kill $HEARTBEAT 2>/dev/null
gzip -6 -c data/curation/phase5_interfaces.csv > /root/phase5_interfaces.csv.gz

publish /root/phase5_interfaces.csv.gz "$B/$OUT/phase5_interfaces.csv.gz"
publish data/curation/phase5_interface_summary.json "$B/$OUT/phase5_interface_summary.json"
[ -f data/curation/phase5_interface_absent.csv ] && \
  publish data/curation/phase5_interface_absent.csv "$B/$OUT/phase5_interface_absent.csv"
publish /var/log/exp145-phase5.log "$B/$OUT/phase5.log"

[ "$RC" -eq 0 ] || die "interfaces_rc$RC"

python3 - <<'PY' || die verify_published
import pyarrow.fs as pafs
fs = pafs.S3FileSystem(region=pafs.resolve_s3_region("marinfold-exp91-usw2"))
for name in ["phase5_interfaces.csv.gz", "phase5_interface_summary.json", "phase5.log"]:
    key = f"marinfold-exp91-usw2/exp145/phase5-output/{name}"
    info = fs.get_file_info(key)
    if info.type == pafs.FileType.NotFound or info.size == 0:
        raise SystemExit(f"{name} did not land in S3 ({info.type}, {info.size} bytes)")
    print(f"verified {name}: {info.size:,} bytes")
PY

touch /root/_marker && publish /root/_marker "$B/$OUT/_DONE_rc0"
shutdown -h now
