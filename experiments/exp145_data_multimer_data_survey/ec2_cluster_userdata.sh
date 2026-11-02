#!/bin/bash
# exp145 phase 3: cluster every val-side chain with one foldseek easy-cluster.
#
# Two stages on one box. Extraction streams the 441 tar shards out of S3 and writes
# one file per distinct UniProt accession (~42 GB); clustering then runs over that
# directory at TM >= 0.5. Both are single-instance work: clustering is linear in
# chains and fits in about a gigabyte, so there is nothing here to fan out.
#
# Headless: no SSH, no credentials in the environment, log published to S3 every
# 5 minutes, and the box terminates itself.
#
# Every failure is loud. There is no path through this script that ends in
# _DONE_rc0 without the cluster tsv and both summaries actually existing in S3;
# anything else publishes _FAILED_<stage> plus a directory listing, because a run
# that dies quietly costs a day of not knowing (exp91).
set -x
set -o pipefail
exec > /var/log/exp145-cluster.log 2>&1

B=marinfold-exp91-usw2
OUT=exp145/cluster-output
SRC=s3://$B/MarinFold/exp145-afdb-dimers/val/
CHAINS=/data/chains
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

# Bail out loudly: dump enough state into the log to diagnose without SSH, publish
# the log, mark the stage that failed, and stop. Never falls through to success.
die() {
  local stage="$1"
  echo "===== FAILED AT STAGE: $stage ====="
  df -h; free -g
  ls -la /root /data 2>&1 | head -50
  ls -la "$CHAINS" 2>/dev/null | head -5
  echo "chain files on disk: $(ls "$CHAINS" 2>/dev/null | wc -l)"
  ls -laR /data/cluster_work 2>/dev/null | head -40
  tail -100 /var/log/exp145-cluster.log
  publish /var/log/exp145-cluster.log "$B/$OUT/cluster.log"
  touch /root/_marker
  publish /root/_marker "$B/$OUT/_FAILED_$stage"
  shutdown -h now
  exit 1
}

mkdir -p /data "$CHAINS" data/curation || die setup

apt-get update -qq || die apt_update
apt-get install -y -qq python3-pip || die apt_install
# pyarrow only. Do NOT install boto3: it wedges these boxes ~60s in (exp91).
pip3 install --quiet pyarrow || die pip_pyarrow

# compression=None is required: open_input_stream defaults to "detect" and would
# silently gunzip the .gz object, which broke the first EC2 smoke test in phase 2.
python3 - <<'PY' || die stage_scripts
import pyarrow.fs as pafs
fs = pafs.S3FileSystem(region=pafs.resolve_s3_region("marinfold-exp91-usw2"))
data = fs.open_input_stream(
    "marinfold-exp91-usw2/exp145/staging/cluster_scripts.tar.gz", compression=None).read()
open("/root/scripts.tar.gz", "wb").write(data)
print("staged", len(data), "bytes")
PY
tar xzf scripts.tar.gz || die untar
for f in curate_extract_chains.py curate_foldseek_scaling.py foldseek_env.py \
         data/curation/dimer_split_assignment_id30_cov50.csv.gz \
         data/curation/fetch_missing_404.csv; do
  [ -f "$f" ] || die "missing_staged_$(basename "$f")"
done

( while true; do sleep 300; publish /var/log/exp145-cluster.log "$B/$OUT/cluster.log"; done ) &
HEARTBEAT=$!

# Stage 1: one chain file per accession onto the data volume, from S3 in-region,
# one worker per vCPU. The script owns the completeness check: it knows which
# models 404'd during the fetch and which accessions those block, and it writes
# every absent accession to /data/chains_absent.csv. The first attempt failed here
# on a hard-coded 858,360 that did not subtract the 762 unfetchable models, so
# there is deliberately no count in this file to get wrong.
python3 curate_extract_chains.py --source "$SRC" --split val \
    --assignment data/curation/dimer_split_assignment_id30_cov50.csv.gz \
    --missing data/curation/fetch_missing_404.csv \
    --workers "$(nproc)" --out "$CHAINS" || die extract

N=$(ls "$CHAINS" | wc -l)
echo "extracted $N chains"
publish /data/chains_absent.csv "$B/$OUT/chains_absent.csv"

# Stage 2: one clustering pass over all of them.
python3 curate_foldseek_scaling.py --chains "$CHAINS" --op cluster \
    --sizes "$N" --threads "$(nproc)" \
    --work /data/cluster_work --out data/curation --tag _full || die cluster

TSV=$(ls /data/cluster_work/n${N}_t*/res_cluster.tsv 2>/dev/null | head -1)
[ -n "$TSV" ] && [ -s "$TSV" ] || die missing_cluster_tsv
gzip -9 -c "$TSV" > /root/chain_cluster.tsv.gz || die gzip_tsv

kill $HEARTBEAT 2>/dev/null
# Every one of these must land. A missing artifact is a failure, not a skip.
publish /root/chain_cluster.tsv.gz          "$B/$OUT/chain_cluster.tsv.gz"
publish data/curation/foldseek_scaling_full.json "$B/$OUT/foldseek_scaling_full.json"
publish /data/chains_summary.json           "$B/$OUT/chains_summary.json"
publish /var/log/exp145-cluster.log         "$B/$OUT/cluster.log"

python3 - <<'PY' || die verify_published
import pyarrow.fs as pafs
fs = pafs.S3FileSystem(region=pafs.resolve_s3_region("marinfold-exp91-usw2"))
want = ["chain_cluster.tsv.gz", "foldseek_scaling_full.json", "chains_summary.json",
        "chains_absent.csv", "cluster.log"]
for name in want:
    key = f"marinfold-exp91-usw2/exp145/cluster-output/{name}"
    info = fs.get_file_info(key)
    if info.type == pafs.FileType.NotFound or info.size == 0:
        raise SystemExit(f"{name} did not land in S3 ({info.type}, {info.size} bytes)")
    print(f"verified {name}: {info.size:,} bytes")
PY

touch /root/_marker && publish /root/_marker "$B/$OUT/_DONE_rc0"
shutdown -h now
