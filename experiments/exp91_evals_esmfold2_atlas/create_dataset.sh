#!/usr/bin/env bash
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end driver for building the ESM Atlas distillation subset.
#
# Stages (pass one as $1; "all" chains prep-ref -> run -> manifest):
#   local-smoke  run the WHOLE chain locally on a tiny Atlas slice (no AWS): funnel
#                (scan->novelty->leakage->cluster->select) + materialize to parquet.
#                Needs mmseqs on PATH; reads the Atlas anonymously. Sanity check only.
#   prep-ref  extract the afdb-24M struct-cluster rep sequences from exp41's
#             MMseqs/Foldseek target DB to FASTA, upload to S3 (novelty ref; once).
#   smoke     launch a capped EC2 run (--limit) to validate the whole chain cheaply.
#   run       launch the full funnel on EC2; writes selected_manifest.csv to S3.
#   manifest  download the finished manifest locally.
#   all       prep-ref + run + manifest.
#
# Everything funnel-side runs in AWS us-west-2, co-located with the Atlas (no
# egress). See README "Building the distillation subset" + run_aws.py for details.
set -euo pipefail

# ============================ CONFIG (edit these) ============================
BUCKET="marinfold-exp91-usw2"             # S3 bucket you own in us-west-2 (name must start "marinfold")
REGION="us-west-2"
IAM_PROFILE="marinfold-exp91-instance-profile"   # EC2 instance profile wrapping role marinfold-exp91-instance-role
INSTANCE_TYPE="r7i.24xlarge"              # 96 vCPU / 768 GB: novelty is alignment-bound so 3x cores ~3x faster
                                          #   (~10h vs ~30h). Scan is capped to 32 workers + bounded Lance
                                          #   readahead so it stays well under 768 GB (see SCAN_WORKERS).
SMOKE_LIMIT="2000000"                     # rows scanned in the smoke run
VOLUME_GB="2000"                          # root EBS: full-scale .m8 + linclust scratch (~1-2 TB)
SMOKE_VOLUME_GB="300"                     # smoke doesn't need the big scratch volume

# --- optional SSH access (both empty = no SSH; the .live log heartbeat is the
#     default way to watch a run) -----------------------------------------------
# Requires an EC2 key pair + a security group opening :22, which in turn need IAM
# perms the current marinfold-exp91 policy does NOT grant (ec2:CreateKeyPair or
# ImportKeyPair, ec2:CreateSecurityGroup, ec2:AuthorizeSecurityGroupIngress). Ask
# the account admin to grant those or to provision a key pair + SG and hand you
# the names, then fill these in.
KEY_NAME=""                               # EC2 key pair name (SSH login)
SECURITY_GROUP_ID=""                      # sg-... allowing inbound :22 from your IP

# --- AFDB novelty reference (afdb-24M struct-cluster reps, via exp41) ----------
# The novelty stage drops Atlas proteins already represented in our training set
# (afdb-24M). The reference is the 1.33M struct-cluster representatives exp41
# isolated; their amino-acid sequences already live in that experiment's MMseqs/
# Foldseek target DB (dbtype 0), so prep-ref extracts them with `convert2fasta`.
# We deliberately do NOT pull `timodonnell/afdb-24M` (1.2 TB, and it has no
# sequence column — only mmCIF text) just to recover sequences.
EXP41_REP_DB="../exp41_evals_foldseek_train_similarity/db_full/db/targetDB"

# --- funnel thresholds (mirror funnel.py / pipeline.py defaults) --------------
# pLDDT is a QUALITY FLOOR, not the size dial. 0.70 = the paper's "high confidence"
# line (~320M survive here, before clustering). Keep it loose so we don't delete
# the sole member of a novel structural cluster; clustering + per-cluster rep
# selection (CLUSTER_ID / REPS_PER_CLUSTER below) sets the final published size and
# picks the best-pLDDT exemplar per cluster. Tighten CLUSTER_ID, not this, to shrink.
MIN_PLDDT="0.70"
MIN_PTM="0.50"
MAX_AFDB_SEQ_ID="0.40"                    # novelty: drop >=40% id to AFDB
EVAL_MAX_SEQ_ID="0.40"                    # leakage: drop >=40% id to eval set
CLUSTER_ID="0.40"                         # linclust identity (size dial)
REPS_PER_CLUSTER="1"

# --- mmseqs scaling (the novelty search is the heavy stage, not the scan) ------
# The first full run wedged because a single easy-search over ~150M survivors blew
# past RAM+disk. pipeline.py now searches the query set in chunks of this size and
# deletes each chunk's scratch immediately. SPLIT_MEMORY_LIMIT is an extra RAM cap.
QUERY_CHUNK_SEQS="5000000"                # survivor seqs per novelty/leakage chunk
SPLIT_MEMORY_LIMIT="180G"                 # cap mmseqs RAM (leaves ~76 GB on the 256 GB box for OS + id-sets)
SEARCH_SENSITIVITY="4.0"                  # mmseqs -s (5.7 default); 4.0 = faster, may miss a few near-40%-id hits
SCAN_WORKERS="32"                         # parallel scan procs. 64 OOM-wedged the 768 GB box (Lance S3
                                          #   read-ahead x per_residue_plddt marched MemAvailable to 0);
                                          #   pipeline.py now also caps batch/fragment readahead. 32 is the
                                          #   safety-margin choice (scan is minutes; novelty dominates runtime).

# --- mid-scale test run (validate scaling before a full run) ------------------
TEST_LIMIT="40000000"                     # ~40M rows (~20x the smoke, ~3M survivors)
TEST_INSTANCE_TYPE="r7i.8xlarge"          # same box class as the full run; test just scans fewer rows
TEST_VOLUME_GB="600"

# --- materialize (stage 3: decode selected reps' structures) ------------------
# Reads only the ~1.65 TB of blobs we need (Lance take), decodes to mmCIF, writes
# ~3.2 TB of parquet parts streamed to S3 (each part is a resume checkpoint). RAM
# is NOT the constraint here (per-worker ~1.5 GB), so a cheaper compute-optimized
# box beats the funnel's big-RAM r7i.
MATERIALIZE_INSTANCE_TYPE="c7i.24xlarge"  # 96 vCPU / 192 GB; decode is CPU + S3-I/O bound
MATERIALIZE_VOLUME_GB="4000"              # holds the full ~3.2 TB of parts locally + headroom
CHUNK_SIZE="20000"                        # reps per plan chunk / output part; matches the plan
                                          # already in S3. Memory is bounded by _release_memory()
                                          # per sub-batch + 64 workers (diagnostic: RAM floor ~48 GB
                                          # even at 20k), so smaller chunks are not needed for safety.
# Decode is S3-read-latency bound (a local probe: ~6 ms/structure decode vs ~200 ms
# take), so the 96 vCPUs sit ~97% idle waiting on S3 -> throughput scales with the
# number of concurrent workers, capped by RAM (each take buffers ~GBs of Lance pages).
# TAKE_BATCH is small so many workers fit; DECODE_WORKERS oversubscribes the vCPUs.
# The `probe` target measures the sweet spot before a full run.
# Probe (i-0e27544233032e562, c7i.24xlarge) @96 workers: take_batch 512->3005/s (126 GB RAM
# free), 256->2697/s; both fall off past 96 workers (512: 160->1835, 224->1323; 256: 160->1684,
# 224->1206, 288->960). One box saturates its S3/network bandwidth at ~96 workers (~3000/s,
# ~6.2h, safe RAM); more workers only add contention. Go faster by sharding across boxes
# (MAT_NUM_SHARDS), not by raising DECODE_WORKERS.
# 96 workers OOM-cascaded the 192 GB box ~halfway through the first chunk (Arrow/glibc
# memory creep x 96 synchronized workers). 64 completed parts cleanly in attempt 2, so
# 64 is the proven-safe count; combined with per-sub-batch _release_memory() + 10k chunks.
DECODE_WORKERS="64"                       # proven safe (attempt 2); 96 OOMs over sustained runs
TAKE_BATCH="512"                          # dense sorted window -> far less page-read waste than 2000
MAT_NUM_SHARDS="${MAT_NUM_SHARDS:-1}"     # boxes to fan chunks across (1 = single box, ~7h)
MAT_STAGGER_SECONDS="${MAT_STAGGER_SECONDS:-90}"  # gap between shard launches (desync waves)
MATERIALIZE_LIMIT="2000"                  # materialize-smoke: only this many reps
# --- probe (throughput sweep) knobs -------------------------------------------
PROBE_WORKERS="${PROBE_WORKERS:-96 160 224 288}"  # worker counts to sweep
PROBE_TAKE_BATCH="${PROBE_TAKE_BATCH:-256 512}"   # take/decode sub-batch sizes to sweep
PROBE_SECONDS="${PROBE_SECONDS:-90}"              # wall-clock window per config

# --- local-smoke knobs --------------------------------------------------------
LOCAL_SMOKE_LIMIT="50000"                 # Atlas rows scanned in the local smoke
LOCAL_WORK="${LOCAL_WORK:-_localsmoke}"   # scratch dir (gitignored); wiped each run
EVAL_SEQS_CSV="../exp65_evals_low_msa_depth_proteins/data/candidate_sequences.csv"
# ============================================================================

# Derived locations
STAGING="s3://${BUCKET}/exp91/staging"
OUT="s3://${BUCKET}/exp91/out"
SMOKE_OUT="s3://${BUCKET}/exp91/smoke"
TEST_OUT="s3://${BUCKET}/exp91/test"
STRUCT_OUT="s3://${BUCKET}/exp91/structures"          # materialized structure parquet parts
STRUCT_SMOKE_OUT="s3://${BUCKET}/exp91/structures_smoke"
MANIFEST_S3="${OUT}/selected_manifest.csv"            # produced by the funnel 'run'
AFDB_REF_S3="s3://${BUCKET}/refs/afdb_ref.fasta"
AFDB_LOCAL_DIR="${AFDB_LOCAL_DIR:-data/_afdb_ref}"     # local scratch (gitignored)
AFDB_FASTA="${AFDB_LOCAL_DIR}/afdb_ref.fasta"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

local_smoke() {  # whole pipeline on a tiny slice, no AWS — does it run + emit valid output?
  command -v mmseqs >/dev/null || { echo "[local-smoke] mmseqs not on PATH" >&2; exit 1; }
  local w="$LOCAL_WORK"
  rm -rf "$w"; mkdir -p "$w"
  echo "[local-smoke] building eval + tiny AFDB references"
  EVAL_SEQS_CSV="$EVAL_SEQS_CSV" SMOKE_DIR="$w" uv run python - <<'PY'
import os, pandas as pd
w = os.environ["SMOKE_DIR"]
df = pd.read_csv(os.environ["EVAL_SEQS_CSV"])
with open(f"{w}/eval_seqs.fasta", "w") as f:
    for r in df.itertuples():
        f.write(f">{r.stem}\n{r.sequence}\n")
# Tiny stand-in AFDB novelty reference (exercises the mmseqs search path). The
# real run uses prep-ref's full afdb_ref.fasta; for a local sanity check two
# arbitrary sequences are enough.
with open(f"{w}/afdb_ref.fasta", "w") as f:
    f.write(">ref1\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR\n")
    f.write(">ref2\nMSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWQPEVKLDLDTASSQLADDVYEVVLRVTVTASLGEETAFLCEVQQGGIFSI\n")
print(f"[local-smoke] eval seqs: {len(df)}, afdb ref: 2")
PY

  echo "[local-smoke] funnel (limit ${LOCAL_SMOKE_LIMIT})"
  uv run python pipeline.py --work-dir "$w/work" --stage all --limit "$LOCAL_SMOKE_LIMIT" \
    --afdb-ref "$w/afdb_ref.fasta" --eval-ref "$w/eval_seqs.fasta" --compute-plddt-std \
    --min-plddt "$MIN_PLDDT" --min-ptm "$MIN_PTM" \
    --max-afdb-seq-id "$MAX_AFDB_SEQ_ID" --eval-max-seq-id "$EVAL_MAX_SEQ_ID" \
    --cluster-id "$CLUSTER_ID" --reps-per-cluster "$REPS_PER_CLUSTER"

  # Materialize only a small, row-ordered slice of the manifest so the (heavy,
  # cross-region) structure_blob reads finish quickly; early-exit stops the scan
  # once those reps are found.
  echo "[local-smoke] mini manifest (first 30 reps) + materialize"
  SMOKE_DIR="$w" uv run python - <<'PY'
import os, pandas as pd
w = os.environ["SMOKE_DIR"]
m = pd.read_parquet(f"{w}/work/survivors_meta.parquet").head(30).copy()
m["cluster_id"] = m["protein_hash"]; m["cluster_size"] = 1
m.to_csv(f"{w}/mini_manifest.csv", index=False)
PY
  uv run python materialize.py full --manifest "$w/mini_manifest.csv" \
    --work-dir "$w/mat" --limit 30 --chunk-size 15 --workers 2 --batch-rows 5000

  SMOKE_DIR="$w" uv run python - <<'PY'
import os, glob, pandas as pd, gemmi
w = os.environ["SMOKE_DIR"]
parts = sorted(glob.glob(f"{w}/mat/parts/part_*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
ok = sum(1 for c in df.cif_content if len(gemmi.read_structure_string(c)[0][0]) > 0)
# afdb-24M-aligned names (entry_id/cif_content/seq_len/global_plddt/seq_cluster_id/split)
# plus our extras must all be present.
need = {"entry_id", "cif_content", "seq_len", "global_plddt", "seq_cluster_id",
        "split", "sequence", "seq_ok", "source"}
assert need <= set(df.columns), f"schema mismatch: missing {need - set(df.columns)}"
assert (df.split == "train").all(), "split should be constant 'train'"
assert ok == len(df), f"only {ok}/{len(df)} CIFs parse"
assert bool(df.seq_ok.all()), f"{(~df.seq_ok).sum()} reps failed seq_ok"
print(f"[local-smoke] OK: {len(df)} structures across {len(parts)} parts, "
      f"schema + all CIFs valid + seq_ok all True -> {w}/mat/parts/")
PY
}

prep_ref() {
  # Extract the afdb-24M struct-cluster rep sequences (exp41's target DB, dbtype 0)
  # to FASTA and upload as the novelty reference. Run once; ~1.33M seqs, ~350 MB.
  command -v mmseqs >/dev/null || { echo "[prep-ref] mmseqs not on PATH" >&2; exit 1; }
  [ -e "$EXP41_REP_DB" ] || {
    echo "[prep-ref] rep DB not found: ${EXP41_REP_DB}" >&2
    echo "[prep-ref] sync exp41's db_full first (see exp41 README)" >&2
    exit 1; }
  mkdir -p "$AFDB_LOCAL_DIR"
  echo "[prep-ref] mmseqs convert2fasta ${EXP41_REP_DB} -> ${AFDB_FASTA}"
  mmseqs convert2fasta "$EXP41_REP_DB" "$AFDB_FASTA"
  echo "[prep-ref] extracted $(grep -c '^>' "$AFDB_FASTA") sequences"

  echo "[prep-ref] uploading -> ${AFDB_REF_S3}"
  aws s3 cp "$AFDB_FASTA" "$AFDB_REF_S3" --region "$REGION"
  echo "[prep-ref] done. Novelty reference ready at ${AFDB_REF_S3}"
}

launch() {  # $1 = output prefix, remaining args appended to run_aws.py
  local out_prefix="$1"; shift
  uv run python run_aws.py \
    --s3-staging "$STAGING" \
    --s3-output  "$out_prefix" \
    --afdb-ref-uri "$AFDB_REF_S3" \
    --iam-instance-profile "$IAM_PROFILE" \
    --region "$REGION" \
    --instance-type "$INSTANCE_TYPE" \
    --volume-size-gb "$VOLUME_GB" \
    ${KEY_NAME:+--key-name "$KEY_NAME"} \
    ${SECURITY_GROUP_ID:+--security-group-id "$SECURITY_GROUP_ID"} \
    --min-plddt "$MIN_PLDDT" --min-ptm "$MIN_PTM" \
    --max-afdb-seq-id "$MAX_AFDB_SEQ_ID" --eval-max-seq-id "$EVAL_MAX_SEQ_ID" \
    --cluster-id "$CLUSTER_ID" --reps-per-cluster "$REPS_PER_CLUSTER" \
    --query-chunk-seqs "$QUERY_CHUNK_SEQS" \
    --scan-workers "$SCAN_WORKERS" \
    ${SPLIT_MEMORY_LIMIT:+--split-memory-limit "$SPLIT_MEMORY_LIMIT"} \
    ${SEARCH_SENSITIVITY:+--search-sensitivity "$SEARCH_SENSITIVITY"} \
    --compute-plddt-std --watch "$@"
}

smoke() {
  echo "[smoke] capped run (--limit ${SMOKE_LIMIT}) -> ${SMOKE_OUT}"
  # Trailing --volume-size-gb overrides the full-run VOLUME_GB (argparse: last wins).
  launch "$SMOKE_OUT" --limit "$SMOKE_LIMIT" --volume-size-gb "$SMOKE_VOLUME_GB"
  echo "[smoke] check: aws s3 ls ${SMOKE_OUT}/"
}

test-run() {
  # Mid-scale validation before committing to the full 1.1B run: ~40M rows on a
  # smaller box, so the chunked novelty search + logging get exercised at a scale
  # where the old code would already have thrashed. Watch pipeline.log.live.
  echo "[test] ${TEST_LIMIT} rows on ${TEST_INSTANCE_TYPE} -> ${TEST_OUT}"
  # Trailing overrides win (argparse: last wins) over the full-run defaults.
  launch "$TEST_OUT" --limit "$TEST_LIMIT" \
    --instance-type "$TEST_INSTANCE_TYPE" --volume-size-gb "$TEST_VOLUME_GB"
  echo "[test] live log: aws s3 cp ${TEST_OUT}/pipeline.log.live -"
}

run() {
  echo "[run] full funnel -> ${OUT}"
  launch "$OUT"
  echo "[run] manifest -> ${OUT}/selected_manifest.csv"
}

manifest() {
  aws s3 cp "${OUT}/selected_manifest.csv" ./selected_manifest.csv --region "$REGION"
  echo "[manifest] saved ./selected_manifest.csv ($(wc -l < selected_manifest.csv) lines)"
}

launch_materialize() {  # $1 = output prefix, remaining args appended to run_aws.py
  local out_prefix="$1"; shift
  uv run python run_aws.py --task materialize \
    --s3-staging "$STAGING" \
    --s3-output  "$out_prefix" \
    --manifest-uri "$MANIFEST_S3" \
    --iam-instance-profile "$IAM_PROFILE" \
    --region "$REGION" \
    --instance-type "$MATERIALIZE_INSTANCE_TYPE" \
    --volume-size-gb "$MATERIALIZE_VOLUME_GB" \
    --chunk-size "$CHUNK_SIZE" \
    --scan-workers "$SCAN_WORKERS" \
    --decode-workers "$DECODE_WORKERS" \
    --take-batch "$TAKE_BATCH" \
    ${KEY_NAME:+--key-name "$KEY_NAME"} \
    ${SECURITY_GROUP_ID:+--security-group-id "$SECURITY_GROUP_ID"} \
    "$@"
}

probe() {
  # Measure decode throughput + RAM floor on the real target box before a full run:
  # sweep (workers x take_batch), each for PROBE_SECONDS, reading the plan already in
  # S3 at STRUCT_OUT/plan. Writes probe.log(.live); self-terminates (1h cap). Cheap
  # (~15 min of one c7i.24xlarge). No --watch (probe emits probe.log, not _DONE).
  echo "[probe] sweep workers=[$PROBE_WORKERS] x take_batch=[$PROBE_TAKE_BATCH] -> ${STRUCT_OUT}/probe.log"
  uv run python run_aws.py --task probe \
    --s3-staging "$STAGING" \
    --s3-output  "$STRUCT_OUT" \
    --iam-instance-profile "$IAM_PROFILE" \
    --region "$REGION" \
    --instance-type "$MATERIALIZE_INSTANCE_TYPE" \
    --volume-size-gb 100 \
    --probe-workers $PROBE_WORKERS \
    --probe-take-batch $PROBE_TAKE_BATCH \
    --probe-seconds "$PROBE_SECONDS" \
    ${KEY_NAME:+--key-name "$KEY_NAME"} \
    ${SECURITY_GROUP_ID:+--security-group-id "$SECURITY_GROUP_ID"}
  echo "[probe] live log: aws s3 cp ${STRUCT_OUT}/probe.log.live -"
}

materialize() {
  local n="$MAT_NUM_SHARDS"
  if [ "$n" -le 1 ]; then
    echo "[materialize] decode all reps in ${MANIFEST_S3} -> ${STRUCT_OUT} (1 box, ~7h)"
    launch_materialize "$STRUCT_OUT" --watch
  else
    # Shard the 3,338 plan chunks across N boxes (chunk_id %% N). Each box owns a
    # disjoint chunk set, writes non-colliding part_*.parquet into the SAME parts/
    # prefix, and self-terminates writing _DONE_shard_<i>. No --watch (can't watch N);
    # monitor with: aws s3 ls ${STRUCT_OUT}/ | grep _DONE_shard. Per-box EBS only needs
    # ~3.2 TB / N of local part storage. A failed shard is a resumable relaunch of that id.
    local vol=$(( 3200 / n + 400 ))
    echo "[materialize] sharding across ${n} boxes -> ${STRUCT_OUT} (~$(( 7 / n ))-$(( 14 / n ))h wall), ${vol} GB EBS each"
    for i in $(seq 0 $(( n - 1 ))); do
      echo "[materialize] launching shard ${i}/${n}"
      launch_materialize "$STRUCT_OUT" \
        --mat-num-shards "$n" --mat-shard-id "$i" --volume-size-gb "$vol"
      # Stagger starts so the boxes do not all hit S3 (and finish their equal-size
      # chunks) in lockstep -- desynchronizes the read + heartbeat-sync waves. Steady
      # -state concurrency is unchanged; this only smooths the transients.
      if [ "$i" -lt $(( n - 1 )) ]; then sleep "$MAT_STAGGER_SECONDS"; fi
    done
    echo "[materialize] all ${n} shards launched. Done when ${n} markers exist:"
    echo "               aws s3 ls ${STRUCT_OUT}/ | grep _DONE_shard"
    echo "[materialize] per-shard live logs: ${STRUCT_OUT}/pipeline_shard_<i>.log.live"
    echo "[materialize] parts: aws s3 ls ${STRUCT_OUT}/parts/ | wc -l"
    return
  fi
  echo "[materialize] live log: aws s3 cp ${STRUCT_OUT}/pipeline.log.live -"
  echo "[materialize] parts:    aws s3 ls ${STRUCT_OUT}/parts/ | wc -l"
}

materialize_diag() {
  # Bounded, heavily-instrumented single-box run to confirm the S3-read-timeout fix
  # stops the wedge (2- and 4-box runs both wedged ~10 min in: CPU->1%, NetworkIn->0,
  # workers blocked forever in ds.take with no timeout). 1 box x DECODE_WORKERS at the
  # real load, --log-io (per-sub-batch take/decode timing) + always-on SLOW-take warns,
  # --max-chunks so it stops after ~DIAG_MAX_CHUNKS (self-terminates). Writes REAL parts
  # to STRUCT_OUT (they count as progress + resume); reads the existing plan there.
  # Watch: does CPU stay up past ~15 min, do chunks complete, do SLOW takes recover?
  local mc="${DIAG_MAX_CHUNKS:-200}"
  echo "[materialize-diag] 1 box, ${DECODE_WORKERS} workers, take_batch ${TAKE_BATCH}, --log-io, max ${mc} chunks -> ${STRUCT_OUT}"
  uv run python run_aws.py --task materialize \
    --s3-staging "$STAGING" --s3-output "$STRUCT_OUT" --manifest-uri "$MANIFEST_S3" \
    --iam-instance-profile "$IAM_PROFILE" --region "$REGION" \
    --instance-type "$MATERIALIZE_INSTANCE_TYPE" --volume-size-gb 300 \
    --chunk-size "$CHUNK_SIZE" --scan-workers "$SCAN_WORKERS" \
    --decode-workers "$DECODE_WORKERS" --take-batch "$TAKE_BATCH" \
    --mat-max-chunks "$mc" --log-io \
    ${KEY_NAME:+--key-name "$KEY_NAME"} \
    ${SECURITY_GROUP_ID:+--security-group-id "$SECURITY_GROUP_ID"}
  echo "[materialize-diag] live log: aws s3 cp ${STRUCT_OUT}/pipeline.log.live -"
  echo "[materialize-diag] watch SLOW takes: ... | grep -E 'SLOW take|chunk [0-9]{5}:'"
}

materialize_smoke() {
  # Cheap cloud validation of the materialize path: decode only MATERIALIZE_LIMIT
  # reps on a small box, so plan + take + decode + checkpointing + self-terminate
  # all get exercised for ~$1 before the full ~3.2 TB run. Trailing flags win.
  echo "[materialize-smoke] ${MATERIALIZE_LIMIT} reps -> ${STRUCT_SMOKE_OUT}"
  launch_materialize "$STRUCT_SMOKE_OUT" --limit "$MATERIALIZE_LIMIT" \
    --instance-type r7i.2xlarge --volume-size-gb 100 --decode-workers 4
  echo "[materialize-smoke] live log: aws s3 cp ${STRUCT_SMOKE_OUT}/pipeline.log.live -"
}

case "${1:-}" in
  local-smoke)       local_smoke ;;
  prep-ref)          prep_ref ;;
  smoke)             smoke ;;
  test)              test-run ;;
  run)               run ;;
  manifest)          manifest ;;
  materialize)       materialize ;;
  materialize-smoke) materialize_smoke ;;
  materialize-diag)  materialize_diag ;;
  probe)             probe ;;
  all)               prep_ref; run; manifest ;;
  *) echo "usage: $0 {local-smoke|prep-ref|smoke|test|run|manifest|materialize|materialize-smoke|materialize-diag|probe|all}" >&2; exit 2 ;;
esac
