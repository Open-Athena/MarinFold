# Runbook: fetching the val-eligible dimers on AWS

Phase 2 of [`CURATION_PLAN.md`](CURATION_PLAN.md). Fetches **881,010 val-eligible
complexes** (103 GB gzipped) from EBI into tar shards on S3.

No sequence collapse. PINDER clusters every system in its corpus rather than a
sequence-collapsed subset, and following PINDER is the decision recorded for this
dataset, so we fetch the whole val side. The collapse experiment and its numbers
stay in `val_cluster_summary.json` as a measured alternative, not as the path
taken.

## Shape of the job

| | |
|---|---|
| complexes | 881,010, from `dimer_split_assignment_id30_cov50.csv.gz` where `split == val` |
| transfer | 103 GB gzipped (117.0 kB/model measured over 8,012 models) |
| objects | 441 tars of 2,000 models each, ~234 MB per object, plus a sidecar JSON each |
| wall time | ~21 h at 11.5 files/s |
| cost | ~$2 EC2 + $0 ingress + ~$0.005 PUT + $2.37/mo storage |

**One instance, eight workers.** The pilot showed throughput linear in workers to
8 with no throttling, because `alphafold.ebi.ac.uk/files/` is GCS-backed. We do
not fan out across hosts: EBI publishes no rate limit, 11.5 files/s is the rate we
have demonstrated is safe, and 21 hours on a $0.09/h instance is not worth
multiplying our footprint against a public resource. The `--shard` / `--n-shards`
flags exist for the full-corpus case and are left at their defaults here.

## Launch

- **Instance**: `c7i.large` (2 vCPU, 4 GiB), on demand. The job is network-bound,
  not CPU-bound. One shard is assembled in memory before it is written, which is
  ~234 MB, so 4 GiB is ample.
- **Disk**: 20 GB root is enough. Structures stream to S3; nothing large is staged
  locally.
- **Region**: `us-west-2`, matching the account's existing
  `marinfold-exp91-usw2` bucket. Ingress from EBI is free from anywhere, so the
  region to pick is the one the data will be *read* from later, and exp91's data
  is already there.
- **IAM**: `--iam-instance-profile Name=marinfold-exp91-instance-profile`. This
  account cannot discover it (`iam:ListInstanceProfiles` is denied for
  `user/external-jsilterra`); the name comes from exp91's untracked
  `run_upload.local.sh`, which is why it is written down here. **Do not put credentials in the
  environment.** exp91's short-lived CloudShell STS creds expire in ~15 minutes,
  which will not survive a 21-hour run; the instance role auto-refreshes and is
  the only thing that works here.

**Never commit the credential file.** `aws_keys.env` is gitignored at the repo
root; a `git add -A` in an experiment directory will otherwise sweep it into a
commit.

## Bootstrap

Do **not** `pip install boto3`. It reliably wedges these boxes about 60 seconds in
and the instance self-terminates; exp91 lost two days to this. `pyarrow` installs
cleanly and picks up the instance role on its own.

```bash
sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip git
pip3 install --quiet pyarrow
git clone --depth 1 --branch exp145/multimer-data-survey \
    https://github.com/Open-Athena/MarinFold.git
cd MarinFold/experiments/exp145_data_multimer_data_survey
```

## Verified end to end on 2026-09-05

Instance `i-0792976cea76c8882`, `c7i.large` in us-west-2a, launched with
`ec2_smoke_userdata.sh` as user-data and no SSH key. It staged the scripts from
S3 with the instance role, fetched 25 val complexes (25/25 ok, 7.79 files/s),
wrote a 3.07 MB tar and its sidecar, read a member back to
`data_AF-0000000065760132-model_v1` with `chains: ['A', 'B']`, published its log
and a `_DONE_rc0` marker, and terminated itself.

Launch command, AMI `ami-07b3d2f97d89e29a4` (Ubuntu 22.04, us-west-2):

```bash
aws ec2 run-instances \
    --image-id ami-07b3d2f97d89e29a4 --instance-type c7i.large --count 1 \
    --subnet-id subnet-0a393c46ff06d81c3 \
    --iam-instance-profile Name=marinfold-exp91-instance-profile \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3,DeleteOnTermination=true}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=exp145-fetch}]' \
    --user-data file://ec2_smoke_userdata.sh
```

No key pair exists on this account, so everything runs from user-data and the
result is read from S3. `_DONE_rc<N>` carries the exit code; an instance that
terminates without a marker died before publishing, which is a failure to
investigate rather than a slow run.

## Step 1: smoke test one shard

Run this on the instance role before committing to 21 hours. The procedure was
validated from a workstation on 2026-09-05 against `s3://marinfold-exp91-usw2/`:
23 models written as a 2.7 MB tar plus a 330-byte sidecar, read back to valid
mmCIF, resume correctly skipping the existing shard. What it has *not* been run
against is an EC2 instance role, which is the only credential source that
survives 21 hours.

```bash
python3 curate_fetch_structures.py --split val \
    --ids data/curation/dimer_split_assignment_id30_cov50.csv.gz \
    --out s3://marinfold-exp91-usw2/MarinFold/exp145-afdb-dimers/val/ \
    --shard 0 --n-shards 40000 --shard-size 25 --workers 8
```

Expect `n_ok: 23, n_missing_404: 0, n_failed: 0` and two objects at the
destination. Then confirm a member is real mmCIF:

```bash
python3 -c "
import io, tarfile, gzip, pyarrow.fs as pafs
fs = pafs.S3FileSystem(region=pafs.resolve_s3_region('marinfold-exp91-usw2'))
raw = fs.open_input_stream('marinfold-exp91-usw2/MarinFold/exp145-afdb-dimers/val/shard_000_00000.tar').read()
t = tarfile.open(fileobj=io.BytesIO(raw))
name = t.getnames()[0]
text = gzip.decompress(t.extractfile(name).read()).decode()
print(name, text.splitlines()[0])
# label_asym_id is field 6 and both chains must appear. Field 7 is
# label_entity_id, legitimately '1' for both chains of a homodimer, so checking
# that one makes a correct two-chain file look like it has only one.
print('chains:', sorted({l.split()[6] for l in text.splitlines() if l.startswith('ATOM')}))
"
```

Delete those two test objects before step 2, or the real run will skip
`shard_000_00000.tar` and silently omit 2,000 models.

## Step 2: the full run

Headless, same shape as the smoke test but with `ec2_fetch_userdata.sh`. There is
no SSH into this account (no key pairs), so the run is driven entirely from
user-data and observed through S3. Three things that script does which the smoke
test did not:

- **Stages a compact ID list.** `val_ids.csv.gz` is `model_entity_id,kind,split`
  only, 881,010 rows in 2.6 MB, rather than the 26 MB split-assignment table.
- **Publishes the log to S3 every 5 minutes.** A 19-hour run whose log only
  appears at the end is not observable.
- **Publishes `fetch_timings.csv` at the end**, before `shutdown -h now`. It
  exists only on the instance root volume, which is deleted on termination.

```bash
tar czf scripts.tar.gz curate_fetch_structures.py val_ids.csv.gz
aws s3 cp scripts.tar.gz s3://marinfold-exp91-usw2/exp145/staging/scripts.tar.gz
aws ec2 run-instances --region us-west-2 \
    --image-id ami-07b3d2f97d89e29a4 --instance-type c7i.large \
    --subnet-id subnet-0a393c46ff06d81c3 \
    --iam-instance-profile Name=marinfold-exp91-instance-profile \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3}' \
    --instance-initiated-shutdown-behavior terminate \
    --user-data file://ec2_fetch_userdata.sh \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=exp145-fetch-full}]'
```

Clear the output prefix first. A stale tar in the destination is silently skipped
by the resume logic, which is correct on a rerun and wrong on a fresh run.

## A pyarrow trap worth knowing

`open_input_stream` and `open_output_stream` both default to `compression="detect"`
and act on the *file name*. Reading an object called `something.tar.gz` hands back
the decompressed bytes with no indication it happened, so writing them to a file of
the same name produces something `tar xzf` rejects. This broke the first EC2 smoke
test. Pass `compression=None` whenever you want the object's literal bytes;
`Destination.write` in the fetcher already does.

## Resume

Interrupt-safe at shard granularity. Rerunning the same command lists the
destination once, skips every tar already present, and continues. At most one
shard's work (2,000 models, ~3 minutes) is redone. A shard with a non-404 failure
after all five retries raises and is **not** written, so a partial tar can never
be mistaken for a complete one.

## What to check when it finishes

- **441 tars and 441 sidecar JSONs.** Fewer means a shard is missing, not that
  the run was short.
- **The 404 manifest.** Aggregate `missing_404` across the sidecars. The pilot saw
  28 in 2,000 in one ID band and 0 in 400 sampled across the whole confident set,
  so expect a small clustered count. This list ships with the dataset: the
  dataset is defined by what actually downloaded.
- **`fetch_timings.csv`.** `ec2_fetch_userdata.sh` publishes it to
  `exp145/fetch-output/` before shutting down; it exists nowhere else once the
  root volume goes. `AGENTS.md` requires per-input timings for every predictor and
  fetch run. At 880k rows it is too large for git, so it belongs beside the shards.
- **The instance is gone.** `--instance-initiated-shutdown-behavior terminate`
  plus `shutdown -h now` should leave nothing. Confirm with `describe-instances`
  and `describe-volumes` in us-west-2: a detached EBS volume bills whether or not
  an instance is attached to it. This account is shared with other projects, so
  filter by our region and tags rather than assuming every instance you see is
  ours.

## Run record: the full val fetch, 2026-09-05 to 2026-09-06

Instance `i-00dbfc381ec876cfc`, `c7i.large` in us-west-2a, launched 17:20 UTC
2026-09-05. Published `_DONE_rc0` and self-terminated at 12:13 UTC 2026-09-06:
**18.9 hours**, against a 21 h estimate.

880,248 of 881,010 fetched, 762 404s (0.09%), **zero non-404 failures**, 117
retries. 441 tars + 441 sidecars, 103.7 GiB, at
`s3://marinfold-exp91-usw2/MarinFold/exp145-afdb-dimers/val/`. Median shard rate
10.34 files/s (min 8.93, max 18.68).

Cost: $1.69 of compute, $0.004 of PUTs, $2.37/month of storage.

**84.2% of replies arrived uncompressed** despite `Accept-Encoding: gzip`, so the
real transfer was 396 GiB rather than the 103 GB projected. See phase 2 of
`CURATION_PLAN.md`. Those rows carry `http_status = -200`.

Nothing had to be restarted, and the resume path was never exercised on this run.
