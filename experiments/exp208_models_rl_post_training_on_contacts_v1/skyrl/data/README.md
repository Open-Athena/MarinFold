# exp208 SkyRL datasets

`skyrl_train_2k.parquet` (4 MB) is **committed**: it is the training input to ten
of the eleven scored runs in [../../RESULTS.md](../../RESULTS.md), so committing it
makes those results reproducible from the repo alone.

`skyrl_train_10k.parquet` (20 MB) is **not committed**. It was the input to arm C v4
only — the run that diverged (KL 3.96) and was stopped at step 270 — and it is
exactly regenerable:

```
python ../build_dataset.py --out skyrl_train_10k.parquet --n 10000
```

`build_dataset.py` takes the first N targets from exp200's pool in order, so the
generation is deterministic and the first 2,000 rows of the 10k file are byte-wise
the 2k file. Regenerating needs GCS read access to
`gs://marin-us-central1/protein-structure/MarinFold/exp200/train/`.

`skyrl_smoke.parquet` (86 KB) is a small slice for launch smoke tests.
