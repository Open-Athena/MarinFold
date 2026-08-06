# Summary slides — exp: optimize 1.5B contacts-v1 AFDB + ESM models

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Optimize the 1.5B no-crops contacts-v1 model trained on AFDB and ESM-Atlas data. The first milestone is a ten-step DCLM nano smoke test of MarinFold's package-only Marin training path on a `v6e-4` in `europe-west4`.

## Why

The protein sweep is expensive. A minimal run first verifies Iris dispatch, the packaged Marin stack, regional storage routing, TPU availability, and checkpoint logging without coupling the experiment to the sibling Marin source checkout.

## Data safety

The smoke script adopts the completed regional Llama-3-tokenized DCLM cache as an immutable external artifact. It contains no tokenization recipe, so missing data causes failure rather than retokenization.

## Results so far

The DCLM preflight passed (1,024 complete Llama-3-tokenized shards), and the graph contained only an adopted cache plus training. Iris launched `run_levanter_train_lm-ad08751d`; the runtime confirmed a `v6e-4` and `auto_build_caches=False`. The run stopped before step 0 because the authenticated account lacks Models write access to `open-athena`. Grant that permission, then rerun unchanged.
