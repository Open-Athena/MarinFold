# Summary slides — Foldseek train-set similarity tool

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Building reusable code (issue #41) that answers, for any candidate protein
structure: how structurally close is it to anything in MarinFold's training
set? Candidates are FoldBench monomers, de novo designs, and low-MSA natural
proteins we may want in an eval set.

## Why

The training set (afdb-24M) is already Foldseek-clustered: every structure
has a struct_cluster_id and a train/val/test split hashed from it. What we
lacked was code to take an EXTERNAL candidate and map it back onto those
clusters. The mechanism: foldseek easy-search the candidate against a DB of
the training-set cluster representatives (TM-score), then read off the
nearest train representative, its TM, and its split.

## How it works

query_similarity.py: candidate cifs -> foldseek easy-search (alignment-type
1 = TM-align) -> join each hit's representative to its split -> per-candidate
CSV with the nearest training rep, its TM-score, the free sequence-identity
signal, and a verdict (redundant >= 0.9, same_fold >= 0.5, else novel_fold).
build_db_modal.py builds the representative DB + split manifest on Modal
(all 12,005 afdb-24M shards); fetch_db.py pulls it local.

## Full DB built, real measurement

The full representative DB is built (build_db_modal.py on Modal, all 12,005
afdb-24M shards): 1,331,330 cluster representatives, a 2.6 GB foldseek DB,
splits 0.980/0.0099/0.0099 (matches the ~0.98/0.01/0.01 split-hash). Querying
the FoldBench-100 monomers against the full 1.33M-rep training set: 48
redundant (qtm>=0.9), 51 same_fold, 1 novel_fold. So 99/100 FoldBench
monomers fall in a fold MarinFold trained on; the lone novel is 7xcd_A
(qtm 0.485). Median nearest-train qtmscore is 0.895. Unit + real-foldseek
integration tests pass.

## Results: near-total structural overlap

99/100 FoldBench monomers have a same-fold-or-closer match in training, and
48 are structurally near-identical (qtm>=0.9). The prototype's 229-rep slice
had reported 68 novel / 32 same -- that "novelty" was a sparse-DB artifact,
which is why the full build mattered. Crucially the overlap is structural,
not sequence: 65 of the 99 matches sit below 0.30 sequence identity to their
nearest rep. A sequence-only leakage filter would clear most of these as
"novel" while structurally they are near-duplicates of trained folds.

## Next

Optional: upload the compact DB + manifest to the HF bucket so others fetch
it without rebuilding. Then reuse the same tool to score de novo designs and
low-MSA natural proteins, and sub-stratify any eval-set candidate pool by its
nearest-train qtmscore (median here 0.895) per .dev/eval-strategy-summary.md.
