# MarinFold eval strategy — recommendation (short version)

> One-page proposal for a train/val/test split. Full background, dataset
> tables, and references in [`eval-dataset-design.md`](eval-dataset-design.md).

## TL;DR

The training set is still mutable, so **structural cluster-holdout is the
primary leakage control** (stronger than any date filter):

1. **Foldseek-cluster everything once** — the AFDB training pool + the
   experimental eval candidates — into fold-level clusters.
2. **Choose val/test clusters** from the *experimental* structures
   (stratified for fold diversity; force-include designed proteins).
3. **Delete those clusters from the training set** and train on the rest.
   Train and test now share no fold *by construction* — no reliance on
   deposition dates.

Two invariants:

- **Test = experimental** structures (true ground truth). **Train =
  AFDB** (abundant, predicted). Testing against predicted structures
  would be circular, so the eval target is always experimental.
- **Time is a secondary, orthogonal check**, not the primary defense:
  tag a post-cutoff "blind" slice of test to mimic deployment and to
  catch any Foldseek mis-clustering.

Your instinct was right, just inverted: test is chosen first, then we
**purge its folds from train** (we remove from train, rather than move
the test item out).

## Why structure-holdout beats a date filter

- **Time alone overestimates the model.** A protein deposited last week
  can be the 1,000th instance of a fold we trained on constantly —
  provenance-clean but trivially easy. A date does not give novelty.
- **Holdout gives novelty directly.** Removing a fold's clusters from
  train *guarantees* the model never saw that fold — exactly the
  generalization question we care about.
- Time still earns its keep as a cheap, independent guarantee (it does
  not depend on clustering sensitivity) and as a realistic blind subset.

|  | Fold left in training | Fold held out of training |
|---|---|---|
| **Eval target** | **"seen-fold" eval** → memorization ceiling | **"novel-fold" test** → real generalization ← gold |

We deliberately keep *both*: with active holdout, the novel test is novel
by construction, so the seen-fold eval (experimental structures whose
fold we intentionally left in train) is what gives us the ceiling. The
gap between them = folding vs retrieval.

## Pipeline (pseudocode)

```python
# ---- Inputs ----
AFDB_ALL          # all AFDB structures available to train on (the pool)
PDB_EXPERIMENTAL  # experimental structures = true ground truth for eval
CUTOFF            # optional date for the "blind" temporal subset

# ---- Step 1: STRUCTURE clustering — one Foldseek pass over EVERYTHING --
clusters = foldseek_easy_cluster(AFDB_ALL + PDB_EXPERIMENTAL,
                                 alignment_type=1, coverage=0.6)
# clusters: structure_id -> cluster_representative_id (a "fold" bucket)

# ---- Step 2: choose eval clusters from the experimental structures ----
# Stratify for fold diversity + length; force-include designed-protein
# clusters and anything we specifically want to probe. These define the
# NOVEL eval set.
exp_clusters   = {clusters[s.id] for s in PDB_EXPERIMENTAL}
eval_clusters  = choose_holdout_clusters(exp_clusters,
                                         n_val=..., n_test=...,
                                         stratify="fold + length + designed")
val_clusters, test_clusters = partition(eval_clusters)
reserved = val_clusters | test_clusters

# ---- Step 3: build splits, cluster-disjoint by construction ----
test = [s for s in PDB_EXPERIMENTAL if clusters[s.id] in test_clusters]
val  = [s for s in PDB_EXPERIMENTAL if clusters[s.id] in val_clusters]

# THE KEY MOVE (enabled by a mutable train set): remove reserved folds.
train = [s for s in AFDB_ALL if clusters[s.id] not in reserved]
# train now shares NO fold-cluster with val/test -> leakage-free.

# ---- Step 4: seen-fold eval = memorization ceiling ----
# Experimental structures whose fold we LEFT in train. Easy by design;
# gap vs `test` quantifies folding-not-retrieval. Hold these aside so
# they aren't used for hyperparameter selection.
seen_fold_eval = [s for s in PDB_EXPERIMENTAL if clusters[s.id] not in reserved]

# ---- Step 5: optional temporal cross-check (orthogonal) ----
test_blind = [s for s in test if s.deposit_date > CUTOFF]

# ---- Step 6: designed-protein subset (same machinery) ----
designed = rcsb_query(struct_keywords="DE NOVO PROTEIN")
# its clusters are force-included in eval_clusters at Step 2; report as
# its own slice of `test`.
```

```bash
# The Foldseek call in Step 1, concretely. --alignment-type 1 = TM mode;
# tune -c (coverage) and cluster threshold to land near fold-level.
foldseek easy-cluster all_structures/ clusterRes tmp \
    --alignment-type 1 -c 0.6
# -> clusterRes_cluster.tsv : every chain -> its cluster representative.
```

> If you later want to train on experimental structures too, the same
> cluster map tells you which are safe: any experimental structure whose
> cluster is **not** reserved can be added to `train`.

## Two eval sets, one headline metric

Because we actively hold folds out of training, the two sets are defined
by *which side of the holdout* a structure's fold sits on:

- **Novel-fold test** — fold removed from training. The real
  generalization number.
- **Seen-fold eval** — fold deliberately left in training. The
  memorization ceiling.

Report both; the **gap** is the headline metric — it tells us whether
MarinFold is *folding* or *retrieving*. (Foldseek's nearest-training-
cluster TM-score also lets us sub-stratify the novel set by *how* far it
is from anything seen.)

## Priority special subset: designed proteins

Cheapest high-signal add-on. Pull the PDB `DE NOVO PROTEIN` class
(~2,007 entries, one API call) plus RFdiffusion deposits. These have no
evolutionary homologs, so it's a rare apples-to-apples comparison vs
MSA-based models, and most are outside AFDB's training universe. Run them
through the same cluster filter and report separately.

