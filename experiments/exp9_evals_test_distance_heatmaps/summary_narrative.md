# Summary slides — zero-shot distance heatmap eval on 10 random test proteins (1B model)

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

A small, interactive, runnable check of how the 1B MarinFold model
predicts CA-CA distance maps on 10 random held-out AFDB structures
from the `contacts-and-distances-v1-5x` test split. Anyone training a
new checkpoint can drop it in and see how it stacks up.

Four variants run on the same 10 proteins:
V1 zero-shot heatmaps; V2 beam-2 search over long-range GT contacts;
V3 pure-greedy search over all three contact ranges; V4 directed
search by predicted CB-CB distance (no per-candidate inference,
LDDT-CB only).

## Why

Existing eval surfaces are either the training-time tagged loss on
13 fixed PDB targets, or a one-PDB offline distogram script. This is
the smallest useful "did the model learn the test distribution?"
check. Hypothesis: zero-shot CA-CA MAE comparable to FoldBench
(~3-4 Å); heatmaps look like noisy versions of GT — diagonal correct,
off-diagonal placed roughly, detail degrades at long range.

## Results — zero-shot (V1)

**Macro CA-CA MAE = 3.29 Å** across 10 AFDB proteins (40,442 pairs),
on RTX A5000, ~5 min end-to-end. Per-protein MAE ranges 1.03 →
4.81 Å. Source CSV `data/per_protein_mae.csv`; per-protein heatmaps
in `plots/`.

## Results — seeded long-range only (V2)

Beam-2 search over GT long-range contacts (CB-CB ≤ 8 Å, sep ≥ 24),
target MAE < 1.0 Å, MAX_CONTACTS=30. **2 / 10 proteins cross the
target** (k=7 and k=18). The two crossers are the same two with the
best zero-shot MAE. "Hardness" tracks number of GT long-range
contacts.

## Results — seeded all ranges (V3)

Pure greedy, candidate pool opened to long + medium + short. **4 / 10
proteins cross the 1 Å target** (vs 2 / 10 under V2), and *every*
protein improves vs V2. Headline: letting the search pick from all
three contact ranges matters more than the choice of search
algorithm.

## Results — directed search by predicted CB-CB (V4)

Single full-matrix prediction per round; pick next contact by sorting
remaining candidates by current predicted CB-CB distance and taking
the largest. Tracks LDDT-CB only (no MAE column). About ⅔ the wall
time of V3 + LDDT replay. Per-protein LDDT-CB curves in
`plots/contact_directed_search_trace.png`.

## Conclusion

Zero-shot CA-CA MAE 3.29 Å is consistent with the ~3-4 Å hypothesis
and close to FoldBench-monomer (~3.6 Å). With all-range seeding,
4 / 10 proteins cross 1 Å MAE; the hardest cases need either >30
contacts or a stronger model. The two notebooks are the deliverable —
the future "did the new model learn?" check is: add a nickname to
`MODELS.yaml` and re-run them.
