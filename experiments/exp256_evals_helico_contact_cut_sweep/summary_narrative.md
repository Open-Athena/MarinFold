# Summary slides — exp: how many contacts should we hand Helico? sweep the cut past top-L, up to every pair any rollout proposed

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Does handing Helico **more than top-L** MarinFold contacts improve folding
accuracy, and where does the lDDT-versus-k curve actually turn?

## Why

Two measurements point at a gap nobody has looked in.

**From [#254](https://github.com/Open-Athena/MarinFold/issues/254):** the 100
rollouts behind a MarinFold contact prediction collectively propose **92 % of
the true contacts** (union recall 0.923 all-range, 0.900 long-range) using only
~15.7×R distinct pairs. Ranking them by vote count recovers 0.52 at the R cut,
0.67 at 2R and 0.79 at 5R. There is a lot of true signal sitting between rank L
and rank 5L that the top-L cut throws away, and #254 established that no
pointwise re-ranking recovers it (best +0.0015, a tie).

**From helico's side:** the existing cut sweep only ever went *down* — top-L/5,
top-L/2, top-L — and is already flat at the top end (lDDT 0.480 / 0.508 / 0.513
on FoldBench; on natural-pooled and designed targets it reverses). Nobody has
run a cut above L.

Helico's `contact-list` conditioning marks unlisted pairs **UNKNOWN, not
ABSENT** (`src/helico/contacts.py`), so a longer list does not overwrite true
negatives — it only trades precision for recall. It was trained with precision
sampled down to `MIN_SAMPLED_PRECISION = 0.4`.

So the prediction is a **shallow interior optimum between L and 2L**: at 2R the
list is ~0.33 precision, just under Helico's training floor but close to it,
while recall rises 0.52 → 0.67. By 5L (precision ~0.16) and certainly at the
full union (**~6 % precision**, ~10× outside anything Helico saw in training)
lDDT should fall. If the curve is instead flat or still rising at 3L, the
"emit fewer, better contacts" framing that top-L encodes is wrong.

## Results so far

_(Fill in as results come in.)_
