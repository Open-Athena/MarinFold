# Summary slides — exp: generate training dataset that includes pause tokens

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Implementing the training-data generator for
`contacts-and-distances-v2`: byte-identical to v1, except `<think>`
pause tokens are spliced in between statements. Only generation —
running at scale and training-on-it are deferred to follow-up
issues.

## Why

Pause tokens let a model spend extra forward-pass cycles without
conditioning on additional content tokens (Goyal et al. 2023,
arxiv 2310.02226). Loss-masking at `<think>` positions during
training lets the model use them as scratch space at inference.

## What changed vs v1

- Vocab grows by 2 (`<contacts-and-distances-v2>`, `<think>`),
  appended at the end so every v1 id stays stable — a v1
  checkpoint can be warm-started on v2 by growing the embedding
  table by 2 rows.
- Initial run after `<begin_statements>`: P=0.75 gate, `k1 ~ Geom(0.13)`.
- `max(int(Uniform(-4, 4)), 0)` additional runs at random slots
  between statements, each length `~ Geom(0.25)`.
- Total think-token cost is subtracted from the 8192 budget *before*
  statements are allocated — docs still fit and still end with `<end>`.

## Results so far

`uv run pytest tests/ -v -m 'not network'` → 44 passed. End-to-end
CLI smoke on a 60-residue synthetic chain produced a v2 doc with 15
`<think>` tokens, a 9-token initial run, and exact 8192-token
length. No follow-up data-gen or training has run yet — that's the
next experiment.
