# Summary slides — exp: Replace learned residue location tokens with position embedding

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Should we use a positional embedding like RoPe (rather than learned embeddings) for residue position tokens?

## Why

Intuitively, absolute positions of residues is not as meaningful as relative positions (distances). If we use an embedding designed to make it easy to compute distance, rather than learned per-position embeddings, it could improve model efficiency for two reasons: (1) fewer parameters to learn, and (2) possibly a more flexible and re-usable embedding

## Results so far

_(Fill in as results come in.)_
