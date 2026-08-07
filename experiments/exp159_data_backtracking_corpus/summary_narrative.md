# Summary slides — exp: build a contacts-v1 backtracking corpus by re-conditioning the base model on its own corrected contacts

<!-- Feeds plots/summary.pdf via build_summary.py.
     One `## ` heading per slide; body text becomes the slide.
     Keep this current as the experiment progresses. -->

## What we're doing

Can we build a contacts-v1 training corpus of coherent **self-correction traces** by generating with the base model in the loop — emitting contacts, retracting the wrong ones once the model's own posterior turns against them, and continuously re-conditioning the model on its currently-live (corrected) contact set?

## Why

The right way to synthesise retraction traces is **not** to post-hoc splice `<retract>` into a finished rollout — that leaves the whole tail of the document conditioned on a context that still contains the mistake and never contains the correction, i.e. off-distribution. Instead, generate incrementally and, after every retraction, **rebuild the base model's prompt from the live (post-retraction) contact set and continue**. Then every contact the model emits is conditioned on a coherent set, so all contacts stay on-distribution; the only synthetic tokens in the corpus are the `<retract>` statements themselves — exactly the behaviour we want to install. If, additionally, the *timing* of each retraction is driven by the base model's own collapsing posterior on the queued contact, the corrected-away contacts will be enriched for false positives **using a signal computable purely from context** — the property the trained model needs in order to learn *when* to retract.

## Results so far

_(Fill in as results come in.)_
